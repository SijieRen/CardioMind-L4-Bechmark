"""
Main evaluation function and CLI interface for medical diagnosis evaluation.

This module provides the main entry point for the evaluation system,
including backwards compatibility with the original API.
"""

import json
import asyncio
import argparse
import concurrent.futures
from typing import Dict, Any, List
from tqdm import tqdm
from core.config import EvaluationConfig
from evaluators.base import MedicalDiagnosisEvaluator
from reporting.report_generator import create_evaluation_report, create_detailed_report
from reporting.report_generator import create_csv_report


async def evaluate_metric(emr: str, ref_json: Dict, pred_json: Dict, 
                         eval_mode: bool = False, eval_evidence: bool = False, 
                         eval_selection: bool = False, pipeline_parallel: bool = False, sample_count: int = 9) -> Dict[str, Any]:
    """
    Main evaluation function - refactored version
    
    Args:
        emr: Electronic Medical Record text
        ref_json: Reference/ground truth JSON
        pred_json: Predicted JSON  
        evaluate_all_types: Whether to evaluate all diagnosis types (SD + DD)
        eval_evidence: Whether to enable evidence evaluation
        eval_selection: Whether to enable selection evaluation
        
    Returns:
        Dictionary containing evaluation results
    """
    config = EvaluationConfig(
        enable_evidence_eval=eval_evidence,
        enable_selection_eval=eval_selection,
        sample_count=sample_count
    )
    
    evaluator = MedicalDiagnosisEvaluator(config)
    return await evaluator.evaluate_all(emr, ref_json, pred_json, eval_mode, pipeline_parallel=pipeline_parallel)


def load_and_validate_cases(jsonl_file: str,required_fields: List[str] = []) -> List[Dict[str, Any]]:
    """
    Load all cases from JSONL file and validate/supplement fields
    
    Args:
        jsonl_file: Path to input JSONL file
        
    Returns:
        List of validated cases
    """
    cases = []
    fixed_seqid_list = ["e225127b-79ef-4c38-86f7-7895954f5b0a", 
                        "796cab04-9882-40a3-8685-ae7b4d4a1566", "6ecb3c11-a360-4293-8603-3d3f4df1cb6c", 
                        "ce2d4e50-8a8b-4b87-a0f4-2f41d8579bbe", "d77c8d08-5c4c-40dc-9180-55a0656d8b15"]
    
    # First, count total lines for progress bar
    total_lines = sum(1 for _ in open(jsonl_file, 'r', encoding='utf-8'))
    
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        with tqdm(total=total_lines, desc="Loading and validating cases", unit="line") as pbar:
            for line_num, line in enumerate(f, 1):
                pbar.update(1)
                # Parse each line as JSON
                case = json.loads(line.strip())
                case_valid = True

                if case["seq_id"] not in fixed_seqid_list: 
                    pbar.set_postfix(status="skipped", seq_id=case.get("seq_id", "unknown"))
                    continue
                
                for field in required_fields:
                    if field not in case:
                        print(f"Warning: Missing field '{field}' in line {line_num}")
                        case_valid = False
                        break
                
                if not case_valid:
                    print(f"Skipping invalid case on line {line_num}")
                    pbar.set_postfix(status="invalid", line=line_num)
                    continue
                    
                cases.append(case)
                pbar.set_postfix(status="loaded", seq_id=case['seq_id'], total_valid=len(cases))
                print(f"✓ Loaded and validated case {case['seq_id']} (line {line_num})")
    
    print(f"Successfully loaded {len(cases)} valid cases from {jsonl_file}")
    return cases


def save_incremental_results(results: list, output_file: str, output_format: str, detailed: bool):
    """
    Save results incrementally to output file
    
    Args:
        results: List of evaluation results
        output_file: Path to output file
        output_format: Output format (json, text, csv)
        detailed: Whether to generate detailed report
    """
    try:
        if output_format == "json":
            # Save as JSON array
            with open(output_file, 'a+', encoding='utf-8') as fw:
                fw.write(json.dumps(results, ensure_ascii=False) + "\n")
        
        elif output_format == "csv":
            # Save as CSV
            
            csv_content = create_csv_report(results)
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(csv_content)
        
        else:  # text format
            # Save as text report
            text_content = ""
            for i, result in enumerate(results):
                text_content += f"\n{'='*50}\n"
                text_content += f"Case {i+1}: {result.get('seq_id', 'Unknown')}\n"
                text_content += f"{'='*50}\n"
                
                if detailed:
                    text_content += create_detailed_report(result)
                else:
                    text_content += create_evaluation_report(result)
                text_content += "\n"
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(text_content)
                
    except Exception as e:
        print(f"Warning: Failed to save results - {e}")

async def process_case_async(case: Dict, required_fields: List[str], 
                             eval_mode: bool, evidence: bool, 
                             selection: bool, pipeline_parallel: bool, output_file: str, 
                             output_format: str, detailed: bool, sample_count: int = 9) -> Dict[str, Any]:
    """
    Process a single case asynchronously
    
    Args:
        case: Case data dictionary
        eval_mode: Evaluation mode
        evidence: Whether to enable evidence evaluation
        selection: Whether to enable selection evaluation  
        pipeline_parallel: Whether to enable pipeline parallel evaluation
        
    Returns:
        Case result dictionary
    """
    try:
        print(f"Processing case {case['seq_id']}...")
        emr = case[required_fields[1]]
        ref_json = case[required_fields[2]]
        pred_json = case[required_fields[3]]
        # Run evaluation for this case
        case_result = await evaluate_metric(
            emr, 
            ref_json, 
            pred_json,
            eval_mode=eval_mode,
            eval_evidence=evidence,
            eval_selection=selection,
            pipeline_parallel=pipeline_parallel,
            sample_count=sample_count
        )

        # Add metadata to result
        case_result['seq_id'] = case[required_fields[0]]
        case_result['line_number'] = case.get('line_number', 0)
        print(f"✓ Completed case {case['seq_id']}, with result: {case_result}")
        
        # Save all results at once
        save_incremental_results(case_result, output_file, output_format, detailed)
        print(f"✓ Saved case {case['seq_id']} to {output_file}")
        return case_result
        
    except Exception as e:
        print(f"Error processing case {case['seq_id']}: {e}")
        return {
            'seq_id': case['seq_id'],
            'line_number': case.get('line_number', 0),
            'error': str(e),
            'status': 'failed'
        }


def process_cases_parallel(cases: List[Dict[str, Any]], required_fields: List[str], 
                                 eval_mode: bool, evidence: bool, selection: bool, 
                                 pipeline_parallel: bool, parallel_count: int=3,
                                 output_file: str = None, output_format: str = None, detailed: bool = False, sample_count: int = 9) -> List[Dict[str, Any]]:
    """
    Process multiple cases in parallel using ThreadPoolExecutor with limited concurrency
    
    Args:
        cases: List of validated cases
        eval_mode: Evaluation mode
        evidence: Whether to enable evidence evaluation
        selection: Whether to enable selection evaluation
        pipeline_parallel: Whether to enable pipeline parallel evaluation
        
    Returns:
        List of case results
    """
    
    def run_case_in_thread(case):
        """Wrapper to run async function in thread"""
        return asyncio.run(process_case_async(case, required_fields, eval_mode, evidence, selection, pipeline_parallel, output_file, output_format, detailed, sample_count=sample_count))
    
    print(f"Starting parallel processing of {len(cases)} cases with max {parallel_count} workers...")
    
    # Use ThreadPoolExecutor with max 3 workers
    with concurrent.futures.ThreadPoolExecutor(max_workers=parallel_count) as executor:
        # Submit all tasks
        future_to_case = {executor.submit(run_case_in_thread, case): case for case in cases}
        
        processed_results = []
        # Use tqdm for progress bar in parallel processing
        with tqdm(total=len(cases), desc="Processing cases", unit="case") as pbar:
            for future in concurrent.futures.as_completed(future_to_case):
                case = future_to_case[future]
                try:
                    result = future.result()
                    processed_results.append(result)
                    pbar.set_postfix(case_id=case['seq_id'])
                    pbar.update(1)
                    print(f"✓ Completed case {case['seq_id']}")
                except Exception as e:
                    processed_results.append({
                        'seq_id': case['seq_id'],
                        'line_number': case.get('line_number', 0),
                        'error': str(e),
                        'status': 'failed'
                    })
                    pbar.set_postfix(case_id=case['seq_id'], status="FAILED")
                    pbar.update(1)
                    print(f"✗ Failed case {case['seq_id']}: {e}")
    
    return processed_results


def process_diagnosis_cases(jsonl_file: str, output_file: str, eval_mode: bool = False, 
                       evidence: bool = False, selection: bool = False, 
                       output_format: str = "json", detailed: bool = False, args: argparse.Namespace = None):
    """
    Process JSONL file batch by batch and save results incrementally
    
    Args:
        jsonl_file: Path to input JSONL file
        output_file: Path to output file
        eval_mode: Evaluation mode
        evidence: Whether to enable evidence evaluation
        selection: Whether to enable selection evaluation
        output_format: Output format (json, text, csv)
        detailed: Whether to generate detailed report
        args: Command line arguments
    """
    try:
        # Step 1: Load and validate all cases
        print("=== Step 1: Loading and validating cases ===")
        cases = load_and_validate_cases(jsonl_file, )
        
        if not cases:
            print("No valid cases found in input file")
            return 1
        
        # Step 2: Process cases
        print("=== Step 2: Processing cases ===")
        required_fields = args.required_fields
        
        if not args.case_parallel:
            # Original sequential processing (when multi_thread is True)
            print("Using sequential processing mode...")
            results = []
            for case in tqdm(cases, desc="Processing cases sequentially", unit="case"):
                case_result = asyncio.run(process_case_async(
                    case, required_fields, eval_mode, evidence, selection, args.pipeline_parallel, output_file, output_format, detailed, sample_count=args.sample_count
                ))
        else:
            # New parallel processing mode (when multi_thread is False)
            print("Using parallel processing mode...")
            results = process_cases_parallel(
                cases, required_fields, eval_mode, evidence, selection, args.pipeline_parallel, parallel_count=args.case_parallel_count,
                output_file=output_file, output_format=output_format, detailed=detailed,sample_count=args.sample_count
            )
        
        print(f"=== Processing completed: {len(results)} cases processed ===")
        return 0
        
    except Exception as e:
        print(f"Error in batch processing: {e}")
        return 1




def main():
    """Command-line interface for the medical diagnosis evaluator"""
    parser = argparse.ArgumentParser(
        description="Medical Diagnosis Evaluation System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
        # Process JSONL file with basic evaluation
        /usr/bin/python3 main.py --input example_input_cotv3.jsonl --output results_cotv3.jsonl --eval_mode "PD_SD_DD" --evidence --selection --case_parallel --pipeline_parallel
        /usr/bin/python3 main.py --input example_input_cotv3.jsonl --output results_cotv3.jsonl --eval_mode "PD_SD_DD" --evidence --selection --case_parallel --pipeline_parallel
        
        # Process with all diagnosis types and evidence evaluation
        python main.py --input data.jsonl --output results.json --all --evidence
        
        # Generate detailed text report
        python main.py --input data.jsonl --output results.txt --format text --detailed
        
        # Legacy mode (single case)
        python main.py --emr "patient_emr.txt" --ref "reference.json" --pred "prediction.json"
                """
    )
    
    # New JSONL mode arguments
    parser.add_argument("--input", help="Path to input JSONL file (each line: {seq_id, emr, ref, pred})", default="example_input_cotv3.jsonl")
    parser.add_argument("--output", help="Path to output file (required for JSONL mode)",default="results_cotv3.jsonl")
    
    # Evaluation options
    parser.add_argument("--eval_mode", help="Evaluate all diagnosis types (PD + SD + DD)", default="PD_SD_DD")
    parser.add_argument("--evidence", action="store_true", help="Enable evidence evaluation")
    parser.add_argument("--selection", action="store_true", help="Enable selection evaluation")
    parser.add_argument("--detailed", action="store_true", help="Generate detailed report")
    parser.add_argument("--format", choices=["text", "json", "csv"], default="json", 
                       help="Output format")
    parser.add_argument("--case_parallel", action="store_true", help="Enable case parallel evaluation")
    parser.add_argument("--pipeline_parallel", action="store_true", help="Enable pipeline parallel evaluation")
    parser.add_argument("--case_parallel_count", type=int, default=3, help="Number of parallel workers")
    parser.add_argument("--required_fields", type=str, default=["seq_id","raw_query","ans_label","response"], 
                        help="Required fields")
    parser.add_argument("--sample_count", type=int, default=9, help="Number of samples")
    
    args = parser.parse_args()
    
    # Determine mode: JSONL vs Legacy
    if args.input:
        # JSONL mode
        if not args.output:
            print("Error: --output is required when using --input (JSONL mode)")
            return 1
            
        return process_diagnosis_cases(
            args.input, args.output,
            eval_mode=args.eval_mode,
            evidence=args.evidence,
            selection=args.selection,
            output_format=args.format,
            detailed=args.detailed,
            args=args
        )
    
    else:
        print("Error: Either use --input for JSONL mode, or provide --emr, --ref, --pred for legacy mode")
        return 1


if __name__ == "__main__":
    exit(main())