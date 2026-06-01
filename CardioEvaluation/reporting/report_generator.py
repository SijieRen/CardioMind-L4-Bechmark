"""
Report generation utilities for medical diagnosis evaluation.

This module contains functions for creating formatted evaluation reports
with detailed metrics and summaries.
"""

from typing import Dict, Any, List


def create_evaluation_report(results: Dict[str, Any]) -> str:
    """Create a formatted evaluation report"""
    report = ["Medical Diagnosis Evaluation Report", "=" * 40, ""]
    
    if "PD" in results:
        pd = results["PD"]
        report.extend([
            f"Primary Diagnosis:",
            f"  Predicted: {pd.get('pred_diagnosis', 'N/A')}",
            f"  Reference: {pd.get('ref_diagnosis', 'N/A')}", 
            f"  Top-1 Precision: {pd.get('top_1_precision', 0):.4f}",
            f"  Top-K Precision: {pd.get('top_k_precision', 0):.4f}",
            f"  Type Precision: {pd.get('type_precision', 0):.4f}",
            f"  Score: {pd.get('score', 0):.4f}",
            ""
        ])
    
    if "SD" in results:
        sd = results["SD"]
        report.extend([
            f"Secondary Diagnosis:",
            f"  Predictions: {len(sd.get('pred_diagnosis_list', []))}",
            f"  References: {len(sd.get('ref_diagnosis_list', []))}",
            f"  Precision: {sd.get('diagnosis_precision', 0):.4f}",
            f"  Recall: {sd.get('diagnosis_recall', 0):.4f}",
            f"  IoU: {sd.get('diagnosis_iou', 0):.4f}",
            f"  Score: {sd.get('score', 0):.4f}",
            ""
        ])
    
    if "DD" in results:
        dd = results["DD"]
        report.extend([
            f"Differential Diagnosis:",
            f"  Predictions: {len(dd.get('pred_diagnosis_list', []))}",
            f"  References: {len(dd.get('ref_diagnosis_list', []))}",
            f"  Precision: {dd.get('diagnosis_precision', 0):.4f}",
            f"  Recall: {dd.get('diagnosis_recall', 0):.4f}",
            f"  IoU: {dd.get('diagnosis_iou', 0):.4f}",
            f"  Exclusion Precision: {dd.get('exclusion_precision', 0):.4f}",
            f"  Score: {dd.get('score', 0):.4f}",
            ""
        ])
    
    report.extend([
        f"Overall Score: {results.get('score', 0):.4f}"
    ])
    
    return "\n".join(report)


def create_detailed_report(results: Dict[str, Any]) -> str:
    """Create a detailed evaluation report with all metrics"""
    report = ["Detailed Medical Diagnosis Evaluation Report", "=" * 50, ""]
    
    # Primary Diagnosis Details
    if "PD" in results:
        pd = results["PD"]
        report.extend([
            "PRIMARY DIAGNOSIS EVALUATION",
            "-" * 30,
            f"Predicted Diagnosis: {pd.get('pred_diagnosis', 'N/A')}",
            f"Reference Diagnosis: {pd.get('ref_diagnosis', 'N/A')}",
            f"Predicted Type: {pd.get('pred_type', 'N/A')}",
            f"Reference Type: {pd.get('ref_type', 'N/A')}",
            "",
            "Metrics:",
            f"  Top-1 Precision: {pd.get('top_1_precision', 0):.4f}",
            f"  Top-K Precision: {pd.get('top_k_precision', 0):.4f}",
            f"  Type Precision: {pd.get('type_precision', 0):.4f}",
            f"  Diagnosis Score: {pd.get('diagnosis_score', 0):.4f}",
            f"  Final Score: {pd.get('score', 0):.4f}",
            "",
            f"Top-K Candidates: {pd.get('top_k_list', [])}",
            ""
        ])
    
    # Secondary Diagnosis Details
    if "SD" in results:
        sd = results["SD"]
        report.extend([
            "SECONDARY DIAGNOSIS EVALUATION",
            "-" * 32,
            f"Predicted Count: {len(sd.get('pred_diagnosis_list', []))}",
            f"Reference Count: {len(sd.get('ref_diagnosis_list', []))}",
            f"Matched Predictions: {len(sd.get('diagnosis_precision_list', []))}",
            f"Matched References: {len(sd.get('diagnosis_recall_list', []))}",
            "",
            "Metrics:",
            f"  Precision: {sd.get('diagnosis_precision', 0):.4f}",
            f"  Recall: {sd.get('diagnosis_recall', 0):.4f}",
            f"  IoU: {sd.get('diagnosis_iou', 0):.4f}",
            f"  Evidence F1: {sd.get('evidence_f1', 0):.4f}",
            f"  Final Score: {sd.get('score', 0):.4f}",
            "",
            f"Predicted Diagnoses: {sd.get('pred_diagnosis_list', [])}",
            f"Reference Diagnoses: {sd.get('ref_diagnosis_list', [])}",
            f"Matched Predictions: {sd.get('diagnosis_precision_list', [])}",
            f"Matched References: {sd.get('diagnosis_recall_list', [])}",
            ""
        ])
    
    # Differential Diagnosis Details
    if "DD" in results:
        dd = results["DD"]
        report.extend([
            "DIFFERENTIAL DIAGNOSIS EVALUATION",
            "-" * 35,
            f"Predicted Count: {len(dd.get('pred_diagnosis_list', []))}",
            f"Reference Count: {len(dd.get('ref_diagnosis_list', []))}",
            f"Matched Predictions: {len(dd.get('diagnosis_precision_list', []))}",
            f"Matched References: {len(dd.get('diagnosis_recall_list', []))}",
            "",
            "Core Metrics:",
            f"  Diagnosis Precision: {dd.get('diagnosis_precision', 0):.4f}",
            f"  Diagnosis Recall: {dd.get('diagnosis_recall', 0):.4f}",
            f"  Diagnosis IoU: {dd.get('diagnosis_iou', 0):.4f}",
            f"  Exclusion Precision: {dd.get('exclusion_precision', 0):.4f}",
            "",
            "Evidence Metrics:",
            f"  Positive Evidence F1: {dd.get('pos_evidence_f1', 0):.4f}",
            f"  Negative Evidence F1: {dd.get('neg_evidence_f1', 0):.4f}",
            f"  Check Recommendations F1: {dd.get('check_f1', 0):.4f}",
            f"  Selection Evidence Precision: {dd.get('sel_evidence_precision', 0):.4f}",
            "",
            f"Final Score: {dd.get('score', 0):.4f}",
            "",
            f"Predicted Diagnoses: {dd.get('pred_diagnosis_list', [])}",
            f"Reference Diagnoses: {dd.get('ref_diagnosis_list', [])}",
            f"Matched Predictions: {dd.get('diagnosis_precision_list', [])}",
            f"Matched References: {dd.get('diagnosis_recall_list', [])}",
            ""
        ])
    
    # Overall Summary
    report.extend([
        "OVERALL SUMMARY",
        "-" * 15,
        f"Final Overall Score: {results.get('score', 0):.4f}",
        ""
    ])
    
    return "\n".join(report)


def create_csv_report(results: Dict[str, Any]) -> str:
    """Create a CSV format report for data analysis"""
    headers = [
        "metric_type", "diagnosis_type", "precision", "recall", "f1_score", 
        "iou", "evidence_score", "type_precision", "final_score"
    ]
    
    rows = [",".join(headers)]
    
    if "PD" in results:
        pd = results["PD"]
        row = [
            "primary", "primary",
            str(pd.get('top_1_precision', 0)),
            str(pd.get('top_k_precision', 0)),
            "0",  # No F1 for primary
            "0",  # No IoU for primary
            "0",  # Evidence score handled separately
            str(pd.get('type_precision', 0)),
            str(pd.get('score', 0))
        ]
        rows.append(",".join(row))
    
    if "SD" in results:
        sd = results["SD"]
        row = [
            "secondary", "secondary",
            str(sd.get('diagnosis_precision', 0)),
            str(sd.get('diagnosis_recall', 0)),
            str(sd.get('evidence_f1', 0)),
            str(sd.get('diagnosis_iou', 0)),
            str(sd.get('evidence_f1', 0)),
            "0",  # No type precision for secondary
            str(sd.get('score', 0))
        ]
        rows.append(",".join(row))
    
    if "DD" in results:
        dd = results["DD"]
        row = [
            "differential", "differential",
            str(dd.get('diagnosis_precision', 0)),
            str(dd.get('diagnosis_recall', 0)),
            str((dd.get('pos_evidence_f1', 0) + dd.get('neg_evidence_f1', 0)) / 2),
            str(dd.get('diagnosis_iou', 0)),
            str((dd.get('pos_evidence_f1', 0) + dd.get('neg_evidence_f1', 0)) / 2),
            "0",  # No type precision for differential
            str(dd.get('score', 0))
        ]
        rows.append(",".join(row))
    
    return "\n".join(rows) 