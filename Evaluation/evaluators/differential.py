"""
Differential diagnosis evaluator for medical diagnosis evaluation.

This module contains the specialized evaluator for differential diagnosis
assessment, including exclusion logic and comprehensive evidence evaluation.
"""

from typing import Dict
from core.config import EvaluationConfig
from core.models import DiagnosisResult
from core.constants import DiagnosisFields
from utils.text_processing import TextProcessor
from utils.matching import DiagnosisMatchingUtils
from utils.evidence import EvidenceEvaluator
import numpy as np
import time


class DifferentialDiagnosisEvaluator:
    """Evaluator for differential diagnosis"""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.text_processor = TextProcessor()
        self.matching_utils = DiagnosisMatchingUtils()
        self.evidence_evaluator = EvidenceEvaluator()
    
    async def evaluate(self, emr: str, ref_json: Dict, pred_json: Dict,sample_count: int=9) -> DiagnosisResult:
        """Evaluate differential diagnosis"""
        print("Evaluating Differential Diagnosis...")
        duration = {
            "diagnosis_matching_eval_time": 0,
            "pos_evidence_eval_time": 0,
            "neg_evidence_eval_time": 0,
            "check_eval_time": 0,
            "selection_eval_time": 0
        }
        
        pred_diagnoses = pred_json[DiagnosisFields.DIFFERENTIAL_DIAGNOSIS]
        ref_diagnoses = ref_json[DiagnosisFields.DIFFERENTIAL_DIAGNOSIS]
        
        pred_diagnosis_list = self.text_processor.extract_diagnosis_list(pred_diagnoses)
        ref_diagnosis_list = self.text_processor.extract_diagnosis_list(ref_diagnoses)
        
        # Extract excludable predictions
        pred_excludable_list = [str(item[DiagnosisFields.EXCLUDABLE]) for item in pred_diagnoses]
        
        matched_predictions = []
        matched_references = []
        correct_exclusions = 0
        evidence_metrics_sum = {
            'pos_f1': [], 'neg_f1':[], 'check_f1': [], 'sel_precision': [], 'pos_hall':[], 'neg_hall':[]
        }
        
        # Match and evaluate each reference diagnosis
        for idx, ref_diagnosis_item in enumerate(ref_diagnoses):
            start_time = time.time()
            ref_diagnosis = self.text_processor.clean_diagnosis_name(ref_diagnosis_item[DiagnosisFields.NAME])
            is_match, matched_pred, matched_idx = self.matching_utils.evaluate_diagnosis_match(
                pred_diagnosis_list, ref_diagnosis, emr, sample_count=sample_count
            )
            duration["diagnosis_matching_eval_time"] += time.time() - start_time
            
            if is_match:
                ## todo: 已经匹配的结果，应该从pred_diagnosis_list中删除?
                matched_predictions.append(pred_diagnosis_list[matched_idx])
                matched_references.append(ref_diagnosis)
                
                # Check exclusion accuracy
                ref_excludable = ref_diagnosis_item[DiagnosisFields.EXCLUDABLE]
                pred_excludable = pred_excludable_list[matched_idx]

                if self.matching_utils.check_exact_match(ref_excludable, pred_excludable):
                    correct_exclusions += 1
                
                # Evaluate evidence if enabled
                if self.config.enable_evidence_eval:
                    pred_diagnosis_item = pred_diagnoses[matched_idx]
                    start_time = time.time()
                    # Supporting evidence
                    pos_metrics = self.evidence_evaluator.evaluate_evidence(
                        pred_diagnosis_item[DiagnosisFields.SUPPORTING_EVIDENCE],
                        ref_diagnosis_item[DiagnosisFields.SUPPORTING_EVIDENCE],
                        emr,
                    )
                    evidence_metrics_sum['pos_f1'].append(pos_metrics['f1'])
                    evidence_metrics_sum['pos_hall'].append(pos_metrics['hallucination'])
                    duration["pos_evidence_eval_time"] += time.time() - start_time
                    start_time = time.time()
                    # Opposing evidence  
                    neg_metrics = self.evidence_evaluator.evaluate_evidence(
                        pred_diagnosis_item[DiagnosisFields.OPPOSING_EVIDENCE],
                        ref_diagnosis_item[DiagnosisFields.OPPOSING_EVIDENCE],
                        emr
                    )
                    evidence_metrics_sum['neg_f1'].append(neg_metrics['f1'])
                    evidence_metrics_sum['neg_hall'].append(neg_metrics['hallucination'])
                    duration["neg_evidence_eval_time"] += time.time() - start_time
                    start_time = time.time()
                    # 检查推荐评估
                    _, pred_tests = self.text_processor.get_test_recommendations(pred_diagnosis_item)
                    _, ref_tests = self.text_processor.get_test_recommendations(ref_diagnosis_item)
                    
                    ## 这里不需要计算幻觉，只用计算f1即可(增加enable_hallucination_check=False)
                    check_metrics = self.evidence_evaluator.evaluate_evidence(
                        pred_tests, 
                        ref_tests, 
                        emr,
                        enable_hallucination_check=False,
                        use_similarity=False
                    )
                    evidence_metrics_sum['check_f1'].append(check_metrics['f1'])
                    duration["check_eval_time"] += time.time() - start_time
                    start_time = time.time()
                    
                    # Selection basis (if enabled)
                    if self.config.enable_selection_eval:
                        sel_metrics = self.evidence_evaluator.evaluate_evidence(
                            pred_diagnosis_item[DiagnosisFields.SELECTION_BASIS],
                            ref_diagnosis_item[DiagnosisFields.SELECTION_BASIS],
                            emr,
                            enable_hallucination_check=False,
                            use_similarity=False
                        )
                
                        evidence_metrics_sum['sel_precision'].append(sel_metrics['precision'])
                    duration["selection_eval_time"] += time.time() - start_time
        
        # Handle predictions marked as excludable but not matched
        for idx, pred_diagnosis_item in enumerate(pred_diagnoses):
            pred_diagnosis = self.text_processor.clean_diagnosis_name(pred_diagnosis_item[DiagnosisFields.NAME])
            if pred_diagnosis not in matched_predictions and str(pred_diagnosis_item[DiagnosisFields.EXCLUDABLE]) == "True":
                correct_exclusions += 1
                matched_predictions.append(pred_diagnosis)
        
        # Remove duplicates
        matched_predictions = self.text_processor.remove_duplicates(matched_predictions)
        matched_references = self.text_processor.remove_duplicates(matched_references)
        
        # Calculate metrics
        num_exact_matches = len([p for p in matched_predictions if p in ref_diagnosis_list])
        
        diagnosis_iou = round(
            len(matched_predictions) / (len(pred_diagnosis_list) + len(ref_diagnosis_list) - num_exact_matches), 4
        )
        diagnosis_precision = round(len(matched_predictions) / len(pred_diagnosis_list), 4)
        diagnosis_recall = round(len(matched_references) / len(ref_diagnosis_list), 4)
        ## 匹配到ref列表，且是否可排除正确的 + 未匹配到ref列表，但提示可排除的
        exclusion_precision = round(correct_exclusions / len(pred_diagnosis_list), 4)
        
        # Average evidence metrics:（macro-f1）
        avg_pos_f1 = round(np.mean(evidence_metrics_sum["pos_f1"]),4) if len(evidence_metrics_sum["pos_f1"]) > 0 else 0
        avg_neg_f1 = round(np.mean(evidence_metrics_sum["neg_f1"]),4) if len(evidence_metrics_sum["neg_f1"]) > 0 else 0
        avg_check_f1 = round(np.mean(evidence_metrics_sum["check_f1"]),4) if len(evidence_metrics_sum["check_f1"]) > 0 else 0
        avg_sel_precision = round(np.mean(evidence_metrics_sum["sel_precision"]),4) if len(evidence_metrics_sum["sel_precision"]) > 0 else 0
        avg_pos_hall = round(np.mean(evidence_metrics_sum["pos_hall"]),4) if len(evidence_metrics_sum["pos_hall"]) > 0 else 0
        avg_neg_hall = round(np.mean(evidence_metrics_sum["neg_hall"]),4) if len(evidence_metrics_sum["neg_hall"]) > 0 else 0
        
        # Final score calculation
        final_score = round(
            diagnosis_iou * 0.4 + exclusion_precision * 0.2 + 
            avg_pos_f1 * 0.1 + avg_neg_f1 * 0.1 + 
            avg_check_f1 * 0.1 + avg_sel_precision * 0.1, 4
        )
        res = DiagnosisResult(
            diagnosis_name=pred_diagnosis_list,
            task_type="differential",
            precision=diagnosis_precision,
            recall=diagnosis_recall,
            iou=diagnosis_iou,
            f1_score=final_score,
            additional_metrics={
                'ref_diagnosis_list': ref_diagnosis_list,
                'pred_diagnosis_list': pred_diagnosis_list,
                'final_right_predict_list': matched_predictions,
                'final_right_recall_list': matched_references,
                'diagnosis_iou': diagnosis_iou,
                'diagnosis_precision': diagnosis_precision,
                'diagnosis_recall': diagnosis_recall,
                'exclusion_precision': exclusion_precision,
                'sel_evidence_precision': avg_sel_precision,
                'pos_evidence_f1': avg_pos_f1,
                'pos_hallucination': avg_pos_hall,
                'neg_evidence_f1': avg_neg_f1,
                'neg_hallucination': avg_neg_hall,
                'check_f1': avg_check_f1,
                'total_score': final_score,
                "sample_count": sample_count
            },
            duration=duration
        ) 
        print("endding: differential diagnosis evaluation with result:{}".format(res))
        return res