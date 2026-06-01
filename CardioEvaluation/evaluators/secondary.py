"""
Secondary diagnosis evaluator for medical diagnosis evaluation.

This module contains the specialized evaluator for secondary diagnosis
assessment, including precision, recall, and IoU calculations.
"""

from typing import Dict
from core.config import EvaluationConfig
from core.models import DiagnosisResult
from core.constants import DiagnosisFields
from utils.text_processing import TextProcessor
from utils.matching import DiagnosisMatchingUtils
from utils.evidence import EvidenceEvaluator
import time
import numpy as np


class SecondaryDiagnosisEvaluator:
    """Evaluator for secondary diagnosis"""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.text_processor = TextProcessor()
        self.matching_utils = DiagnosisMatchingUtils()
        self.evidence_evaluator = EvidenceEvaluator()
    
    async def evaluate(self, emr: str, ref_json: Dict, pred_json: Dict,sample_count: int=9) -> DiagnosisResult:
        """Evaluate secondary diagnosis"""
        print("Evaluating Secondary Diagnosis...")
        duration = {
            "matching_eval_time": 0,
            "evidence_eval_time": 0
        }
        
        pred_diagnoses = pred_json[DiagnosisFields.INITIAL_DIAGNOSIS][DiagnosisFields.SECONDARY_DIAGNOSIS]
        ref_diagnoses = ref_json[DiagnosisFields.INITIAL_DIAGNOSIS][DiagnosisFields.SECONDARY_DIAGNOSIS]
        
        pred_diagnosis_list = self.text_processor.extract_diagnosis_list(pred_diagnoses)
        ref_diagnosis_list = self.text_processor.extract_diagnosis_list(ref_diagnoses)
        
        matched_predictions = []
        matched_references = []
        evidence_score = {
            "f1":[],
            "hallucination":[]
        }

        ## todo: 这里一旦有预测诊断列表中的元素被检测到，应该剔除？
        
        # Match predictions with references
        for ref_diagnosis_item in ref_diagnoses:
            start_time = time.time()
            ref_diagnosis = self.text_processor.clean_diagnosis_name(ref_diagnosis_item[DiagnosisFields.NAME])
            is_match, matched_pred, matched_idx = self.matching_utils.evaluate_diagnosis_match(
                pred_diagnosis_list, ref_diagnosis, emr, sample_count=sample_count
            )
            duration["matching_eval_time"] += time.time() - start_time
            start_time = time.time()
            
            if is_match:
                matched_predictions.append(pred_diagnosis_list[matched_idx])
                matched_references.append(ref_diagnosis)
                
                # Evaluate evidence if enabled
                if self.config.enable_evidence_eval:
                    pred_evidence = pred_diagnoses[matched_idx][DiagnosisFields.DIAGNOSIS_EVIDENCE]
                    ref_evidence = ref_diagnosis_item[DiagnosisFields.DIAGNOSIS_EVIDENCE]
                    evidence_metrics = self.evidence_evaluator.evaluate_evidence(pred_evidence, ref_evidence, emr)
                    evidence_score["f1"].append(evidence_metrics['f1'])
                    evidence_score["hallucination"].append(evidence_metrics["hallucination"])
                duration["evidence_eval_time"] += time.time() - start_time
        
        # Remove duplicates?
        # todo: 这里references 是否需要考虑是否 需要去重
        matched_predictions = self.text_processor.remove_duplicates(matched_predictions)
        matched_references = self.text_processor.remove_duplicates(matched_references)
        
        # Calculate metrics
        ## init metrics by dict format
        metrics = {
            'precision': 0,
            'recall': 0,
            'iou': 0,
            'evidence_f1': 0,
            "evidence_hallucination": 0,
            'final_score': 0
        }

        if len(matched_predictions) != 0:
            metrics['precision'] = round(len(matched_predictions) / len(pred_diagnosis_list), 4)
            metrics['recall'] = round(len(matched_references) / len(ref_diagnosis_list), 4)
            metrics['iou'] = self.evidence_evaluator.calculate_iou_score(
                len(pred_diagnosis_list), len(ref_diagnosis_list), len(matched_predictions)
            )
            metrics['evidence_f1'] = round(np.mean(evidence_score["f1"]), 4)
            metrics["evidence_hallucination"] = round(np.mean(evidence_score["hallucination"]), 4)
            metrics['final_score'] = round((metrics['iou'] * 0.67 + metrics['evidence_f1'] * 0.33), 4)

        res = DiagnosisResult(
            diagnosis_name=pred_diagnosis_list,
            task_type="secondary",
            precision=metrics['precision'],
            recall=metrics['recall'],
            iou=metrics['iou'],
            f1_score=metrics['final_score'],
            evidence_score=metrics['evidence_f1'],
            additional_metrics={
                'ref_diagnosis_list': ref_diagnosis_list,
                'pred_diagnosis_list': pred_diagnosis_list,
                'diagnosis_precision_list': matched_predictions,
                'diagnosis_recall_list': matched_references,
                'diagnosis_iou': metrics['iou'],
                'diagnosis_precision': metrics['precision'],
                'diagnosis_recall': metrics['recall'],
                'evidence_f1': metrics['evidence_f1'],
                'evidence_hallucination': metrics['evidence_hallucination'],
                'total_score': metrics['final_score'],
                "sample_count": sample_count
            },
            duration=duration
        )
        print("endding: secondary diagnosis evaluation with result:{}".format(res))
        return res