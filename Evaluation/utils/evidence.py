"""
Evidence evaluation utilities for medical diagnosis evaluation.

This module contains functions for evaluating the quality of 
evidence and supporting information in medical diagnoses.
"""

from typing import Dict
from utils.model_metric_utils import (
    split_medical_text_items,
    evaluate_medical_evidence,
    calculate_iou,
    calculate_f1_score
)


class EvidenceEvaluator:
    """Utility class for evidence evaluation operations"""
    # todo: 为了对齐结果，将use_similarity临时设置为False，后续需要修改
    
    @staticmethod
    def evaluate_evidence(pred_evidence: str, ref_evidence: str, emr: str = None, enable_hallucination_check: bool = True,use_similarity: bool = False) -> Dict[str, float]:
        """Evaluate evidence matching"""
        pred_num, pred_list = split_medical_text_items(pred_evidence)
        ref_num, ref_list = split_medical_text_items(ref_evidence)
        
        evaluation_results = evaluate_medical_evidence(
            pred_num, ref_num, pred_list, ref_list, emr, enable_hallucination_check=enable_hallucination_check, use_similarity=use_similarity
        )
        
        return {
            'f1': evaluation_results.f1_score,
            'iou': evaluation_results.iou, 
            'precision': evaluation_results.precision,
            'recall': evaluation_results.recall,
            'hallucination': evaluation_results.hallucination_rate
        }
    
    @staticmethod
    def calculate_f1_score(precision: float, recall: float) -> float:
        """Calculate F1 score from precision and recall"""
        return calculate_f1_score(precision, recall)
    
    @staticmethod
    def calculate_iou_score(pred_count: int, ref_count: int, matched_count: int) -> float:
        """Calculate Intersection over Union (IoU) score"""
        return calculate_iou(pred_count, ref_count, matched_count) 