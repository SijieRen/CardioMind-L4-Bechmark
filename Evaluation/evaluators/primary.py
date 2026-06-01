"""
Primary diagnosis evaluator for medical diagnosis evaluation.

This module contains the specialized evaluator for primary diagnosis
assessment, including top-1 and top-k precision evaluation.
"""

from typing import Dict
from core.config import EvaluationConfig
from core.models import DiagnosisResult
from core.constants import DiagnosisFields
from utils.text_processing import TextProcessor
from utils.matching import DiagnosisMatchingUtils
from utils.evidence import EvidenceEvaluator
import time


class PrimaryDiagnosisEvaluator:
    """Evaluator for primary diagnosis"""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.text_processor = TextProcessor()
        self.matching_utils = DiagnosisMatchingUtils()
        self.evidence_evaluator = EvidenceEvaluator()
    
    async def evaluate(self, emr: str, ref_json: Dict, pred_json: Dict,sample_count: int=9) -> DiagnosisResult:
        """Evaluate primary diagnosis"""
        print("Evaluating Primary Diagnosis...")
        start_time = time.time()
        duration = {
            "top1_eval_time": 0,
            "type_eval_time": 0,
            "evidence_eval_time": 0,
            "topk_eval_time": 0
        }
        
        # Extract diagnosis data
        # 1. 提取诊断名称
        pred_path = [DiagnosisFields.INITIAL_DIAGNOSIS, DiagnosisFields.PRIMARY_DIAGNOSIS]
        ref_path = [DiagnosisFields.INITIAL_DIAGNOSIS, DiagnosisFields.PRIMARY_DIAGNOSIS]
        pred_diagnosis_data = pred_json[pred_path[0]][pred_path[1]]
        ref_diagnosis_data = ref_json[ref_path[0]][ref_path[1]]
        # 2. 清洗诊断名称
        pred_diagnosis = pred_diagnosis_data[DiagnosisFields.NAME].strip(" ")
        ref_diagnosis = ref_diagnosis_data[DiagnosisFields.NAME].strip(" ")
        # 3. 提取病因类型etiology
        pred_etiology = self.text_processor.get_disease_type(pred_diagnosis_data)
        ref_etiology = self.text_processor.get_disease_type(ref_diagnosis_data)

        
        # 4. 提取预测鉴别诊断（计算top-k：看鉴别诊断有没有命中主要诊断）
        differential_diagnosis_list, _ = self.matching_utils.extract_differential_diagnosis_data(pred_json)

        # 5. 初始化指标
        metrics = {
            'top_1_precision': 0,
            'top_k_precision': 0,
            'type_precision': 0,
            'evidence_metrics': {'f1': 0, 'hallucination': 0}
        }
        
        # 6. 评估主要诊断匹配
        primary_diagnosis_matched, _, _ = self.matching_utils.evaluate_diagnosis_match([pred_diagnosis], ref_diagnosis, emr, sample_count=sample_count)
        duration["top1_eval_time"] = time.time() - start_time
        start_time = time.time()
        if primary_diagnosis_matched:
            metrics['top_1_precision'] = 1
            metrics['top_k_precision'] = 1
            
            # 7.评估病因分型匹配
            type_is_match, _, _ = self.matching_utils.evaluate_diagnosis_match([pred_etiology], ref_etiology, emr, sample_count=sample_count)
            metrics['type_precision'] = 1 if type_is_match else 0

            duration["type_eval_time"] = time.time() - start_time
            start_time = time.time()
            
            # 8.评估诊断依据匹配
            if self.config.enable_evidence_eval:
                metrics['evidence_metrics'] = self.evidence_evaluator.evaluate_evidence(
                    pred_diagnosis_data[DiagnosisFields.DIAGNOSIS_EVIDENCE],
                    ref_diagnosis_data[DiagnosisFields.DIAGNOSIS_EVIDENCE],
                    emr,
                )
            duration["evidence_eval_time"] = time.time() - start_time
            start_time = time.time()
        else:
            # 9.检查预测鉴别诊断是否命中主要诊断
            topk_diagnosis_matched, _, _ = self.matching_utils.evaluate_diagnosis_match(differential_diagnosis_list, ref_diagnosis, emr, sample_count=sample_count)
            metrics['top_k_precision'] = 1 if topk_diagnosis_matched else 0

            duration["topk_eval_time"] = time.time() - start_time
            start_time = time.time()
        
        # Calculate scores
        diagnosis_score = round((metrics['top_1_precision'] + metrics['top_k_precision']) / 2, 4)

        ## todo: 如果不计算evidence,这里final_score就不对了
        final_score = round(
            (metrics['top_1_precision'] * 0.25 + metrics['top_k_precision'] * 0.25 + 
             metrics['type_precision'] * 0.25 + metrics['evidence_metrics']['f1'] * 0.25), 4
        )
        res = DiagnosisResult(
            diagnosis_name=pred_diagnosis,
            task_type="primary",
            precision=metrics['top_1_precision'],
            f1_score=final_score,
            evidence_score=metrics['evidence_metrics']['f1'],
            type_precision=metrics['type_precision'],
            additional_metrics={
                'ref_diagnosis': ref_diagnosis,
                'pred_diagnosis': pred_diagnosis,
                'top_k_list': differential_diagnosis_list,
                'ref_etiology': ref_etiology,
                'pred_etiology': pred_etiology,
                'top_1_precision': metrics['top_1_precision'],
                'top_k_precision': metrics['top_k_precision'],
                'diagnosis_score': diagnosis_score,
                'type_score': metrics['type_precision'],
                'evidence_f1': metrics['evidence_metrics']['f1'],
                'evidence_hallucination': metrics['evidence_metrics']['hallucination'],
                'total_score': final_score
            },
            duration=duration
        ) 
        print("endding: primary diagnosis evaluation with result:{}".format(res))
        return res