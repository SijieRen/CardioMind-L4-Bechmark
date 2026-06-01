"""
Base evaluator class for medical diagnosis evaluation.

This module contains the main MedicalDiagnosisEvaluator class that
orchestrates the evaluation of different diagnosis types.
"""

import asyncio
import threading
from typing import Dict, Any
from core.config import EvaluationConfig
from core.models import DiagnosisResult
from evaluators.primary import PrimaryDiagnosisEvaluator
from evaluators.secondary import SecondaryDiagnosisEvaluator
from evaluators.differential import DifferentialDiagnosisEvaluator


class MedicalDiagnosisEvaluator:
    """Main evaluator class for medical diagnosis assessment"""
    
    def __init__(self, config: EvaluationConfig = None):
        self.config = config or EvaluationConfig()
        self.primary_evaluator = PrimaryDiagnosisEvaluator(self.config)
        self.secondary_evaluator = SecondaryDiagnosisEvaluator(self.config)
        self.differential_evaluator = DifferentialDiagnosisEvaluator(self.config)
        self.sample_count = self.config.sample_count

    def evaluate_all_pipeline(self, emr: str, ref_json: Dict, pred_json: Dict, 
                          eval_mode: str = "PD_SD_DD", evaluator_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Evaluate all diagnosis types"""
        threads = []
        results = {}
        
        # 通用执行函数
        def run_evaluator(evaluator, result_key, eval_name):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(evaluator.evaluate(emr, ref_json, pred_json, sample_count=self.sample_count))
                # 确保result是DiagnosisResult对象
                if isinstance(result, DiagnosisResult):
                    results[result_key] = result
                else:
                    # 如果不是DiagnosisResult对象，尝试处理其他类型
                    results[result_key] = {"error": f"{eval_name} evaluation returned unexpected type: {type(result)}"}
            except Exception as e:
                results[result_key] = {"error": f"{eval_name} evaluation failed: {e}"}
            finally:
                loop.close()
        
        # 根据eval_mode启动相应的线程
        task_types = []
        for eval_type, (evaluator, eval_name) in evaluator_config.items():
            if eval_type in eval_mode:
                thread = threading.Thread(target=run_evaluator, 
                                        args=(evaluator, eval_type, eval_name))
                threads.append(thread)
                task_types.append(eval_type)
                thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()

        return results,task_types
    
    async def evaluate_all_single(self, emr: str, ref_json: Dict, pred_json: Dict, 
                            eval_mode: str, evaluator_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Evaluate all diagnosis types"""
        results = {}
        task_types = []
        for eval_type, (evaluator, _) in evaluator_config.items():
            if eval_type in eval_mode:
                result = await evaluator.evaluate(emr, ref_json, pred_json, sample_count=self.sample_count)
                task_types.append(eval_type)
                results[eval_type] = result
        return results, task_types
    
    async def evaluate_all(self, emr: str, ref_json: Dict, pred_json: Dict, 
                          eval_mode: str = "PD_SD_DD", pipeline_parallel: bool = False) -> Dict[str, Any]:
        """Evaluate all diagnosis types"""
        # 评估器配置
        evaluator_config = {
            "PD": (self.primary_evaluator, "Primary"),
            "SD": (self.secondary_evaluator, "Secondary"),
            "DD": (self.differential_evaluator, "Differential")
        }
        # 并行执行
        if pipeline_parallel: 
            results,task_types = self.evaluate_all_pipeline(emr, ref_json, pred_json, eval_mode, evaluator_config)
        else:
            results,task_types = await self.evaluate_all_single(emr, ref_json, pred_json, eval_mode, evaluator_config)

        # 处理结果
        evaluation_results = {}
        total_weighted_score = 0.0
        total_weight = 0.0
        
        # 映射task_type到config权重键
        weight_mapping = {
            "PD": "primary",
            "SD": "secondary", 
            "DD": "differential"
        }
        
        for task_type in task_types:
            if task_type in results:
                result = results[task_type]
                
                if isinstance(result, dict) and "error" in result:
                    # 处理异常情况
                    evaluation_results[task_type] = result
                elif isinstance(result, DiagnosisResult):
                    # 正确的DiagnosisResult对象
                    evaluation_results[task_type] = result.additional_metrics
                    evaluation_results[task_type]["duration"] = result.duration
                    
                    # 计算加权分数
                    weight_key = weight_mapping.get(task_type, "primary")
                    weight = self.config.weights.get(weight_key, 0)
                    total_weighted_score += result.f1_score * weight
                    total_weight += weight
                else:
                    # 处理其他意外类型
                    evaluation_results[task_type] = {"error": f"Unexpected result type: {type(result)}"}
        
        # 计算总分
        overall_score = round(total_weighted_score, 4) if total_weight > 0 else 0.0
        print("here",overall_score)
        return {
            **evaluation_results,
            "score": overall_score
        }
    