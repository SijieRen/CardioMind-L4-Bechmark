"""
Medical Diagnosis Evaluation System

A comprehensive system for evaluating medical diagnosis predictions
against reference standards with support for primary, secondary,
and differential diagnoses.
"""

from .core.models import DiagnosisResult, DiagnosisType
from .core.config import EvaluationConfig
from .evaluators.base import MedicalDiagnosisEvaluator
from .main import evaluate_metric
from .reporting.report_generator import create_evaluation_report

__version__ = "1.0.0"
__author__ = "Medical AI Team"

__all__ = [
    "DiagnosisResult",
    "DiagnosisType", 
    "EvaluationConfig",
    "MedicalDiagnosisEvaluator",
    "evaluate_metric",
    "create_evaluation_report"
] 