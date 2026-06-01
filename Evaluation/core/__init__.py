"""
Core components for the medical diagnosis evaluation system.

This package contains the fundamental data structures, configuration classes,
and constants used throughout the evaluation system.
"""

from .models import DiagnosisResult, DiagnosisType
from .config import EvaluationConfig
from .constants import DiagnosisFields

__all__ = [
    "DiagnosisResult",
    "DiagnosisType",
    "EvaluationConfig", 
    "DiagnosisFields"
] 