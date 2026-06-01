"""
Data models and enumerations for medical diagnosis evaluation.

This module contains the core data structures used throughout the
evaluation system, including result containers and type definitions.
"""

from typing import Dict, Any
from dataclasses import dataclass
from enum import Enum


class DiagnosisType(Enum):
    """Enumeration for different diagnosis types"""
    PRIMARY = "主要诊断"
    SECONDARY = "次要诊断"
    DIFFERENTIAL = "鉴别诊断"


@dataclass
class DiagnosisResult:
    """Container for diagnosis evaluation results"""
    diagnosis_name: str
    task_type: str
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    iou: float = 0.0
    evidence_score: float = 0.0
    type_precision: float = 0.0
    additional_metrics: Dict[str, Any] = None
    duration: Dict[str, float] = None
    
    def __post_init__(self):
        if self.additional_metrics is None:
            self.additional_metrics = {} 