"""
Configuration classes for the medical diagnosis evaluation system.

This module contains configuration classes that control the behavior
of the evaluation system, including feature flags and weights.
"""

from typing import Dict
from dataclasses import dataclass


@dataclass
class EvaluationConfig:
    """Configuration for evaluation parameters"""
    enable_evidence_eval: bool = False
    enable_selection_eval: bool = False
    sample_count: int = 9
    weights: Dict[str, float] = None
    
    def __post_init__(self):
        if self.weights is None:
            self.weights = {
                'primary': 0.5,
                'secondary': 0.3, 
                'differential': 0.2
            } 