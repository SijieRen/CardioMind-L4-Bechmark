"""
Diagnosis matching utilities for medical diagnosis evaluation.

This module contains functions for matching predicted diagnoses
with reference diagnoses using various comparison methods.
"""

import sys
import os
# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Tuple
from utils.model_metric_utils import check_diagnosis_match_with_emr, is_exact_match

# Import from our package
try:
    from core.constants import DiagnosisFields
    from utils.text_processing import TextProcessor
except ImportError:
    # Fallback for direct execution
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.constants import DiagnosisFields
    from utils.text_processing import TextProcessor


class DiagnosisMatchingUtils:
    """Utility class for diagnosis matching operations"""
    
    @staticmethod
    def evaluate_diagnosis_match(pred_list: List[str], ref_diagnosis: str, emr: str, sample_count=9) -> Tuple[bool, str, int]:
        """Evaluate if prediction matches reference diagnosis"""
        result, matched_diagnosis, idx = check_diagnosis_match_with_emr(pred_list, ref_diagnosis, emr, sample_count=sample_count)
        return result, matched_diagnosis, idx
    
    @staticmethod
    def extract_differential_diagnosis_data(pred_json: dict) -> Tuple[List[str], List[str]]:
        """Extract differential diagnosis data from prediction"""
        dd_list = []
        excludable_list = []
        
        for dd_item in pred_json[DiagnosisFields.DIFFERENTIAL_DIAGNOSIS]:
            dd_list.append(TextProcessor.clean_diagnosis_name(dd_item[DiagnosisFields.NAME]))
            if str(dd_item[DiagnosisFields.EXCLUDABLE]) == "False":
                excludable_list.append(TextProcessor.clean_diagnosis_name(dd_item[DiagnosisFields.NAME]))
                
        return dd_list, excludable_list
    
    @staticmethod
    def check_exact_match(value1: str, value2: str) -> bool:
        """Check if two values match exactly"""
        return is_exact_match(value1, value2)