"""
Text processing utilities for medical diagnosis evaluation.

This module contains functions for cleaning, normalizing, and processing
text data used in medical diagnosis evaluation.
"""

from typing import List, Dict
from core.constants import DiagnosisFields


class TextProcessor:
    """Utility class for text processing operations"""
    
    @staticmethod
    def remove_duplicates(items: List[str]) -> List[str]:
        """Remove duplicates while preserving order"""
        return list(dict.fromkeys(items))
    
    @staticmethod
    def clean_diagnosis_name(name: str) -> str:
        """Clean and normalize diagnosis name"""
        return name.strip(" ")
    
    @staticmethod
    def extract_diagnosis_list(diagnosis_section: List[Dict], field: str = DiagnosisFields.NAME) -> List[str]:
        """Extract diagnosis names from a section"""
        return [TextProcessor.clean_diagnosis_name(item[field]) for item in diagnosis_section]
    
    @staticmethod
    def get_disease_type(diagnosis_data: Dict, fallback_field: str = None) -> str:
        """Get disease type with fallback logic"""
        try:
            return diagnosis_data[DiagnosisFields.DISEASE_TYPE]
        except KeyError:
            fallback = fallback_field or DiagnosisFields.ETIOLOGY_TYPE
            return diagnosis_data.get(fallback, "")
    
    @staticmethod
    def get_test_recommendations(diagnosis_data: Dict) -> tuple[int, List]:
        """Get test recommendations with fallback logic"""
        for field in [DiagnosisFields.RECOMMENDED_TESTS, DiagnosisFields.ALT_RECOMMENDED_TESTS]:
            if field in diagnosis_data:
                recommendations = diagnosis_data[field]
                return len(recommendations), recommendations
        return 0, [] 