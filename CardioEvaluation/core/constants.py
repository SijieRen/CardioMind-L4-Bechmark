"""
Constants used throughout the medical diagnosis evaluation system.

This module contains field names, default values, and other constants
that are used across multiple modules in the evaluation system.
"""


class DiagnosisFields:
    """Constants for diagnosis field names"""
    INITIAL_DIAGNOSIS = "初步诊断"
    PRIMARY_DIAGNOSIS = "主要诊断"
    SECONDARY_DIAGNOSIS = "次要诊断" 
    DIFFERENTIAL_DIAGNOSIS = "鉴别诊断"
    
    NAME = "名称"
    DISEASE_TYPE = "疾病分型"
    ETIOLOGY_TYPE = "病因分型"
    DIAGNOSIS_EVIDENCE = "诊断依据"
    EXCLUDABLE = "是否可排除"
    SUPPORTING_EVIDENCE = "支持依据"
    OPPOSING_EVIDENCE = "不支持依据"
    RECOMMENDED_TESTS = "对应检查推荐"
    ALT_RECOMMENDED_TESTS = "检查推荐"
    SELECTION_BASIS = "选取依据" 