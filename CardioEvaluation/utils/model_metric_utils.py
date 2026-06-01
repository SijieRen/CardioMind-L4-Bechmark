# -*- coding: utf-8 -*-
"""
医疗诊断评估器 - 核心处理函数模块

该模块包含用于医疗诊断和证据匹配评估的核心功能函数，主要用于：
1. 文本处理和分割
2. 诊断匹配评估
3. 证据匹配评估
4. 幻觉检测
5. 评估指标计算

Author: Medical AI Team
Date: 2024
"""

import re
import json
import warnings
from difflib import SequenceMatcher
from typing import List, Tuple, Union, Dict, Any
from utils.r1_request import get_response_diagnosis
from config.prompt import diagnosis_list_prompt, evidence_hallucination_prompt, trace_query_prompt_o


# ====================================================================
# 数学计算和指标评估函数
# ====================================================================

def calculate_f1_score(precision: float, recall: float) -> float:
    """
    计算F1分数
    
    Args:
        precision: 精确率 (0-1之间的值)
        recall: 召回率 (0-1之间的值)
        
    Returns:
        F1分数，保留4位小数
    """
    if precision + recall == 0:
        return 0.0  # 避免除以零
    
    f1 = 2 * (precision * recall) / (precision + recall)
    return round(f1, 4)


def calculate_iou(list_a_length: int, list_b_length: int, intersection_length: int) -> float:
    """
    计算IoU (Intersection over Union)
    
    Args:
        list_a_length: 列表A的长度 (>=0的整数)
        list_b_length: 列表B的长度 (>=0的整数)
        intersection_length: 两个列表的交集长度 (>=0的整数)
        
    Returns:
        IoU值 (0-1之间的浮点数)
        
    Raises:
        ValueError: 当输入参数不符合要求时
    """
    # 参数验证
    if list_a_length < 0 or list_b_length < 0 or intersection_length < 0:
        raise ValueError("输入长度必须为非负整数")
    
    if intersection_length > min(list_a_length, list_b_length):
        raise ValueError("交集长度不能超过任一列表的长度")
    
    union_length = list_a_length + list_b_length - intersection_length
    
    if union_length == 0:
        return 0.0  # 避免除以零
    
    iou = intersection_length / union_length
    return round(iou, 4)


def round_number(number: float, decimal_places: int = 2) -> float:
    """数值四舍五入"""
    return round(number, decimal_places)


# ====================================================================
# 文本处理和分割函数
# ====================================================================

def split_medical_text_items(input_list: List[str]) -> Tuple[int, List[str]]:
    """
    分割医学文本项目，提取有效的医学信息片段
    
    Args:
        input_list: 输入的文本列表
        
    Returns:
        Tuple[结果列表长度, 结果列表]
    """
    DELIMITERS = ["；", "。"]
    MEDICAL_PREFIXES = [
        "症状：", "查体：", "辅助检查：", "既往史：", 
        "体格检查：", "病史：", "绝对排除证据：", "待验证证据："
    ]
    
    result = []
    
    for item in input_list:
        # 使用多个分隔符进行分割
        processed_item = item
        for delimiter in DELIMITERS:
            processed_item = processed_item.replace(delimiter, "|")
        
        # 分割后的子字符串列表
        substrings = processed_item.split("|")
        
        for substring in substrings:
            # 跳过空字符串
            if not substring.strip():
                continue
            
            # 移除医学前缀
            cleaned_substring = _remove_medical_prefix(substring, MEDICAL_PREFIXES)
            if not cleaned_substring:
                continue
            
            # 按逗号分割并添加到结果
            sub_parts = cleaned_substring.split("，")
            result.extend(sub_parts)
    
    return len(result), result


def _remove_medical_prefix(text: str, prefixes: List[str]) -> str:
    """
    移除医学前缀的辅助函数
    
    Args:
        text: 待处理的文本
        prefixes: 前缀列表
        
    Returns:
        移除前缀后的文本，如果没有匹配的前缀则返回空字符串
    """
    for prefix in prefixes:
        if text.startswith(prefix):
            return text[len(prefix):].strip()
    return ""


def remove_duplicates_preserve_order(input_list: List[str]) -> List[str]:
    """
    去除列表中的重复项，保持原有顺序
    
    Args:
        input_list: 输入列表
        
    Returns:
        去重后的列表
    """
    return list(dict.fromkeys(input_list))


# ====================================================================
# 大模型响应函数
# ====================================================================

def get_hallucination_response(emr: str, query: str, pattern=None, sample_count=1) -> Tuple[Any, Any, str]:
    """获取幻觉检测的大模型响应"""
    prompt = evidence_hallucination_prompt
    question = prompt.format(emr, query)
    return get_response_diagnosis(question, pattern=pattern, sample_count=sample_count)


def get_evidence_reference_response(reference_list: List[str], query: str, sample_count: int=1) -> Tuple[Any, Any, str]:
    """获取(证据、检查、选取依据)参考列表的大模型响应"""
    prompt = trace_query_prompt_o
    question = prompt.format(reference_list, query)
    return get_response_diagnosis(question, sample_count=sample_count)


def get_diagnosis_emr_response(predict_list: List[str], reference: str, emr: str, sample_count=1) -> Tuple[Any, Any, str]:
    """获取基于EMR的诊断匹配响应"""
    prompt = diagnosis_list_prompt
    question = prompt.format(predict_list, reference, emr)
    return get_response_diagnosis(question, temperature=0.0,sample_count=sample_count)


# ====================================================================
# 匹配检测函数
# ====================================================================

def is_exact_match(reference: str, query: str) -> bool:
    """检查是否完全匹配"""
    return query == reference

def check_diagnosis_match_with_emr(predict_list: List[str], reference: str, emr: str, sample_count=1, diagnosis_type: str="") -> Tuple[bool, str, Union[str, int]]:
    """
    检查诊断列表是否与参考诊断匹配（考虑EMR）
    
    Args:
        query: 查询诊断列表
        reference: 参考诊断名称
        emr: 电子病历
        
    Returns:
        Tuple[是否匹配, 正确诊断, 正确诊断索引]
    """
    # 首先尝试精确匹配
    for idx, diagnosis in enumerate(predict_list):
        if is_exact_match(reference, diagnosis) or diagnosis in reference:
            return True, diagnosis, idx
    
    # 是否带诊断类型
    if len(diagnosis_type) > 0:
        predict_list = [f"{diagnosis}({diagnosis_type})" for diagnosis in predict_list]

    # 如果没有精确匹配，使用大模型判断
    _, _, llm_response = get_diagnosis_emr_response(predict_list, reference, emr, sample_count=sample_count)
    
    # 解析大模型响应
    match_pattern = r'预测推理.*?是否存在与.*?一致的诊断：([是|否])'
    match_results = re.findall(match_pattern, llm_response)
    
    if not match_results or match_results[0] != "是":
        return False, "", ""
    
    # 提取正确诊断和索引
    try:
        diagnosis_pattern = r'预测推理.*?的情况\)[:：]\[?([^\]]*)\]?'
        diagnosis_matches = re.findall(diagnosis_pattern, llm_response)
        correct_diagnosis = re.split(r',|，', diagnosis_matches[0])[0] if diagnosis_matches else ""
        
        index_pattern = r'判定为一致的诊断在.*?开始编码\）[:：]\[?([^\]]*)\]?'
        index_matches = re.findall(index_pattern, llm_response)
        correct_diagnosis_idx = int(re.split(r',|，', index_matches[0])[0]) if index_matches else ""
        
        return True, correct_diagnosis, correct_diagnosis_idx
    except (IndexError, ValueError):
        return True, "", ""


def check_hallucination(emr: str, query: str,sample_count=1) -> bool:
    """
    检查查询是否为幻觉
    
    Args:
        emr: 电子病历
        query: 查询内容
        
    Returns:
        是否为幻觉
    """
    _, _, llm_response = get_hallucination_response(emr, query, pattern=r'是否为幻觉：([是|否])')
    hallucination_results = re.findall(r'是否为幻觉：([是|否])', llm_response)
    print("幻觉检测结果: {}".format(hallucination_results))
    
    #todo: 这里没有结果的话，是不是应该返回True更稳妥？
    if not hallucination_results:
        return False
    
    return hallucination_results[0] == "是"



# ====================================================================
# 列表匹配和检索函数
# ====================================================================

def levenshtein_ratio(str1: str, str2: str) -> float:
    """
    计算两个字符串的相似度比（使用difflib.SequenceMatcher）
    
    Args:
        str1: 第一个字符串
        str2: 第二个字符串
        
    Returns:
        相似度比，范围0-1，1表示完全相同
    """
    if str1 == str2:
        return 1.0
    
    if len(str1) == 0 or len(str2) == 0:
        return 0.0
    
    # 使用difflib计算相似度
    similarity = SequenceMatcher(None, str1, str2).ratio()
    return round(similarity, 4)


def find_query_evidence_in_reference_list(reference_list: List[str], query: str) -> List[int]:
    """
    在参考列表中查找匹配的证据（原始版本）
    
    Args:
        reference_list: 参考列表
        query: 查询内容
        
    Returns:
        匹配的索引列表
    """
    matched_indices = []
    for idx, reference in enumerate(reference_list):
        if query == reference or query in reference:
            matched_indices.append(idx)
    return matched_indices


def find_query_evidence_in_reference_list_with_similarity(reference_list: List[str], query: str, use_similarity: bool=True) -> Tuple[List[int], float]:
    """
    在参考列表中查找匹配的证据，使用精确匹配和相似度比较
    
    Args:
        reference_list: 参考列表
        query: 查询内容
        
    Returns:
        Tuple[匹配的索引列表, 最高的相似度比]
    """
    matched_indices = {}
    
    for idx, reference in enumerate(reference_list):
        # 首先尝试精确匹配
        if query == reference:
            matched_indices[reference] = 1.0
        elif use_similarity and query in reference:
            matched_indices[reference] = 1.0
        else:
            # 计算相似度比
            ratio = levenshtein_ratio(query, reference)
            matched_indices[reference] = ratio
    return matched_indices


def find_query_diagnosis_in_reference_list(reference_list: List[str], query: str) -> List[int]:
    """
    在参考列表中查找匹配的诊断（只返回第一个匹配）
    
    Args:
        reference_list: 参考列表
        query: 查询内容
        
    Returns:
        匹配的索引列表（最多包含一个元素）
    """
    for idx, reference in enumerate(reference_list):
        if query == reference or query in reference:
            return [idx]
    return []


def match_evidence_in_reference_list(reference_list: List[str], query: str, sample_count: int=1, use_similarity: bool=True) -> List[str]:
    """
    在参考列表中匹配证据/检查/选取依据（带相似度优化）
    
    Args:
        reference_list: 参考列表
        query: 查询内容
        sample_count: 采样次数
        
    Returns:
        匹配的索引列表（字符串格式）
        
    逻辑说明:
        1. 如果存在相似度 > 0.9 的匹配，直接返回
        2. 如果所有相似度 < 0.5，跳过大模型调用
        3. 相似度在 0.5-0.9 之间时，调用大模型进一步判断
    """
    # 首先尝试直接匹配和相似度匹配
    match_similarity = find_query_evidence_in_reference_list_with_similarity(reference_list, query, use_similarity=use_similarity)

    direct_matches = []
    may_matched = []
    may_matched_idx = {}
    for idx, reference_item in enumerate(reference_list):
        ratio = match_similarity[reference_item]
        if use_similarity:
            if ratio > 0.9:
                direct_matches.append(idx)
            elif ratio > 0.5:
                may_matched.append(reference_item)
                may_matched_idx[reference_item] = idx
        else:
            if ratio == 1.0:
                direct_matches.append(idx)
            else:
                may_matched.append(reference_item)
                may_matched_idx[reference_item] = idx

    # 如果有直接匹配或相似度大于0.9的匹配，直接返回
    if direct_matches:
        return direct_matches
    
    # 如果相似度在0.5-0.9之间，使用大模型
    candidate_traces = []
    for idx, evidence in enumerate(may_matched):
        candidate_traces.append(f"s{idx}:'{evidence.strip()}'")
    
    _, _, llm_response = get_evidence_reference_response(candidate_traces, query, sample_count=sample_count)
    
    # 提取匹配结果
    matches = re.findall(r"'s(\d+)'", str(llm_response))
    for match in matches:
        match = int(match)
        assert match < len(may_matched)
        direct_matches.append(may_matched_idx[may_matched[match]])
    
    # if len(matches) > 1:
    #     warnings.warn("发现多个参考匹配!\nquery={},candidate_trace={},matches={}\n".format(query, candidate_traces,matches))
    
    return direct_matches


# ====================================================================
# 综合评估函数
# ====================================================================

class EvaluationResults:
    """评估结果数据类"""
    def __init__(self):
        self.f1_score: float = 0.0
        self.iou: float = 0.0
        self.precision: float = 0.0
        self.recall: float = 0.0
        self.hallucination_rate: float = 0.0
        self.precision_list: List[str] = []
        self.hallucination_list: List[str] = []
        self.recall_list: List[str] = []


def evaluate_medical_evidence(
    predicted_count:int,
    reference_count:int,
    predicted_evidence_list: List[str],
    reference_evidence_list: List[str],
    emr: str,
    enable_hallucination_check: bool = True,
    use_similarity: bool = True
) -> EvaluationResults:
    """
    评估医疗证据的匹配情况
    
    Args:
        predicted_evidence_list: 预测的证据列表
        reference_evidence_list: 参考证据列表
        emr: 电子病历
        enable_hallucination_check: 是否启用幻觉检测，默认为True
        
    Returns:
        评估结果对象
    """
    results = EvaluationResults()
    
    # 处理边界情况
    if predicted_count == 0 and reference_count == 0:
        results.f1_score = results.iou = results.precision = results.recall = 1.0
        results.hallucination_rate = 0.0
        return results
    
    # label不为空，预测为空
    if predicted_count == 0:
        return results
    
    # label为空，但有预测结果
    if reference_count == 0:
        # 只检查幻觉
        if enable_hallucination_check:
            for evidence in predicted_evidence_list:
                if check_hallucination(emr, evidence):
                    results.hallucination_list.append(evidence)
            results.hallucination_rate = round( len(results.hallucination_list) / predicted_count, 4)
        else:
            results.hallucination_rate = 0.0        
        return results
    
    # 正常评估流程
    for predicted_evidence in predicted_evidence_list:
        matches = match_evidence_in_reference_list(reference_evidence_list, predicted_evidence, sample_count=1, use_similarity=use_similarity)
        
        if not matches:
            # 没有匹配，检查是否为幻觉
            if enable_hallucination_check and check_hallucination(emr, predicted_evidence):
                results.hallucination_list.append(predicted_evidence)
        else:
            # 有匹配，记录为正确预测
            results.precision_list.append(predicted_evidence)
            
            # 记录被召回的参考证据
            for match_idx in matches:
                idx = int(match_idx)
                results.recall_list.append(reference_evidence_list[idx])
            ## todo: 这里匹配之后，应该从reference_evidence_list中删除已经匹配的证据
    
    # 去重处理
    results.hallucination_list = remove_duplicates_preserve_order(results.hallucination_list)
    results.precision_list = remove_duplicates_preserve_order(results.precision_list)
    results.recall_list = remove_duplicates_preserve_order(results.recall_list)
    
    # 检查是否有重复项被移除
    if (len(results.hallucination_list) != len([e for e in predicted_evidence_list if e in results.hallucination_list]) or
        len(results.precision_list) != len([e for e in predicted_evidence_list if e in results.precision_list])):
        warnings.warn("在不同测试中可能存在重复项!", UserWarning)
    
    # 计算指标
    precision_count = len(results.precision_list)
    recall_count = len(results.recall_list)
    hallucination_count = len(results.hallucination_list)
    
    results.precision = round(precision_count / predicted_count, 4)
    results.recall = round(recall_count / reference_count, 4)
    if enable_hallucination_check:
        results.hallucination_rate = round(hallucination_count / predicted_count, 4)
    else:   
        results.hallucination_rate = 0.0
        
    results.f1_score = calculate_f1_score(results.precision, results.recall)
    
    # 计算IoU（处理特殊情况）
    if precision_count > reference_count:
        results.iou = 1.0
    else:
        results.iou = calculate_iou(predicted_count, reference_count, precision_count)
    
    return results