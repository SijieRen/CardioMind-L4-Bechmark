import json
import requests
import concurrent.futures
from tqdm import tqdm
import re
# 请求的 URL
url = "http://10.21.76.144:1025/v1/chat/completions"

# 请求头
headers = {
    "Content-Type": "application/json"
}

def call_r1_api_and_parse(data, headers):
    """
    调用R1 API并解析响应结果
    返回: (think_ans, final_ans)
    """
    response = requests.post(url, json=data, headers=headers, stream=True)
    response.raise_for_status()

    think_ans = ""
    final_ans = ""
    for response_data in response.iter_lines():
        if response_data:
            response_str = response_data.decode("utf-8")
            if response_str.startswith("data: "):
                response_str = response_str[len("data: "):]
            try:
                response_json = json.loads(response_str)
                if "content" not in response_json["choices"][0]["delta"]:
                    content = response_json["choices"][0]["delta"].get("reasoning_content", "")
                    think_ans += content
                else:
                    content = response_json["choices"][0]["delta"].get("content", "")
                    final_ans += content
            except:
                pass
    
    return think_ans, final_ans

def get_response_diagnosis(content, temperature=0.0, sample_count=1, pattern=r'<预测推理诊断列表>中是否存在与<标准诊断结果>一致的诊断：([是|否])'):
    """
    获取R1诊断响应
    Args:
        content: 请求内容
        temperature: 温度参数
        sample_count: 采样次数，默认为1次
    """
    # 发送请求
    try:
        data = {
            "model": "model",
            "messages": [
                {
                    "role": "user",
                    "content": content
                }
            ],
            "stream": True,
            "temperature": temperature
        }
        
        # 如果只采样1次，直接返回结果
        if sample_count == 1:
            think_ans, final_ans = call_r1_api_and_parse(data, headers)
            return "successful", think_ans, final_ans
        
        # 多次采样的逻辑
        pos_ans = "" 
        pos_think = ""
        neg_ans = ""
        neg_think = ""
        pos_count = 0
        neg_count = 0
        
        for idx in range(sample_count):
            think_ans, final_ans = call_r1_api_and_parse(data, headers)

            res = re.findall(pattern, final_ans)
            if not res: continue
                
            res = res[0]

            if res == "是":
                pos_count += 1
                if pos_count == 1:
                    pos_ans = final_ans
                    pos_think = think_ans
            else:
                neg_count += 1
                if neg_count == 1:
                    neg_ans = final_ans
                    neg_think = think_ans
            
            # 达到多数票决条件时提前返回
            majority_threshold = (sample_count + 1) // 2
            if pos_count >= majority_threshold:
                return "successful", pos_think, pos_ans
            if neg_count >= majority_threshold:
                return "successful", neg_think, neg_ans

        # 如果没有达到多数票决，返回票数更多的结果
        if pos_count > neg_count:
            return "successful", pos_think, pos_ans
        else:
            return "successful", neg_think, neg_ans

    except requests.exceptions.RequestException as e:
        print(f"请求失败：{e}")
        return "error", "", str(e)
