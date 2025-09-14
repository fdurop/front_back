import requests
import re

def ask_ollama(question):
    payload = {
        "model": "deepseek-r1:1.5b",
        "prompt": question,
        "stream": False
    }
    response = requests.post("http://localhost:11434/api/generate", json=payload)
    data = response.json()
    print(data)  # 打印查看结构
    return data.get('response', 'No response key in output')

# 调用函数，得到完整文本
# raw_text = ask_ollama("说一句李白的诗")


def extract_final_answer(raw_text: str) -> str:
    """
    从模型返回的原始文本中提取去除 <think> 标签后的最终回答。

    参数：
    - raw_text: str，模型返回的包含 <think> 思考过程和最终回答的字符串。

    返回：
    - str，纯净的最终回答文本。
    """
    match = re.split(r'</think>\s*', raw_text)
    if len(match) > 1:
        final_answer = match[1].strip()
    else:
        final_answer = raw_text.strip()
    return final_answer

# 示例用法
# raw_text = 模型返回的文本
# print(extract_final_answer(raw_text))
