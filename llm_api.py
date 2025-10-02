# models/ollama_api.py
import requests
import re
from .get_prompt import get_prompt_from_question
from .logger import logger

OLLAMA_URL = "http://10.48.62.143:11434/api/generate"


def build_prompt(question: str, context: str) -> str:
    """
    构建传给大模型的提示词 (Prompt)
    :param question: 用户问题
    :param context: RAG 检索结果
    :return: 拼接后的完整 Prompt
    """
    prompt = f"""你是一名智能问答助手。
下面给出用户的问题和从知识库中检索到的参考信息。
请结合检索内容进行回答，如果参考信息中没有相关内容，请基于你的知识回答。

【用户问题】：
{question}

【参考信息】：
{context}

【回答要求】：
- 若参考信息不相关，请忽略
"""
    return prompt

def extract_after_think(text: str) -> str:
    """
    提取文本中最后一个 </think> 标签之后的内容
    """
    # 使用正则匹配 </think> 后面的内容
    match = re.search(r"</think>\s*(.*)", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()  # 如果没找到，返回原始文本

def call_ollama(question):
    """
    调用本地 Ollama API
    :param prompt: 用户输入
    :param model: 使用的模型名称 (如 llama2、llama3、mistral)
    :return: 模型返回的回答
    """

    # 从知识库获取 context
    try:
        context_list = get_prompt_from_question(question)
        context = "\n".join(context_list) if context_list else ""
        logger.info(f"生成 RAG context，长度 {len(context_list)} 条")
    except Exception as e:
        context = ""
        logger.error(f"获取 RAG context 出错: {e}")

    prompt = build_prompt(question, context)
    logger.info(f"发送给 Ollama 的 Prompt:\n{prompt}")

    payload = {
        "model": "deepseek-r1:70b",
        "prompt": prompt,
        "stream": False  # 不启用流式，返回完整结果
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload)
        resp.raise_for_status()
        data = resp.json()
        answer = data.get("response", "No response key in output")
        logger.info(f"Ollama 返回结果: {answer[:200]}{'...' if len(answer) > 200 else ''}")
        return answer
    except Exception as e:
        logger.error(f"调用 Ollama 出错: {e}")
        return f"调用 Ollama 出错: {e}"
