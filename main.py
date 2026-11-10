"""
=====================================================================
V4 —— RAG 检索增强生成（精简框架）
=====================================================================

详细教程在对话里，照着教程把 TODO 补全即可运行。

新知识（需要你自己写）：
- embed：文字 → 向量
- cosine_similarity：余弦相似度
- add_notes：知识入库
- search_notes：检索

复习（V1/V3 学过，框架里已给完整代码或提示）：
- 工具函数、工具名册、映射表、Agent 循环
=====================================================================
"""

import os
import sys
import json
import math
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI

sys.stdout.reconfigure(encoding="utf-8")
load_dotenv(override=True)

# ------------------------------------------------------------------
# 1. 两个客户端（样板代码，照抄即可）
# ------------------------------------------------------------------
chat_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY", ""),
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
)
model_name = os.getenv("MODEL_NAME", "gpt-4o-mini")

embed_client = OpenAI(
    api_key=os.getenv("EMBEDDING_API_KEY", ""),
    base_url=os.getenv("EMBEDDING_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
)
embed_model = os.getenv("EMBEDDING_MODEL", "text-embedding-v3")

# ------------------------------------------------------------------
# 2. 知识库：两个列表，下标一一对应
# ------------------------------------------------------------------
documents = []   # 存文本
vectors = []     # 存向量


# ------------------------------------------------------------------
# 3. TODO① embed：文字 → 向量
# ------------------------------------------------------------------
def embed(text: str) -> list:
    # 提示：调用 embed_client.embeddings.create(model=embed_model, input=text)
    #      然后返回 resp.data[0].embedding
    resp = embed_client.embeddings.create(
        model=embed_model,
        input=text
    )
    return resp.data[0].embedding


# ------------------------------------------------------------------
# 4. TODO② 余弦相似度：衡量两个向量方向有多接近
# ------------------------------------------------------------------
def cosine_similarity(a: list, b: list) -> float:
    # 提示：
    #   dot = sum(x * y for x, y in zip(a, b))        # 点积
    #   norm_a = math.sqrt(sum(x * x for x in a))     # a 的长度
    #   norm_b = math.sqrt(sum(x * x for x in b))     # b 的长度
    #   返回 dot / (norm_a * norm_b)（注意分母为 0 的情况）
    dot = sum(x*y for x,y in zip(a,b))
    norm_a = math.sqrt(sum(x*x for x in a))
    norm_b = math.sqrt(sum(x*x for x in b))
    if norm_a * norm_b == 0:
        return 0
    else:
        return dot / (norm_a * norm_b)



# ------------------------------------------------------------------
# 5. TODO③ 知识入库：把每条笔记向量化后存进两个列表
# ------------------------------------------------------------------
def add_notes(notes: list):
    # 提示：for note in notes:
    #           documents.append(note)       # 文本存进 documents
    #           vectors.append(embed(note))  # 向量存进 vectors
    for note in notes:
        documents.append(note)
        vectors.append(embed(note))


# ------------------------------------------------------------------
# 6. TODO④ 检索：根据问题找到最相关的笔记片段
# ------------------------------------------------------------------
def search_notes(query: str) -> str:
    # 提示：
    #   1. query_vec = embed(query)                     # 问题向量化
    #   2. scores = [cosine_similarity(query_vec, v) for v in vectors]  # 算相似度
    #   3. ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:3]
    #   4. 返回 "\n".join(f"- {documents[i]}" for i in ranked)
    query_vec = embed(query)
    scores = [cosine_similarity(query_vec,v) for v in vectors]
    ranked = sorted(range(len(scores)),key = lambda i:scores[i],reverse=True)[:3]
    return "\n".join(f"-{documents[i]}" for i in ranked)


# ------------------------------------------------------------------
# 7. 原有工具（V1 学过，直接给你）
# ------------------------------------------------------------------
def get_current_time() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def calculate(expression: str) -> str:
    try:
        allowed_chars = set("0123456789+-*/(). ")
        if not set(expression).issubset(allowed_chars):
            return "错误：表达式包含非法字符"
        return str(eval(expression))
    except Exception as e:
        return f"计算出错: {str(e)}"


# ------------------------------------------------------------------
# 8. 工具名册：TODO⑤ 把 search_notes 的 schema 补进去
# ------------------------------------------------------------------
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "当用户询问当前时间、日期或今天几号几点时调用此工具。",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "用于计算数学表达式，例如加减乘除算式。",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "需要计算的数学表达式字符串",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_notes",
            "description": "当用户询问自己的笔记/文档内容时，调用此工具检索知识库",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "需要检索的问题"
                    }
                },
                "required": ["query"]
            }
        },
    }
]


# ------------------------------------------------------------------
# 9. 映射表：TODO⑥ 把 search_notes 映射进去
# ------------------------------------------------------------------
available_tools = {
    "get_current_time": get_current_time,
    "calculate": calculate,
    # TODO⑥：补上 "search_notes": search_notes
    "search_notes": search_notes
}


# ------------------------------------------------------------------
# 10. Agent 循环（V3 学过，直接给你）
# ------------------------------------------------------------------
messages = [
    {"role": "system", "content": "你是一个智能助理。如果用户的问题需要借助工具，请主动调用工具。"}
]


def run_agent(user_query: str):
    messages.append({"role": "user", "content": user_query})

    max_steps = 10
    step = 0
    final_reply = "处理超时"

    while step < max_steps:
        step += 1
        response = chat_client.chat.completions.create(
            model=model_name,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
        response_message = response.choices[0].message

        if not response_message.tool_calls:
            final_reply = response_message.content
            break

        messages.append(response_message)
        for tool_call in response_message.tool_calls:
            func_name = tool_call.function.name
            func_args = json.loads(tool_call.function.arguments)
            func_to_call = available_tools.get(func_name)

            print(f"🛠️ [调用工具] {func_name}")

            if func_to_call:
                tool_output = func_to_call(**func_args)
            else:
                tool_output = "工具调用失败"

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(tool_output),
            })

    messages.append({"role": "assistant", "content": final_reply})
    print(f"🤖 助手: {final_reply}")


# ------------------------------------------------------------------
# 11. 启动：加载示例笔记 + 主循环
# ------------------------------------------------------------------
sample_notes = [
    "Python 是一种解释型、面向对象的高级编程语言，由 Guido van Rossum 在 1989 年发明。",
    "Agent 是指能够自主感知环境、做出决策并执行动作的智能程序。",
    "函数调用（Function Calling）让大模型能通过调用外部工具来完成它自己无法完成的任务。",
    "RAG 是检索增强生成，先检索相关文档，再把文档喂给大模型来回答问题。",
    "向量数据库用于存储和检索 embedding 向量，常见的有 Chroma 和 FAISS。",
]

add_notes(sample_notes)

print("RAG Agent 已启动！输入 exit / 退出 结束对话")

while True:
    user_input = input("\n👤 你: ")

    if user_input.strip().lower() in ["exit", "quit", "退出", "q"]:
        print("👋 再见！")
        break

    run_agent(user_input)
