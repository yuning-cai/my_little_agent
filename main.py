"""
=====================================================================
V2 —— 多轮对话记忆（给 Agent 装上"大脑"）
=====================================================================

【V1 的问题】
每次调用 run_agent，messages 都是新建的、用完就丢，
所以 Agent 记不住上一轮说过什么。

【V2 的目标】
让 messages 变成"长期记忆"，实现持续对话：
1. messages 提升到函数外面，只初始化一次（放一条 system 人设）
2. 加 while True 主循环，用 input() 持续对话
3. 每轮结束后，把 assistant 的回复也记进 messages

【学完后你该感受到】
先问"现在几点了"，下一轮问"你刚才说几点了？"
它还能答上来 —— 这就是"记忆"。

【退出方式】输入 exit / quit / 退出 结束对话
=====================================================================
"""

import os
import sys
import json
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI

sys.stdout.reconfigure(encoding="utf-8")
load_dotenv(override=True)

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY", ""),
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
)
model_name = os.getenv("MODEL_NAME", "gpt-4o-mini")


# ------------------------------------------------------------------
# 工具函数（和 V1 完全一样）
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
]

available_tools = {
    "get_current_time": get_current_time,
    "calculate": calculate,
}


# ==================================================================
# 【关键改动 1】messages 提升到函数外面，成为"长期记忆"
# 这里只放一条 system 人设，用户消息在 run_agent 里动态追加
# ==================================================================
messages = [
    {"role": "system", "content": "你是一个智能助理。如果用户的问题需要借助工具，请主动调用工具。"}
]


# ==================================================================
# 核心函数：不再新建 messages，而是持续使用外面的长期记忆
# ==================================================================
def run_agent(user_query: str):
    # TODO 1：把用户这句话追加进长期记忆
    # 提示：messages.append({"role": "user", "content": user_query})
    messages.append(
        {"role":"user","content":user_query}
    )
    # TODO 2：第一次调用大模型（带 tools，和 V1 一样）
    # 提示：response = client.chat.completions.create(model=model_name, messages=messages, tools=tools, tool_choice="auto")
    #      response_message = response.choices[0].message
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )

    response_message = response.choices[0].message

    # TODO 3：如果 tool_calls 非空，走工具流程（和 V1 完全一样）
    #   a. messages.append(response_message)
    #   b. for tool_call in ...: 执行工具，拿结果
    #   c. messages.append({"role": "tool", ...})
    #   d. 第二次调用，得到 final_reply
    # TODO 4：否则 final_reply = response_message.content
    if response_message.tool_calls:
        messages.append(response_message)
        for tool_call in response_message.tool_calls:
            func_name = tool_call.function.name
            func_args = json.loads(tool_call.function.arguments)
            func_to_call = available_tools.get(func_name)

            if func_to_call:
                tool_output = func_to_call(**func_args)
            else:
                tool_output = "调用工具失败"

            messages.append({
                "role":"tool",
                "tool_call_id":tool_call.id,
                "content":tool_output
            })

        second_response = client.chat.completions.create(
            model=model_name,
            messages=messages,
        )

        final_reply = second_response.choices[0].message.content

    else:
        final_reply = response_message.content


    # TODO 5：【关键改动 2】把最终回复也 append 进 messages
    # 为什么：否则下一轮军师不记得自己上一轮说过什么
    # 提示：messages.append({"role": "assistant", "content": final_reply})
    messages.append(
        {"role":"assistant","content":final_reply}
    )

    # TODO 6：打印最终回复
    # 提示：print(f"🤖 助手: {final_reply}")
    print(final_reply)



# ==================================================================
# 【关键改动 3】主循环：持续对话，直到用户输入 exit/退出
# ==================================================================
print("多轮对话 Agent 已启动！输入 exit / 退出 结束对话")

while True:
    user_input = input("\n👤 你: ")

    if user_input.strip().lower() in ["exit", "quit", "退出", "q"]:
        print("👋 再见！")
        break

    run_agent(user_input)
