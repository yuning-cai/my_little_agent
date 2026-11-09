"""
=====================================================================
V3 —— ReAct 自主推理循环（让 Agent 学会"连续干活"）
=====================================================================

【V2 的问题】
工具调用是"一次性"的：if tool_calls 只执行一轮工具就总结。
所以复合任务（如"查时间 + 算数"）它只会做第一步。

【V3 的目标】
把 if 改成 while 循环，让 Agent 能连续调用多个工具，
直到它自己判断"任务完成、不用再调工具了"才停。

【核心思路】
while step < max_steps:
    调用模型(带 tools)
    if 模型不再甩单:  # 任务完成
        拿到最终答案，break
    否则:  # 模型甩单了，执行工具，继续循环
        执行工具，把结果追加进 messages

【必须加的安全阀】
max_steps 防死循环：模型万一一直甩单，超过上限就强制停止。

【学完后你该感受到】
问它"现在几点了？再过 2 小时 30 分钟是几点？"
它能连续调 2 次工具，最终给出完整答案。

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
# 工具函数（和 V2 完全一样）
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
# 长期记忆（和 V2 一样，提升到函数外面）
# ==================================================================
messages = [
    {"role": "system", "content": "你是一个智能助理。如果用户的问题需要借助工具，请主动调用工具。"}
]


# ==================================================================
# 核心函数：V3 的关键就是把 V2 的 if 改成 while 循环
# ==================================================================
def run_agent(user_query: str):
    # 先把用户这句话追加进长期记忆
    messages.append({"role": "user", "content": user_query})

    # TODO 1：设置防死循环的变量
    # 提示：
    #   max_steps = 10          # 最多循环几次就强制停止
    #   step = 0                # 计数器
    #   final_reply = "处理超时"  # 兜底：万一循环耗尽还没拿到答案，就用这个
    max_steps = 10
    step = 0
    final_reply = "处理超时"

    # TODO 2：写 while 循环（核心！把 V2 的 if 换成 while）
    # 提示：while step < max_steps:
    #   2a. step += 1  （每循环一次，计数器 +1）
    #
    #   2b. 调用模型（带 tools），拿到 response_message
    #       提示：和 V2 一样 client.chat.completions.create(...)
    #
    #   2c. if not response_message.tool_calls:   （模型不再甩单 → 任务完成）
    #           final_reply = response_message.content
    #           break                              （退出循环！）
    #
    #   2d. 否则（模型甩单了）：
    #           messages.append(response_message)   （记下指令单）
    #           for tool_call in ...:               （执行每个工具）
    #               执行工具，messages.append(tool 结果)
    #           （注意：这里不 break，让循环继续，让模型接着干下一步）
    while step < max_steps:
        step += 1
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )
        response_message = response.choices[0].message
        if not response_message.tool_calls:
            final_reply = response_message.content
            break
        else:
            messages.append(response_message)
            for tool_call in response_message.tool_calls:
                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments)
                func_to_call = available_tools.get(func_name)

                if func_to_call:
                    tool_output = func_to_call(**func_args)
                else:
                    tool_output = "工具调用失败"

                messages.append({
                    "role":"tool",
                    "tool_call_id":tool_call.id,
                    "content":tool_output,
                })



    # TODO 3：循环结束后（break 出来后走到这里）
    #   把最终回复记进长期记忆 + 打印
    # 提示：
    #   messages.append({"role": "assistant", "content": final_reply})
    #   print(f"🤖 助手: {final_reply}")
    messages.append({
        "role":"assistant",
        "content":final_reply
    })
    print(final_reply)



# ==================================================================
# 主循环（和 V2 一样）
# ==================================================================
print("ReAct Agent 已启动！输入 exit / 退出 结束对话")

while True:
    user_input = input("\n👤 你: ")

    if user_input.strip().lower() in ["exit", "quit", "退出", "q"]:
        print("👋 再见！")
        break

    run_agent(user_input)
