# llm_interface.py

import os
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage
from config import read_configs

configs = read_configs("/home/raha/agent_factory/coder-agent/config.json")
os.environ["OPENAI_API_KEY"] = configs["chatbot"]["OPENAI_API_KEY"]

model = ChatOpenAI(model="gpt-4.1-2025-04-14", streaming=True)

def predict(message, history):
    history_langchain_format = []
    for msg in history:
        if msg["role"] == "user":
            history_langchain_format.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            history_langchain_format.append(AIMessage(content=msg["content"]))
    history_langchain_format.append(HumanMessage(content=message))

    response = ""
    for chunk in model.stream(history_langchain_format):
        response += chunk.content
        yield response  # Gradio will stream tokens as they arrive!