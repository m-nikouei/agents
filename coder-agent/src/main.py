import os
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
import gradio as gr
import json
from langchain_core.prompts import PromptTemplate
from langchain.schema import HumanMessage, AIMessage, SystemMessage

def read_configs(config_file):
    f = open(config_file)
    configs = json.load(f)
    f.close()
    return configs

configs = read_configs("/home/raha/agent_factory/coder-agent/config.json")
# Initialize OpenAI LLM
os.environ["OPENAI_API_KEY"] = configs["chatbot"]["OPENAI_API_KEY"]

model = ChatOpenAI(model="gpt-4.1-2025-04-14")

def predict(message,history):
    history_langchain_format = []
    for msg in history:
        if msg["role"] == "user":
            history_langchain_format.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            history_langchain_format.append(AIMessage(content=msg["content"]))
    history_langchain_format.append(HumanMessage(content=message))
    get_response = model.invoke(history_langchain_format)
    return get_response.content

demo = gr.ChatInterface(predict,type="messages")
os.environ["GRADIO_SERVER_PORT"] = "7060"
demo.launch()