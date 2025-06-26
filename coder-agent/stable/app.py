# app.py

import gradio as gr
import os
from llm_interface import predict

os.environ["GRADIO_SERVER_PORT"] = "7060"

demo = gr.ChatInterface(
    predict,
    type="messages",
    title="Coder Agent Chatbot"
)


demo.launch()