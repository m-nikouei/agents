# app.py

import gradio as gr
import os
from llm_interface import ChatBackend

if __name__ == "__main__":

    backend = ChatBackend()

    os.environ["GRADIO_SERVER_PORT"] = "9060"

    demo = gr.ChatInterface(
        backend.predict,
        type="messages",
        title="Coder Agent Chatbot",
        chat_history=backend.history
    )

    demo.launch()