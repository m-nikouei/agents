# app_frontend.py
import gradio as gr
import sys
import os
from llm_backend import (
    set_api_key_and_initialize,
    get_chat_response,
    get_initialization_status
)
from utils import read_configs
from config import STABLE_PORT, DEV_PORT

def launch_ui(model_name_to_display: str):
    initialized, status_message = get_initialization_status()
    
    title_text = f"Google AI Chatbot ({model_name_to_display})"
    
    description_text_md = f"A simple chatbot powered by Langchain and Google AI ({model_name_to_display})."
    if not initialized:
        description_text_md += f"\n\n**WARNING:** Chatbot may not be functional. Backend status: {status_message}"
        if "GOOGLE_API_KEY" in status_message or "API key" in status_message:
             description_text_md += "\nPlease ensure your GOOGLE_API_KEY is correctly set in your config file or as an environment variable."

    with gr.Blocks(theme="soft", fill_height=True) as demo:
        gr.Markdown(f"# {title_text}")
        gr.Markdown(description_text_md)
        
        chatbot_component = gr.Chatbot(
            bubble_full_width=False,
            type="messages",
            show_label=False 
        )
        
        with gr.Row():
            with gr.Column(scale=8):
                textbox_component = gr.Textbox(
                    placeholder="Ask me anything...",
                    show_label=False,
                    container=False,
                )
            with gr.Column(scale=1, min_width=80):
                submit_button = gr.Button("Send")

        async def handle_submit_fn(message: str, history_from_chatbot: list[dict[str, str | None]]):
            if not message.strip():
                yield history_from_chatbot, "" 
                return

            current_gradio_history = list(history_from_chatbot) if history_from_chatbot else []
            current_gradio_history.append({"role": "user", "content": message})
            current_gradio_history.append({"role": "assistant", "content": ""}) 
            
            yield current_gradio_history, ""

            bot_response_content = await get_chat_response(message, [])

            if not isinstance(bot_response_content, str):
                print(f"Warning: Backend response was not a string (type: {type(bot_response_content)}, value: {bot_response_content}). Using an error message.")
                bot_response_content = "Error: Received an unexpected response from the AI backend."
            
            if current_gradio_history and current_gradio_history[-1]["role"] == "assistant":
                current_gradio_history[-1]["content"] = bot_response_content
            else:
                current_gradio_history.append({"role": "assistant", "content": bot_response_content})
            
            yield current_gradio_history, ""

        textbox_component.submit(
            fn=handle_submit_fn,
            inputs=[textbox_component, chatbot_component],
            outputs=[chatbot_component, textbox_component],
            show_progress="hidden" 
        )
        
        submit_button.click(
            fn=handle_submit_fn,
            inputs=[textbox_component, chatbot_component],
            outputs=[chatbot_component, textbox_component],
            show_progress="hidden"
        )

    print("Launching Gradio Chat Interface with Blocks...")
    demo.launch(server_name="0.0.0.0",server_port=STABLE_PORT)

if __name__ == "__main__":
    loaded_api_key = None
    model_name_from_config = None
    config_file_path = None

    if len(sys.argv) > 1 and sys.argv[1]:
        config_file_path = sys.argv[1]
        if os.path.exists(config_file_path):
            configs = read_configs(config_file_path) 
            if isinstance(configs, dict) and "chatbot" in configs and \
               isinstance(configs["chatbot"], dict):
                loaded_api_key = configs["chatbot"].get("GOOGLE_API_KEY")
                model_name_from_config = configs["chatbot"].get("GOOGLE_AI_MODEL_NAME")
            else:
                print(f"Warning: Config structure in {config_file_path} is incorrect or missing 'chatbot' section.")
        else:
            print(f"Warning: Config file '{config_file_path}' not found. Skipping.")
            
    if not loaded_api_key:
        loaded_api_key = os.environ.get("GOOGLE_API_KEY")
        if loaded_api_key:
            print("Info: API key loaded from environment variable GOOGLE_API_KEY.")

    env_model_name = os.environ.get("GOOGLE_AI_MODEL_NAME")
    final_model_name_for_ui = model_name_from_config or env_model_name or "Google AI Model"
    
    initialization_successful = False
    if loaded_api_key:
        print(f"API Key found, attempting to initialize backend. (UI will display model: '{final_model_name_for_ui}')")
        initialization_successful = set_api_key_and_initialize(loaded_api_key)
        if not initialization_successful:
            _, status_msg = get_initialization_status()
            print(f"Frontend: Backend initialization failed. Reason: {status_msg}")
    else:
        print("\n" + "="*50)
        print("ERROR: GOOGLE_API_KEY is not set.")
        print("Please set it via config file or environment variable.")
        print("Chatbot will not work until this is configured.")
        print("="*50 + "\n")
        set_api_key_and_initialize(None) 

    launch_ui(model_name_to_display=final_model_name_for_ui)