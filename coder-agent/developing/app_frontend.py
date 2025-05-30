# app_frontend.py
import gradio as gr
import sys
import os
from llm_backend import (
    set_api_key_and_initialize,
    get_chat_response,
    get_initialization_status,
    change_model_and_reinitialize
)
from config import DEV_PORT, read_configs, AVAILABLE_MODELS, GOOGLE_AI_MODEL_NAME as DEFAULT_CONFIG_MODEL

def generate_description_markdown_text(model_name_for_display: str, initialized: bool, status_message: str) -> str:
    desc = f"A simple chatbot powered by Langchain and Google AI ({model_name_for_display})."
    if not initialized:
        desc += f"\n\n**WARNING:** Chatbot may not be functional. Backend status: {status_message}"
        if "GOOGLE_API_KEY" in status_message or "API key" in status_message:
            desc += "\nPlease ensure your GOOGLE_API_KEY is correctly set in your config file or as an environment variable."
    return desc

async def chat_interface_fn(message: str, history: list[list[str | None]]):
    if not message.strip():
        yield ""
        return

    is_initialized, status_msg, _ = get_initialization_status()
    if not is_initialized:
        yield f"**Chatbot not available:** {status_msg}"
        return

    assistant_response_content = ""
    stream_had_content = False
    
    processed_history = []
    for user_msg, bot_msg in history:
        if user_msg:
            processed_history.append({"role": "user", "content": user_msg})
        if bot_msg:
            processed_history.append({"role": "assistant", "content": bot_msg})

    async for chunk in get_chat_response(message, processed_history): 
        if isinstance(chunk, str):
            assistant_response_content += chunk
            stream_had_content = True
            yield assistant_response_content
    
    if not stream_had_content and not assistant_response_content:
        _init_ok_final, _status_msg_final, _ = get_initialization_status()
        error_msg_display = "Error: Received an empty or unexpected response from the AI backend."
        if not _init_ok_final: 
            error_msg_display = f"**Chatbot not available:** {_status_msg_final}"
        yield error_msg_display

def launch_ui_chat_interface(initial_model_name: str):
    initial_is_initialized, initial_status_msg, actual_initial_model_in_use = get_initialization_status()
    current_display_model_name = actual_initial_model_in_use or initial_model_name

    with gr.Blocks(theme="soft", fill_height=True) as demo:
        current_model_gr_state = gr.State(current_display_model_name)
        settings_visible_state = gr.State(False)
        
        with gr.Column(scale=1):
            with gr.Row():
                with gr.Column(scale=0, min_width=50):
                    hamburger_button = gr.Button("☰", variant="secondary", size="sm")
                with gr.Column(scale=9):
                    title_md = gr.Markdown(f"# Google AI Chatbot ({current_display_model_name})")
            
            description_md = gr.Markdown(
                generate_description_markdown_text(current_display_model_name, initial_is_initialized, initial_status_msg)
            )

            settings_accordion = gr.Accordion("Settings", open=False, visible=False)
            with settings_accordion:
                settings_status_md = gr.Markdown("")
                model_selector_radio = gr.Radio(
                    choices=AVAILABLE_MODELS,
                    value=current_display_model_name,
                    label="Select Language Model"
                )
            
            chat_interface_component = gr.ChatInterface(
                fn=chat_interface_fn,
                fill_height=True,
                chatbot=gr.Chatbot(show_label=False,container=False,height="120%"),
            )

        def toggle_settings_accordion(current_visibility):
            new_visibility = not current_visibility
            radio_update_val = gr.update()
            status_text_update = ""
            if new_visibility:
                _, _, current_backend_model = get_initialization_status()
                radio_update_val = gr.update(value=current_backend_model)
                status_text_update = "Select model or adjust settings."
            
            return new_visibility, gr.update(visible=new_visibility, open=new_visibility), radio_update_val, status_text_update

        hamburger_button.click(
            fn=toggle_settings_accordion,
            inputs=[settings_visible_state],
            outputs=[settings_visible_state, settings_accordion, model_selector_radio, settings_status_md]
        )

        async def handle_model_change_interface(selected_new_model: str, current_model_in_state: str):
            feedback_msg_for_settings = ""
            
            if selected_new_model == current_model_in_state:
                is_init, status_msg, model_name_b = get_initialization_status()
                new_title_str = f"# Google AI Chatbot ({model_name_b})"
                new_desc_str = generate_description_markdown_text(model_name_b, is_init, status_msg)
                feedback_msg_for_settings = "Settings refreshed. Model remains the same."
                return model_name_b, new_title_str, new_desc_str, feedback_msg_for_settings, gr.update(value=model_name_b)

            success, msg, actual_model_name = change_model_and_reinitialize(selected_new_model)
            
            new_title_str = f"# Google AI Chatbot ({actual_model_name})"
            is_overall_init, overall_status_msg, _ = get_initialization_status()
            new_desc_str = generate_description_markdown_text(actual_model_name, is_overall_init, overall_status_msg)
            
            if success:
                feedback_msg_for_settings = f"Successfully switched to: {actual_model_name}. You may need to clear chat history."
            else:
                feedback_msg_for_settings = f"Failed to switch to {selected_new_model}. Error: {msg}. Current model: {actual_model_name}."
            
            return actual_model_name, new_title_str, new_desc_str, feedback_msg_for_settings, gr.update(value=actual_model_name)

        model_selector_radio.change(
            fn=handle_model_change_interface,
            inputs=[model_selector_radio, current_model_gr_state],
            outputs=[
                current_model_gr_state, 
                title_md, 
                description_md, 
                settings_status_md, 
                model_selector_radio
            ]
        )

    print("Launching Gradio Chat Interface...")
    demo.launch(server_name="0.0.0.0", server_port=DEV_PORT)

if __name__ == "__main__":
    loaded_api_key = None
    model_name_from_config_json = None
    config_file_path_arg = None

    if len(sys.argv) > 1 and sys.argv[1]:
        config_file_path_arg = sys.argv[1]
        if os.path.exists(config_file_path_arg):
            configs = read_configs(config_file_path_arg) 
            if isinstance(configs, dict) and "chatbot" in configs and \
               isinstance(configs["chatbot"], dict):
                loaded_api_key = configs["chatbot"].get("GOOGLE_API_KEY")
                model_name_from_config_json = configs["chatbot"].get("GOOGLE_AI_MODEL_NAME")
            else:
                print(f"Warning: Config structure in {config_file_path_arg} is incorrect or missing 'chatbot' section.")
        else:
            print(f"Warning: Config file '{config_file_path_arg}' not found. Skipping.")
            
    if not loaded_api_key:
        loaded_api_key = os.environ.get("GOOGLE_API_KEY")
        if loaded_api_key:
            print("Info: API key loaded from environment variable GOOGLE_API_KEY.")

    env_model_name = os.environ.get("GOOGLE_AI_MODEL_NAME")
    initial_model_to_attempt = model_name_from_config_json or env_model_name or DEFAULT_CONFIG_MODEL
    
    if loaded_api_key:
        print(f"API Key found. Attempting to initialize backend with model: '{initial_model_to_attempt}'")
        set_api_key_and_initialize(loaded_api_key, initial_model_to_attempt)
    else:
        print("\n" + "="*50)
        print("ERROR: GOOGLE_API_KEY is not set.")
        print("Please set it via config file or environment variable.")
        print("Chatbot will not work until this is configured.")
        print("="*50 + "\n")
        set_api_key_and_initialize(None, initial_model_to_attempt) 

    _, _, model_name_for_ui_launch = get_initialization_status()
    launch_ui_chat_interface(initial_model_name=model_name_for_ui_launch or DEFAULT_CONFIG_MODEL)