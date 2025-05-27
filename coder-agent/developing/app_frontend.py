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
from config import DEV_PORT, STABLE_PORT, read_configs, AVAILABLE_MODELS, GOOGLE_AI_MODEL_NAME as DEFAULT_CONFIG_MODEL

def generate_description_markdown_text(model_name_for_display: str, initialized: bool, status_message: str) -> str:
    desc = f"A simple chatbot powered by Langchain and Google AI ({model_name_for_display})."
    if not initialized:
        desc += f"\n\n**WARNING:** Chatbot may not be functional. Backend status: {status_message}"
        if "GOOGLE_API_KEY" in status_message or "API key" in status_message:
            desc += "\nPlease ensure your GOOGLE_API_KEY is correctly set in your config file or as an environment variable."
    return desc

def launch_ui(initial_model_name: str):
    
    initial_is_initialized, initial_status_msg, actual_initial_model_in_use = get_initialization_status()
    current_display_model_name = actual_initial_model_in_use or initial_model_name
    
    custom_css = """
    html, body {
        height: 100%; 
        margin: 0;
        overflow: hidden; 
    }
    .gradio-container {
        padding-left: 10% !important;
        padding-right: 10% !important;
        padding-top: 0.2em !important; 
        padding-bottom: 0.2em !important;
        box-sizing: border-box !important;
        height: 100% !important; 
        display: flex !important;
        flex-direction: column !important;
    }
    #main_layout_row {
        flex-grow: 1; 
        display: flex;
        min-height: 0; 
    }
    #settings_panel_col {
        min-height: 0;
        overflow-y: auto;
        flex-shrink: 0;
    }
    #chat_area_col {
        display: flex !important;
        flex-direction: column !important;
        flex-grow: 1; /* Takes up space in its parent flex row */
        height: 100%; /* Fills height allocated by main_layout_row */
        min-height: 0;
        overflow: hidden !important;
    }

    /* Rows within chat_area_col */
    #title_bar_row, #description_content_row {
        flex-grow: 0;
        flex-shrink: 0;
        padding-top: 0.1em !important;
        padding-bottom: 0.1em !important;
        margin: 0 !important;
    }
    
    #chatbot_wrapper_col { /* Row containing the chatbot component */
        flex-grow: 1; 
        display: flex;
        flex-direction: column;
        min-height: 0; /* Critical for flex-grow to work and allow shrinking */
        overflow: hidden; 
        margin: 0 !important;
    }
    #chatbot_wrapper_col > div[data-testid="chatbot"] { /* The actual gr.Chatbot UI element */
        flex-grow: 1; 
        min-height: 0; 
    }
    
    #input_message_row {
        flex-grow: 0;
        flex-shrink: 0;
        padding-top: 0.5em !important;
        padding-bottom: 0.2em !important;
        margin: 0 !important;
    }
    #input_message_row > div[data-testid="column"] {
        padding: 0 !important;
    }

    footer {visibility: hidden; display: none !important;}
    
    .prose p { 
        margin-top: 0.1em !important;
        margin-bottom: 0.1em !important;
    }
    #main-title p { 
        margin-top: 0.1em !important; 
        margin-bottom: 0.1em !important;
        padding-top: 0 !important;
        padding-bottom: 0 !important;
        line-height: 1.2;
    }
    #description-markdown p {
        margin-top: 0.05em !important;
        margin-bottom: 0.05em !important;
        font-size: 0.85em; 
        line-height: 1.2;
    }
    #settings_panel_col .prose h3 { 
        margin-top: 0.3em !important;
        margin-bottom: 0.1em !important;
    }
    """

    with gr.Blocks(theme="soft", fill_height=True, css=custom_css) as demo:
        current_model_gr_state = gr.State(current_display_model_name)
        settings_panel_visible_state = gr.State(False)

        with gr.Row(equal_height=True, elem_id="main_layout_row"): 
            with gr.Column(scale=1, min_width=280, visible=False, elem_id="settings_panel_col") as settings_panel:
                gr.Markdown("## Settings")
                gr.Markdown("### Model Selection")
                settings_panel_status_md = gr.Markdown("")
                model_selector_radio = gr.Radio(
                    choices=AVAILABLE_MODELS,
                    value=current_display_model_name,
                    label="Language Model"
                )
                gr.Markdown("### Theme")
                gr.Markdown("_(Theme selection may require app reload if changed via URL parameters or Gradio's native settings if footer is enabled)_")
                gr.Markdown("---")
                gr.Markdown("### Links")
                gr.Markdown("<div style='margin-bottom: 5px;'><a href='#' target='_blank' style='text-decoration: none; color: #007bff;'>Use via API</a></div>")
                gr.Markdown("<div style='margin-bottom: 5px;'><a href='https://gradio.app' target='_blank' style='text-decoration: none; color: #007bff;'>Built with Gradio</a></div>")
                gr.Markdown("---")
                close_settings_button = gr.Button("Close Settings")

            with gr.Column(scale=4, elem_id="chat_area_col"): 
                with gr.Row(equal_height=False, elem_id="title_bar_row"): 
                    with gr.Column(scale=0, min_width=50): 
                        hamburger_button = gr.Button("☰", variant="secondary", size="sm")
                    with gr.Column(scale=9): 
                        title_markdown = gr.Markdown(f"# Google AI Chatbot ({current_display_model_name})", elem_id="main-title")
                
                with gr.Row(elem_id="description_content_row"):
                    description_markdown = gr.Markdown(
                        generate_description_markdown_text(current_display_model_name, initial_is_initialized, initial_status_msg),
                        elem_id="description-markdown"
                    )
                
                with gr.Row(elem_id="chatbot_wrapper_col"): 
                    chatbot_component = gr.Chatbot(
                        type="messages",
                        show_label=False,
                        # height="100%" # Could be added if needed, but CSS flex should handle
                    )
                
                with gr.Row(equal_height=False, elem_id="input_message_row"): 
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

            assistant_response_content = ""
            stream_had_content = False
            async for chunk in get_chat_response(message, []):
                if isinstance(chunk, str):
                    assistant_response_content += chunk
                    current_gradio_history[-1]["content"] = assistant_response_content
                    stream_had_content = True
                    yield current_gradio_history, ""
            
            if not stream_had_content and current_gradio_history[-1]["content"] == "":
                _init_ok, _status_msg, _ = get_initialization_status()
                error_msg_display = "Error: Received an empty or unexpected response from the AI backend."
                if not _init_ok: error_msg_display = f"Chatbot not available: {_status_msg}"
                elif not assistant_response_content: error_msg_display = "Error: AI backend did not provide a response."
                current_gradio_history[-1]["content"] = error_msg_display
                yield current_gradio_history, ""

        textbox_component.submit(fn=handle_submit_fn, inputs=[textbox_component, chatbot_component], outputs=[chatbot_component, textbox_component], show_progress="hidden")
        submit_button.click(fn=handle_submit_fn, inputs=[textbox_component, chatbot_component], outputs=[chatbot_component, textbox_component], show_progress="hidden")

        async def handle_model_change(selected_new_model: str, current_model_in_state: str):
            feedback_msg = ""
            updated_radio_val = selected_new_model

            if selected_new_model == current_model_in_state:
                is_init, status_msg, model_name_b = get_initialization_status()
                new_title = f"# Google AI Chatbot ({model_name_b})"
                new_desc = generate_description_markdown_text(model_name_b, is_init, status_msg)
                feedback_msg = "Settings refreshed. Model remains the same."
                updated_radio_val = model_name_b
                return model_name_b, new_title, new_desc, feedback_msg, gr.update(value=updated_radio_val)

            success, msg, actual_model_name = change_model_and_reinitialize(selected_new_model)
            new_title_str = f"# Google AI Chatbot ({actual_model_name})"
            is_overall_init, overall_status_msg, _ = get_initialization_status()
            new_desc_str = generate_description_markdown_text(actual_model_name, is_overall_init, overall_status_msg)
            updated_radio_val = actual_model_name
            
            if success: feedback_msg = f"Successfully switched to: {actual_model_name}."
            else: feedback_msg = f"Failed to switch to {selected_new_model}. {msg}"
            
            return actual_model_name, new_title_str, new_desc_str, feedback_msg, gr.update(value=updated_radio_val)

        model_selector_radio.change(
            fn=handle_model_change,
            inputs=[model_selector_radio, current_model_gr_state],
            outputs=[current_model_gr_state, title_markdown, description_markdown, settings_panel_status_md, model_selector_radio]
        )
        
        def toggle_settings_panel(current_visibility_state_value: bool):
            new_visibility = not current_visibility_state_value
            radio_update_val = gr.update() 
            status_text_update = ""
            if new_visibility: 
                _, _, current_backend_model = get_initialization_status()
                radio_update_val = gr.update(value=current_backend_model)
                status_text_update = "Select model or close settings."
            return new_visibility, gr.update(visible=new_visibility), radio_update_val, status_text_update

        hamburger_button.click(
            fn=toggle_settings_panel,
            inputs=[settings_panel_visible_state], 
            outputs=[settings_panel_visible_state, settings_panel, model_selector_radio, settings_panel_status_md]
        )
        
        close_settings_button.click(
            lambda: (False, gr.update(visible=False), gr.update(), ""), 
            inputs=None, 
            outputs=[settings_panel_visible_state, settings_panel, model_selector_radio, settings_panel_status_md]
        )

    print("Launching Gradio Chat Interface with Blocks...")
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
    launch_ui(initial_model_name=model_name_for_ui_launch)