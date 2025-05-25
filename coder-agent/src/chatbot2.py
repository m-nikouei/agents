# main_app.py
import gradio as gr
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts.prompt import PromptTemplate # <--- ADDED THIS IMPORT
from utils import read_configs
import sys


# IMPORTANT: Replace with your desired Google AI model
# Examples: "gemini-1.0-pro", "gemini-1.5-flash-latest", "gemini-pro" (generic latest)
GOOGLE_AI_MODEL_NAME = "gemini-2.5-pro-preview-05-06"

# --- ADD YOUR CHATBOT'S SYSTEM PROMPT HERE ---
# This prompt will be used by the LLM for every response.
# It helps define the chatbot's persona, tone, or specific instructions.
# For example:
# CHATBOT_SYSTEM_PROMPT = "You are a friendly and helpful assistant. Always answer concisely."
# CHATBOT_SYSTEM_PROMPT = "You are a pirate chatbot. Speak like a pirate and seek treasure in your answers!"
CHATBOT_SYSTEM_PROMPT = """
You are a coding chatbot. The code that you generate should follow the following rules:
1. Don't add any comments or explanation to the code.
2. Don't use try/except blocks in the code. Try to write the code in a way that it doesn't throw any errors or handle errors through conditional statements.
3. Use only well-known libraries and packages. Don't use any custom or unknown libraries.
4. Try to split the code into multiple files if the code is loner than 200 lines, or if any config exists in the code, or if the code contains both frontend and backend code.
The explanation code outside of the generated code should be in markdown format and follow the following rules:
1. Explanations should be very short and concise.
2. Don't explain all the changes. Only explain the changes that are not obvious.
"""
# --- END OF SYSTEM PROMPT DEFINITION ---

GOOGLE_API_KEY = None
chat_llm_chain = None


def initialize_llm_and_chain():
    """
    Initializes the Google AI LLM, conversation memory, and conversation chain.
    Returns the conversation chain.
    """
    if not GOOGLE_API_KEY:
        raise gr.Error(
            "GOOGLE_API_KEY not found. Please set it as an environment variable "
            "or directly in the script. Get a key from https://aistudio.google.com/app/apikey"
        )

    try:
        # Initialize the Google AI Chat Model
        llm = ChatGoogleGenerativeAI(
            model=GOOGLE_AI_MODEL_NAME,
            google_api_key=GOOGLE_API_KEY,
            temperature=0.7,
        )

        # Initialize conversation memory
        memory = ConversationBufferMemory()

        # --- MODIFIED SECTION: Define and use a custom prompt ---
        # The template string now includes our CHATBOT_SYSTEM_PROMPT.
        # {history} will be filled by ConversationBufferMemory.
        # {input} will be the user's current message.
        _template = f"""{CHATBOT_SYSTEM_PROMPT}

Current conversation:
{{history}}
Human: {{input}}
AI:"""
        
        CUSTOM_PROMPT = PromptTemplate(
            input_variables=["history", "input"], template=_template
        )

        # Initialize the conversation chain with the custom prompt
        conversation_chain = ConversationChain(
            llm=llm,
            memory=memory,
            prompt=CUSTOM_PROMPT, # <--- PASSING THE CUSTOM PROMPT
            verbose=False # Set to False to reduce console output
        )
        # --- END OF MODIFIED SECTION ---

        print(f"Successfully initialized Google AI model: {GOOGLE_AI_MODEL_NAME} with custom prompt.")
        return conversation_chain

    except Exception as e:
        print(f"Error initializing LLM or chain: {e}")
        raise gr.Error(f"Failed to initialize AI model. Please check your GOOGLE_API_KEY and Model Name. Error: {e}")

def initialize_chat(api_key_value): # Renamed GOOGLE_API_KEY to avoid conflict with global
    """
    Initializes the chat components.
    """
    global GOOGLE_API_KEY # Ensure we are setting the global GOOGLE_API_KEY
    GOOGLE_API_KEY = api_key_value
    
    chat_chain_instance = None # Renamed for clarity
    try:
        if not GOOGLE_API_KEY:
            print("WARNING: GOOGLE_API_KEY environment variable not set. The application might fail to initialize the AI model.")
            # initialize_llm_and_chain will handle the gr.Error
        else:
            chat_chain_instance = initialize_llm_and_chain()
    except Exception as e:
        # This exception will be caught by Gradio if it occurs during app setup
        print(f"Critical error during initial setup: {e}")
        # No need to raise gr.Error here, as Gradio will catch it or initialize_llm_and_chain will.
    return chat_chain_instance


async def chat_fn(message: str, history: list) -> str:
    """
    Handles the chat interaction.
    Args:
        message: The user's input message.
        history: The chat history (managed by Gradio).
    Returns:
        The bot's response as a string.
    """
    global chat_llm_chain # <--- MOVED TO THE TOP OF THE FUNCTION

    if chat_llm_chain is None:
        print("Chatbot chain not initialized. Attempting to re-initialize...")
        # We check the module-level GOOGLE_API_KEY.
        # No need for `global GOOGLE_API_KEY` here if we are only reading it,
        # as Python will find it in the global scope.
        # Using globals().get() is a safe way to read it if unsure.
        current_api_key = globals().get("GOOGLE_API_KEY")

        if current_api_key: # Check if the API key is available
            try:
                print(f"Re-initializing with API key: {'*' * (len(current_api_key) - 4) + current_api_key[-4:] if current_api_key else 'None'}")
                # initialize_llm_and_chain() will use the global GOOGLE_API_KEY
                temp_chain = initialize_llm_and_chain()
                if temp_chain is None:
                    raise gr.Error("Re-initialization attempt failed to produce a chain. Chatbot is not available.")
                chat_llm_chain = temp_chain # Assign to the global chat_llm_chain
                print("Chatbot re-initialized successfully.")
            except Exception as e:
                print(f"Error during re-initialization: {e}")
                raise gr.Error(f"Re-initialization failed: {e}")
        else:
            print("Cannot re-initialize: GOOGLE_API_KEY is missing.")
            raise gr.Error("Chatbot is not initialized, and GOOGLE_API_KEY is missing. Please set it and check console logs.")

    # If after all that, it's still None (e.g., re-init failed and raised, or wasn't possible)
    # The raise gr.Error above would have already stopped execution, but as a safeguard:
    if chat_llm_chain is None:
        # This should ideally not be reached if the logic above is correct
        # because an error should have been raised.
        return "Sorry, the chatbot is critically uninitialized. Please check the logs."

    try:
        print(f"User message: {message}")
        response = await chat_llm_chain.apredict(input=message)
        print(f"Bot response: {response}")
        return response

    except Exception as e:
        print(f"Error during model prediction: {e}")
        if "API key not valid" in str(e):
             return "Sorry, there's an issue with the API key. Please ensure it's correct and valid."
        # Check for API key missing during prediction (if somehow it got unset or was never valid)
        elif "API key is not set" in str(e) or "GOOGLE_API_KEY" in str(e).upper(): # More generic check
            return "Sorry, there seems to be an issue with the Google API Key configuration."
        return f"Sorry, I encountered an error: {e}"

# --- Gradio Interface ---
if __name__ == "__main__":
    # Initialize module-level globals that will be set here
    # GOOGLE_API_KEY is already None by default from top of script
    # chat_llm_chain is already None by default from top of script

    if len(sys.argv) > 1 and sys.argv[1]:
        configs = read_configs(sys.argv[1])
        loaded_api_key = configs["chatbot"].get("GOOGLE_API_KEY")
        if loaded_api_key:
            # The initialize_chat function I provided earlier handles setting the global GOOGLE_API_KEY
            # and then returns the chain.
            chat_llm_chain = initialize_chat(loaded_api_key)
        else:
            print("Error: GOOGLE_API_KEY not found in the config file.")
            # chat_llm_chain remains None, initialize_chat won't be called with a key.
            # We can call initialize_chat with None to let it handle the warning.
            chat_llm_chain = initialize_chat(None)
    else:
        print("Warning: No config file path provided. Attempting to use environment variables.")
        import os
        env_api_key = os.environ.get("GOOGLE_API_KEY")
        # initialize_chat will handle if env_api_key is None or valid
        chat_llm_chain = initialize_chat(env_api_key)

    # The GOOGLE_API_KEY global is now set (or not) by the initialize_chat call.
    # The chat_llm_chain global is now set (or not) by the initialize_chat call.

    if not GOOGLE_API_KEY: # Check the global that initialize_chat was supposed to set
        print("\n" + "="*50)
        print("IMPORTANT: Your GOOGLE_API_KEY is not set from config or environment variables.")
        print("Please set it by running: export GOOGLE_API_KEY='YOUR_API_KEY'")
        print("Or ensure it's in your config file.")
        print("You can get a key from Google AI Studio: https://aistudio.google.com/app/apikey")
        print("The chatbot will likely not work until this is configured.")
        print("="*50 + "\n")

    print("Launching Gradio Chat Interface with Google AI Studio models...")
    # ... (rest of your __main__ block, it looked okay)
    title_model_name = GOOGLE_AI_MODEL_NAME if 'GOOGLE_AI_MODEL_NAME' in globals() and GOOGLE_AI_MODEL_NAME else "Google AI Model"

    chat_interface = gr.ChatInterface(
        fn=chat_fn,
        title=f"Google AI Chatbot ({title_model_name})",
        description=f"A simple chatbot powered by Langchain and Google AI Studio ({title_model_name}). Now with a custom system prompt!",
        chatbot=gr.Chatbot(height=600),
        textbox=gr.Textbox(placeholder="Ask me anything...", container=False, scale=7),
        submit_btn="Send",
        theme="soft"
    )

    chat_interface.launch(server_name="0.0.0.0", server_port=7060)