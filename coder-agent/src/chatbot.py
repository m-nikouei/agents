# main_app.py
import gradio as gr
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from utils import read_configs
import sys


# IMPORTANT: Replace with your desired Google AI model
# Examples: "gemini-1.0-pro", "gemini-1.5-flash-latest", "gemini-pro" (generic latest)
GOOGLE_AI_MODEL_NAME = "gemini-2.5-pro-preview-05-06"


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
        # You can adjust parameters like temperature, top_p, etc.
        llm = ChatGoogleGenerativeAI(
            model=GOOGLE_AI_MODEL_NAME,
            google_api_key=GOOGLE_API_KEY,
            temperature=0.7,  # Controls randomness
            # convert_system_message_to_human=True # Use if facing issues with system messages
        )

        # Initialize conversation memory
        memory = ConversationBufferMemory()

        # Initialize the conversation chain
        conversation_chain = ConversationChain(
            llm=llm,
            memory=memory,
            verbose=False # Set to False to reduce console output
        )
        print(f"Successfully initialized Google AI model: {GOOGLE_AI_MODEL_NAME}")
        return conversation_chain

    except Exception as e:
        print(f"Error initializing LLM or chain: {e}")
        raise gr.Error(f"Failed to initialize AI model. Please check your GOOGLE_API_KEY and Model Name. Error: {e}")

def initialize_chat(GOOGLE_API_KEY):
    try:
        if not GOOGLE_API_KEY:
            print("WARNING: GOOGLE_API_KEY environment variable not set. The application might fail to initialize the AI model.")
            # We'll let initialize_llm_and_chain handle the gr.Error if it's truly missing at runtime.
            chat_llm_chain = None
        else:
            chat_llm_chain = initialize_llm_and_chain()
    except Exception as e:
        # This exception will be caught by Gradio if it occurs during app setup
        chat_llm_chain = None
        print(f"Critical error during initial setup: {e}")
    return chat_llm_chain

async def chat_fn(message: str, history: list) -> str:
    """
    Handles the chat interaction.
    Args:
        message: The user's input message.
        history: The chat history (managed by Gradio).
    Returns:
        The bot's response as a string.
    """
    if chat_llm_chain is None:
        # This check is important if initialization failed.
        raise gr.Error("Chatbot is not initialized. Please ensure your GOOGLE_API_KEY is set and check console logs for errors.")

    try:
        print(f"User message: {message}")
        # Use the chain to get a response
        response = await chat_llm_chain.apredict(input=message)
        print(f"Bot response: {response}")
        return response

    except Exception as e:
        print(f"Error during model prediction: {e}")
        # Check for specific API errors if possible
        if "API key not valid" in str(e):
             return "Sorry, there's an issue with the API key. Please ensure it's correct and valid."
        return f"Sorry, I encountered an error: {e}"

# --- Gradio Interface ---
if __name__ == "__main__":
    if sys.argv[1]:
        configs = read_configs(sys.argv[1])
        GOOGLE_API_KEY = configs["chatbot"]["GOOGLE_API_KEY"]
        chat_llm_chain = initialize_chat(GOOGLE_API_KEY)
    else:
        print("Error: Please provide the config file path as a command line argument.")
        exit(1)
    if not GOOGLE_API_KEY:
        print("\n" + "="*50)
        print("IMPORTANT: Your GOOGLE_API_KEY is not set as an environment variable.")
        print("Please set it by running: export GOOGLE_API_KEY='YOUR_API_KEY'")
        print("You can get a key from Google AI Studio: https://aistudio.google.com/app/apikey")
        print("The chatbot will likely not work until this is configured.")
        print("="*50 + "\n")

    print("Launching Gradio Chat Interface with Google AI Studio models...")
    print("If the interface loads but the bot doesn't respond, please check the console for errors,")
    print("especially regarding your GOOGLE_API_KEY or model access.")

    chat_interface = gr.ChatInterface(
        fn=chat_fn,
        title=f"Google AI Chatbot ({GOOGLE_AI_MODEL_NAME})",
        description=f"A simple chatbot powered by Langchain and Google AI Studio ({GOOGLE_AI_MODEL_NAME}).",
        chatbot=gr.Chatbot(height=600),
        textbox=gr.Textbox(placeholder="Ask me anything...", container=False, scale=7),
        submit_btn="Send",
        theme="soft"
    )

    chat_interface.launch(server_name="0.0.0.0", server_port=7860)
    # For local use only:
    # chat_interface.launch()
