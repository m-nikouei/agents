# llm_backend.py
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts.prompt import PromptTemplate
from config import GOOGLE_AI_MODEL_NAME, CHATBOT_SYSTEM_PROMPT

_BACKEND_GOOGLE_API_KEY = None
_BACKEND_CHAT_LLM_CHAIN = None
_BACKEND_INITIALIZATION_ERROR = None

def _is_google_api_key_potentially_valid(api_key):
    if not api_key:
        return False, "API key is missing."
    if not isinstance(api_key, str):
        return False, "API key must be a string."
    if len(api_key.strip()) < 30: 
        return False, "API key appears to be too short or invalid."
    return True, ""

def _initialize_llm_resources_unsafe():
    global _BACKEND_CHAT_LLM_CHAIN, _BACKEND_INITIALIZATION_ERROR
    
    llm = ChatGoogleGenerativeAI(
        model=GOOGLE_AI_MODEL_NAME,
        google_api_key=_BACKEND_GOOGLE_API_KEY,
        temperature=0.7,
    )

    memory = ConversationBufferMemory()
    _template = f"""{CHATBOT_SYSTEM_PROMPT}

Current conversation:
{{history}}
Human: {{input}}
AI:"""
    
    CUSTOM_PROMPT = PromptTemplate(
        input_variables=["history", "input"], template=_template
    )

    _BACKEND_CHAT_LLM_CHAIN = ConversationChain(
        llm=llm,
        memory=memory,
        prompt=CUSTOM_PROMPT,
        verbose=False 
    )
    
    _BACKEND_INITIALIZATION_ERROR = None
    print(f"Backend: Successfully initialized Google AI model: {GOOGLE_AI_MODEL_NAME}.")
    return True

def set_api_key_and_initialize(api_key_value: str):
    global _BACKEND_GOOGLE_API_KEY, _BACKEND_CHAT_LLM_CHAIN, _BACKEND_INITIALIZATION_ERROR
    _BACKEND_GOOGLE_API_KEY = api_key_value
    
    if not _BACKEND_GOOGLE_API_KEY:
        _BACKEND_INITIALIZATION_ERROR = "GOOGLE_API_KEY was not provided. Cannot initialize AI model."
        _BACKEND_CHAT_LLM_CHAIN = None
        print(f"Backend Warning: {_BACKEND_INITIALIZATION_ERROR}")
        return False

    is_valid_key_format, validity_message = _is_google_api_key_potentially_valid(_BACKEND_GOOGLE_API_KEY)
    if not is_valid_key_format:
        _BACKEND_INITIALIZATION_ERROR = f"GOOGLE_API_KEY validation failed: {validity_message}"
        _BACKEND_CHAT_LLM_CHAIN = None
        print(f"Backend Error: {_BACKEND_INITIALIZATION_ERROR}")
        return False
    
    # Direct call to initialization; exceptions from underlying libraries are not caught.
    # This adheres to the "no try/except" rule, but means errors during LLM init might crash.
    return _initialize_llm_resources_unsafe()


async def get_chat_response(message: str, history: list) -> str:
    global _BACKEND_CHAT_LLM_CHAIN, _BACKEND_INITIALIZATION_ERROR, _BACKEND_GOOGLE_API_KEY

    if _BACKEND_INITIALIZATION_ERROR:
        return f"Chatbot not available due to initialization error: {_BACKEND_INITIALIZATION_ERROR}"

    if _BACKEND_CHAT_LLM_CHAIN is None:
        if _BACKEND_GOOGLE_API_KEY:
            print("Backend: Chatbot chain is None, attempting re-initialization...")
            if _initialize_llm_resources_unsafe(): # Direct call
                print("Backend: Re-initialization successful.")
            else:
                return f"Chatbot re-initialization failed: {_BACKEND_INITIALIZATION_ERROR if _BACKEND_INITIALIZATION_ERROR else 'Unknown reason.'}"
        else:
            _BACKEND_INITIALIZATION_ERROR = "Chatbot is not initialized: GOOGLE_API_KEY is missing for re-attempt."
            return _BACKEND_INITIALIZATION_ERROR
            
    if _BACKEND_CHAT_LLM_CHAIN is None: # Check again after re-init attempt
         return "Sorry, the chatbot is critically uninitialized. Please check server logs."

    print(f"Backend: User message: {message}")
    bot_response = await _BACKEND_CHAT_LLM_CHAIN.apredict(input=message)

    if isinstance(bot_response, str) and bot_response.strip():
        print(f"Backend: Bot response: {bot_response}")
        return bot_response
    else:
        error_detail = f"LLM returned an empty or invalid response: '{str(bot_response)}'"
        print(f"Backend: {error_detail}")
        # This path might be taken if apredict returns None/empty for non-exception errors.
        # API errors from apredict itself will likely be unhandled exceptions.
        return f"Sorry, I received an unexpected or empty response from the AI. Details: {error_detail}"

def get_backend_google_api_key():
    global _BACKEND_GOOGLE_API_KEY
    return _BACKEND_GOOGLE_API_KEY

def get_initialization_status():
    global _BACKEND_CHAT_LLM_CHAIN, _BACKEND_INITIALIZATION_ERROR
    if _BACKEND_INITIALIZATION_ERROR:
        return False, _BACKEND_INITIALIZATION_ERROR
    if _BACKEND_CHAT_LLM_CHAIN is None:
        # This state implies API key might be set, but chain init failed for other reasons or hasn't run.
        return False, "Chat LLM chain not initialized (API key might be set but other init step failed or pending)."
    return True, "Successfully initialized."