# llm_backend.py
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts.prompt import PromptTemplate
from config import GOOGLE_AI_MODEL_NAME as DEFAULT_MODEL_NAME, CHATBOT_SYSTEM_PROMPT

_BACKEND_GOOGLE_API_KEY = None
_BACKEND_CHAT_LLM_CHAIN = None
_BACKEND_INITIALIZATION_ERROR = None
_BACKEND_CURRENT_MODEL_NAME = DEFAULT_MODEL_NAME

def _is_google_api_key_potentially_valid(api_key):
    if not api_key:
        return False, "API key is missing."
    if not isinstance(api_key, str):
        return False, "API key must be a string."
    if len(api_key.strip()) < 30:
        return False, "API key appears to be too short or invalid."
    return True, ""

def _initialize_llm_resources_unsafe(model_to_init: str):
    global _BACKEND_CHAT_LLM_CHAIN, _BACKEND_INITIALIZATION_ERROR, _BACKEND_CURRENT_MODEL_NAME
    
    _BACKEND_CURRENT_MODEL_NAME = model_to_init
    print(f"Backend: Attempting to initialize Google AI model: {model_to_init}...")
    llm = ChatGoogleGenerativeAI(
        model=model_to_init,
        google_api_key=_BACKEND_GOOGLE_API_KEY,
        temperature=0.7
        # streaming=True removed, as astream_events implies and manages streaming
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
    print(f"Backend: Successfully initialized Google AI model: {model_to_init}.")
    return True

def set_api_key_and_initialize(api_key_value: str, initial_model_name: str = None):
    global _BACKEND_GOOGLE_API_KEY, _BACKEND_CHAT_LLM_CHAIN, _BACKEND_INITIALIZATION_ERROR, _BACKEND_CURRENT_MODEL_NAME
    _BACKEND_GOOGLE_API_KEY = api_key_value
    
    model_for_first_init = initial_model_name or DEFAULT_MODEL_NAME
    _BACKEND_CURRENT_MODEL_NAME = model_for_first_init

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
    
    return _initialize_llm_resources_unsafe(model_for_first_init)

def change_model_and_reinitialize(new_model_name: str):
    global _BACKEND_GOOGLE_API_KEY, _BACKEND_INITIALIZATION_ERROR, _BACKEND_CURRENT_MODEL_NAME
    
    if not _BACKEND_GOOGLE_API_KEY:
        _BACKEND_INITIALIZATION_ERROR = "Cannot change model: GOOGLE_API_KEY is not set."
        _BACKEND_CURRENT_MODEL_NAME = new_model_name
        return False, _BACKEND_INITIALIZATION_ERROR, _BACKEND_CURRENT_MODEL_NAME

    print(f"Backend: Request to change model to: {new_model_name}")
    _BACKEND_CURRENT_MODEL_NAME = new_model_name
    success = _initialize_llm_resources_unsafe(new_model_name)
    
    if success:
        return True, f"Successfully switched to model: {new_model_name}.", _BACKEND_CURRENT_MODEL_NAME
    else:
        err_msg = _BACKEND_INITIALIZATION_ERROR or f"Failed to initialize model: {new_model_name} for an unknown reason."
        return False, err_msg, _BACKEND_CURRENT_MODEL_NAME


async def get_chat_response(message: str, history: list):
    global _BACKEND_CHAT_LLM_CHAIN, _BACKEND_INITIALIZATION_ERROR, _BACKEND_GOOGLE_API_KEY, _BACKEND_CURRENT_MODEL_NAME

    if _BACKEND_INITIALIZATION_ERROR:
        yield f"Chatbot not available (using model: {_BACKEND_CURRENT_MODEL_NAME}). Error: {_BACKEND_INITIALIZATION_ERROR}"
        return

    if _BACKEND_CHAT_LLM_CHAIN is None:
        if _BACKEND_GOOGLE_API_KEY:
            print(f"Backend: Chatbot chain is None (model: {_BACKEND_CURRENT_MODEL_NAME}), attempting re-initialization...")
            if _initialize_llm_resources_unsafe(_BACKEND_CURRENT_MODEL_NAME):
                print("Backend: Re-initialization successful.")
            else:
                error_msg = _BACKEND_INITIALIZATION_ERROR if _BACKEND_INITIALIZATION_ERROR else 'Unknown re-initialization reason.'
                yield f"Chatbot re-initialization failed for model {_BACKEND_CURRENT_MODEL_NAME}: {error_msg}"
                return
        else:
            _BACKEND_INITIALIZATION_ERROR = "Chatbot is not initialized: GOOGLE_API_KEY is missing for re-attempt."
            yield _BACKEND_INITIALIZATION_ERROR
            return
            
    if _BACKEND_CHAT_LLM_CHAIN is None:
         yield f"Sorry, the chatbot (model: {_BACKEND_CURRENT_MODEL_NAME}) is critically uninitialized. Please check server logs."
         return

    print(f"Backend: User message: {message} (to model: {_BACKEND_CURRENT_MODEL_NAME})")
    
    stream_produced_content = False
    full_bot_response_for_log = ""
    async for event in _BACKEND_CHAT_LLM_CHAIN.astream_events(
        {"input": message}, version="v2"
    ):
        kind = event.get("event")
        if kind == "on_chat_model_stream":
            chunk_data = event.get("data", {}).get("chunk")
            if hasattr(chunk_data, 'content'):
                response_piece = chunk_data.content
                if isinstance(response_piece, str) and response_piece:
                    full_bot_response_for_log += response_piece
                    yield response_piece
                    stream_produced_content = True
    
    if stream_produced_content:
        print(f"Backend: Bot full streamed response (from model {_BACKEND_CURRENT_MODEL_NAME}): {full_bot_response_for_log}")
    else:
        print(f"Backend: LLM stream (astream_events, model {_BACKEND_CURRENT_MODEL_NAME}) was empty or did not produce 'on_chat_model_stream' content.")
        yield "Sorry, I received an empty or unexpected response from the AI."

def get_initialization_status():
    global _BACKEND_CHAT_LLM_CHAIN, _BACKEND_INITIALIZATION_ERROR, _BACKEND_CURRENT_MODEL_NAME
    if _BACKEND_INITIALIZATION_ERROR:
        return False, _BACKEND_INITIALIZATION_ERROR, _BACKEND_CURRENT_MODEL_NAME
    if _BACKEND_CHAT_LLM_CHAIN is None:
        return False, f"Chat LLM chain not initialized for model '{_BACKEND_CURRENT_MODEL_NAME}'. API key might be set but other init step failed or pending.", _BACKEND_CURRENT_MODEL_NAME
    return True, f"Successfully initialized with model '{_BACKEND_CURRENT_MODEL_NAME}'.", _BACKEND_CURRENT_MODEL_NAME