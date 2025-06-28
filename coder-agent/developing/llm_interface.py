import os
import json
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from config import read_configs

class ChatBackend():

    def load_conversation_state(self,state_file):
        if os.path.exists(state_file):
            with open(state_file, "r") as f:
                return json.load(f)
        print("Files does not exist, returning empty history")
        return []

    def __init__(self, config_address="/home/raha/agent_factory/coder-agent/config.json"):
        configs = read_configs(config_address)
        os.environ["OPENAI_API_KEY"] = configs["chatbot"]["OPENAI_API_KEY"]

        self.model = ChatOpenAI(model="gpt-4.1-mini", streaming=True)

        self.state_path = configs["chatbot"].get("CONV_LOG_PATH", "conversation_state.json")
        self.history = self.load_conversation_state(self.state_path)
        print("Read history:")
        print(self.history)
        print("*************************************************")

        system_prompt = SystemMessage(content="""You are Coder Agent, an expert AI assistant who writes clean and easy to understand code. At each step you need write the best code possible. different options are not necessary. Don't add comments to code. The code should be production ready. Demos, incomplete code or code that requires further work is not acceptable. When a piece of code is provided, you should not change it unless the user asks you to do so. If the user asks you to change a piece of code, you should only change necessary part of the code and not the rest of the code.""")
        self.history_langchain_format = [system_prompt]
        for msg in self.history:
            if msg["role"] == "user":
                self.history_langchain_format.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                self.history_langchain_format.append(AIMessage(content=msg["content"]))

    def save_conversation_state(self,history):
        with open(self.state_path, "w") as f:
            json.dump(history, f, indent=2)

    def predict(self,message):

        self.history_langchain_format.append(HumanMessage(content=message))

        response = ""
        for chunk in self.model.stream(self.history_langchain_format):
            response += chunk.content
            yield response

        new_history = self.history + [{"role": "user", "content": message}, {"role": "assistant", "content": response}]
        self.save_conversation_state(new_history)

    def reset_conversation(self):
        self.save_conversation_state([])