import os

import dotenv
from langchain_deepseek import ChatDeepSeek

dotenv.load_dotenv()

os.environ['DEEPSEEK_API_KEY'] = os.getenv('DEEPSEEK_API_KEY')
os.environ['DEEPSEEK_API_BASE'] = os.getenv('DEEPSEEK_BASE_URL')

def get_llm():
    """Get a chat model from deepseek api key."""
    return ChatDeepSeek(
        model='deepseek-chat',
        temperature=0.4
    )
