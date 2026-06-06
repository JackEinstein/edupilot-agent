import os

import dotenv
from langchain_deepseek import ChatDeepSeek

dotenv.load_dotenv()

DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY') or os.getenv('API_KEY')
DEEPSEEK_BASE_URL = os.getenv('DEEPSEEK_BASE_URL') or os.getenv('BASE_URL')

if DEEPSEEK_API_KEY:
    os.environ['DEEPSEEK_API_KEY'] = DEEPSEEK_API_KEY

if DEEPSEEK_BASE_URL:
    os.environ['DEEPSEEK_API_BASE'] = DEEPSEEK_BASE_URL


def get_llm():
    """Get a chat model from deepseek api key."""
    if not DEEPSEEK_API_KEY:
        raise RuntimeError(
            'Missing DeepSeek API key. Set DEEPSEEK_API_KEY or API_KEY in your .env file.'
        )

    return ChatDeepSeek(
        model='deepseek-chat',
        temperature=0.4
    )
