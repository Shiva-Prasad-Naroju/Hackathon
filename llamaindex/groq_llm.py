from llama_index.llms.openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

llm = OpenAI(
    model="llama3-8b-8192",
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
    temperature=0.0
)
