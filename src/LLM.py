from google import genai
import os
from dotenv import load_dotenv
load_dotenv()

def generate_response(prompt):
    # Initialize the GenAI client
    client = genai.Client()

    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=prompt,
    )

    # Log usage information (epochs/tokens processed)
    if hasattr(response, 'usage_metadata'):
        usage = response.usage_metadata
        print(f"🔄 LLM Processing - Input tokens: {getattr(usage, 'prompt_token_count', 'N/A')}, "
              f"Output tokens: {getattr(usage, 'candidates_token_count', 'N/A')}, "
              f"Total tokens: {getattr(usage, 'total_token_count', 'N/A')}")
    else:
        print("🔄 LLM Processing - Usage metadata not available")

    return response.text