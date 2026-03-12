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
    return response.text