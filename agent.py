import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# Load environment variables from .env
load_dotenv()

# Get the Hugging Face API key
hf_token = os.getenv("HF_TOKEN")

if not hf_token:
    raise ValueError("Missing Hugging Face API token. Set HF_TOKEN in .env file.")

# Initialize the client
client = InferenceClient("meta-llama/Llama-3.2-3B-Instruct", token=hf_token)

# Ask the user for a question
user_question = input("Ask a political question: ")

# Structured prompt for better responses
prompt = f"""
You are an AI specializing in political analysis. 
Provide a well-structured, unbiased answer to the following question: 

Question: {user_question}

Your response should be clear, factual, and insightful.
"""

# Generate AI response
output = client.text_generation(prompt, max_new_tokens=200)

print("\n🔹 AI Response:\n", output)