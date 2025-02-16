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

# Test if the model responds
output = client.text_generation("The current political situation in Germany is", max_new_tokens=100)

print(output)
