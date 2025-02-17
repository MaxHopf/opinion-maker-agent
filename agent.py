import os
from dotenv import load_dotenv
from smolagents import HfApiModel

# Load environment variables from .env
load_dotenv()
hf_token = os.getenv("HF_TOKEN")

if not hf_token:
    raise ValueError("Missing Hugging Face API token. Set HF_TOKEN in .env file.")

# Initialize AI Model using SmolAgents
model = HfApiModel(
    model_id="meta-llama/Llama-3.2-3B-Instruct",
    max_tokens=200,
    temperature=0.5,
    token=hf_token
)

print("✅ AI Model Loaded Successfully!")

# Ask the user for a question
user_question = input("Ask a political question: ")

# Format the message properly with a role
messages = [
    {
        "role": "user",
        "content": user_question
    }
]

# Generate AI response
try:
    response = model(messages)
    # Extract just the content from the response
    if hasattr(response, 'content'):
        output = response.content
    else:
        # If response is more deeply nested, try to get content
        output = response.raw.choices[0].message.content
    
    print("\n🔹 AI Response:", output)
except Exception as e:
    print(f"Error: {str(e)}")