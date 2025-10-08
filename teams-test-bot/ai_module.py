# ai_module.py
import os
import requests

def call_openrouter_model(prompt: str) -> str:
    """Send prompt to OpenRouter and return the reply."""
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
        "Content-Type": "application/json",
    }

    body = {
        "model": "qwen/qwen3-8b",
        "messages": [
            {"role": "system", "content": "You are a concise and helpful assistant."},
            {"role": "user", "content": prompt},
        ],
    }

    try:
        response = requests.post(url, headers=headers, json=body, timeout=60)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Error calling model: {e}")
        return "Sorry, I couldn’t process your request."
