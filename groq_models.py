import requests
import os

api_key = os.environ.get("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")
url = "https://api.groq.com/openai/v1/models"

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

response = requests.get(url, headers=headers)

print(response.json())