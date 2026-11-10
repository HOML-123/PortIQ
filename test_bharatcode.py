
import os
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("BHARATCODE_API_KEY")

if not api_key:
    raise SystemExit("ERROR: BHARATCODE_API_KEY not found in .env")

url = "https://bharatcode.ai/api/model/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

payload = {
    "model": "qwen-3.8-27b",
    "messages": [
        {
            "role": "system",
            "content": "You are a logistics operations analyst."
        },
        {
            "role": "user",
            "content": "In two sentences, explain how port congestion affects container shipment delays."
        }
    ],
    "temperature": 0.3,
    "max_tokens": 120
}

try:
    response = requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=60
    )

    print("HTTP STATUS:", response.status_code)

    if response.ok:
        result = response.json()
        print("MODEL:", result.get("model", "Not specified"))
        print("QWEN RESPONSE:")
        print(result["choices"][0]["message"]["content"])
    else:
        print("API ERROR:")
        print(response.text[:1500])

except requests.exceptions.Timeout:
    print("ERROR: Request timed out.")

except requests.exceptions.RequestException as error:
    print("NETWORK ERROR:", str(error))
