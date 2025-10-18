import requests
import typing as t
import json

OLLAMA_URL = "http://127.0.0.1:11434"  # Using direct IP to avoid name resolution issues

class OllamaClient:
    def __init__(self, base_url: str = OLLAMA_URL, model: str = "gemma:9b") -> None:
        self.base = base_url.rstrip("/")
        self.model = model

    def generate(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.0) -> str:
        """
        Calls Ollama's /api/chat endpoint with a prompt and returns the generated text.
        """
        url = f"{self.base}/api/chat"
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            },
            "stream": False
        }

        # Bypasses potential issues with misconfigured system-wide proxies.
        proxies = {
            "http": None,
            "https": None,
        }

        try:
            # **THE FIX:** Increased the timeout to 300 seconds (5 minutes).
            # This gives large models like gemma:9b enough time to load on the first run.
            resp = requests.post(url, json=payload, timeout=300, proxies=proxies)
            resp.raise_for_status()
            
            j = resp.json()
            if j.get("message") and isinstance(j["message"], dict):
                return j["message"].get("content", "").strip()
            else:
                return "Error: Unexpected response format from Ollama."

        except requests.exceptions.ReadTimeout:
            print(f"Ollama model loading timed out. The model '{self.model}' is likely still loading in the background.")
            return f"Error: The Ollama model '{self.model}' is taking a long time to load. This is expected on the first run. Please try asking your question again in a minute or two."
        except requests.exceptions.RequestException as e:
            print(f"Error connecting to Ollama: {e}")
            return f"Error: Could not connect to Ollama at {self.base}. Is it running?"
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return "An unexpected error occurred while communicating with Ollama."

# convenience singleton
client = OllamaClient()

