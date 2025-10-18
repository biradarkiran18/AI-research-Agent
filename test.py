import requests

try:
    r = requests.get("http://127.0.0.1:11434")
    print("Ollama reachable:", r.text)
except Exception as e:
    print("Ollama is not reachable:", e)
