from backend.ollama_client import OllamaClient
from backend.retriever import Retriever
from backend.tools.web_search import wiki_search
from backend.tools.summarizer import summarize_context
from typing import List, Dict

class AgentController:
    def __init__(self, data_dir: str = "../data"):
        self.retriever = Retriever(data_dir)

    def plan_and_execute(self, query: str, top_k: int = 4) -> Dict:
        # Step 1: retrieve from local index
        local_hits = self.retriever.retrieve(query, top_k=top_k)
        
        # Step 2: if local evidence weak (few hits or low scores), enrich via web search
        need_web = len(local_hits) < 2 or (len(local_hits) > 0 and local_hits[0]["score"] < 0.3)
        
        web_hits = []
        if need_web:
            print(f"Local context is weak, searching web for: {query}")
            web_hits_raw = wiki_search(query)
            # Convert web hits to same shape as local hits for consistency
            for w in web_hits_raw:
                web_hits.append({
                    "id": None, 
                    "source": w.get("link") or w.get("title"), 
                    "offset": 0, 
                    "text": w.get("snippet"), 
                    "score": 0.0
                })

        # Combine local and web results
        combined = (local_hits or []) + web_hits
        
        # Step 3: Summarize using Ollama
        summary = summarize_context(query, combined, max_tokens=1024)
        
        return {"answer": summary, "sources": combined}

