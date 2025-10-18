from backend.ollama_client import client # CORRECTED IMPORT
from typing import List, Dict

def summarize_context(question: str, contexts: List[Dict], max_tokens: int = 400) -> str:
    """
    contexts: list of {"source":..., "text":..., "score":...}
    """
    if not contexts:
        return "No information found to answer the question."

    blocks = []
    for c in contexts:
        s = c.get("source", "unknown")
        text = c.get("text", "")[:2000]  # keep reasonable length
        blocks.append(f"Source: {s}\n\n{text}")

    prompt = (
        "You are a concise research assistant. Use the context blocks below to answer the question with direct, cited points.\n\n"
        "CONTEXT:\n" + "\n\n---\n\n".join(blocks) +
        f"\n\nQUESTION: {question}\n\nINSTRUCTIONS:\n- Provide a short answer (3-6 sentences).\n- Then provide 3 bullet key points with the source for each (source name or link).\n- If the answer is not supported by the provided context, write: 'No reliable answer found in provided sources.'\n\nAnswer:"
    )

    out = client.generate(prompt, max_tokens=max_tokens, temperature=0.0)
    return out.strip()
