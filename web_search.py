# backend/tools/web_search.py
import requests
from typing import List, Dict
from urllib.parse import quote

WIKI_OPENSEARCH = "https://en.wikipedia.org/w/api.php?action=opensearch&limit=5&format=json&search={q}"
WIKI_SUMMARY = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"

def wiki_search(query: str) -> List[Dict]:
    q = quote(query)
    url = WIKI_OPENSEARCH.format(q=q)
    r = requests.get(url, timeout=10)
    if r.status_code != 200:
        return []
    j = r.json()
    # j[1] titles, j[2] descriptions, j[3] links
    titles = j[1]
    descs = j[2]
    links = j[3]
    results = []
    for t, d, l in zip(titles, descs, links):
        # fetch summary for richer text
        try:
            sj = requests.get(WIKI_SUMMARY.format(title=quote(t))).json()
            extract = sj.get("extract") or d or ""
        except Exception:
            extract = d or ""
        results.append({"title": t, "snippet": extract, "link": l})
    return results
