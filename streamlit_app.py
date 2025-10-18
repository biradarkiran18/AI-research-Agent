import streamlit as st
import requests

# It's good practice to get the backend URL from secrets or environment variables
# but for simplicity, we'll keep it here.
BACKEND = st.secrets.get("backend_url", "http://localhost:8000")

st.set_page_config(page_title="Ollama Research Agent", layout="wide")
st.title("Ollama Research Agent")

# Ingest new documents
with st.expander("Ingest PDF"):
    uploaded = st.file_uploader("Upload a PDF to add to the knowledge base", type=["pdf"])
    if uploaded:
        # Prepare the file for the POST request
        files = {"file": (uploaded.name, uploaded.getvalue(), "application/pdf")}
        try:
            # Set a timeout for the request
            resp = requests.post(f"{BACKEND}/ingest", files=files, timeout=120)
            if resp.status_code == 200:
                st.success(f"Ingested: {resp.json()}")
            else:
                st.error(f"Error: {resp.status_code} {resp.text}")
        except requests.exceptions.RequestException as e:
            st.error(f"Failed to connect to backend: {e}")

st.markdown("---")

# Query the agent
query = st.text_input("Ask a research question")
top_k = st.slider("Number of sources to retrieve (top_k)", min_value=1, max_value=10, value=4)

if st.button("Ask") and query:
    payload = {"query": query, "top_k": top_k}
    try:
        # **THE FIX:** Increased the timeout to 300 seconds (5 minutes)
        # This gives the backend more time to process, especially on the first run.
        resp = requests.post(f"{BACKEND}/query", json=payload, timeout=300)
        
        if resp.status_code == 200:
            data = resp.json()
            st.subheader("Answer")
            st.write(data.get("answer"))
            
            st.subheader("Sources")
            sources = data.get("sources", [])
            if sources:
                for s in sources[:10]:
                    # Using an expander for each source to keep the UI clean
                    with st.expander(f"**{s.get('source','unknown')}** (score: {s.get('score',0):.3f})"):
                        # Display a snippet of the text
                        snippet = s.get("text", "")
                        st.write(snippet[:800] + ("..." if len(snippet) > 800 else ""))
            else:
                st.write("No sources were found for this answer.")
        else:
            st.error(f"Error from backend: {resp.status_code} {resp.text}")

    except requests.exceptions.ReadTimeout:
        st.error("The request timed out. The backend is taking too long to respond. This might happen on the first query as models are loaded. Please try again in a moment.")
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to connect to backend: {e}")
