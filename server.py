# backend/server.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import tempfile, os
from .ingest import ingest_pdf_to_retriever
from .agent_controller import AgentController
from pydantic import BaseModel

app = FastAPI(title="Ollama Agent Research Backend")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

agent = AgentController(data_dir="../data")

class QueryRequest(BaseModel):
    query: str
    top_k: int = 4

@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF allowed")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    tmp.write(await file.read())
    tmp.flush()
    tmp.close()
    try:
        ids = ingest_pdf_to_retriever(tmp.name, source_name=file.filename, data_dir="../data")
        return {"status": "ok", "ingested_ids": ids}
    finally:
        os.unlink(tmp.name)

@app.post("/query")
async def query(req: QueryRequest):
    q = req.query.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Empty query")
    res = agent.plan_and_execute(q, top_k=req.top_k)
    return res

@app.get("/status")
def status():
    return {"status": "running"}

if __name__ == "__main__":
    uvicorn.run("backend.server:app", host="0.0.0.0", port=8000, reload=True)
