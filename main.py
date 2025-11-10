from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import faiss
import numpy as np
import pandas as pd
import pickle
import re
import time
import logging
import requests
import os

# ------------------------------------------------------------
# 🚀 APP INITIALIZATION
# ------------------------------------------------------------
app = FastAPI(
    title="SHL Assessment Recommender API (Gemini + FAISS)",
    description="Semantic retrieval API for SHL assessments using Gemini embeddings + FAISS",
    version="2.0.0"
)

# ------------------------------------------------------------
# 🔧 CONFIGURATION & LOGGING
# ------------------------------------------------------------
logging.basicConfig(filename="trace.log", level=logging.INFO)

INDEX_FILE = "index.faiss"
META_FILE = "meta.pkl"

try:
    index = faiss.read_index(INDEX_FILE)
    df_meta = pickle.load(open(META_FILE, "rb"))
except Exception as e:
    raise RuntimeError(f"❌ Failed to load index or metadata: {e}")


# ------------------------------------------------------------
# 🔠 TEST TYPE MAPPING
# ------------------------------------------------------------
TEST_TYPE_MAP = {
    "A": "Ability and Aptitude",
    "B": "Biodata and Situational Judgement",
    "C": "Competencies",
    "D": "Development and 360",
    "E": "Assessment Exercises",
    "K": "Knowledge and Skills",
    "P": "Personality and Behavior",
    "S": "Simulations"
}

def expand_test_type(tt: str):
    """Expand test type abbreviations into full descriptive names."""
    if not isinstance(tt, str):
        return ""
    expanded = []
    for t in tt.replace(" ", "").split(","):
        if t in TEST_TYPE_MAP:
            expanded.append(TEST_TYPE_MAP[t])
    return ", ".join(expanded)

# ------------------------------------------------------------
# ⏱️ HELPER FUNCTIONS
# ------------------------------------------------------------
def parse_duration_from_query(q: str):
    """Extract desired duration in minutes from the query."""
    q = q.lower()
    if "minute" in q or "min" in q:
        m = re.search(r"(\d+)", q)
        if m:
            return int(m.group(1))
    if "hour" in q:
        m = re.search(r"(\d+)", q)
        if m:
            return int(m.group(1)) * 60
    if "short" in q:
        return 20
    if "long" in q or "detailed" in q:
        return 60
    return None


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set in environment")

# Correct endpoint for gemini-embedding-001
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent"

def get_gemini_embedding(text: str) -> np.ndarray:
    """
    Call Gemini gemini-embedding-001 and return a float32 numpy array.
    Raises HTTPException on any non-200 response (so your FastAPI route returns a 500).
    """
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": GEMINI_API_KEY   # recommended header auth
    }

    payload = {
        "model": "gemini-embedding-001",
        "content": {"parts": [{"text": text}]}
    }

    try:
        resp = requests.post(GEMINI_URL, headers=headers, json=payload, timeout=30)
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Embedding request failed: {e}")

    if resp.status_code != 200:
        # include body so logs show the provider error
        raise HTTPException(status_code=500, detail=f"Gemini API error: {resp.status_code}: {resp.text}")

    j = resp.json()
    # expected shape: {"embedding": {"values": [...]}} or similar nested shapes
    # try a couple of reasonable paths
    for path in (("embedding","values"), ("data",0,"embedding","values"), ("data",0,"embedding")):
        cur = j
        try:
            for k in path:
                cur = cur[k]
            arr = np.array(cur, dtype=np.float32)
            return arr
        except Exception:
            continue

    # if we reach here, response format was unexpected
    raise HTTPException(status_code=500, detail=f"Unexpected Gemini response structure: {j}")
# ------------------------------------------------------------
# 📦 DATA MODELS
# ------------------------------------------------------------
class RecommendRequest(BaseModel):
    query: str

class Assessment(BaseModel):
    url: str
    adaptive_support: str
    description: str
    duration: int
    remote_support: str
    test_type: List[str]

class RecommendResponse(BaseModel):
    recommended_assessments: List[Assessment]

# ------------------------------------------------------------
# 🌐 ENDPOINTS
# ------------------------------------------------------------
@app.get("/")
def root():
    return {"message": "✅ SHL Recommender API (Gemini + FAISS) is live."}

@app.get("/health")
def health_check():
    return {"status": "healthy", "items_loaded": len(df_meta)}

# ------------------------------------------------------------
# 🔍 RECOMMENDATION ENDPOINT (Gemini-powered)
# ------------------------------------------------------------
@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    """Return top 10 SHL assessments matching the input query using Gemini embeddings."""
    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query is required")

    start_time = time.time()
    logging.info(f"Received query: {query}")

    # ---- 1️⃣ Get embedding from Gemini API ----
    try:
        q_emb = get_gemini_embedding(query).reshape(1, -1)
        faiss.normalize_L2(q_emb)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding generation failed: {e}")

    # ---- 2️⃣ Search FAISS ----
    D, I = index.search(q_emb, 30)
    results = df_meta.iloc[I[0]].copy()
    results["semantic_score"] = D[0][:len(results)]

    # ---- 3️⃣ Duration-aware re-ranking ----
    desired_time = parse_duration_from_query(query)
    if desired_time:
        def duration_score(d):
            try:
                val = int(re.search(r"(\d+)", str(d)).group(1))
                return 1.0 - abs(val - desired_time) / max(desired_time, 1)
            except:
                return 0
        results["duration_score"] = results["Assessment length"].apply(duration_score)
        results["combined_score"] = 0.8 * results["semantic_score"] + 0.2 * results["duration_score"]
    else:
        results["combined_score"] = results["semantic_score"]

    # ---- 4️⃣ Get Top 10 ----
    top = results.sort_values("combined_score", ascending=False).head(10)
    print(top)
    recs = []
    for _, row in top.iterrows():
        try:
            duration = int(re.search(r"(\d+)", str(row.get("Assessment length", "0"))).group(1))
        except:
            duration = 0

        recs.append({
            "url": str(row.get("URL", "")),
            "adaptive_support": str(row.get("Adaptive/IRT", "Unknown")),
            "description": str(row.get("Description", "")),
            "duration": duration,
            "remote_support": str(row.get("Remote Testing", "Unknown")),
            "test_type": [
                t.strip() for t in expand_test_type(row.get("Test Type", "")).split(",") if t.strip()
            ],
        })

    elapsed = time.time() - start_time
    logging.info(f"Processed query in {elapsed:.2f}s, returning {len(recs)} results")

    return {"recommended_assessments": recs}
