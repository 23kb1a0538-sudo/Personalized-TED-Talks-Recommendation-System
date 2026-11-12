"""
FastAPI REST API for TED Talks Recommendation Engine
Provides programmatic access to recommendation algorithms, talk metadata,
search, and evaluation benchmarks.
"""

import sys
import os
import json
import pandas as pd
from typing import List, Optional
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

sys.path.append("/workspace")
from src.models import HybridTEDRecommender

app = FastAPI(
    title="TED Talks Recommendation API",
    description="High-performance machine learning recommendation engine combining Content-Based and Collaborative Filtering.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables loaded on startup
recommender: Optional[HybridTEDRecommender] = None
df_talks: Optional[pd.DataFrame] = None
dataset_summary: dict = {}
evaluation_summary: dict = {}


@app.on_event("startup")
def startup_event():
    global recommender, df_talks, dataset_summary, evaluation_summary
    print("Initializing Recommender in FastAPI...")
    recommender = HybridTEDRecommender()
    df_talks = pd.read_csv("/workspace/data/processed/ted_clean.csv")
    with open("/workspace/data/processed/dataset_summary.json") as f:
        dataset_summary = json.load(f)
    with open("/workspace/data/processed/evaluation_summary.json") as f:
        evaluation_summary = json.load(f)
    print("FastAPI ready to serve requests.")


class RecommendationRequest(BaseModel):
    talk_id: Optional[int] = Field(None, description="Seed talk ID for content-based similarity")
    user_id: Optional[int] = Field(None, description="User ID (1-800) for collaborative filtering")
    query: Optional[str] = Field(None, description="Free-text semantic search query")
    alpha: float = Field(0.55, ge=0.0, le=1.0, description="Weight for content vs collaborative (1.0 = content, 0.0 = collab)")
    top_n: int = Field(6, ge=1, le=50, description="Number of recommendations to return")
    tags_filter: Optional[List[str]] = Field(None, description="Optional tag filters")


@app.get("/")
def root():
    return {
        "service": "TED Talks Machine Learning Recommendation Engine",
        "status": "online",
        "version": "1.0.0",
        "total_talks": len(df_talks) if df_talks is not None else 0,
        "docs_url": "/docs"
    }


@app.get("/summary")
def get_dataset_summary():
    """Returns dataset summary metrics and emotional distribution."""
    return dataset_summary


@app.get("/metrics")
def get_evaluation_metrics():
    """Returns offline evaluation benchmarks across models."""
    return evaluation_summary


@app.get("/talks")
def list_talks(
    query: Optional[str] = None,
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0)
):
    """Lists talks with optional keyword filtering and pagination."""
    if df_talks is None:
        raise HTTPException(status_code=503, detail="Service initializing")

    filtered = df_talks
    if query:
        q = query.lower()
        filtered = filtered[
            filtered['title'].str.lower().str.contains(q) |
            filtered['main_speaker'].str.lower().str.contains(q) |
            filtered['tag_string'].str.lower().str.contains(q)
        ]

    total = len(filtered)
    page_data = filtered.iloc[offset:offset + limit]

    results = []
    for _, row in page_data.iterrows():
        results.append({
            "talk_id": int(row['talk_id']),
            "title": row['title'],
            "main_speaker": row['main_speaker'],
            "speaker_occupation": row['speaker_occupation'],
            "duration_min": float(row['duration_min']),
            "views": int(row['views']),
            "top_emotion": row.get('top_emotion', 'Inspiring'),
            "tags": eval(row['tags']) if isinstance(row['tags'], str) else row['tags'],
            "url": row['url']
        })

    return {"total": total, "offset": offset, "limit": limit, "items": results}


@app.get("/talks/{talk_id}")
def get_talk_detail(talk_id: int):
    """Retrieves full metadata for a specific talk."""
    if df_talks is None:
        raise HTTPException(status_code=503, detail="Service initializing")

    matching = df_talks[df_talks['talk_id'] == talk_id]
    if matching.empty:
        raise HTTPException(status_code=404, detail=f"Talk ID {talk_id} not found")

    row = matching.iloc[0]
    return {
        "talk_id": int(row['talk_id']),
        "title": row['title'],
        "main_speaker": row['main_speaker'],
        "speaker_occupation": row['speaker_occupation'],
        "description": row['description'],
        "duration_min": float(row['duration_min']),
        "views": int(row['views']),
        "film_year": int(row['film_year']),
        "top_emotion": row.get('top_emotion', 'Inspiring'),
        "tags": eval(row['tags']) if isinstance(row['tags'], str) else row['tags'],
        "url": row['url'],
        "transcript_lead": row.get('transcript_lead', '')
    }


@app.post("/recommend")
def recommend_talks(req: RecommendationRequest):
    """Generates personalized TED recommendations."""
    if recommender is None:
        raise HTTPException(status_code=503, detail="Recommender engine initializing")

    recs = recommender.recommend(
        talk_id=req.talk_id,
        user_id=req.user_id,
        query=req.query,
        tags_filter=req.tags_filter,
        alpha=req.alpha,
        top_n=req.top_n
    )
    return {
        "status": "success",
        "count": len(recs),
        "parameters": {
            "talk_id": req.talk_id,
            "user_id": req.user_id,
            "query": req.query,
            "alpha": req.alpha,
            "top_n": req.top_n
        },
        "recommendations": recs
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
