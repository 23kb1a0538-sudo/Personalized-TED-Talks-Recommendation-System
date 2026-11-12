# Personalized TED Talks Recommendation System Using Machine Learning

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.67.0-FF4B4B.svg)](https://streamlit.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.9.1-F7931E.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **SpringBoard Internship Program Capstone Project**  
> An end-to-end, production-ready machine learning recommendation engine designed to overcome content overload across TED Talks. Combines **TF-IDF n-grams**, **Latent Semantic Analysis (LSA TruncatedSVD)**, **14-dimensional emotional resonance vectors**, and **Matrix Factorization (SVD)** into a dynamic hybrid scoring architecture with automated cold-start fallback and recommendation explainability.

---

## 🚀 Live Services & Demos

- **Streamlit Interactive Web Application:** [TED Recommender App](https://winter-blessed-rhino.8501.dev.raccoonai.tech)
- **FastAPI Interactive Swagger REST API:** [API Documentation](https://winter-blessed-rhino.8000.dev.raccoonai.tech/docs)
- **Executive Presentation Slides (PDF):** `ted_talks_presentation.pdf`
- **Interactive Presentation Deck (HTML):** `slides/presentation.html`
- **Complete Pipeline Jupyter Notebook:** `notebooks/ted_recommendation_pipeline.ipynb`

---

## 🌟 Key Features

1. **4 Flexible Recommendation Modes:**
   - 🎯 **Hybrid Recommendation:** Dynamic linear blend of content similarity and collaborative matrix factorization with user-controlled $\alpha$ slider.
   - 📖 **Content-Based Similarity:** Seed talk matching using composite similarity across title, tags, description, speaker occupation, transcript, and emotional tone.
   - 👤 **Collaborative Filtering:** Matrix Factorization ($K=40$) predicting ratings based on user personas and viewing histories.
   - 🔍 **Natural Language Semantic Search:** Free-form conceptual queries vectorized against the talk corpus.
2. **Audience Emotional Resonance Modeling:**
   - Normalizes audience feedback across 14 emotional dimensions (*Inspiring, Fascinating, Informative, Funny, Ingenious, Persuasive, Courageous, Beautiful, Jaw-dropping, etc.*).
3. **Transparent Explainability Engine:**
   - Every recommendation provides human-readable explanations ("Shares 4 common tags: *neuroscience, brain, psychology*", "92% emotional tone match: *Inspiring*", "Preferred by viewers with similar tastes").
4. **Interactive Analytics & Explorer:**
   - Searchable catalog of 2,628 talks with sorting, filtering, and transcript previews.
   - EDA dashboards for talk durations, views vs. comments correlation, top tags, and emotional distributions.
5. **Production REST API:**
   - High-throughput FastAPI service with endpoints for talk details, search, batch recommendations, and benchmark metrics.

---

## 📊 Offline Benchmark Evaluation

The system was evaluated against 500 benchmark talks using TED editorial/viewer ground truth related talks, and 5,486 held-out test user interactions:

### Content-Based Retrieval Benchmark (Ground Truth Related Talks)
| Model | Precision@5 | Precision@10 | Recall@10 | NDCG@10 | MAP@10 | Catalog Coverage |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| TF-IDF Cosine | 0.2368 | 0.1752 | 0.2979 | 0.5607 | 0.1758 | 78.54% |
| Semantic LSA (100 dims) | 0.1948 | 0.1494 | 0.2537 | 0.4709 | 0.1381 | 78.01% |
| **Composite Hybrid Content** ⭐ | **0.2400** | **0.1824** | **0.3095** | **0.5703** | **0.1829** | **78.58%** |

### Hybrid Alpha ($\alpha$) Optimization Curve (Held-Out User Interactions)
| Configuration | Alpha ($\alpha$) | Precision@10 | Recall@10 | NDCG@10 | Hit Rate@10 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Pure Collaborative (SVD) | 0.0 | 0.0106 | 0.0191 | 0.0464 | 10.37% |
| Hybrid (Balanced) | 0.4 | 0.0121 | 0.0213 | 0.0507 | **11.63%** |
| **Hybrid (Optimal)** ⭐ | **0.55** | **0.0119** | **0.0205** | **0.0507** | **11.38%** |
| Pure Content | 1.0 | 0.0073 | 0.0115 | 0.0364 | 7.12% |

*The Hybrid approach achieves a **+63% relative increase in Hit Rate@10** compared to pure content-based filtering, while avoiding collaborative cold-start degradation.*

---

## 📂 Project Repository Structure

```
├── app/
│   ├── streamlit_app.py        # Interactive Streamlit Web Application (Port 8501)
│   └── api.py                  # FastAPI REST Service (Port 8000)
├── src/
│   ├── preprocessing.py        # Ingestion, cleaning, metadata soup, emotion extraction
│   ├── eda.py                  # Statistical analysis and plot generation
│   ├── vectorization.py        # TF-IDF, TruncatedSVD LSA, user-item interaction matrix
│   ├── models.py               # ContentBased, Collaborative (SVD), Hybrid Recommender
│   └── evaluation.py           # Precision@K, Recall@K, NDCG@K, MAP@K, alpha curve
├── docs/
│   ├── ARCHITECTURE.md         # Full mathematical & architectural specification
│   ├── EVALUATION_REPORT.md    # Detailed metrics, benchmark tables, and trade-off analysis
│   └── assets/                 # Generated EDA and evaluation plots
├── slides/
│   └── presentation.html       # 8-slide presentation deck (16:9 widescreen)
├── notebooks/
│   └── ted_recommendation_pipeline.ipynb # End-to-end runnable Jupyter notebook
├── tests/
│   └── test_recommender.py     # Unit test suite (6 passing tests)
├── data/
│   ├── raw/                    # Raw TED datasets (ted_main.csv, transcripts.csv)
│   └── processed/              # Cleaned parquet/csv and JSON summary stats
├── models/                     # Serialized artifacts (TF-IDF, SVD, embeddings, matrices)
├── ted_talks_presentation.pdf  # Compiled Capstone presentation slides (PDF)
├── ted_recommendation_system.zip # Complete packaged deliverable archive
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

## 🛠️ Installation & Quickstart

### 1. Clone & Set Up Environment
```bash
git clone <repository_url>
cd ted-talks-recommendation-system
pip install -r requirements.txt
```

### 2. Run Data Processing & Train Models
```bash
python3 src/preprocessing.py
python3 src/vectorization.py
python3 src/evaluation.py
```

### 3. Launch Web Application (Streamlit)
```bash
streamlit run app/streamlit_app.py --server.port 8501 --server.address 0.0.0.0
```
Open your browser at `http://localhost:8501`.

### 4. Launch REST API (FastAPI)
```bash
python3 app/api.py
```
API Documentation available at `http://localhost:8000/docs`.

### 5. Run Unit Tests
```bash
python3 -m unittest tests/test_recommender.py
```

---

## 💻 Python API Usage Example

```python
from src.models import HybridTEDRecommender

# Initialize recommender engine
engine = HybridTEDRecommender()

# 1. Content-based recommendation for Ken Robinson's Creativity talk (ID: 0)
content_recs = engine.recommend(talk_id=0, top_n=5)

# 2. Collaborative recommendation for User 1
collab_recs = engine.recommend(user_id=1, top_n=5)

# 3. Hybrid recommendation (60% Content, 40% Collaborative)
hybrid_recs = engine.recommend(talk_id=0, user_id=1, alpha=0.6, top_n=5)

# 4. Natural Language Semantic Query
query_recs = engine.recommend(query="artificial intelligence neuroscience memory", top_n=5)
```

---

## 📜 SpringBoard Internship Milestone Mapping

- **Milestone 1 (Week 1–2):** Data collection, cleaning, missing value imputation, 14-D emotion extraction, EDA visualizations.
- **Milestone 2 (Week 3–4):** TF-IDF n-grams (12k features), Latent Semantic Analysis (TruncatedSVD $K=100$), sparse CSR user-item matrix.
- **Milestone 3 (Week 5–6):** Content-Based, SVD Collaborative, and Dynamic Hybrid models with Precision@K, Recall@K, NDCG@K, MAP@K benchmarks.
- **Milestone 4 (Week 7–8):** Streamlit web deployment, FastAPI REST API, presentation slides, comprehensive technical documentation, and deliverable bundle.
