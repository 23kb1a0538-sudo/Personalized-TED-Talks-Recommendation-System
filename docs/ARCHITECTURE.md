# System Architecture & Technical Specifications

## 1. Executive Overview

The **Personalized TED Talks Recommendation System** is designed to address information overload in multimedia content discovery. Combining natural language processing (NLP), latent semantic decomposition, and collaborative filtering, the system surfaces personalized, serendipitous, and highly relevant TED talks tailored to individual viewer tastes.

```
                      +---------------------------------------+
                      | Raw TED Corpus (Metadata + Transcripts)|
                      +---------------------------------------+
                                          |
                                          v
                      +---------------------------------------+
                      |   Data Cleaning & Feature Pipeline    |
                      | - HTML removal, tokenization, soup    |
                      | - 14-D Emotion Vector Normalization   |
                      +---------------------------------------+
                                          |
                   +----------------------+----------------------+
                   |                                             |
                   v                                             v
    +------------------------------+             +-------------------------------+
    |  Content Vectorization Layer |             |  Collaborative Factorization  |
    | - TF-IDF n-grams (12k dims)  |             | - Sparse User-Item CSR Matrix |
    | - Semantic LSA (100 dims)    |             | - TruncatedSVD (40 factors)   |
    | - Emotional Cosine Sim       |             | - Predicted User Ratings      |
    +------------------------------+             +-------------------------------+
                   |                                             |
                   +----------------------+----------------------+
                                          |
                                          v
                      +---------------------------------------+
                      |       Dynamic Hybrid Re-ranking       |
                      |   S = α·S_content + (1-α)·Norm(S_svd) |
                      |   - Cold-Start Fallback Router        |
                      |   - Topic & Tag Re-weighting          |
                      +---------------------------------------+
                                          |
                         +----------------+----------------+
                         |                                 |
                         v                                 v
        +--------------------------------+ +--------------------------------+
        |   Streamlit Interactive App    | |      FastAPI REST API          |
        |   (Port 8501: Modern Web UI)   | |   (Port 8000: /docs Swagger)   |
        +--------------------------------+ +--------------------------------+
```

---

## 2. Data Pipeline & Representation

### 2.1 Ingestion & Cleaning
- **Raw Data Ingestion:** 2,628 talks from `ted_main.csv` merged with 2,542 full text transcripts from `transcripts.csv` via TED URL keys.
- **Deduplication:** Sanitized unique talk records.
- **Text Normalization:**
  - Removal of URLs, HTML tags, punctuation, and extraneous whitespace.
  - Lowercase normalization.
  - Extraction of the initial 250 words of each transcript to preserve core premise without narrative noise.
- **Content Metadata Soup:**
  $$\text{Soup} = 3 \times (\text{Title}) + 3 \times (\text{Tags}) + \text{Occupation} + \text{Description} + \text{Transcript Lead}$$

### 2.2 14-Dimensional Emotional Resonance Modeling
Every talk is rated by audience members across 14 emotional dimensions:
- *Positive/Stimulating:* Inspiring, Fascinating, Informative, Funny, Ingenious, Persuasive, Courageous, Beautiful, Jaw-dropping.
- *Neutral/Critical:* OK, Confusing, Longwinded, Unconvincing, Obnoxious.

The raw vote counts are mapped into a normalized probability distribution vector:
$$\mathbf{e}_i = \left[ \frac{c_{i,1}}{\sum_k c_{i,k}}, \frac{c_{i,2}}{\sum_k c_{i,k}}, \dots, \frac{c_{i,14}}{\sum_k c_{i,k}} \right]$$

---

## 3. Mathematical Formulations of Recommender Layers

### 3.1 Content-Based Modeling
1. **TF-IDF Vectorization:**
   - Vocabulary: 12,000 unigrams and bigrams.
   - Sublinear term frequency scaling: $w = 1 + \log(\text{tf})$.
   - Document frequency bounds: $\text{min\_df} = 2$, $\text{max\_df} = 0.85$.
   - Stopword removal via English lexicon.

2. **Semantic Latent Semantic Analysis (LSA):**
   - Truncated Singular Value Decomposition reduces the 12,000-dimensional sparse TF-IDF space into $K=100$ dense latent semantic concepts:
     $$X_{\text{TF-IDF}} \approx U_k \Sigma_k V_k^T$$
   - Embeddings are $L_2$-normalized:
     $$\hat{\mathbf{v}}_i = \frac{\mathbf{v}_i}{\|\mathbf{v}_i\|_2}$$

3. **Composite Content Similarity:**
   $$S_{\text{content}}(i, j) = 0.70 \cdot \cos(\mathbf{x}_i^{\text{tfidf}}, \mathbf{x}_j^{\text{tfidf}}) + 0.20 \cdot (\hat{\mathbf{v}}_i \cdot \hat{\mathbf{v}}_j) + 0.10 \cdot (\hat{\mathbf{e}}_i \cdot \hat{\mathbf{e}}_j)$$

### 3.2 Collaborative Filtering (Matrix Factorization via SVD)
Given a sparse User-Item interaction matrix $R \in \mathbb{R}^{M \times N}$ ($M=800$ users, $N=2,628$ talks):
1. Low-rank decomposition with $d=40$ latent factors:
   $$R \approx P \cdot Q^T$$
   Where $P \in \mathbb{R}^{M \times d}$ represents latent user preferences and $Q \in \mathbb{R}^{N \times d}$ represents latent item characteristics.
2. Predicted rating for user $u$ on talk $i$:
   $$\hat{r}_{u,i} = \mathbf{p}_u \cdot \mathbf{q}_i^T$$

### 3.3 Dynamic Hybrid Blending Formula
To combine content similarity with collective viewer behavior:
$$S_{\text{hybrid}}(u, i; \text{seed}) = \alpha \cdot S_{\text{content}}(\text{seed}, i) + (1 - \alpha) \cdot \text{Norm}(\hat{r}_{u,i})$$
Where:
- $\alpha \in [0.0, 1.0]$ is the user-adjustable balance slider (default $\alpha = 0.55$).
- $\text{Norm}(\hat{r}_{u,i}) = \frac{\hat{r}_{u,i} - \min(\hat{R}_u)}{\max(\hat{R}_u) - \min(\hat{R}_u)}$ standardizes predicted ratings to $[0, 1]$.
- Already-watched talks are masked to $-\infty$.

### 3.4 Cold-Start Strategy
- **New User (No history / $< 3$ interactions):** The engine seamlessly falls back to pure content-based similarity or topic-guided browsing, bypassing collaborative factorization.
- **Unseen Talk:** Utilizes TF-IDF, LSA semantic embeddings, and tag overlap for instantaneous retrieval without needing prior viewing history.

---

## 4. Serving & Deployment Architecture
- **Web Interface:** Built on Streamlit, served on port `8501`. Features interactive controls, explanation callouts, EDA charts, and catalog browsing.
- **REST API:** High-throughput FastAPI application running with Uvicorn on port `8000`. Exposes full CRUD and recommendation endpoints with automatic Swagger docs (`/docs`).
