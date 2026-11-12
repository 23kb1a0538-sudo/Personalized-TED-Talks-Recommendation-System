"""
Feature Engineering and Vectorization Pipeline
Converts text, metadata, and user interactions into mathematical representations:
1. TF-IDF vectorization with n-grams
2. Latent Semantic Analysis (TruncatedSVD) dense semantic embeddings
3. Normalized emotional response feature vectors
4. User-Item interaction matrix (sparse CSR representation)
5. Precomputed similarity matrices for low-latency retrieval
"""

import os
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize


def build_features(
    clean_data_path: str = "/workspace/data/processed/ted_clean.csv",
    interactions_path: str = "/workspace/data/processed/user_interactions.csv",
    models_dir: str = "/workspace/models"
):
    os.makedirs(models_dir, exist_ok=True)
    print("Loading preprocessed data for feature engineering...")
    df = pd.read_csv(clean_data_path)
    df_interactions = pd.read_csv(interactions_path)

    # 1. TF-IDF Feature Extraction
    print("Building TF-IDF model...")
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=12000,
        sublinear_tf=True,
        min_df=2,
        max_df=0.85,
        stop_words='english'
    )
    tfidf_matrix = tfidf.fit_transform(df['content_soup'].fillna(''))
    print(f"TF-IDF Matrix shape: {tfidf_matrix.shape}")

    # 2. Semantic Embedding (TruncatedSVD / LSA)
    print("Building Semantic Topic Embeddings (LSA)...")
    n_components = 100
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    lsa_embeddings = svd.fit_transform(tfidf_matrix)
    lsa_embeddings = normalize(lsa_embeddings, norm='l2')
    print(f"LSA Embeddings shape: {lsa_embeddings.shape}, Explained variance ratio sum: {svd.explained_variance_ratio_.sum():.4f}")

    # 3. Emotional Profile Feature Matrix
    print("Extracting Emotional Feature Vectors...")
    emotion_cols = [c for c in df.columns if c.startswith('emo_')]
    emotion_matrix = df[emotion_cols].values
    # L2 normalize so cosine similarity equals dot product
    emotion_matrix_norm = normalize(emotion_matrix, norm='l2')

    # 4. Content Similarity Matrices
    print("Precomputing Cosine Similarity matrices...")
    # TF-IDF Cosine Similarity
    content_cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix).astype(np.float32)

    # Semantic LSA Cosine Similarity
    semantic_sim = np.dot(lsa_embeddings, lsa_embeddings.T).astype(np.float32)

    # Emotion Cosine Similarity
    emotion_sim = np.dot(emotion_matrix_norm, emotion_matrix_norm.T).astype(np.float32)

    # Composite Content Similarity (70% TF-IDF + 20% Semantic LSA + 10% Emotional tone)
    composite_content_sim = (
        0.70 * content_cosine_sim +
        0.20 * semantic_sim +
        0.10 * emotion_sim
    )

    # 5. User-Item Interaction Matrix for Collaborative Filtering
    print("Constructing User-Item Interaction Matrix...")
    num_users = df_interactions['user_id'].max()
    num_talks = len(df)

    # Map user_id to 0-indexed rows
    user_ids = df_interactions['user_id'].values - 1
    talk_ids = df_interactions['talk_id'].values
    ratings = df_interactions['rating'].values

    interaction_sparse = csr_matrix((ratings, (user_ids, talk_ids)), shape=(num_users, num_talks), dtype=np.float32)
    print(f"User-Item Matrix shape: {interaction_sparse.shape}, Non-zero entries: {interaction_sparse.nnz}")

    # 6. Save all artifacts for fast serving in deployment
    print("Saving feature matrices and models to disk...")
    joblib.dump(tfidf, os.path.join(models_dir, "tfidf_vectorizer.joblib"))
    joblib.dump(svd, os.path.join(models_dir, "svd_model.joblib"))
    np.save(os.path.join(models_dir, "lsa_embeddings.npy"), lsa_embeddings)
    np.save(os.path.join(models_dir, "emotion_matrix.npy"), emotion_matrix)
    np.save(os.path.join(models_dir, "content_cosine_sim.npy"), content_cosine_sim)
    np.save(os.path.join(models_dir, "composite_content_sim.npy"), composite_content_sim)
    save_npz(os.path.join(models_dir, "interaction_matrix.npz"), interaction_sparse)

    print("Feature engineering completed successfully!")
    return {
        "tfidf": tfidf,
        "svd": svd,
        "tfidf_matrix": tfidf_matrix,
        "lsa_embeddings": lsa_embeddings,
        "content_cosine_sim": content_cosine_sim,
        "composite_content_sim": composite_content_sim,
        "interaction_matrix": interaction_sparse
    }


if __name__ == "__main__":
    build_features()
