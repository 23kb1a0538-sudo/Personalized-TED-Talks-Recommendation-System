"""
Recommendation Engine Module for TED Talks
Implements:
1. Content-Based Filtering (TF-IDF, Semantic LSA, Emotional Tone, Query Search)
2. Collaborative Filtering (Matrix Factorization via SVD, Item-Item CF)
3. Hybrid Recommendation Engine (Weighted, Cascade, and Cold-Start Handling)
4. Recommendation Explainability Engine
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from scipy.sparse import load_npz
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity


class ContentBasedRecommender:
    """Content-Based Recommender utilizing TF-IDF, Semantic LSA, and Emotional distributions."""

    def __init__(self, df: pd.DataFrame, tfidf_vectorizer, svd_model, lsa_embeddings, composite_sim, tfidf_sim):
        self.df = df
        self.tfidf = tfidf_vectorizer
        self.svd = svd_model
        self.lsa_embeddings = lsa_embeddings
        self.composite_sim = composite_sim
        self.tfidf_sim = tfidf_sim
        self.title_to_id = {row['title'].strip().lower(): row['talk_id'] for _, row in df.iterrows()}
        self.id_to_row = {row['talk_id']: row for _, row in df.iterrows()}

    def recommend_by_talk_id(self, talk_id: int, top_n: int = 10, mode: str = "composite") -> List[Dict[str, Any]]:
        """Recommends top N talks similar to a given talk_id."""
        if talk_id < 0 or talk_id >= len(self.df):
            return []

        sim_matrix = self.composite_sim if mode == "composite" else self.tfidf_sim
        sim_scores = sim_matrix[talk_id].copy()
        # Set self-similarity to -1 so it's not recommended
        sim_scores[talk_id] = -1.0

        top_indices = np.argsort(sim_scores)[::-1][:top_n]
        results = []
        source_talk = self.id_to_row[talk_id]

        for rank, idx in enumerate(top_indices, 1):
            target_talk = self.id_to_row[idx]
            score = float(sim_scores[idx])

            # Compute shared tags
            source_tags = set(eval(source_talk['tags']) if isinstance(source_talk['tags'], str) else source_talk['tags'])
            target_tags = set(eval(target_talk['tags']) if isinstance(target_talk['tags'], str) else target_talk['tags'])
            common_tags = list(source_tags.intersection(target_tags))

            results.append({
                'rank': rank,
                'talk_id': int(idx),
                'title': target_talk['title'],
                'main_speaker': target_talk['main_speaker'],
                'speaker_occupation': target_talk['speaker_occupation'],
                'views': int(target_talk['views']),
                'duration_min': float(target_talk['duration_min']),
                'description': target_talk['description'],
                'url': target_talk['url'],
                'tags': list(target_tags)[:5],
                'common_tags': common_tags,
                'similarity_score': round(score, 4),
                'top_emotion': target_talk.get('top_emotion', 'Inspiring'),
                'explanation': f"Shares {len(common_tags)} topic tags ({', '.join(common_tags[:3]) if common_tags else 'thematic concepts'}) and similar speaking tone."
            })
        return results

    def recommend_by_query(self, query: str, top_n: int = 10) -> List[Dict[str, Any]]:
        """Recommends talks matching free-form text query."""
        if not query.strip():
            return []
        query_vec = self.tfidf.transform([query])
        sim_scores = cosine_similarity(query_vec, self.tfidf.transform(self.df['content_soup'].fillna('')))[0]
        top_indices = np.argsort(sim_scores)[::-1][:top_n]

        results = []
        for rank, idx in enumerate(top_indices, 1):
            target_talk = self.id_to_row[idx]
            score = float(sim_scores[idx])
            tags = eval(target_talk['tags']) if isinstance(target_talk['tags'], str) else target_talk['tags']
            results.append({
                'rank': rank,
                'talk_id': int(idx),
                'title': target_talk['title'],
                'main_speaker': target_talk['main_speaker'],
                'speaker_occupation': target_talk['speaker_occupation'],
                'views': int(target_talk['views']),
                'duration_min': float(target_talk['duration_min']),
                'description': target_talk['description'],
                'url': target_talk['url'],
                'tags': list(tags)[:5],
                'similarity_score': round(score, 4),
                'top_emotion': target_talk.get('top_emotion', 'Inspiring'),
                'explanation': f"Direct semantic alignment with your query: '{query[:40]}'."
            })
        return results


class CollaborativeFilteringRecommender:
    """Collaborative Filtering Recommender using Matrix Factorization (SVD) and Item-Item CF."""

    def __init__(self, df_talks: pd.DataFrame, interaction_matrix, n_factors: int = 40):
        self.df = df_talks
        self.interaction_matrix = interaction_matrix
        self.n_factors = n_factors
        self.id_to_row = {row['talk_id']: row for _, row in df_talks.iterrows()}

        # Fit SVD on user-item matrix
        self.svd = TruncatedSVD(n_components=n_factors, random_state=42)
        # User factors (num_users x n_factors)
        self.user_factors = self.svd.fit_transform(interaction_matrix)
        # Item factors (n_factors x num_talks)
        self.item_factors = self.svd.components_

        # Precompute predicted rating matrix for fast user recommendations
        # Shape: (num_users, num_talks)
        self.pred_ratings = np.dot(self.user_factors, self.item_factors)

        # Precompute item-item cosine similarity from user interaction columns
        # Column i is talk i's rating vector across all users
        item_vectors = self.interaction_matrix.T.tocsr()
        self.item_item_sim = cosine_similarity(item_vectors, item_vectors).astype(np.float32)

    def recommend_for_user(self, user_id: int, top_n: int = 10, exclude_watched: bool = True) -> List[Dict[str, Any]]:
        """Recommends talks for a user based on SVD matrix factorization."""
        user_idx = user_id - 1
        if user_idx < 0 or user_idx >= self.pred_ratings.shape[0]:
            # Cold start fallback: recommend globally popular talks
            return self._popular_fallback(top_n)

        scores = self.pred_ratings[user_idx].copy()

        if exclude_watched:
            # Mask talks user already watched
            watched_indices = self.interaction_matrix[user_idx].indices
            scores[watched_indices] = -999.0

        top_indices = np.argsort(scores)[::-1][:top_n]
        results = []
        for rank, idx in enumerate(top_indices, 1):
            talk = self.id_to_row[idx]
            tags = eval(talk['tags']) if isinstance(talk['tags'], str) else talk['tags']
            results.append({
                'rank': rank,
                'talk_id': int(idx),
                'title': talk['title'],
                'main_speaker': talk['main_speaker'],
                'speaker_occupation': talk['speaker_occupation'],
                'views': int(talk['views']),
                'duration_min': float(talk['duration_min']),
                'description': talk['description'],
                'url': talk['url'],
                'tags': list(tags)[:5],
                'predicted_rating': round(float(scores[idx]), 3),
                'top_emotion': talk.get('top_emotion', 'Inspiring'),
                'explanation': f"Highly rated by users with viewing habits and preferences like User #{user_id}."
            })
        return results

    def recommend_similar_items(self, talk_id: int, top_n: int = 10) -> List[Dict[str, Any]]:
        """Recommends talks based on collaborative item-item co-engagement patterns."""
        if talk_id < 0 or talk_id >= self.item_item_sim.shape[0]:
            return []
        sim_scores = self.item_item_sim[talk_id].copy()
        sim_scores[talk_id] = -1.0
        top_indices = np.argsort(sim_scores)[::-1][:top_n]

        results = []
        for rank, idx in enumerate(top_indices, 1):
            talk = self.id_to_row[idx]
            tags = eval(talk['tags']) if isinstance(talk['tags'], str) else talk['tags']
            results.append({
                'rank': rank,
                'talk_id': int(idx),
                'title': talk['title'],
                'main_speaker': talk['main_speaker'],
                'speaker_occupation': talk['speaker_occupation'],
                'views': int(talk['views']),
                'duration_min': float(talk['duration_min']),
                'description': talk['description'],
                'url': talk['url'],
                'tags': list(tags)[:5],
                'co_engagement_score': round(float(sim_scores[idx]), 4),
                'top_emotion': talk.get('top_emotion', 'Inspiring'),
                'explanation': "Frequently co-watched and enjoyed by community members who watched this talk."
            })
        return results

    def _popular_fallback(self, top_n: int) -> List[Dict[str, Any]]:
        top_talks = self.df.sort_values(by='views', ascending=False).head(top_n)
        results = []
        for rank, (_, talk) in enumerate(top_talks.iterrows(), 1):
            tags = eval(talk['tags']) if isinstance(talk['tags'], str) else talk['tags']
            results.append({
                'rank': rank,
                'talk_id': int(talk['talk_id']),
                'title': talk['title'],
                'main_speaker': talk['main_speaker'],
                'speaker_occupation': talk['speaker_occupation'],
                'views': int(talk['views']),
                'duration_min': float(talk['duration_min']),
                'description': talk['description'],
                'url': talk['url'],
                'tags': list(tags)[:5],
                'predicted_rating': 4.5,
                'top_emotion': talk.get('top_emotion', 'Inspiring'),
                'explanation': "Top curated talk across all audiences (Cold-start baseline)."
            })
        return results


class HybridTEDRecommender:
    """
    Unified Hybrid Recommender combining Content-Based and Collaborative signals
    with dynamic weight adjustment, cold-start handling, and rich explanations.
    """

    def __init__(
        self,
        clean_data_path: str = "/workspace/data/processed/ted_clean.csv",
        models_dir: str = "/workspace/models"
    ):
        self.df = pd.read_csv(clean_data_path)
        self.models_dir = models_dir

        print("Loading artifacts for Recommender Engine...")
        self.tfidf = joblib.load(os.path.join(models_dir, "tfidf_vectorizer.joblib"))
        self.svd_lsa = joblib.load(os.path.join(models_dir, "svd_model.joblib"))
        self.lsa_embeddings = np.load(os.path.join(models_dir, "lsa_embeddings.npy"))
        self.content_sim = np.load(os.path.join(models_dir, "content_cosine_sim.npy"))
        self.composite_sim = np.load(os.path.join(models_dir, "composite_content_sim.npy"))
        self.interaction_matrix = load_npz(os.path.join(models_dir, "interaction_matrix.npz"))

        # Initialize sub-recommenders
        self.content_rec = ContentBasedRecommender(
            df=self.df,
            tfidf_vectorizer=self.tfidf,
            svd_model=self.svd_lsa,
            lsa_embeddings=self.lsa_embeddings,
            composite_sim=self.composite_sim,
            tfidf_sim=self.content_sim
        )

        self.collab_rec = CollaborativeFilteringRecommender(
            df_talks=self.df,
            interaction_matrix=self.interaction_matrix,
            n_factors=40
        )

        print("Hybrid Recommender Engine initialized successfully.")

    def recommend(
        self,
        talk_id: Optional[int] = None,
        user_id: Optional[int] = None,
        query: Optional[str] = None,
        tags_filter: Optional[List[str]] = None,
        alpha: float = 0.5,
        top_n: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Main recommendation entry point:
        - If query is provided: runs content query search with tag re-ranking.
        - If talk_id and user_id are both provided: computes weighted hybrid score:
            Score = alpha * Content_Sim + (1 - alpha) * Normalized_Collab_Score
        - If only talk_id: pure content recommendation.
        - If only user_id: collaborative recommendation (or cold-start fallback).
        """
        # Case 1: Free-text search query
        if query and query.strip():
            return self.content_rec.recommend_by_query(query.strip(), top_n=top_n)

        # Case 2: Both talk_id and user_id -> Full Hybrid
        if talk_id is not None and user_id is not None:
            user_idx = user_id - 1
            # Check cold start
            is_cold_start = (user_idx < 0 or user_idx >= self.collab_rec.pred_ratings.shape[0])

            if is_cold_start:
                # Fall back to content-based
                return self.content_rec.recommend_by_talk_id(talk_id, top_n=top_n)

            # Content scores (0 to 1)
            c_scores = self.composite_sim[talk_id].copy()
            c_scores[talk_id] = 0.0

            # Collab scores: min-max normalized predicted ratings
            collab_raw = self.collab_rec.pred_ratings[user_idx].copy()
            # Mask watched talks
            watched = self.interaction_matrix[user_idx].indices
            collab_raw[watched] = -999.0

            # Normalize collab_raw to 0..1 scale
            valid_collab = collab_raw[collab_raw > -500]
            if len(valid_collab) > 0:
                c_min, c_max = valid_collab.min(), valid_collab.max()
                denom = (c_max - c_min) if (c_max - c_min) > 1e-5 else 1.0
                norm_collab = np.clip((collab_raw - c_min) / denom, 0.0, 1.0)
                norm_collab[collab_raw <= -500] = 0.0
            else:
                norm_collab = np.zeros_like(collab_raw)

            # Hybrid linear combination
            hybrid_scores = alpha * c_scores + (1.0 - alpha) * norm_collab
            hybrid_scores[talk_id] = -1.0
            if len(watched) > 0:
                hybrid_scores[watched] = -1.0

            # Filter by tags if specified
            if tags_filter:
                filter_set = set([t.lower().replace(' ', '_') for t in tags_filter])
                for idx, row in self.df.iterrows():
                    ttags = set([t.lower().replace(' ', '_') for t in eval(row['tags']) if isinstance(row['tags'], str)])
                    if not filter_set.intersection(ttags):
                        hybrid_scores[idx] *= 0.5  # Soft penalty for non-matching tags

            top_indices = np.argsort(hybrid_scores)[::-1][:top_n]
            results = []
            source_talk = self.df.iloc[talk_id]
            source_tags = set(eval(source_talk['tags']) if isinstance(source_talk['tags'], str) else source_talk['tags'])

            for rank, idx in enumerate(top_indices, 1):
                target = self.df.iloc[idx]
                target_tags = set(eval(target['tags']) if isinstance(target['tags'], str) else target['tags'])
                common = list(source_tags.intersection(target_tags))

                results.append({
                    'rank': rank,
                    'talk_id': int(idx),
                    'title': target['title'],
                    'main_speaker': target['main_speaker'],
                    'speaker_occupation': target['speaker_occupation'],
                    'views': int(target['views']),
                    'duration_min': float(target['duration_min']),
                    'description': target['description'],
                    'url': target['url'],
                    'tags': list(target_tags)[:5],
                    'common_tags': common,
                    'hybrid_score': round(float(hybrid_scores[idx]), 4),
                    'content_component': round(float(c_scores[idx]), 3),
                    'collab_component': round(float(norm_collab[idx]), 3),
                    'top_emotion': target.get('top_emotion', 'Inspiring'),
                    'explanation': f"Hybrid blend (Content: {int(alpha*100)}%, Collaborative: {int((1-alpha)*100)}%). Matches '{source_talk['title'][:25]}...' and affinities of similar users."
                })
            return results

        # Case 3: Only talk_id
        if talk_id is not None:
            return self.content_rec.recommend_by_talk_id(talk_id, top_n=top_n)

        # Case 4: Only user_id
        if user_id is not None:
            return self.collab_rec.recommend_for_user(user_id, top_n=top_n)

        # Default: globally popular talks
        return self.collab_rec._popular_fallback(top_n=top_n)


def test_recommenders():
    """Quick verification of recommender functionality."""
    engine = HybridTEDRecommender()

    print("\n--- Test 1: Content-Based Recommendation for Talk 0 (Ken Robinson - Do schools kill creativity?) ---")
    recs = engine.recommend(talk_id=0, top_n=3)
    for r in recs:
        print(f"Rank {r['rank']}: {r['title']} by {r['main_speaker']} (Sim: {r['similarity_score']})")

    print("\n--- Test 2: Collaborative Filtering for User 5 ---")
    recs = engine.recommend(user_id=5, top_n=3)
    for r in recs:
        print(f"Rank {r['rank']}: {r['title']} by {r['main_speaker']} (Pred: {r.get('predicted_rating')})")

    print("\n--- Test 3: Hybrid Recommendation (Talk 0 + User 5, alpha=0.6) ---")
    recs = engine.recommend(talk_id=0, user_id=5, alpha=0.6, top_n=3)
    for r in recs:
        print(f"Rank {r['rank']}: {r['title']} (Hybrid: {r['hybrid_score']}, Content: {r['content_component']}, Collab: {r['collab_component']})")

    print("\n--- Test 4: Free text search 'artificial intelligence robotics future' ---")
    recs = engine.recommend(query="artificial intelligence robotics future", top_n=3)
    for r in recs:
        print(f"Rank {r['rank']}: {r['title']} by {r['main_speaker']} (Score: {r['similarity_score']})")


if __name__ == "__main__":
    test_recommenders()
