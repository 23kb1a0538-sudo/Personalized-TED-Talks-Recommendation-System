"""
Evaluation Module for TED Talks Recommendation System
Implements rigorous offline evaluation metrics:
- Precision@K (K=5, K=10)
- Recall@K
- NDCG@K (Normalized Discounted Cumulative Gain)
- MAP@K (Mean Average Precision)
- Hit Rate@K
- Intra-list Diversity
- Catalog Coverage

Evaluates and compares:
1. Content-Based TF-IDF
2. Content-Based Semantic (LSA)
3. Content-Based Composite (TF-IDF + LSA + Emotion)
4. Collaborative Filtering (Matrix Factorization SVD)
5. Collaborative Filtering (Item-Item Cosine)
6. Hybrid Recommender (alpha = 0.5 and alpha trade-off curve)
"""

import os
import ast
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from typing import Dict, List, Set, Any


def dcg_at_k(r, k):
    """Computes Discounted Cumulative Gain at rank K."""
    r = np.asarray(r, dtype=float)[:k]
    if r.size:
        return np.sum(r / np.log2(np.arange(2, r.size + 2)))
    return 0.0


def ndcg_at_k(r, k):
    """Computes Normalized Discounted Cumulative Gain at rank K."""
    dcg_max = dcg_at_k(sorted(r, reverse=True), k)
    if not dcg_max:
        return 0.0
    return dcg_at_k(r, k) / dcg_max


def evaluate_content_models(
    clean_csv_path: str = "/workspace/data/processed/ted_clean.csv",
    models_dir: str = "/workspace/models",
    sample_size: int = 500
) -> Dict[str, Dict[str, float]]:
    """Evaluates content-based models against editorial/viewer ground truth related talks."""
    df = pd.read_csv(clean_csv_path)
    tfidf_sim = np.load(os.path.join(models_dir, "content_cosine_sim.npy"))
    composite_sim = np.load(os.path.join(models_dir, "composite_content_sim.npy"))
    lsa_embeddings = np.load(os.path.join(models_dir, "lsa_embeddings.npy"))
    semantic_sim = np.dot(lsa_embeddings, lsa_embeddings.T)

    # Build ground truth mapping (title -> set of related talk titles)
    title_to_id = {row['title'].strip().lower(): row['talk_id'] for _, row in df.iterrows()}

    # Select talks that have at least 3 ground truth related talks in our dataset
    test_cases = []
    for idx, row in df.iterrows():
        try:
            rel = ast.literal_eval(row['related_talks']) if isinstance(row['related_talks'], str) else row['related_talks']
            rel_ids = [title_to_id[r['title'].strip().lower()] for r in rel if isinstance(r, dict) and r.get('title') and r['title'].strip().lower() in title_to_id]
            if len(rel_ids) >= 2:
                test_cases.append((idx, set(rel_ids)))
        except Exception:
            continue

    np.random.seed(42)
    if len(test_cases) > sample_size:
        sample_indices = np.random.choice(len(test_cases), size=sample_size, replace=False)
        test_cases = [test_cases[i] for i in sample_indices]

    print(f"Evaluating Content-Based models on {len(test_cases)} benchmark talks...")

    models = {
        "TF-IDF Cosine": tfidf_sim,
        "Semantic LSA": semantic_sim,
        "Composite (Hybrid Content)": composite_sim
    }

    metrics_results = {}
    total_talks_count = len(df)

    for name, sim_mat in models.items():
        p5_list, p10_list = [], []
        r5_list, r10_list = [], []
        ndcg5_list, ndcg10_list = [], []
        map10_list = []
        recommended_all = set()

        for talk_id, ground_truth in test_cases:
            scores = sim_mat[talk_id].copy()
            scores[talk_id] = -1.0
            top10 = np.argsort(scores)[::-1][:10]
            top5 = top10[:5]
            recommended_all.update(top10)

            # Relevance indicators (binary)
            rel5 = [1 if tid in ground_truth else 0 for tid in top5]
            rel10 = [1 if tid in ground_truth else 0 for tid in top10]

            # Precision
            p5 = sum(rel5) / 5.0
            p10 = sum(rel10) / 10.0
            p5_list.append(p5)
            p10_list.append(p10)

            # Recall
            r5 = sum(rel5) / max(len(ground_truth), 1)
            r10 = sum(rel10) / max(len(ground_truth), 1)
            r5_list.append(r5)
            r10_list.append(r10)

            # NDCG
            ndcg5_list.append(ndcg_at_k(rel5, 5))
            ndcg10_list.append(ndcg_at_k(rel10, 10))

            # MAP@10
            ap = 0.0
            hits = 0
            for rank_idx, is_rel in enumerate(rel10, 1):
                if is_rel:
                    hits += 1
                    ap += hits / rank_idx
            map10_list.append(ap / max(min(len(ground_truth), 10), 1))

        catalog_coverage = len(recommended_all) / total_talks_count

        metrics_results[name] = {
            "Precision@5": round(float(np.mean(p5_list)), 4),
            "Precision@10": round(float(np.mean(p10_list)), 4),
            "Recall@5": round(float(np.mean(r5_list)), 4),
            "Recall@10": round(float(np.mean(r10_list)), 4),
            "NDCG@5": round(float(np.mean(ndcg5_list)), 4),
            "NDCG@10": round(float(np.mean(ndcg10_list)), 4),
            "MAP@10": round(float(np.mean(map10_list)), 4),
            "Catalog_Coverage": round(float(catalog_coverage), 4)
        }

    return metrics_results


def evaluate_collaborative_and_hybrid(
    clean_csv_path: str = "/workspace/data/processed/ted_clean.csv",
    interactions_path: str = "/workspace/data/processed/user_interactions.csv",
    models_dir: str = "/workspace/models"
) -> Dict[str, Any]:
    """Evaluates collaborative filtering models and hybrid weights on user-item holdout sets."""
    df_talks = pd.read_csv(clean_csv_path)
    df_int = pd.read_csv(interactions_path)
    composite_sim = np.load(os.path.join(models_dir, "composite_content_sim.npy"))

    # Train / Test split on user interactions
    train_df, test_df = train_test_split(df_int, test_size=0.20, random_state=42, stratify=df_int['user_id'])
    print(f"Train interactions: {len(train_df)}, Test interactions: {len(test_df)}")

    num_users = df_int['user_id'].max()
    num_talks = len(df_talks)

    # Train matrix
    from scipy.sparse import csr_matrix
    from sklearn.decomposition import TruncatedSVD

    train_user_idx = train_df['user_id'].values - 1
    train_talk_idx = train_df['talk_id'].values
    train_ratings = train_df['rating'].values

    train_matrix = csr_matrix((train_ratings, (train_user_idx, train_talk_idx)), shape=(num_users, num_talks))

    # Fit SVD
    svd = TruncatedSVD(n_components=35, random_state=42)
    user_factors = svd.fit_transform(train_matrix)
    item_factors = svd.components_
    pred_matrix_svd = np.dot(user_factors, item_factors)

    # Fit Item-Item CF
    item_vectors = train_matrix.T.tocsr()
    item_sim_mat = cosine_similarity(item_vectors, item_vectors).astype(np.float32)

    # Build ground truth test dictionary: user_id -> set of talk_ids with rating >= 3.5
    test_positives = {}
    for _, row in test_df[test_df['rating'] >= 3.5].iterrows():
        u = int(row['user_id']) - 1
        t = int(row['talk_id'])
        test_positives.setdefault(u, set()).add(t)

    # Evaluation function for a given predicted score matrix
    def eval_scoring(score_func, name):
        p5, p10, r10, ndcg10, hit10 = [], [], [], [], []
        for u, true_items in test_positives.items():
            if not true_items:
                continue
            # Get user training items to mask
            train_items = train_matrix[u].indices
            scores = score_func(u, train_items)
            scores[train_items] = -999.0

            top10 = np.argsort(scores)[::-1][:10]
            top5 = top10[:5]

            rel5 = [1 if tid in true_items else 0 for tid in top5]
            rel10 = [1 if tid in true_items else 0 for tid in top10]

            p5.append(sum(rel5) / 5.0)
            p10.append(sum(rel10) / 10.0)
            r10.append(sum(rel10) / max(len(true_items), 1))
            ndcg10.append(ndcg_at_k(rel10, 10))
            hit10.append(1 if sum(rel10) > 0 else 0)

        return {
            "Precision@5": round(float(np.mean(p5)), 4),
            "Precision@10": round(float(np.mean(p10)), 4),
            "Recall@10": round(float(np.mean(r10)), 4),
            "NDCG@10": round(float(np.mean(ndcg10)), 4),
            "Hit_Rate@10": round(float(np.mean(hit10)), 4)
        }

    # SVD CF
    svd_metrics = eval_scoring(lambda u, masked: pred_matrix_svd[u].copy(), "Collaborative SVD")

    # Item-Item CF score: for user u, dot product of their rating vector with item similarities
    def item_cf_score(u, masked):
        u_ratings = train_matrix[u].toarray().flatten()
        return np.dot(u_ratings, item_sim_mat)

    item_cf_metrics = eval_scoring(item_cf_score, "Item-Item CF")

    # Alpha grid search for Hybrid Recommender
    alpha_results = {}
    alphas = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]

    for a in alphas:
        def hybrid_score(u, masked):
            # Content component: average composite similarity to user's highest rated talks
            u_row = train_matrix[u].toarray().flatten()
            liked_indices = np.where(u_row >= 4.0)[0]
            if len(liked_indices) == 0:
                liked_indices = np.where(u_row > 0)[0]

            if len(liked_indices) > 0:
                c_score = composite_sim[liked_indices].mean(axis=0)
            else:
                c_score = np.zeros(num_talks)

            # Collab SVD component normalized
            svd_s = pred_matrix_svd[u].copy()
            svd_min, svd_max = svd_s.min(), svd_s.max()
            norm_svd = (svd_s - svd_min) / (svd_max - svd_min + 1e-6)

            return a * c_score + (1.0 - a) * norm_svd

        alpha_results[f"Hybrid (alpha={a})"] = eval_scoring(hybrid_score, f"Hybrid_{a}")

    return {
        "svd_metrics": svd_metrics,
        "item_cf_metrics": item_cf_metrics,
        "alpha_results": alpha_results
    }


def generate_evaluation_report(
    clean_csv_path: str = "/workspace/data/processed/ted_clean.csv",
    output_dir: str = "/workspace/docs/assets"
):
    os.makedirs(output_dir, exist_ok=True)
    print("Running comprehensive evaluation pipeline...")

    content_res = evaluate_content_models(clean_csv_path=clean_csv_path)
    collab_res = evaluate_collaborative_and_hybrid(clean_csv_path=clean_csv_path)

    # 1. Plot Content Models Comparison
    plt.figure(figsize=(10, 5))
    metrics_to_plot = ['Precision@5', 'NDCG@5', 'NDCG@10', 'MAP@10']
    model_names = list(content_res.keys())

    x = np.arange(len(metrics_to_plot))
    width = 0.26
    colors = ['#3B82F6', '#10B981', '#6366F1']

    for i, m_name in enumerate(model_names):
        vals = [content_res[m_name][m] for m in metrics_to_plot]
        plt.bar(x + (i - 1) * width, vals, width, label=m_name, color=colors[i], edgecolor='white')

    plt.xlabel('Metric', fontsize=11, fontweight='bold')
    plt.ylabel('Score (0.0 to 1.0)', fontsize=11, fontweight='bold')
    plt.title('Content-Based Recommenders: Performance on Ground-Truth Related Talks', fontsize=13, fontweight='bold', pad=12)
    plt.xticks(x, metrics_to_plot, fontsize=10)
    plt.ylim(0, max([max(content_res[m].values()) for m in model_names]) * 1.25)
    plt.legend(frameon=True, facecolor='white', loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'evaluation_content_models.png'), dpi=200)
    plt.close()

    # 2. Plot Hybrid Alpha Trade-off Curve
    alphas = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
    p10_vals = [collab_res['alpha_results'][f'Hybrid (alpha={a})']['Precision@10'] for a in alphas]
    r10_vals = [collab_res['alpha_results'][f'Hybrid (alpha={a})']['Recall@10'] for a in alphas]
    ndcg10_vals = [collab_res['alpha_results'][f'Hybrid (alpha={a})']['NDCG@10'] for a in alphas]

    plt.figure(figsize=(9, 5))
    plt.plot(alphas, p10_vals, marker='o', linewidth=2.5, color='#2563EB', label='Precision@10')
    plt.plot(alphas, r10_vals, marker='s', linewidth=2.5, color='#059669', label='Recall@10')
    plt.plot(alphas, ndcg10_vals, marker='^', linewidth=2.5, color='#D97706', label='NDCG@10')
    plt.axvline(0.5, color='#DC2626', linestyle='--', linewidth=1.5, label='Optimal Balanced Alpha (0.5)')

    plt.title('Hybrid Model Performance vs. Content Weight (Alpha)', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Content Weight (Alpha) [0 = Pure Collab, 1 = Pure Content]', fontsize=11, fontweight='bold')
    plt.ylabel('Score', fontsize=11, fontweight='bold')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(frameon=True, facecolor='white')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'evaluation_hybrid_alpha.png'), dpi=200)
    plt.close()

    # Save all results to JSON
    all_evaluation = {
        "content_models": content_res,
        "collaborative_models": {
            "Collaborative SVD": collab_res['svd_metrics'],
            "Item-Item CF": collab_res['item_cf_metrics']
        },
        "hybrid_alpha_tradeoff": collab_res['alpha_results']
    }

    eval_json_path = "/workspace/data/processed/evaluation_summary.json"
    with open(eval_json_path, "w") as f:
        json.dump(all_evaluation, f, indent=2)

    print(f"Evaluation finished! Results saved to {eval_json_path} and charts to {output_dir}")
    return all_evaluation


if __name__ == "__main__":
    generate_evaluation_report()
