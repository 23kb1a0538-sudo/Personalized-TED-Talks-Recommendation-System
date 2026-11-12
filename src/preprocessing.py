"""
Data Preprocessing Pipeline for TED Talks Recommendation System
Handles data ingestion, cleaning, parsing of JSON/literal structures,
text normalization, feature extraction, and user-item interaction generation.
"""

import os
import ast
import re
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


def clean_text(text: str) -> str:
    """Cleans text by removing special chars, URLs, excess whitespace, and lowercasing."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def parse_literal(val, default):
    """Safely evaluates string literals."""
    if pd.isna(val):
        return default
    try:
        return ast.literal_eval(val)
    except Exception:
        return default


def preprocess_data(
    ted_main_path: str = "/workspace/data/raw/ted_main.csv",
    transcripts_path: str = "/workspace/data/raw/transcripts.csv",
    output_dir: str = "/workspace/data/processed"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads raw TED talks metadata and transcripts, cleans text, extracts features,
    and constructs user-item interaction logs.
    """
    os.makedirs(output_dir, exist_ok=True)
    print("Loading raw datasets...")
    df_main = pd.read_csv(ted_main_path, encoding='latin-1')
    df_trans = pd.read_csv(transcripts_path, encoding='latin-1')

    print(f"Raw main: {df_main.shape}, Raw transcripts: {df_trans.shape}")

    # Remove duplicates
    df_main = df_main.drop_duplicates(subset=['url']).reset_index(drop=True)
    df_trans = df_trans.drop_duplicates(subset=['url']).reset_index(drop=True)

    # Merge metadata with transcripts
    df = pd.merge(df_main, df_trans, on='url', how='left')
    print(f"Merged shape: {df.shape}")

    # Add numeric talk_id (0 to N-1)
    df['talk_id'] = np.arange(len(df))

    # Fill NA text
    df['description'] = df['description'].fillna('')
    df['speaker_occupation'] = df['speaker_occupation'].fillna('Unknown')
    df['transcript'] = df['transcript'].fillna('')
    df['main_speaker'] = df['main_speaker'].fillna('Unknown')
    df['title'] = df['title'].fillna('Untitled')

    # Parse tags
    df['parsed_tags'] = df['tags'].apply(lambda x: parse_literal(x, []))
    df['tag_string'] = df['parsed_tags'].apply(lambda tags: " ".join([t.replace(' ', '_') for t in tags]))

    # Parse related talks
    df['parsed_related'] = df['related_talks'].apply(lambda x: parse_literal(x, []))
    df['related_titles'] = df['parsed_related'].apply(lambda rel: [r.get('title') for r in rel if isinstance(r, dict)])

    # Parse ratings and normalize 14 standard TED emotions
    all_emotions = [
        'Inspiring', 'Fascinating', 'Informative', 'Funny', 'Ingenious',
        'Persuasive', 'Courageous', 'Beautiful', 'Jaw-dropping', 'OK',
        'Confusing', 'Longwinded', 'Unconvincing', 'Obnoxious'
    ]

    def extract_emotions(ratings_str):
        rats = parse_literal(ratings_str, [])
        emotion_dict = {emo: 0 for emo in all_emotions}
        total_votes = 0
        for item in rats:
            if isinstance(item, dict):
                name = item.get('name')
                cnt = item.get('count', 0)
                if name in emotion_dict:
                    emotion_dict[name] = cnt
                    total_votes += cnt
        # Normalize to probability distribution
        if total_votes > 0:
            for k in emotion_dict:
                emotion_dict[k] = round(emotion_dict[k] / total_votes, 4)
        return emotion_dict

    emotions_list = df['ratings'].apply(extract_emotions).tolist()
    emotions_df = pd.DataFrame(emotions_list)
    for col in all_emotions:
        df[f'emo_{col.lower()}'] = emotions_df[col]

    # Primary emotion
    df['top_emotion'] = emotions_df.idxmax(axis=1)

    # Word counts & duration
    df['transcript_word_count'] = df['transcript'].apply(lambda x: len(x.split()))
    df['duration_min'] = round(df['duration'] / 60.0, 1)

    # Clean text components
    df['clean_title'] = df['title'].apply(clean_text)
    df['clean_description'] = df['description'].apply(clean_text)
    df['clean_occupation'] = df['speaker_occupation'].apply(clean_text)

    # Extract first 300 words of transcript for dense feature representation without noise
    def get_transcript_lead(t):
        words = t.split()[:250]
        return clean_text(" ".join(words))

    df['transcript_lead'] = df['transcript'].apply(get_transcript_lead)

    # Create rich metadata "content soup" for TF-IDF and embeddings
    # We weight tags and title heavily by repeating them, then description and transcript lead
    df['content_soup'] = (
        (df['clean_title'] + " ") * 3 +
        (df['tag_string'].apply(lambda s: s.replace('_', ' ')) + " ") * 3 +
        df['clean_occupation'] + " " +
        df['clean_description'] + " " +
        df['transcript_lead']
    )

    # Format human-readable tag display
    df['tags_display'] = df['parsed_tags'].apply(lambda tags: ", ".join(tags[:6]))

    # Clean dates
    df['film_year'] = pd.to_datetime(df['film_date'], unit='s', errors='coerce').dt.year.fillna(2010).astype(int)
    df['published_year'] = pd.to_datetime(df['published_date'], unit='s', errors='coerce').dt.year.fillna(2010).astype(int)

    # Generate synthetic realistic user interactions for Collaborative Filtering
    # To model realistic collaborative signals across 800 users and 2600 talks:
    # Users have affinities to certain clusters of tags (e.g. Technology/AI, Psychology/Health, Business/Economics, Art/Design)
    np.random.seed(42)
    num_users = 800
    user_personas = [
        {"name": "Tech & AI Innovator", "tags": ['technology', 'ai', 'computers', 'future', 'robots', 'science']},
        {"name": "Psychology & Well-being", "tags": ['psychology', 'happiness', 'brain', 'mental_health', 'mind', 'health']},
        {"name": "Business & Strategy", "tags": ['business', 'leadership', 'economics', 'work', 'finance', 'entrepreneur']},
        {"name": "Art & Design Thinker", "tags": ['design', 'art', 'creativity', 'music', 'culture', 'storytelling']},
        {"name": "Science & Universe", "tags": ['science', 'physics', 'space', 'astronomy', 'biology', 'nature', 'oceans']},
        {"name": "Global Reform & Activism", "tags": ['global_issues', 'social_change', 'activism', 'climate_change', 'education', 'humanity']}
    ]

    interactions = []
    # Index talks by primary tag sets
    talk_tags_map = {row['talk_id']: set([t.lower().replace(' ', '_') for t in row['parsed_tags']]) for _, row in df.iterrows()}
    popular_talk_ids = df.sort_values(by='views', ascending=False)['talk_id'].head(100).tolist()

    for user_id in range(1, num_users + 1):
        persona = user_personas[user_id % len(user_personas)]
        target_tags = set(persona['tags'])

        # Candidate talks matching persona
        affinity_talks = [tid for tid, ttags in talk_tags_map.items() if len(ttags.intersection(target_tags)) > 0]
        if len(affinity_talks) < 20:
            affinity_talks = popular_talk_ids

        # User interacts with 15 - 45 talks
        n_watches = np.random.randint(15, 45)
        chosen_affinity = np.random.choice(affinity_talks, size=min(n_watches, len(affinity_talks)), replace=False)
        # Also watches a couple popular talks
        chosen_popular = np.random.choice(popular_talk_ids, size=min(5, len(popular_talk_ids)), replace=False)
        watched_talks = list(set(list(chosen_affinity) + list(chosen_popular)))

        for tid in watched_talks:
            ttags = talk_tags_map.get(tid, set())
            overlap = len(ttags.intersection(target_tags))
            # Base rating centered around 4 with noise; high overlap gives 4-5
            if overlap >= 2:
                rating = np.random.choice([4.0, 4.5, 5.0], p=[0.2, 0.4, 0.4])
                watch_ratio = np.random.uniform(0.85, 1.0)
            elif overlap == 1:
                rating = np.random.choice([3.0, 3.5, 4.0, 4.5], p=[0.2, 0.3, 0.3, 0.2])
                watch_ratio = np.random.uniform(0.6, 0.95)
            else:
                rating = np.random.choice([2.5, 3.0, 3.5, 4.0], p=[0.2, 0.4, 0.3, 0.1])
                watch_ratio = np.random.uniform(0.3, 0.8)

            interactions.append({
                'user_id': user_id,
                'talk_id': tid,
                'rating': rating,
                'watch_ratio': round(watch_ratio, 2),
                'persona': persona['name']
            })

    df_interactions = pd.DataFrame(interactions)
    print(f"Generated {len(df_interactions)} user interactions across {num_users} users.")

    # Save cleaned metadata and interactions
    clean_csv_path = os.path.join(output_dir, "ted_clean.csv")
    interactions_path = os.path.join(output_dir, "user_interactions.csv")
    df.to_csv(clean_csv_path, index=False)
    df_interactions.to_csv(interactions_path, index=False)

    print(f"Saved processed dataset to {clean_csv_path}")
    print(f"Saved interactions to {interactions_path}")

    # Summary statistics
    summary = {
        "total_talks": len(df),
        "total_speakers": int(df['main_speaker'].nunique()),
        "total_interactions": len(df_interactions),
        "total_users": num_users,
        "avg_duration_minutes": round(float(df['duration_min'].mean()), 2),
        "avg_views": round(float(df['views'].mean()), 2),
        "median_views": round(float(df['views'].median()), 2),
        "top_emotions": df['top_emotion'].value_counts().to_dict(),
        "tags_count": int(len(set([t for tags in df['parsed_tags'] for t in tags])))
    }
    with open(os.path.join(output_dir, "dataset_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("Preprocessing completed successfully.")
    return df, df_interactions


if __name__ == "__main__":
    preprocess_data()
