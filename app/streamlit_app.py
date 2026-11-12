"""
TED Talks Personalized Recommendation System - Interactive Web Application
Built with Streamlit and Scikit-Learn
Supports Hybrid, Content-Based, Collaborative, and Semantic Query Recommendations.
"""

import os
import ast
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import sys

# Ensure src/ is in path
sys.path.append("/workspace")
from src.models import HybridTEDRecommender

# Set page config
st.set_page_config(
    page_title="TED Talks AI Recommender",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (TED Red Accent #E62B1E, Solid Slate, High Contrast, NO Purple Gradients)
st.markdown("""
<style>
    /* Global accents */
    :root {
        --ted-red: #E62B1E;
        --ted-dark: #111827;
        --ted-slate: #1E293B;
        --card-bg: #FFFFFF;
        --border-color: #E2E8F0;
    }
    
    .main-header {
        background-color: #111827;
        color: #FFFFFF;
        padding: 24px 32px;
        border-radius: 12px;
        margin-bottom: 24px;
        border-left: 8px solid #E62B1E;
    }
    
    .main-header h1 {
        color: #FFFFFF !important;
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
    }
    
    .main-header p {
        color: #9CA3AF !important;
        font-size: 1.05rem;
        margin-top: 6px;
        margin-bottom: 0;
    }
    
    .recommendation-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 18px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    
    .recommendation-card:hover {
        border-color: #CBD5E1;
        box-shadow: 0 6px 12px rgba(0,0,0,0.08);
    }
    
    .card-rank {
        display: inline-block;
        background-color: #E62B1E;
        color: #FFFFFF;
        font-weight: 700;
        font-size: 0.85rem;
        padding: 3px 10px;
        border-radius: 20px;
        margin-right: 8px;
    }
    
    .score-badge {
        display: inline-block;
        background-color: #ECFDF5;
        color: #065F46;
        font-weight: 700;
        font-size: 0.85rem;
        padding: 3px 10px;
        border-radius: 20px;
        border: 1px solid #A7F3D0;
    }

    .emotion-badge {
        display: inline-block;
        background-color: #EFF6FF;
        color: #1E40AF;
        font-weight: 600;
        font-size: 0.85rem;
        padding: 3px 10px;
        border-radius: 20px;
        border: 1px solid #BFDBFE;
    }
    
    .tag-pill {
        display: inline-block;
        background-color: #F1F5F9;
        color: #334155;
        font-size: 0.8rem;
        padding: 2px 8px;
        border-radius: 4px;
        margin-right: 6px;
        margin-bottom: 4px;
        font-weight: 500;
    }
    
    .explanation-box {
        background-color: #F8FAFC;
        border-left: 4px solid #0284C7;
        padding: 10px 14px;
        border-radius: 0 6px 6px 0;
        font-size: 0.9rem;
        color: #0F172A;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_recommender():
    """Initializes and caches the Recommender Engine."""
    return HybridTEDRecommender()


@st.cache_data
def load_data():
    """Loads and caches preprocessed dataset and evaluation summaries."""
    df = pd.read_csv("/workspace/data/processed/ted_clean.csv")
    with open("/workspace/data/processed/dataset_summary.json") as f:
        summary = json.load(f)
    with open("/workspace/data/processed/evaluation_summary.json") as f:
        eval_data = json.load(f)
    return df, summary, eval_data


# Load engine and data
engine = load_recommender()
df_talks, dataset_summary, eval_summary = load_data()

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/a/aa/TED_three_letter_logo.svg/320px-TED_three_letter_logo.svg.png", width=140)
    st.markdown("### **Recommendation Controls**")
    
    rec_mode = st.radio(
        "Select Recommendation Strategy:",
        [
            "🎯 Hybrid (Content + Collaborative)",
            "📖 Content-Based (Seed Talk)",
            "👤 Collaborative (User Persona)",
            "🔍 Semantic Query Search"
        ],
        index=0
    )
    
    st.markdown("---")
    top_n = st.slider("Number of Recommendations (Top-N):", min_value=3, max_value=20, value=6, step=1)
    
    if "Hybrid" in rec_mode:
        alpha = st.slider(
            "Hybrid Weight (α - Content vs Collaborative):",
            min_value=0.0,
            max_value=1.0,
            value=0.55,
            step=0.05,
            help="1.0 = Pure Content-Based (text similarity); 0.0 = Pure Collaborative (user ratings); 0.55 = Balanced Hybrid"
        )
        st.caption(f"**Content weight:** {int(alpha*100)}% | **Collab weight:** {int((1-alpha)*100)}%")
    else:
        alpha = 0.5
        
    st.markdown("---")
    st.markdown("### **Filter by Topic**")
    popular_tags = [
        "technology", "science", "design", "business", "culture",
        "psychology", "education", "global_issues", "health",
        "entertainment", "future", "creativity", "ai", "biology"
    ]
    selected_tags_filter = st.multiselect("Optional Tag Filter:", popular_tags)

    st.markdown("---")
    st.caption("SpringBoard Machine Learning Internship Capstone")
    st.caption("Engine: TF-IDF + TruncatedSVD (LSA) + SVD Collaborative")


# ----------------- HEADER -----------------
st.markdown("""
<div class="main-header">
    <h1>🎙️ Personalized TED Talks Recommendation Engine</h1>
    <p>Discover inspiring, tailored ideas powered by Content-Based Filtering, Matrix Factorization Collaborative Filtering & Dynamic Hybrid Re-ranking.</p>
</div>
""", unsafe_allow_html=True)

# ----------------- TABS -----------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Personalized Recommendations",
    "🔎 Talk Explorer & Catalog",
    "📊 EDA & Visual Analytics",
    "🧠 Model Architecture & Evaluation",
    "📚 Documentation & API"
])


# =========================================================================
# TAB 1: RECOMMENDATIONS
# =========================================================================
with tab1:
    st.markdown("#### Configure Your Recommendation Inputs")
    
    col_input1, col_input2 = st.columns([3, 1])
    
    # 1. Talk selection for Hybrid / Content
    selected_talk_id = 0
    selected_user_id = 1
    query_text = ""
    
    if "Content" in rec_mode or "Hybrid" in rec_mode:
        # Create user-friendly titles
        talk_options = [f"#{row['talk_id']} | {row['main_speaker']} — {row['title']}" for _, row in df_talks.iterrows()]
        
        # Popular defaults
        popular_defaults = {
            "Ken Robinson (Creativity)": 0,
            "Amy Cuddy (Body Language)": 4,
            "Simon Sinek (Leadership)": 201,
            "Brene Brown (Vulnerability)": 837,
            "Elon Musk (Future & Space)": 2437,
            "Hans Rosling (Global Stats)": 5
        }
        
        st.markdown("**Quick Pick Famous Talks:**")
        preset_cols = st.columns(len(popular_defaults))
        for col_idx, (p_name, p_id) in enumerate(popular_defaults.items()):
            if preset_cols[col_idx].button(p_name, key=f"quick_{p_id}", use_container_width=True):
                st.session_state['selected_talk_id'] = p_id
        
        current_talk_id = st.session_state.get('selected_talk_id', 0)
        selected_talk_str = st.selectbox(
            "Select or Search a TED Talk you liked:",
            options=talk_options,
            index=int(current_talk_id)
        )
        selected_talk_id = int(selected_talk_str.split("|")[0].replace("#", "").strip())
        
        # Display selected talk preview
        source_talk = df_talks.iloc[selected_talk_id]
        with st.expander(f"📌 Selected Seed Talk: **{source_talk['title']}** ({source_talk['main_speaker']})", expanded=False):
            st.write(f"**Speaker:** {source_talk['main_speaker']} ({source_talk['speaker_occupation']})")
            st.write(f"**Duration:** {source_talk['duration_min']} mins | **Views:** {source_talk['views']:,} | **Year:** {source_talk['film_year']}")
            st.write(f"**Description:** {source_talk['description']}")
            st.write(f"**Tags:** {source_talk['tags_display']}")
    
    # 2. User Persona selection for Collaborative / Hybrid
    if "Collaborative" in rec_mode or "Hybrid" in rec_mode:
        persona_dict = {
            "Tech & AI Innovator (User #1)": 1,
            "Psychology & Well-being (User #2)": 2,
            "Business & Strategy (User #3)": 3,
            "Art & Design Thinker (User #4)": 4,
            "Science & Universe (User #5)": 5,
            "Global Reform & Activism (User #6)": 6,
            "Custom User ID (1-800)": 0
        }
        
        persona_choice = st.selectbox("Select User Persona or Profile:", list(persona_dict.keys()), index=0)
        if persona_choice == "Custom User ID (1-800)":
            selected_user_id = st.number_input("Enter User ID:", min_value=1, max_value=800, value=42)
        else:
            selected_user_id = persona_dict[persona_choice]
            
    # 3. Query input for Semantic Query Search
    if "Query" in rec_mode:
        query_text = st.text_input(
            "Enter any topic, question, or concept you're curious about:",
            value="How does artificial intelligence impact human creativity and the future of work?",
            placeholder="e.g. quantum computing, resilience, habits of successful leaders, ocean conservation"
        )

    # Action Button
    st.markdown("")
    get_recs_btn = st.button("🚀 Generate Personalized Recommendations", type="primary", use_container_width=True)
    
    # Run recommendation
    if get_recs_btn or True:  # Run by default on first load
        with st.spinner("Computing similarity representations and collaborative latent vectors..."):
            if "Query" in rec_mode:
                recommendations = engine.recommend(
                    query=query_text,
                    tags_filter=selected_tags_filter,
                    top_n=top_n
                )
            elif "Content" in rec_mode:
                recommendations = engine.recommend(
                    talk_id=selected_talk_id,
                    tags_filter=selected_tags_filter,
                    top_n=top_n
                )
            elif "Collaborative" in rec_mode:
                recommendations = engine.recommend(
                    user_id=selected_user_id,
                    tags_filter=selected_tags_filter,
                    top_n=top_n
                )
            else:  # Hybrid
                recommendations = engine.recommend(
                    talk_id=selected_talk_id,
                    user_id=selected_user_id,
                    alpha=alpha,
                    tags_filter=selected_tags_filter,
                    top_n=top_n
                )
                
        st.markdown(f"### 🏆 Top {len(recommendations)} Recommendations For You")
        
        if not recommendations:
            st.warning("No recommendations found matching current filters. Try relaxing the tag filter.")
        else:
            for rec in recommendations:
                st.markdown(f"""
                <div class="recommendation-card">
                    <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 8px;">
                        <div>
                            <span class="card-rank">#{rec['rank']}</span>
                            <span style="font-size: 1.25rem; font-weight: 700; color: #111827;">{rec['title']}</span>
                        </div>
                        <div>
                            <span class="score-badge">Match: {rec.get('hybrid_score', rec.get('similarity_score', rec.get('predicted_rating', 'N/A')))}</span>
                            <span class="emotion-badge">{rec.get('top_emotion', 'Inspiring')}</span>
                        </div>
                    </div>
                    <div style="color: #4B5563; font-size: 0.95rem; margin-bottom: 10px;">
                        <strong>Speaker:</strong> {rec['main_speaker']} &nbsp;|&nbsp; 
                        <strong>Occupation:</strong> {rec['speaker_occupation']} &nbsp;|&nbsp; 
                        <strong>Duration:</strong> {rec['duration_min']} mins &nbsp;|&nbsp; 
                        <strong>Views:</strong> {rec['views']:,}
                    </div>
                    <div style="margin-bottom: 10px;">
                        {' '.join([f'<span class="tag-pill">{t}</span>' for t in rec.get('tags', [])])}
                    </div>
                    <div class="explanation-box">
                        💡 <strong>Why Recommended:</strong> {rec.get('explanation', 'High contextual and semantic match.')}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Card actions & details
                with st.expander(f"View Description, Transcript Excerpt & Watch Link for '{rec['title']}'"):
                    c1, c2 = st.columns([2, 1])
                    with c1:
                        st.markdown(f"**Overview:** {rec['description']}")
                        # Fetch full transcript snippet from dataframe
                        full_row = df_talks[df_talks['talk_id'] == rec['talk_id']].iloc[0]
                        lead = full_row.get('transcript_lead', '')
                        if lead:
                            st.markdown(f"**Key Opening:** *\"{lead[:280]}...\"*")
                    with c2:
                        st.markdown(f"[▶️ Watch Talk on TED.com]({rec['url']})")
                        if 'content_component' in rec and 'collab_component' in rec:
                            st.write(f"- **Content Match:** `{rec['content_component']:.3f}`")
                            st.write(f"- **Collab Signal:** `{rec['collab_component']:.3f}`")
                            st.progress(float(rec.get('hybrid_score', 0.5)))


# =========================================================================
# TAB 2: TALK EXPLORER & CATALOG
# =========================================================================
with tab2:
    st.markdown("### 🔎 Interactive TED Talks Catalog (2,628 Talks)")
    st.write("Browse, filter, and inspect the complete preprocessed TED talks corpus.")
    
    col_f1, col_f2, col_f3, col_f4 = st.columns(4)
    with col_f1:
        search_keyword = st.text_input("Filter by Keyword / Title / Speaker:", "")
    with col_f2:
        emotion_filter = st.selectbox("Filter by Primary Emotion:", ["All"] + list(dataset_summary['top_emotions'].keys()))
    with col_f3:
        min_views = st.number_input("Minimum Views:", min_value=0, max_value=50000000, value=0, step=500000)
    with col_f4:
        sort_by = st.selectbox("Sort By:", ["Views (High to Low)", "Duration (Short to Long)", "Duration (Long to Short)", "Film Year (Newest)"])
        
    # Filtering logic
    filtered_df = df_talks.copy()
    if search_keyword:
        kw = search_keyword.lower()
        filtered_df = filtered_df[
            filtered_df['title'].str.lower().str.contains(kw) |
            filtered_df['main_speaker'].str.lower().str.contains(kw) |
            filtered_df['tag_string'].str.lower().str.contains(kw)
        ]
    if emotion_filter != "All":
        filtered_df = filtered_df[filtered_df['top_emotion'] == emotion_filter]
    if min_views > 0:
        filtered_df = filtered_df[filtered_df['views'] >= min_views]
        
    if sort_by == "Views (High to Low)":
        filtered_df = filtered_df.sort_values(by='views', ascending=False)
    elif sort_by == "Duration (Short to Long)":
        filtered_df = filtered_df.sort_values(by='duration_min', ascending=True)
    elif sort_by == "Duration (Long to Short)":
        filtered_df = filtered_df.sort_values(by='duration_min', ascending=False)
    elif sort_by == "Film Year (Newest)":
        filtered_df = filtered_df.sort_values(by='film_year', ascending=False)
        
    st.markdown(f"**Showing {len(filtered_df):,} matching talks:**")
    st.dataframe(
        filtered_df[['talk_id', 'title', 'main_speaker', 'speaker_occupation', 'views', 'duration_min', 'top_emotion', 'tags_display', 'film_year']],
        column_config={
            "talk_id": st.column_config.NumberColumn("ID", width="small"),
            "title": st.column_config.TextColumn("Title", width="large"),
            "main_speaker": st.column_config.TextColumn("Speaker", width="medium"),
            "views": st.column_config.NumberColumn("Views", format="%d"),
            "duration_min": st.column_config.NumberColumn("Min", format="%.1f min"),
            "top_emotion": st.column_config.TextColumn("Top Tone", width="small"),
            "tags_display": st.column_config.TextColumn("Tags", width="medium"),
            "film_year": st.column_config.NumberColumn("Year", format="%d")
        },
        use_container_width=True,
        height=500
    )


# =========================================================================
# TAB 3: EDA & VISUAL ANALYTICS
# =========================================================================
with tab3:
    st.markdown("### 📊 Exploratory Data Analysis & Corpus Insights")
    st.write("Quantitative patterns across duration, engagement, emotion, and topical distribution.")
    
    # KPI metrics row
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total Talks", f"{dataset_summary['total_talks']:,}")
    m2.metric("Total Speakers", f"{dataset_summary['total_speakers']:,}")
    m3.metric("Avg Duration", f"{dataset_summary['avg_duration_minutes']} min")
    m4.metric("Avg Views", f"{dataset_summary['avg_views']/1e6:.2f}M")
    m5.metric("Total Interactions", f"{dataset_summary['total_interactions']:,}")
    
    st.markdown("---")
    
    c_chart1, c_chart2 = st.columns(2)
    with c_chart1:
        st.markdown("#### ⏳ Talk Duration Distribution")
        fig_dur = px.histogram(
            df_talks,
            x='duration_min',
            nbins=40,
            color_discrete_sequence=['#2563EB'],
            labels={'duration_min': 'Duration (Minutes)'},
            title="Most TED talks are structured between 10 to 18 minutes"
        )
        fig_dur.update_layout(bargap=0.1, template='plotly_white')
        st.plotly_chart(fig_dur, use_container_width=True)
        
    with c_chart2:
        st.markdown("#### 🏷️ Top 15 Topic Tags")
        all_tags = [t for tags in df_talks['parsed_tags'].apply(ast.literal_eval) for t in tags]
        tag_series = pd.Series(all_tags).value_counts().head(15).reset_index()
        tag_series.columns = ['Tag', 'Count']
        fig_tags = px.bar(
            tag_series,
            x='Count',
            y='Tag',
            orientation='h',
            color='Count',
            color_continuous_scale='Blues',
            title="Frequency of Core Thematic Categories"
        )
        fig_tags.update_layout(yaxis={'categoryorder': 'total ascending'}, template='plotly_white')
        st.plotly_chart(fig_tags, use_container_width=True)
        
    c_chart3, c_chart4 = st.columns(2)
    with c_chart3:
        st.markdown("#### 🎭 Audience Emotional Reaction Profile")
        emo_data = pd.Series(dataset_summary['top_emotions']).reset_index()
        emo_data.columns = ['Emotion', 'TalksCount']
        fig_emo = px.pie(
            emo_data,
            names='Emotion',
            values='TalksCount',
            hole=0.4,
            title="Dominant Emotional Takeaway Across Talks",
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        fig_emo.update_layout(template='plotly_white')
        st.plotly_chart(fig_emo, use_container_width=True)
        
    with c_chart4:
        st.markdown("#### 💬 Engagement: Views vs Comments")
        fig_scatter = px.scatter(
            df_talks.sample(n=min(800, len(df_talks)), random_state=42),
            x='views',
            y='comments',
            log_x=True,
            log_y=True,
            hover_name='title',
            color='top_emotion',
            title="Engagement Correlation (Log-Log Scale)",
            labels={'views': 'Views (Log)', 'comments': 'Comments (Log)'}
        )
        fig_scatter.update_layout(template='plotly_white')
        st.plotly_chart(fig_scatter, use_container_width=True)


# =========================================================================
# TAB 4: MODEL ARCHITECTURE & EVALUATION
# =========================================================================
with tab4:
    st.markdown("### 🧠 Recommendation Engine Architecture & Offline Benchmark")
    
    st.markdown("""
    #### Architectural Overview: Multi-Stage Hybrid Architecture
    The recommendation engine unifies content analysis and collaborative matrix decomposition:
    
    1. **Content-Based Filtering Layer:**
       - **Text Preprocessing:** Tokenization, stopword removal, transcript lead extraction.
       - **TF-IDF Vectorization:** Sublinear TF scaling with unigrams & bigrams across 12,000 features.
       - **Dense Semantic LSA:** TruncatedSVD dimensionality reduction ($K=100$) capturing latent topic spaces.
       - **Emotional Tone Embeddings:** 14-dimensional normalized probability vectors of audience sentiments.
       - **Composite Content Similarity:** $0.70 \cdot \text{TF-IDF} + 0.20 \cdot \text{LSA} + 0.10 \cdot \text{Emotion}$.
       
    2. **Collaborative Filtering Layer:**
       - **User-Item Interaction Matrix:** Sparse CSR representation across 800 users and 2,628 talks.
       - **Matrix Factorization (SVD):** Low-rank decomposition with 40 latent factors: $\hat{R} = U \Sigma V^T$.
       - **Item-Item Collaborative Cosine:** Co-engagement patterns between talks.
       
    3. **Dynamic Hybrid Blending & Re-ranking:**
       - $$S_{\text{hybrid}}(u, i) = \alpha \cdot S_{\text{content}}(i_{\text{seed}}, i) + (1 - \alpha) \cdot \text{Norm}(S_{\text{collab}}(u, i))$$
       - **Cold-Start Fallback:** Automatic routing to content-based similarity when user profile has < 3 interactions.
    """)
    
    st.markdown("---")
    st.markdown("#### 📈 Benchmark Results: Content-Based Models vs Ground-Truth Related Talks")
    st.write("Evaluated on 500 benchmark talks against TED editorial and viewer ground truth related talks:")
    
    content_table = pd.DataFrame(eval_summary['content_models']).T
    st.dataframe(
        content_table.style.highlight_max(axis=0, color='#DCFCE7'),
        use_container_width=True
    )
    st.caption("Notice how Composite (Hybrid Content) outperforms single-feature models in Precision@5 (0.2400), NDCG@10 (0.5703), and Recall@10 (0.3095).")
    
    st.markdown("#### ⚖️ Hybrid Alpha (α) Trade-off Analysis")
    st.write("Evaluating the balance between Content signals and Collaborative Matrix Factorization:")
    
    hybrid_alpha_df = pd.DataFrame(eval_summary['hybrid_alpha_tradeoff']).T
    st.dataframe(hybrid_alpha_df, use_container_width=True)
    
    # Plot alpha tradeoff
    alphas = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
    p10 = [eval_summary['hybrid_alpha_tradeoff'][f'Hybrid (alpha={a})']['Precision@10'] for a in alphas]
    r10 = [eval_summary['hybrid_alpha_tradeoff'][f'Hybrid (alpha={a})']['Recall@10'] for a in alphas]
    ndcg10 = [eval_summary['hybrid_alpha_tradeoff'][f'Hybrid (alpha={a})']['NDCG@10'] for a in alphas]
    
    fig_tradeoff = go.Figure()
    fig_tradeoff.add_trace(go.Scatter(x=alphas, y=p10, mode='lines+markers', name='Precision@10', line=dict(color='#2563EB', width=3)))
    fig_tradeoff.add_trace(go.Scatter(x=alphas, y=r10, mode='lines+markers', name='Recall@10', line=dict(color='#059669', width=3)))
    fig_tradeoff.add_trace(go.Scatter(x=alphas, y=ndcg10, mode='lines+markers', name='NDCG@10', line=dict(color='#D97706', width=3)))
    fig_tradeoff.add_vline(x=0.55, line_width=2, line_dash="dash", line_color="#DC2626", annotation_text="Optimal Balance (0.55)")
    fig_tradeoff.update_layout(
        title="Impact of Alpha on Evaluation Metrics (0 = Collab SVD, 1 = Content)",
        xaxis_title="Alpha (Content Weight)",
        yaxis_title="Metric Score",
        template='plotly_white'
    )
    st.plotly_chart(fig_tradeoff, use_container_width=True)


# =========================================================================
# TAB 5: DOCUMENTATION & API GUIDE
# =========================================================================
with tab5:
    st.markdown("### 📚 Project Documentation & API Guide")
    st.markdown("""
    #### Milestone Mapping (SpringBoard Internship Program)
    - **Milestone 1 (Week 1-2): Data Collection & Preprocessing**
      - Ingested 2,628 talks and 2,542 transcripts.
      - Handled missing data, normalized text, extracted emotional profiles, created interaction logs.
    - **Milestone 2 (Week 3-4): Feature Engineering & Vectorization**
      - Built TF-IDF n-gram vectorizer (12,000 features).
      - Trained 100-component TruncatedSVD Latent Semantic Analysis (LSA) dense embeddings.
      - Constructed sparse CSR user-item matrix with 27,430 interaction entries.
    - **Milestone 3 (Week 5-6): Recommendation Models & Evaluation**
      - Implemented Content-Based, SVD Matrix Factorization, and Dynamic Hybrid algorithms.
      - Evaluated Precision@K, Recall@K, NDCG@K, MAP@K, and Catalog Coverage.
    - **Milestone 4 (Week 7-8): Deployment & Documentation**
      - Deployed interactive Streamlit application with explainability and visual analytics.
      
    #### Python Developer API Usage Example:
    ```python
    from src.models import HybridTEDRecommender

    # Initialize recommender engine
    recommender = HybridTEDRecommender()

    # 1. Recommend by liked talk (Content-Based)
    recs = recommender.recommend(talk_id=0, top_n=5)

    # 2. Recommend by User Profile (Collaborative)
    user_recs = recommender.recommend(user_id=1, top_n=5)

    # 3. Hybrid Recommendation (Content + Collab)
    hybrid_recs = recommender.recommend(talk_id=0, user_id=1, alpha=0.55, top_n=5)

    # 4. Natural Language Semantic Search
    query_recs = recommender.recommend(query="artificial intelligence neuroscience memory", top_n=5)
    ```
    """)
