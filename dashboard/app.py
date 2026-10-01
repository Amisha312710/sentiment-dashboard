import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from db.database import load_posts, get_topic_summary
from analyze.sentiment import get_summary_stats, analyze_single_text
from pipeline import run_pipeline

st.set_page_config(page_title='Social Media Sentiment Analytics', page_icon='📊', layout='wide', initial_sidebar_state='expanded')

st.markdown("""
<style>
.stApp { background-color: #0f1117; }
[data-testid="metric-container"] { background: linear-gradient(135deg, #1e2130, #252836); border: 1px solid #2d3250; border-radius: 12px; padding: 16px 20px; }
[data-testid="metric-container"] label { color: #8b92a5 !important; font-size: 12px !important; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color: #ffffff !important; font-size: 28px !important; font-weight: 700 !important; }
.dashboard-header { background: linear-gradient(135deg, #1a1f35 0%, #16213e 100%); border: 1px solid #2d3250; border-radius: 16px; padding: 24px 32px; margin-bottom: 24px; }
.dashboard-title { font-size: 28px; font-weight: 800; background: linear-gradient(90deg, #667eea, #764ba2, #f093fb); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0; }
.dashboard-subtitle { color: #8b92a5; font-size: 14px; margin-top: 4px; }
.section-title { font-size: 16px; font-weight: 600; color: #c8cdd8; border-left: 3px solid #667eea; padding-left: 12px; margin: 24px 0 16px 0; }
[data-testid="stSidebar"] { background: #13151f !important; border-right: 1px solid #2d3250; }
.stTabs [data-baseweb="tab-list"] { background: #1a1f2e; border-radius: 10px; padding: 4px; }
.stTabs [aria-selected="true"] { background: #252836 !important; color: #fff !important; border-radius: 8px; }
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
.live-badge { padding: 4px 10px; border-radius: 8px; font-size: 13px; font-weight: 600; display: inline-block; }
</style>
""", unsafe_allow_html=True)

LAYOUT = dict(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#c8cdd8', font_family='Inter, sans-serif', margin=dict(l=0, r=0, t=36, b=0), legend=dict(bgcolor='rgba(0,0,0,0)', font_color='#8b92a5'))
COLORS = {'Positive': '#34d399', 'Neutral': '#a78bfa', 'Negative': '#f87171'}
TOPIC_COLORS = ['#667eea', '#764ba2', '#f093fb', '#4facfe', '#43e97b']

with st.sidebar:
    st.markdown('### Filters')
    topic_options = ['All Topics', 'AI & Technology', 'Stock Market', 'Climate & Environment', 'Sports', 'Health & Wellness']
    selected_topic = st.selectbox('Topic Domain', topic_options)
    label_options = ['All Sentiments', 'Positive', 'Neutral', 'Negative']
    selected_label = st.selectbox('Sentiment Filter', label_options)
    days_back = st.slider('Days to display', min_value=1, max_value=7, value=7)
    
    st.markdown('---')
    st.markdown('### ETL Orchestration')
    st.caption("Fetches live posts via HackerNews API & generates multi-domain sentiment data.")
    if st.button('Run ETL Pipeline Now', use_container_width=True, type='primary'):
        with st.spinner('Ingesting, transforming & scoring sentiment...'):
            result = run_pipeline(post_count=120)
        if result['status'] == 'success':
            st.success(f"ETL Complete: {result['inserted']} rows inserted ({result['elapsed_s']}s)")
            st.cache_data.clear()
            st.rerun()
        else:
            st.error(f"ETL Error: {result.get('message', 'Failed')}")
            
    st.markdown('---')
    st.markdown("<span style='color:#4b5267;font-size:11px'>Social Media Sentiment Platform<br>Production ETL + NLP Dashboard</span>", unsafe_allow_html=True)

@st.cache_data(ttl=60)
def get_data(topic, label, days):
    t = topic if topic != 'All Topics' else None
    l = label if label != 'All Sentiments' else None
    return load_posts(topic=t, label=l, days=days)

@st.cache_data(ttl=60)
def get_topic_data():
    return get_topic_summary()

df = get_data(selected_topic, selected_label, days_back)
topic_df = get_topic_data()

# Auto-seed database if fresh deployment / empty
if df.empty:
    with st.spinner('Initializing pipeline & auto-seeding sample dataset...'):
        run_pipeline(post_count=120)
        st.cache_data.clear()
        df = get_data(selected_topic, selected_label, days_back)
        topic_df = get_topic_data()

st.markdown('<div class="dashboard-header"><p class="dashboard-title">Social Media Sentiment Analytics Engine</p><p class="dashboard-subtitle">Automated ETL Pipeline & Multi-Model NLP (VADER + TextBlob) across 5 Key Domains</p></div>', unsafe_allow_html=True)

stats = get_summary_stats(df)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric('Total Volume', f"{stats['total_posts']:,}")
c2.metric('Mean Sentiment', f"{stats['avg_score']:+.3f}", delta='VADER Compound', delta_color='normal')
c3.metric('Positive', f"{stats['positive_pct']}%", delta=f"{stats['positive_count']} posts")
c4.metric('Neutral', f"{stats['neutral_pct']}%", delta=f"{stats['neutral_count']} posts")
c5.metric('Negative', f"{stats['negative_pct']}%", delta=f"{stats['negative_count']} posts", delta_color='inverse')

st.markdown('<p class="section-title">Analytics & Intelligence</p>', unsafe_allow_html=True)
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    '  📈 Sentiment Over Time  ',
    '  🌐 Topic Breakdown  ',
    '  🔬 Score Distribution  ',
    '  📋 Posts Explorer  ',
    '  ⚡ Live Sentiment Playground  '
])

with tab1:
    col_a, col_b = st.columns([2, 1])
    score_col = 'vader_compound' if 'vader_compound' in df.columns else 'sentiment_score'
    with col_a:
        daily_trend = df.groupby('date')[score_col].mean().reset_index()
        fig_line = px.line(daily_trend, x='date', y=score_col, title='Daily Mean Sentiment Velocity', markers=True, color_discrete_sequence=['#667eea'])
        fig_line.add_hline(y=0, line_dash='dash', line_color='#4b5267', line_width=1)
        fig_line.update_traces(line_width=2.5)
        fig_line.update_layout(**LAYOUT)
        fig_line.update_xaxes(showgrid=False, color='#4b5267')
        fig_line.update_yaxes(showgrid=True, gridcolor='#1e2130', color='#4b5267', range=[-1, 1])
        st.plotly_chart(fig_line, use_container_width=True)
    with col_b:
        stacked = df.groupby(['date', 'sentiment_label']).size().reset_index(name='count')
        fig_stack = px.bar(stacked, x='date', y='count', color='sentiment_label', title='Daily Volume Breakdown', color_discrete_map=COLORS, barmode='stack')
        fig_stack.update_layout(**LAYOUT)
        fig_stack.update_xaxes(showgrid=False, color='#4b5267')
        fig_stack.update_yaxes(showgrid=True, gridcolor='#1e2130', color='#4b5267')
        st.plotly_chart(fig_stack, use_container_width=True)
        
    hourly = df.groupby(['hour', 'sentiment_label'])[score_col].mean().reset_index()
    fig_hour = px.bar(hourly, x='hour', y=score_col, color='sentiment_label', title='Sentiment Trajectory by Hour of Day (UTC)', color_discrete_map=COLORS, barmode='group')
    fig_hour.update_layout(**LAYOUT)
    fig_hour.update_xaxes(showgrid=False, color='#4b5267', title='Hour (0-23)', dtick=2)
    fig_hour.update_yaxes(showgrid=True, gridcolor='#1e2130', color='#4b5267')
    st.plotly_chart(fig_hour, use_container_width=True)

with tab2:
    col_a, col_b = st.columns(2)
    with col_a:
        topic_avg = df.groupby('topic')[score_col].mean().sort_values()
        fig_topics = go.Figure(go.Bar(x=topic_avg.values, y=topic_avg.index, orientation='h', marker=dict(color=topic_avg.values, colorscale=[[0, '#f87171'], [0.5, '#a78bfa'], [1, '#34d399']], showscale=False)))
        fig_topics.update_layout(**LAYOUT, title='Average Sentiment Index by Domain', xaxis_title='Sentiment Score')
        fig_topics.add_vline(x=0, line_dash='dash', line_color='#4b5267')
        fig_topics.update_xaxes(showgrid=True, gridcolor='#1e2130', color='#4b5267')
        fig_topics.update_yaxes(showgrid=False, color='#c8cdd8')
        st.plotly_chart(fig_topics, use_container_width=True)
    with col_b:
        topic_labels = df.groupby(['topic', 'sentiment_label']).size().reset_index(name='count')
        fig_tl = px.bar(topic_labels, x='topic', y='count', color='sentiment_label', title='Distribution of Sentiments per Domain', color_discrete_map=COLORS, barmode='group')
        fig_tl.update_layout(**LAYOUT)
        fig_tl.update_xaxes(showgrid=False, color='#4b5267', tickangle=-20)
        fig_tl.update_yaxes(showgrid=True, gridcolor='#1e2130', color='#4b5267')
        st.plotly_chart(fig_tl, use_container_width=True)
        
    topic_avg2 = df.groupby('topic')[score_col].mean().sort_values()
    topics_r = topic_avg2.index.tolist()
    scores_n = [(s + 1) / 2 for s in topic_avg2.values.tolist()]
    fig_radar = go.Figure(go.Scatterpolar(r=scores_n + [scores_n[0]], theta=topics_r + [topics_r[0]], fill='toself', fillcolor='rgba(102,126,234,0.15)', line=dict(color='#667eea', width=2)))
    fig_radar.update_layout(**LAYOUT, title='Domain Sentiment Comparison Radar', polar=dict(bgcolor='#1a1f2e', radialaxis=dict(visible=True, range=[0, 1], color='#4b5267', gridcolor='#2d3250'), angularaxis=dict(color='#8b92a5', gridcolor='#2d3250')))
    st.plotly_chart(fig_radar, use_container_width=True)

with tab3:
    col_a, col_b = st.columns(2)
    with col_a:
        fig_hist = px.histogram(df, x=score_col, color='sentiment_label', nbins=40, title='Score Distribution (Compound Density)', color_discrete_map=COLORS, opacity=0.8)
        fig_hist.update_layout(**LAYOUT)
        fig_hist.add_vline(x=0, line_dash='dash', line_color='#4b5267')
        fig_hist.update_xaxes(showgrid=False, color='#4b5267')
        fig_hist.update_yaxes(showgrid=True, gridcolor='#1e2130', color='#4b5267')
        st.plotly_chart(fig_hist, use_container_width=True)
    with col_b:
        fig_scatter = px.scatter(df, x='subjectivity', y=score_col, color='sentiment_label', size='word_count', hover_data=['topic', 'source'], title='Sentiment vs. Subjectivity (Fact vs. Opinion)', color_discrete_map=COLORS, opacity=0.7)
        fig_scatter.add_hline(y=0, line_dash='dash', line_color='#4b5267')
        fig_scatter.add_vline(x=0.5, line_dash='dash', line_color='#4b5267')
        fig_scatter.update_layout(**LAYOUT)
        fig_scatter.update_xaxes(showgrid=True, gridcolor='#1e2130', color='#4b5267')
        fig_scatter.update_yaxes(showgrid=True, gridcolor='#1e2130', color='#4b5267')
        st.plotly_chart(fig_scatter, use_container_width=True)
        
    source_counts = df.groupby('source').size().reset_index(name='count')
    fig_pie = px.pie(source_counts, names='source', values='count', title='Post Ingestion by Data Source (Live APIs & Feeds)', color_discrete_sequence=TOPIC_COLORS, hole=0.5)
    fig_pie.update_layout(**LAYOUT)
    fig_pie.update_traces(textfont_color='#c8cdd8')
    st.plotly_chart(fig_pie, use_container_width=True)

with tab4:
    st.markdown('<p class="section-title">Raw Processed Records Explorer</p>', unsafe_allow_html=True)
    display_df = df[['created_at', 'topic', 'source', 'sentiment_label', score_col, 'text', 'author', 'upvotes', 'comments']].copy()
    display_df = display_df.sort_values('created_at', ascending=False)
    display_df['created_at'] = display_df['created_at'].dt.strftime('%Y-%m-%d %H:%M')
    display_df[score_col] = display_df[score_col].apply(lambda x: f'{x:+.3f}')
    display_df.columns = ['Time', 'Topic', 'Source', 'Label', 'Score', 'Text', 'Author', 'Upvotes', 'Comments']
    st.dataframe(display_df, use_container_width=True, height=450, column_config={'Score': st.column_config.TextColumn('Score', width='small'), 'Label': st.column_config.TextColumn('Label', width='small'), 'Text': st.column_config.TextColumn('Text', width='large')}, hide_index=True)
    
    # Download as CSV button
    csv_bytes = display_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Export Filtered Dataset as CSV",
        data=csv_bytes,
        file_name="sentiment_filtered_dataset.csv",
        mime="text/csv",
        use_container_width=False
    )
    
    st.markdown('---')
    mc1, mc2, mc3 = st.columns(3)
    mc1.markdown(f"Top Positive Sector: **{stats['most_positive_topic']}**")
    mc2.markdown(f"Top Negative Sector: **{stats['most_negative_topic']}**")
    mc3.markdown(f"Mean Subjectivity: **{stats['avg_subjectivity']}**")

with tab5:
    st.markdown('<p class="section-title">Real-Time NLP Inference Playground</p>', unsafe_allow_html=True)
    st.markdown("Test the dual NLP engines (**VADER Rule-Based Social Analyzer** vs. **TextBlob Lexicon Engine**) on any custom text in real-time.")
    
    preset_col1, preset_col2, preset_col3 = st.columns(3)
    if preset_col1.button("🤖 Test AI & Tech Sample", use_container_width=True):
        st.session_state["user_sample"] = "The new Claude 3.7 and reasoning models are an extraordinary leap forward in coding productivity!"
    if preset_col2.button("📉 Test Financial Market Sample", use_container_width=True):
        st.session_state["user_sample"] = "Catastrophic earnings miss. The company guidance is disastrous and stock dropped 25%."
    if preset_col3.button("🌱 Test Climate / Tech Sample", use_container_width=True):
        st.session_state["user_sample"] = "Breakthroughs in sodium-ion battery storage are making renewable energy radically cheaper and cleaner."
        
    default_text = st.session_state.get("user_sample", "The new AI features launched today exceeded our highest expectations! Truly revolutionary and delighting customers.")
    custom_text = st.text_area("Input any custom sentence or social media post:", value=default_text, height=90)
    
    if custom_text:
        res = analyze_single_text(custom_text)
        
        c_res1, c_res2, c_res3, c_res4 = st.columns(4)
        badge_color = "#34d399" if res["label"] == "Positive" else ("#f87171" if res["label"] == "Negative" else "#a78bfa")
        c_res1.markdown(f"**Predicted Sentiment**<br><span style='font-size:24px; font-weight:800; color:{badge_color}'>{res['label']}</span>", unsafe_allow_html=True)
        c_res2.metric("VADER Compound", f"{res['vader_compound']:+.3f}")
        c_res3.metric("TextBlob Polarity", f"{res['tb_polarity']:+.3f}")
        c_res4.metric("Subjectivity Index", f"{res['tb_subjectivity']:.2f}", help="0 = Objective Fact, 1 = Personal Opinion")
        
        col_g1, col_g2 = st.columns([1, 1])
        with col_g1:
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = res['vader_compound'],
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "VADER Compound Score (-1.0 to +1.0)", 'font': {'color': '#c8cdd8', 'size': 15}},
                gauge = {
                    'axis': {'range': [-1, 1], 'tickcolor': "#8b92a5"},
                    'bar': {'color': badge_color},
                    'steps': [
                        {'range': [-1, -0.05], 'color': 'rgba(248, 113, 113, 0.25)'},
                        {'range': [-0.05, 0.05], 'color': 'rgba(167, 139, 250, 0.25)'},
                        {'range': [0.05, 1], 'color': 'rgba(52, 211, 153, 0.25)'}
                    ],
                    'threshold': {
                        'line': {'color': "white", 'width': 3},
                        'thickness': 0.75,
                        'value': res['vader_compound']
                    }
                }
            ))
            fig_gauge.update_layout(**LAYOUT, height=260)
            st.plotly_chart(fig_gauge, use_container_width=True)
            
        with col_g2:
            breakdown_df = pd.DataFrame({
                "Component": ["Positive 😊", "Neutral 😐", "Negative 😡"],
                "Weight": [res['vader_pos'], res['vader_neu'], res['vader_neg']]
            })
            fig_bar = px.bar(
                breakdown_df, x="Component", y="Weight",
                color="Component",
                title="VADER Sentiment Weight Proportions",
                color_discrete_map={"Positive 😊": "#34d399", "Neutral 😐": "#a78bfa", "Negative 😡": "#f87171"}
            )
            fig_bar.update_layout(**LAYOUT, height=260, showlegend=False)
            fig_bar.update_yaxes(range=[0, 1], showgrid=True, gridcolor='#1e2130')
            st.plotly_chart(fig_bar, use_container_width=True)
