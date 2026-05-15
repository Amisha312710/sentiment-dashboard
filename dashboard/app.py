import sys, os
sys.path.insert(0, '/Users/amishabhatia/Desktop/sentiment-dashboard')

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from db.database import load_posts, get_topic_summary
from analyze.sentiment import get_summary_stats
from pipeline import run_pipeline

st.set_page_config(page_title='Social Media Sentiment Dashboard', page_icon='📊', layout='wide', initial_sidebar_state='expanded')

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
</style>
""", unsafe_allow_html=True)

LAYOUT = dict(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#c8cdd8', font_family='Inter, sans-serif', margin=dict(l=0, r=0, t=36, b=0), legend=dict(bgcolor='rgba(0,0,0,0)', font_color='#8b92a5'))
COLORS = {'Positive': '#34d399', 'Neutral': '#a78bfa', 'Negative': '#f87171'}
TOPIC_COLORS = ['#667eea', '#764ba2', '#f093fb', '#4facfe', '#43e97b']

with st.sidebar:
    st.markdown('### Filters')
    topic_options = ['All Topics', 'AI & Technology', 'Stock Market', 'Climate & Environment', 'Sports', 'Health & Wellness']
    selected_topic = st.selectbox('Topic', topic_options)
    label_options = ['All Sentiments', 'Positive', 'Neutral', 'Negative']
    selected_label = st.selectbox('Sentiment', label_options)
    days_back = st.slider('Days to show', min_value=1, max_value=7, value=7)
    st.markdown('---')
    st.markdown('### Pipeline')
    if st.button('Run pipeline now', use_container_width=True, type='primary'):
        with st.spinner('Running ETL pipeline...'):
            result = run_pipeline(post_count=120)
        if result['status'] == 'success':
            st.success(f"Done! {result['inserted']} new posts added.")
        else:
            st.error(f"Error: {result['message']}")
    st.markdown('---')
    st.markdown("<span style='color:#4b5267;font-size:11px'>Social Media Sentiment Dashboard<br>Built with Python + Streamlit</span>", unsafe_allow_html=True)

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

st.markdown('<div class="dashboard-header"><p class="dashboard-title">Social Media Sentiment Dashboard</p><p class="dashboard-subtitle">Real-time sentiment analysis across AI, Markets, Climate, Sports & Health</p></div>', unsafe_allow_html=True)

if df.empty:
    st.warning('No data found. Click Run pipeline now in the sidebar to load posts.')
    st.stop()

stats = get_summary_stats(df)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric('Total Posts', f"{stats['total_posts']:,}")
c2.metric('Avg Sentiment', f"{stats['avg_score']:+.3f}", delta='higher is better', delta_color='normal')
c3.metric('Positive', f"{stats['positive_pct']}%", delta=f"{stats['positive_count']} posts")
c4.metric('Neutral', f"{stats['neutral_pct']}%", delta=f"{stats['neutral_count']} posts")
c5.metric('Negative', f"{stats['negative_pct']}%", delta=f"{stats['negative_count']} posts", delta_color='inverse')

st.markdown('<p class="section-title">Analysis</p>', unsafe_allow_html=True)
tab1, tab2, tab3, tab4 = st.tabs(['  Sentiment Over Time  ', '  Topic Breakdown  ', '  Score Distribution  ', '  Posts Explorer  '])

with tab1:
    col_a, col_b = st.columns([2, 1])
    with col_a:
        fig_line = px.line(df.groupby('date')['sentiment_score'].mean().reset_index(), x='date', y='sentiment_score', title='Average sentiment score per day', markers=True, color_discrete_sequence=['#667eea'])
        fig_line.add_hline(y=0, line_dash='dash', line_color='#4b5267', line_width=1)
        fig_line.update_traces(line_width=2.5)
        fig_line.update_layout(**LAYOUT)
        fig_line.update_xaxes(showgrid=False, color='#4b5267')
        fig_line.update_yaxes(showgrid=True, gridcolor='#1e2130', color='#4b5267', range=[-1, 1])
        st.plotly_chart(fig_line, use_container_width=True)
    with col_b:
        stacked = df.groupby(['date', 'sentiment_label']).size().reset_index(name='count')
        fig_stack = px.bar(stacked, x='date', y='count', color='sentiment_label', title='Daily volume by sentiment', color_discrete_map=COLORS, barmode='stack')
        fig_stack.update_layout(**LAYOUT)
        fig_stack.update_xaxes(showgrid=False, color='#4b5267')
        fig_stack.update_yaxes(showgrid=True, gridcolor='#1e2130', color='#4b5267')
        st.plotly_chart(fig_stack, use_container_width=True)
    hourly = df.groupby(['hour', 'sentiment_label'])['sentiment_score'].mean().reset_index()
    fig_hour = px.bar(hourly, x='hour', y='sentiment_score', color='sentiment_label', title='Sentiment by hour of day', color_discrete_map=COLORS, barmode='group')
    fig_hour.update_layout(**LAYOUT)
    fig_hour.update_xaxes(showgrid=False, color='#4b5267', title='Hour (0-23)', dtick=2)
    fig_hour.update_yaxes(showgrid=True, gridcolor='#1e2130', color='#4b5267')
    st.plotly_chart(fig_hour, use_container_width=True)

with tab2:
    col_a, col_b = st.columns(2)
    with col_a:
        topic_avg = df.groupby('topic')['sentiment_score'].mean().sort_values()
        fig_topics = go.Figure(go.Bar(x=topic_avg.values, y=topic_avg.index, orientation='h', marker=dict(color=topic_avg.values, colorscale=[[0, '#f87171'], [0.5, '#a78bfa'], [1, '#34d399']], showscale=False)))
        fig_topics.update_layout(**LAYOUT, title='Average sentiment score by topic', xaxis_title='Avg sentiment score')
        fig_topics.add_vline(x=0, line_dash='dash', line_color='#4b5267')
        fig_topics.update_xaxes(showgrid=True, gridcolor='#1e2130', color='#4b5267')
        fig_topics.update_yaxes(showgrid=False, color='#c8cdd8')
        st.plotly_chart(fig_topics, use_container_width=True)
    with col_b:
        topic_labels = df.groupby(['topic', 'sentiment_label']).size().reset_index(name='count')
        fig_tl = px.bar(topic_labels, x='topic', y='count', color='sentiment_label', title='Post volume by topic and sentiment', color_discrete_map=COLORS, barmode='group')
        fig_tl.update_layout(**LAYOUT)
        fig_tl.update_xaxes(showgrid=False, color='#4b5267', tickangle=-20)
        fig_tl.update_yaxes(showgrid=True, gridcolor='#1e2130', color='#4b5267')
        st.plotly_chart(fig_tl, use_container_width=True)
    topic_avg2 = df.groupby('topic')['sentiment_score'].mean().sort_values()
    topics_r = topic_avg2.index.tolist()
    scores_n = [(s + 1) / 2 for s in topic_avg2.values.tolist()]
    fig_radar = go.Figure(go.Scatterpolar(r=scores_n + [scores_n[0]], theta=topics_r + [topics_r[0]], fill='toself', fillcolor='rgba(102,126,234,0.15)', line=dict(color='#667eea', width=2)))
    fig_radar.update_layout(**LAYOUT, title='Topic sentiment radar', polar=dict(bgcolor='#1a1f2e', radialaxis=dict(visible=True, range=[0, 1], color='#4b5267', gridcolor='#2d3250'), angularaxis=dict(color='#8b92a5', gridcolor='#2d3250')))
    st.plotly_chart(fig_radar, use_container_width=True)

with tab3:
    col_a, col_b = st.columns(2)
    with col_a:
        fig_hist = px.histogram(df, x='sentiment_score', color='sentiment_label', nbins=40, title='Distribution of sentiment scores', color_discrete_map=COLORS, opacity=0.8)
        fig_hist.update_layout(**LAYOUT)
        fig_hist.add_vline(x=0, line_dash='dash', line_color='#4b5267')
        fig_hist.update_xaxes(showgrid=False, color='#4b5267')
        fig_hist.update_yaxes(showgrid=True, gridcolor='#1e2130', color='#4b5267')
        st.plotly_chart(fig_hist, use_container_width=True)
    with col_b:
        fig_scatter = px.scatter(df, x='subjectivity', y='sentiment_score', color='sentiment_label', size='word_count', hover_data=['topic', 'source'], title='Sentiment vs subjectivity', color_discrete_map=COLORS, opacity=0.7)
        fig_scatter.add_hline(y=0, line_dash='dash', line_color='#4b5267')
        fig_scatter.add_vline(x=0.5, line_dash='dash', line_color='#4b5267')
        fig_scatter.update_layout(**LAYOUT)
        fig_scatter.update_xaxes(showgrid=True, gridcolor='#1e2130', color='#4b5267')
        fig_scatter.update_yaxes(showgrid=True, gridcolor='#1e2130', color='#4b5267')
        st.plotly_chart(fig_scatter, use_container_width=True)
    source_counts = df.groupby('source').size().reset_index(name='count')
    fig_pie = px.pie(source_counts, names='source', values='count', title='Posts by source', color_discrete_sequence=TOPIC_COLORS, hole=0.5)
    fig_pie.update_layout(**LAYOUT)
    fig_pie.update_traces(textfont_color='#c8cdd8')
    st.plotly_chart(fig_pie, use_container_width=True)

with tab4:
    st.markdown('<p class="section-title">Recent posts</p>', unsafe_allow_html=True)
    display_df = df[['created_at', 'topic', 'source', 'sentiment_label', 'sentiment_score', 'text', 'author', 'upvotes', 'comments']].copy()
    display_df = display_df.sort_values('created_at', ascending=False)
    display_df['created_at'] = display_df['created_at'].dt.strftime('%Y-%m-%d %H:%M')
    display_df['sentiment_score'] = display_df['sentiment_score'].apply(lambda x: f'{x:+.3f}')
    display_df.columns = ['Time', 'Topic', 'Source', 'Label', 'Score', 'Text', 'Author', 'Upvotes', 'Comments']
    st.dataframe(display_df, use_container_width=True, height=500, column_config={'Score': st.column_config.TextColumn('Score', width='small'), 'Label': st.column_config.TextColumn('Label', width='small'), 'Text': st.column_config.TextColumn('Text', width='large')}, hide_index=True)
    st.markdown('---')
    mc1, mc2, mc3 = st.columns(3)
    mc1.markdown(f"Most positive topic: **{stats['most_positive_topic']}**")
    mc2.markdown(f"Most negative topic: **{stats['most_negative_topic']}**")
    mc3.markdown(f"Avg subjectivity: **{stats['avg_subjectivity']}**")
