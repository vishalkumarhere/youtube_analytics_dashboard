import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
from youtube_api import YouTubeAPI
from logging_config import logger
import time
import numpy as np
from scipy import stats
import mlflow
from model_monitoring_utils import get_latest_runs, get_run_params, get_run_metrics
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import json
from typing import Dict, Optional, Tuple, Any, List

# Ensure MLflow tracking URI is set
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlruns/mlflow.db"))

# End any active MLflow runs
try:
    if mlflow.active_run():
        mlflow.end_run()
        logger.info("Ended active MLflow run")
except Exception as e:
    logger.error(f"Error ending MLflow run: {str(e)}")

# Set page config with a custom theme
st.set_page_config(
    page_title="YouTube Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize YouTube API
try:
    youtube_api = YouTubeAPI()
except Exception as e:
    logger.error(f"Failed to initialize YouTube API: {str(e)}")
    st.error("Failed to initialize YouTube API. Please check your API key and try again.")
    st.stop()

# Initialize session state for caching
if 'channel_data' not in st.session_state:
    st.session_state.channel_data = {}
if 'trending_data' not in st.session_state:
    st.session_state.trending_data = {}
if 'last_analysis_time' not in st.session_state:
    st.session_state.last_analysis_time = {}

def get_cached_channel_data(channel_name: str) -> Optional[Tuple[Dict, pd.DataFrame, Dict, List, pd.DataFrame, pd.DataFrame]]:
    """Get cached channel data if it exists and hasn't expired"""
    if channel_name in st.session_state.channel_data:
        data, timestamp = st.session_state.channel_data[channel_name]
        if time.time() - timestamp < 3600:  # 1 hour cache expiry
            return data
        del st.session_state.channel_data[channel_name]
    return None

def set_cached_channel_data(channel_name: str, data: Tuple[Dict, pd.DataFrame, Dict, List, pd.DataFrame, pd.DataFrame]):
    """Store channel data in cache with timestamp"""
    st.session_state.channel_data[channel_name] = (data, time.time())

def get_cached_trending_data(region_code: str) -> Optional[Dict]:
    """Get cached trending data if it exists and hasn't expired"""
    if region_code in st.session_state.trending_data:
        data, timestamp = st.session_state.trending_data[region_code]
        if time.time() - timestamp < 3600:  # 1 hour cache expiry
            return data
        del st.session_state.trending_data[region_code]
    return None

def set_cached_trending_data(region_code: str, data: Dict):
    """Store trending data in cache with timestamp"""
    st.session_state.trending_data[region_code] = (data, time.time())

def analyze_channel(channel_name: str) -> Tuple[Optional[Dict], Optional[pd.DataFrame], Optional[Dict], Optional[List], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Analyze a YouTube channel with caching"""
    try:
        # Check cache first
        cached_data = get_cached_channel_data(channel_name)
        if cached_data:
            logger.info(f"Using cached data for channel: {channel_name}")
            return cached_data

        # Get channel ID from name
        channel_id = youtube_api.get_channel_id_by_name(channel_name)
        if not channel_id:
            st.error("Channel not found. Please check the channel name and try again.")
            return None, None, None, None, None, None

        # Get channel statistics
        channel_stats = youtube_api.get_channel_stats(channel_id)
        if not channel_stats:
            st.error("Channel not found or inaccessible")
            return None, None, None, None, None, None

        # Get channel videos
        videos = youtube_api.get_channel_videos(channel_stats['playlist_id'])
        if not videos:
            st.error("No videos found for this channel")
            return None, None, None, None, None, None

        # Convert to DataFrame
        df = pd.DataFrame(videos)
        df['published_at'] = pd.to_datetime(df['published_at'])
        df['views'] = pd.to_numeric(df['views'])
        df['likes'] = pd.to_numeric(df['likes'])
        df['comments'] = pd.to_numeric(df['comments'])

        # Add additional analytics
        df = youtube_api.analyze_engagement(df)
        df['sentiment'] = youtube_api.analyze_sentiment(df)
        clusters, cluster_terms = youtube_api.analyze_content_clusters(df)
        df['content_cluster'] = clusters
        trending_topics = youtube_api.get_trending_topics(df)
        best_hours, best_days = youtube_api.analyze_upload_patterns(df)

        result = (channel_stats, df, cluster_terms, trending_topics, best_hours, best_days)
        set_cached_channel_data(channel_name, result)
        return result
    except Exception as e:
        logger.error(f"Error analyzing channel: {str(e)}")
        st.error(f"An error occurred while analyzing the channel: {str(e)}")
        return None, None, None, None, None, None

def add_regression_line(fig, x, y):
    """Add regression line to scatter plot with enhanced styling"""
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    line_x = np.array([min(x), max(x)])
    line_y = slope * line_x + intercept
    
    fig.add_trace(go.Scatter(
        x=line_x,
        y=line_y,
        mode='lines',
        name=f'Trend Line (R² = {r_value**2:.3f})',
        line=dict(color='#FF0000', dash='dash', width=2)
    ))
    return fig

def create_metric_card(title, value, delta=None):
    """Create a styled metric card"""
    return f"""
    <div class="metric-card">
        <h4 style="color: #666; margin-bottom: 0.5rem;">{title}</h4>
        <h2 style="color: #FF0000; margin: 0;">{value}</h2>
        {f'<p style="color: #28a745; margin: 0;">▲ {delta}</p>' if delta and float(delta.replace('%', '')) > 0 else ''}
        {f'<p style="color: #dc3545; margin: 0;">▼ {delta}</p>' if delta and float(delta.replace('%', '')) < 0 else ''}
    </div>
    """

def format_number(num):
    """Format large numbers for better readability"""
    if num >= 1_000_000:
        return f"{num/1_000_000:.1f}M"
    elif num >= 1_000:
        return f"{num/1_000:.1f}K"
    return str(num)

def display_channel_suggestions(channels):
    if not channels:
        return None
    
    st.write("### Suggested Channels:")
    for channel in channels:
        col1, col2 = st.columns([1, 4])
        with col1:
            st.image(channel['thumbnail'], width=50)
        with col2:
            if st.button(f"Select: {channel['name']}", key=channel['channel_id']):
                return channel['name']
    return None

def display_content_analysis(df):
    st.markdown("""
        <div class="chart-container">
            <h3 style="margin-bottom: 1rem;">📊 Content Analysis</h3>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Video Duration Distribution
        durations = pd.cut(df['duration_minutes'], 
                          bins=[0, 5, 10, 15, 20, 30, float('inf')],
                          labels=['0-5', '5-10', '10-15', '15-20', '20-30', '30+'])
        duration_dist = durations.value_counts().sort_index()
        
        fig_duration = go.Figure(data=[
            go.Bar(
                x=duration_dist.index,
                y=duration_dist.values,
                marker_color='#FF0000'
            )
        ])
        
        fig_duration.update_layout(
            title=dict(
                text='Video Duration Distribution (minutes)',
                x=0.5,
                y=0.95,
                xanchor='center',
                yanchor='top',
                font=dict(size=16, color='#1a1a1a')
            ),
            paper_bgcolor='white',
            plot_bgcolor='#fafafa',
            xaxis=dict(title='Duration Range'),
            yaxis=dict(title='Number of Videos'),
            margin=dict(l=40, r=20, t=60, b=40),
            showlegend=False
        )
        st.plotly_chart(fig_duration, use_container_width=True)
    
    with col2:
        # Title Word Cloud
        text = ' '.join(df['title'].astype(str))
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            colormap='Reds',
            max_words=100
        ).generate(text)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)

def display_sentiment_analysis(df):
    st.markdown("""
        <div class="chart-container">
            <h3 style="margin-bottom: 1rem;">😊 Sentiment Analysis</h3>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Sentiment Distribution
        sentiment_bins = pd.cut(df['sentiment'], 
                              bins=[-1, -0.33, 0.33, 1],
                              labels=['Negative', 'Neutral', 'Positive'])
        sentiment_dist = sentiment_bins.value_counts()
        
        colors = {'Negative': '#ff4d4d', 'Neutral': '#ffd700', 'Positive': '#4CAF50'}
        
        fig_sentiment = go.Figure(data=[
            go.Pie(
                labels=sentiment_dist.index,
                values=sentiment_dist.values,
                marker=dict(colors=[colors[label] for label in sentiment_dist.index]),
                hole=0.4
            )
        ])
        
        fig_sentiment.update_layout(
            title=dict(
                text='Sentiment Distribution',
                x=0.5,
                y=0.95,
                xanchor='center',
                yanchor='top',
                font=dict(size=16, color='#1a1a1a')
            ),
            paper_bgcolor='white',
            showlegend=True,
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=-0.2,
                xanchor='center',
                x=0.5
            ),
            margin=dict(l=20, r=20, t=60, b=60)
        )
        st.plotly_chart(fig_sentiment, use_container_width=True)
    
    with col2:
        # Sentiment vs. Engagement
        fig_scatter = go.Figure(data=[
            go.Scatter(
                x=df['sentiment'],
                y=df['engagement_rate'],
                mode='markers',
                marker=dict(
                    size=8,
                    color=df['viewCount'],
                    colorscale='Reds',
                    showscale=True,
                    colorbar=dict(title='Views')
                ),
                hovertemplate='<b>Sentiment:</b> %{x:.2f}<br>' +
                             '<b>Engagement:</b> %{y:.2f}%<br>' +
                             '<b>Views:</b> %{marker.color:,.0f}<extra></extra>'
            )
        ])
        
        fig_scatter.update_layout(
            title=dict(
                text='Sentiment vs. Engagement',
                x=0.5,
                y=0.95,
                xanchor='center',
                yanchor='top',
                font=dict(size=16, color='#1a1a1a')
            ),
            paper_bgcolor='white',
            plot_bgcolor='#fafafa',
            xaxis=dict(
                title='Sentiment Score',
                showgrid=True,
                gridcolor='#f0f0f0',
                range=[-1, 1]
            ),
            yaxis=dict(
                title='Engagement Rate (%)',
                showgrid=True,
                gridcolor='#f0f0f0'
            ),
            margin=dict(l=40, r=20, t=60, b=40)
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    # Display top positive and negative videos
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
            <div style="background-color: white; padding: 1rem; border-radius: 8px; border: 1px solid #f0f0f0;">
                <h4 style="margin: 0 0 1rem 0; color: #1a1a1a;">Most Positive Videos</h4>
        """, unsafe_allow_html=True)
        
        top_positive = df.nlargest(3, 'sentiment')
        for _, video in top_positive.iterrows():
            st.markdown(f"""
                <div style="margin-bottom: 0.5rem; padding: 0.5rem; background-color: #f8f8f8; border-radius: 4px;">
                    <div style="color: #1a1a1a; font-weight: 500;">{video['title']}</div>
                    <div style="color: #4CAF50; font-size: 0.9em; margin-top: 0.25rem;">
                        Sentiment: {video['sentiment']:.2f}
                    </div>
                </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div style="background-color: white; padding: 1rem; border-radius: 8px; border: 1px solid #f0f0f0;">
                <h4 style="margin: 0 0 1rem 0; color: #1a1a1a;">Most Negative Videos</h4>
        """, unsafe_allow_html=True)
        
        top_negative = df.nsmallest(3, 'sentiment')
        for _, video in top_negative.iterrows():
            st.markdown(f"""
                <div style="margin-bottom: 0.5rem; padding: 0.5rem; background-color: #f8f8f8; border-radius: 4px;">
                    <div style="color: #1a1a1a; font-weight: 500;">{video['title']}</div>
                    <div style="color: #ff4d4d; font-size: 0.9em; margin-top: 0.25rem;">
                        Sentiment: {video['sentiment']:.2f}
                    </div>
                </div>
            """, unsafe_allow_html=True)

def main():
    # Title with YouTube-style branding
    st.markdown("""
        <div style="text-align: center; padding: 2rem 0;">
            <h1>📊 YouTube Analytics Dashboard</h1>
            <p style="font-size: 1.2em; color: #666;">Analyze any channel performance and get data-driven insights</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Add Help Section in Sidebar
    with st.sidebar:
        st.markdown("## 📚 Help & Instructions")
        
        with st.expander("How to Use This Dashboard"):
            st.markdown("""
            ### Getting Started
            1. **Enter a YouTube Channel Name**: Type the name of any YouTube channel in the search box
            2. **Select from Suggestions**: Choose from the auto-suggested channels
            3. **View Analysis**: The dashboard will automatically load and display the channel's analytics

            ### Understanding the Dashboard
            The dashboard is divided into several sections:

            #### 📈 Channel Overview
            - **Key Metrics**: View subscriber count, total views, and video count
            - **Performance Trends**: See how the channel has grown over time

            #### 🎯 Content Analysis
            - **Video Categories**: Distribution of video types
            - **Content Clusters**: Grouping of similar content based on titles and descriptions
            - **Word Cloud**: Most common words in video titles

            #### 📊 Engagement Metrics
            - **View Patterns**: Best days and times for posting
            - **Engagement Rates**: Like and comment ratios
            - **Performance Predictions**: AI-powered predictions for future video performance

            #### 📉 Trend Analysis
            - **Popular Topics**: Current trending subjects
            - **Category Performance**: How different video categories perform
            - **Competitor Analysis**: Compare with similar channels

            ### Interpreting the Results
            - **Green Arrows (↑)**: Indicate positive trends or above-average performance
            - **Red Arrows (↓)**: Indicate negative trends or below-average performance
            - **Blue Lines**: Show predicted values or trends
            - **Orange Lines**: Show actual values or historical data

            ### Tips for Better Analysis
            1. **Compare Time Periods**: Look at different time ranges to spot trends
            2. **Check Multiple Metrics**: Don't rely on a single metric for decisions
            3. **Use Predictions**: The AI models can help forecast future performance
            4. **Monitor Competitors**: Compare your channel with similar ones

            ### Data Sources
            - All data is fetched directly from the YouTube Data API
            - Analysis is performed using machine learning models
            - Results are cached for faster loading

            ### Need More Help?
            - Hover over charts for detailed information
            - Click on data points to see specific values
            - Use the sidebar filters to customize your view
            """)

        with st.expander("Understanding the Metrics"):
            st.markdown("""
            ### Key Performance Indicators (KPIs)

            #### Views
            - **Total Views**: Overall reach of your channel
            - **Average Views**: Typical performance per video
            - **View Growth**: Rate of view increase over time

            #### Engagement
            - **Engagement Rate**: (Likes + Comments) / Views
            - **Like Rate**: Likes / Views
            - **Comment Rate**: Comments / Views

            #### Content Performance
            - **Video Duration**: Optimal length for your audience
            - **Upload Frequency**: Best days and times to post
            - **Category Performance**: Which types of videos perform best

            #### Audience Insights
            - **Subscriber Growth**: Rate of new subscribers
            - **Viewer Retention**: How long viewers watch your videos
            - **Demographics**: Age and location of your audience

            ### How to Improve Your Channel
            1. **Content Strategy**
               - Focus on your best-performing categories
               - Maintain consistent upload schedule
               - Optimize video length based on performance

            2. **Engagement**
               - Respond to comments regularly
               - Encourage viewer interaction
               - Use calls-to-action effectively

            3. **Technical Optimization**
               - Use relevant tags and descriptions
               - Create eye-catching thumbnails
               - Optimize video titles for search

            4. **Growth Strategy**
               - Collaborate with similar channels
               - Cross-promote your content
               - Stay updated with trending topics
            """)

        with st.expander("Troubleshooting"):
            st.markdown("""
            ### Common Issues and Solutions

            #### Data Loading Problems
            - **No Data Found**: Check if the channel name is correct
            - **Slow Loading**: Data is being fetched from YouTube API
            - **Missing Metrics**: Some channels may have limited public data

            #### Analysis Issues
            - **Incomplete Analysis**: Ensure you have enough videos for meaningful analysis
            - **Prediction Errors**: Models need sufficient historical data
            - **Cache Problems**: Try refreshing the page

            #### API Limitations
            - **Rate Limits**: YouTube API has daily quotas
            - **Data Freshness**: Some metrics may be delayed
            - **Privacy Settings**: Private videos won't be included

            ### Getting Support
            - Check the YouTube API documentation
            - Review the error messages in the console
            - Contact support if issues persist
            """)
    
    # Initialize session state for channel suggestions
    if 'suggestions' not in st.session_state:
        st.session_state.suggestions = []
    if 'last_query' not in st.session_state:
        st.session_state.last_query = ""
    if 'last_suggestions_time' not in st.session_state:
        st.session_state.last_suggestions_time = 0
    
    # Search box with styling
    st.markdown("""
        <div style="max-width: 600px; margin: 0 auto;">
            <p style="text-align: center; color: #666; margin-bottom: 1rem;">
                Enter a YouTube channel name to analyze
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    channel_name = st.text_input(
        "Channel Name",
        key="channel_input",
        placeholder="Enter YouTube Channel Name",
        label_visibility="collapsed"
    )
    
    # Update suggestions if the input has changed
    current_time = time.time()
    if channel_name != st.session_state.last_query and len(channel_name) >= 2:
        if current_time - st.session_state.last_suggestions_time > 1:
            try:
                st.session_state.suggestions = youtube_api.search_channels(channel_name)
                st.session_state.last_suggestions_time = current_time
            except Exception as e:
                logger.error(f"Error fetching channel suggestions: {str(e)}")
                st.session_state.suggestions = []
    
    st.session_state.last_query = channel_name
    
    # Display suggestions with enhanced styling
    if channel_name and len(channel_name) >= 2:
        st.markdown("""
            <div style="max-width: 600px; margin: 0 auto;">
                <h3 style="color: #666; font-size: 1.1em; margin: 1rem 0;">Suggested Channels</h3>
            </div>
        """, unsafe_allow_html=True)
        
        for channel in st.session_state.suggestions:
            col1, col2 = st.columns([1, 5])
            with col1:
                st.image(channel['thumbnail'], width=50)
            with col2:
                if st.button(f"📺 {channel['name']}", key=channel['channel_id']):
                    channel_name = channel['name']
    
    if channel_name:
        with st.spinner("📊 Analyzing channel data..."):
            channel_stats, df, cluster_terms, trending_topics, best_hours, best_days = analyze_channel(channel_name)
            
            if channel_stats and df is not None:
                # Channel Overview Section
                st.markdown("""
                    <div style="margin: 2rem 0;">
                        <h2 style="text-align: center; color: #1E1E1E;">Channel Overview</h2>
                    </div>
                """, unsafe_allow_html=True)
                
                # Channel metrics with enhanced styling
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.markdown(create_metric_card(
                        "Channel Name",
                        channel_stats['channel_name']
                    ), unsafe_allow_html=True)
                with col2:
                    st.markdown(create_metric_card(
                        "Subscribers",
                        format_number(int(channel_stats['subscribers']))
                    ), unsafe_allow_html=True)
                with col3:
                    st.markdown(create_metric_card(
                        "Total Views",
                        format_number(int(channel_stats['views']))
                    ), unsafe_allow_html=True)
                with col4:
                    st.markdown(create_metric_card(
                        "Total Videos",
                        format_number(int(channel_stats['total_videos']))
                    ), unsafe_allow_html=True)

                # Create tabs with enhanced styling
                tabs = st.tabs([
                    "📈 Views Trends",
                    "👥 Engagement",
                    "📝 Content",
                    "⏰ Upload Patterns",
                    "🎯 Predictions",
                    "📊 Trends",
                    "🏆 Top Videos",
                    "⚙️ MLflow Tracking"
                ])

                with tabs[0]:
                    st.markdown("""
                        <div class="chart-container">
                            <h2>Views Over Time</h2>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    fig_views = px.line(df, x='published_at', y='views',
                                      title='Channel Growth',
                                      labels={'published_at': 'Publication Date', 'views': 'Views'})
                    fig_views.update_layout(
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        title_x=0.5,
                        title_font_size=20,
                        showlegend=True,
                        hovermode='x unified',
                        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='#f0f0f0'),
                        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='#f0f0f0')
                    )
                    fig_views.update_traces(line_color='#FF0000', line_width=2)
                    st.plotly_chart(fig_views, use_container_width=True)

                    # Add rolling averages
                    df['7_day_avg'] = df['views'].rolling(window=7).mean()
                    df['30_day_avg'] = df['views'].rolling(window=30).mean()
                    
                    fig_avg = go.Figure()
                    fig_avg.add_trace(go.Scatter(x=df['published_at'], y=df['views'],
                                               name='Daily Views', line=dict(color='#FF0000', width=1)))
                    fig_avg.add_trace(go.Scatter(x=df['published_at'], y=df['7_day_avg'],
                                               name='7-Day Average', line=dict(color='#28a745', width=2)))
                    fig_avg.add_trace(go.Scatter(x=df['published_at'], y=df['30_day_avg'],
                                               name='30-Day Average', line=dict(color='#007bff', width=2)))
                    
                    fig_avg.update_layout(
                        title='View Trends with Moving Averages',
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        title_x=0.5,
                        title_font_size=20,
                        showlegend=True,
                        hovermode='x unified',
                        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='#f0f0f0'),
                        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='#f0f0f0')
                    )
                    st.plotly_chart(fig_avg, use_container_width=True)

                with tabs[1]:
                    st.markdown("""
                        <div class="chart-container">
                            <h2>Engagement Analysis</h2>
                            <p style="color: #666;">Understand how your audience interacts with your content</p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Engagement Overview
                    eng_metrics = {
                        "Average Engagement Rate": f"{df['engagement_rate'].mean():.2f}%",
                        "Average Like Rate": f"{df['like_rate'].mean():.2f}%",
                        "Average Comment Rate": f"{df['comment_rate'].mean():.2f}%",
                        "Most Engaging Video": df.loc[df['engagement_rate'].idxmax(), 'title']
                    }
                    
                    eng_cols = st.columns(4)
                    for i, (metric, value) in enumerate(eng_metrics.items()):
                        with eng_cols[i]:
                            st.markdown(create_metric_card(metric, value), unsafe_allow_html=True)
                    
                    # Engagement Trends
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        fig_engagement = px.scatter(df, x='views', y='engagement_rate',
                                                title='Views vs Engagement Rate',
                                                labels={'views': 'Views', 'engagement_rate': 'Engagement Rate (%)'},
                                                color='engagement_rate',
                                                color_continuous_scale='Reds')
                        fig_engagement = add_regression_line(fig_engagement, df['views'], df['engagement_rate'])
                        fig_engagement.update_layout(
                            plot_bgcolor='white',
                            paper_bgcolor='white',
                            title_x=0.5,
                            showlegend=True,
                            coloraxis_showscale=True
                        )
                        st.plotly_chart(fig_engagement, use_container_width=True)
                    
                    with col2:
                        fig_likes = px.scatter(df, x='views', y='like_rate',
                                           title='Views vs Like Rate',
                                           labels={'views': 'Views', 'like_rate': 'Like Rate (%)'},
                                           color='like_rate',
                                           color_continuous_scale='Greens')
                        fig_likes = add_regression_line(fig_likes, df['views'], df['like_rate'])
                        fig_likes.update_layout(
                            plot_bgcolor='white',
                            paper_bgcolor='white',
                            title_x=0.5,
                            showlegend=True,
                            coloraxis_showscale=True
                        )
                        st.plotly_chart(fig_likes, use_container_width=True)
                    
                    with col3:
                        fig_comments = px.scatter(df, x='views', y='comment_rate',
                                              title='Views vs Comment Rate',
                                              labels={'views': 'Views', 'comment_rate': 'Comment Rate (%)'},
                                              color='comment_rate',
                                              color_continuous_scale='Blues')
                        fig_comments = add_regression_line(fig_comments, df['views'], df['comment_rate'])
                        fig_comments.update_layout(
                            plot_bgcolor='white',
                            paper_bgcolor='white',
                            title_x=0.5,
                            showlegend=True,
                            coloraxis_showscale=True
                        )
                        st.plotly_chart(fig_comments, use_container_width=True)

                with tabs[2]:
                    st.markdown("""
                        <div class="chart-container">
                            <h2>Content Analysis</h2>
                            <p style="color: #666;">Deep dive into your content performance and patterns</p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        # Enhanced Sentiment Analysis
                        st.markdown("""
                            <div class="chart-container">
                                <h3>Title Sentiment Analysis</h3>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        sentiment_df = pd.DataFrame({
                            'title': df['title'],
                            'sentiment': df['sentiment'],
                            'views': df['views']
                        })
                        
                        # Sentiment distribution with views correlation
                        fig_sentiment = px.scatter(sentiment_df,
                                               x='sentiment',
                                               y='views',
                                               title='Sentiment vs Views',
                                               color='sentiment',
                                               color_continuous_scale='RdYlGn',
                                               hover_data=['title'])
                        fig_sentiment.update_layout(
                            plot_bgcolor='white',
                            paper_bgcolor='white',
                            title_x=0.5,
                            showlegend=False
                        )
                        st.plotly_chart(fig_sentiment, use_container_width=True)
                        
                        # Sentiment breakdown
                        sentiment_cats = pd.cut(sentiment_df['sentiment'],
                                              bins=[-1, -0.1, 0.1, 1],
                                              labels=['Negative', 'Neutral', 'Positive'])
                        sentiment_stats = sentiment_cats.value_counts()
                        
                        fig_sentiment_pie = px.pie(
                            values=sentiment_stats.values,
                            names=sentiment_stats.index,
                            title='Sentiment Distribution',
                            color=sentiment_stats.index,
                            color_discrete_map={
                                'Positive': '#28a745',
                                'Neutral': '#ffc107',
                                'Negative': '#dc3545'
                            }
                        )
                        fig_sentiment_pie.update_layout(
                            title_x=0.5,
                            showlegend=True
                        )
                        st.plotly_chart(fig_sentiment_pie, use_container_width=True)
                    
                    with col2:
                        # Enhanced Content Clustering
                        st.markdown("""
                            <div class="chart-container">
                                <h3>Content Analysis</h3>
                                <p style="color: #666;">Content clustering analysis is currently disabled</p>
                            </div>
                        """, unsafe_allow_html=True)

                with tabs[3]:
                    # Upload Patterns
                    st.markdown("""
                        <div class="chart-container">
                            <h2 style="color: #1E1E1E; margin-bottom: 1rem;">⏰ Upload Patterns Analysis</h2>
                            <p style="color: #666; margin-bottom: 2rem;">Discover the best times to upload your videos based on historical performance</p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("""
                            <div class="chart-container">
                                <h3 style="color: #1E1E1E; margin-bottom: 1rem;">Best Upload Hours</h3>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        fig_hours = go.Figure()
                        fig_hours.add_trace(go.Bar(
                            x=best_hours['hour_label'],
                            y=best_hours['mean'],
                            text=[f"{int(v):,}" for v in best_hours['mean']],
                            textposition='auto',
                            hovertemplate="<b>Time:</b> %{x}<br>" +
                                        "<b>Avg Views:</b> %{y:,.0f}<br>" +
                                        "<b>Uploads:</b> %{customdata[0]}<br>" +
                                        "<b>Engagement:</b> %{customdata[1]:.2f}%<br>" +
                                        "<b>Share of Views:</b> %{customdata[2]:.1f}%",
                            customdata=list(zip(
                                best_hours['count'],
                                best_hours['engagement_rate'],
                                best_hours['view_percentage']
                            )),
                            marker_color='#FF0000'
                        ))
                        
                        fig_hours.update_layout(
                            title=dict(
                                text="Average Views by Upload Hour",
                                x=0.5,
                                y=0.95,
                                xanchor='center',
                                yanchor='top',
                                font=dict(size=16, color='#1a1a1a')
                            ),
                            xaxis=dict(
                                title="Hour of Day",
                                tickangle=45,
                                showgrid=True,
                                gridcolor='#f0f0f0'
                            ),
                            yaxis=dict(
                                title="Average Views",
                                showgrid=True,
                                gridcolor='#f0f0f0'
                            ),
                            plot_bgcolor='white',
                            paper_bgcolor='white',
                            showlegend=False,
                            margin=dict(t=60, b=80)
                        )
                        st.plotly_chart(fig_hours, use_container_width=True)
                        
                        # Top 3 hours details
                        st.markdown("#### Top 3 Upload Hours")
                        for _, row in best_hours.head(3).iterrows():
                            st.markdown(f"""
                                <div style="background-color: #f8f9fa; padding: 1rem; border-radius: 8px; margin-bottom: 0.5rem;">
                                    <div style="color: #1a1a1a; font-weight: 500; font-size: 1.1em;">
                                        {row['hour_label']} ({row['count']} uploads)
                                    </div>
                                    <div style="color: #666; margin-top: 0.5rem;">
                                        Average Views: {int(row['mean']):,}<br>
                                        Engagement Rate: {row['engagement_rate']:.2f}%<br>
                                        Share of Total Views: {row['view_percentage']}%
                                    </div>
                                </div>
                            """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown("""
                            <div class="chart-container">
                                <h3 style="color: #1E1E1E; margin-bottom: 1rem;">Best Upload Days</h3>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        fig_days = go.Figure()
                        fig_days.add_trace(go.Bar(
                            x=best_days['day_of_week'],
                            y=best_days['mean'],
                            text=[f"{int(v):,}" for v in best_days['mean']],
                            textposition='auto',
                            hovertemplate="<b>Day:</b> %{x}<br>" +
                                        "<b>Avg Views:</b> %{y:,.0f}<br>" +
                                        "<b>Uploads:</b> %{customdata[0]}<br>" +
                                        "<b>Engagement:</b> %{customdata[1]:.2f}%<br>" +
                                        "<b>Share of Views:</b> %{customdata[2]:.1f}%",
                            customdata=list(zip(
                                best_days['count'],
                                best_days['engagement_rate'],
                                best_days['view_percentage']
                            )),
                            marker_color='#4CAF50'
                        ))
                        
                        fig_days.update_layout(
                            title=dict(
                                text="Average Views by Upload Day",
                                x=0.5,
                                y=0.95,
                                xanchor='center',
                                yanchor='top',
                                font=dict(size=16, color='#1a1a1a')
                            ),
                            xaxis=dict(
                                title="Day of Week",
                                showgrid=True,
                                gridcolor='#f0f0f0'
                            ),
                            yaxis=dict(
                                title="Average Views",
                                showgrid=True,
                                gridcolor='#f0f0f0'
                            ),
                            plot_bgcolor='white',
                            paper_bgcolor='white',
                            showlegend=False,
                            margin=dict(t=60, b=40)
                        )
                        st.plotly_chart(fig_days, use_container_width=True)
                        
                        # Top 3 days details
                        st.markdown("#### Top 3 Upload Days")
                        for _, row in best_days.head(3).iterrows():
                            st.markdown(f"""
                                <div style="background-color: #f8f9fa; padding: 1rem; border-radius: 8px; margin-bottom: 0.5rem;">
                                    <div style="color: #1a1a1a; font-weight: 500; font-size: 1.1em;">
                                        {row['day_of_week']} ({row['count']} uploads)
                                    </div>
                                    <div style="color: #666; margin-top: 0.5rem;">
                                        Average Views: {int(row['mean']):,}<br>
                                        Engagement Rate: {row['engagement_rate']:.2f}%<br>
                                        Share of Total Views: {row['view_percentage']}%
                                    </div>
                                </div>
                            """, unsafe_allow_html=True)
                    
                    # Additional insights
                    st.markdown("""
                        <div class="chart-container" style="margin-top: 2rem;">
                            <h3 style="color: #1E1E1E; margin-bottom: 1rem;">📈 Upload Pattern Insights</h3>
                            <ul style="color: #666; margin: 0; padding-left: 1.2rem;">
                    """, unsafe_allow_html=True)
                    
                    if not best_hours.empty and not best_days.empty:
                        best_hour = best_hours.iloc[0]
                        best_day = best_days.iloc[0]
                        st.markdown(f"""
                            <li style="margin-bottom: 0.5rem;">Best performing uploads tend to be during {best_hour['hour_label']} with an average of {int(best_hour['mean']):,} views</li>
                            <li style="margin-bottom: 0.5rem;">{best_day['day_of_week']} shows the highest average performance with {int(best_day['mean']):,} views</li>
                            <li style="margin-bottom: 0.5rem;">Videos uploaded during peak hours show {best_hour['engagement_rate']:.1f}% engagement rate</li>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                            <li style="margin-bottom: 0.5rem;">Not enough data available to analyze upload patterns</li>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("""
                            </ul>
                        </div>
                    """, unsafe_allow_html=True)

                with tabs[4]:
                    st.markdown("""
                        <div class="chart-container">
                            <h2 style="color: #1E1E1E; margin-bottom: 1rem;">🎯 Predictions & Insights</h2>
                            <p style="color: #666; margin-bottom: 2rem;">Machine learning-powered predictions and insights to optimize your content strategy</p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # View Prediction Model
                    st.markdown("""
                        <div class="chart-container">
                            <h3 style="color: #1E1E1E; margin-bottom: 1rem;">📈 View Performance Predictor</h3>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    view_model = youtube_api.train_view_predictor(df)
                    if view_model:
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            # Feature importance visualization
                            fig_view_importance = go.Figure()
                            fig_view_importance.add_trace(go.Bar(
                                x=view_model['feature_importance']['importance'],
                                y=view_model['feature_importance']['feature'],
                                orientation='h',
                                marker_color='#FF0000'
                            ))
                            
                            fig_view_importance.update_layout(
                                title=dict(
                                    text='Feature Importance for View Prediction',
                                    x=0.5,
                                    y=0.95,
                                    xanchor='center',
                                    yanchor='top',
                                    font=dict(size=16, color='#1a1a1a')
                                ),
                                xaxis_title="Importance Score",
                                yaxis_title="Feature",
                                plot_bgcolor='white',
                                paper_bgcolor='white',
                                yaxis={'categoryorder': 'total ascending'},
                                margin=dict(l=20, r=20, t=40, b=20),
                                height=400
                            )
                            st.plotly_chart(fig_view_importance, use_container_width=True)
                        
                        with col2:
                            # Model performance metrics
                            st.markdown("""
                                <div style="background-color: #f8f9fa; padding: 1.5rem; border-radius: 8px; margin-bottom: 1rem;">
                                    <h4 style="color: #1a1a1a; margin-bottom: 1rem;">Model Performance</h4>
                                    <div style="color: #666;">
                                        <p style="margin-bottom: 0.5rem;">
                                            <span style="font-weight: 500;">R² Score:</span><br/>
                                            <span style="font-size: 1.5em; color: #FF0000;">{:.3f}</span>
                                        </p>
                                        <p style="margin-bottom: 0.5rem;">
                                            <span style="font-weight: 500;">RMSE:</span><br/>
                                            <span style="font-size: 1.5em; color: #1a1a1a;">{:,.0f}</span> views
                                        </p>
                                    </div>
                                </div>
                            """.format(view_model['r2_score'], view_model['rmse']), unsafe_allow_html=True)
                            
                            # Top predictive features
                            st.markdown("""
                                <div style="background-color: #f8f9fa; padding: 1.5rem; border-radius: 8px;">
                                    <h4 style="color: #1a1a1a; margin-bottom: 1rem;">Top Predictive Features</h4>
                                    <div style="color: #666;">
                            """, unsafe_allow_html=True)
                            
                            for _, row in view_model['feature_importance'].head(5).iterrows():
                                st.markdown(f"""
                                    <div style="margin-bottom: 0.5rem;">
                                        <div style="font-weight: 500;">{row['feature']}</div>
                                        <div style="color: #FF0000;">Importance: {row['importance']:.3f}</div>
                                    </div>
                                """, unsafe_allow_html=True)
                            
                            st.markdown("</div></div>", unsafe_allow_html=True)
                    
                    # Engagement Prediction Model
                    st.markdown("""
                        <div class="chart-container" style="margin-top: 2rem;">
                            <h3 style="color: #1E1E1E; margin-bottom: 1rem;">👥 Engagement Predictor</h3>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    engagement_model = youtube_api.predict_engagement(df)
                    if engagement_model:
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            # Feature importance visualization
                            fig_eng_importance = go.Figure()
                            fig_eng_importance.add_trace(go.Bar(
                                x=engagement_model['feature_importance']['importance'],
                                y=engagement_model['feature_importance']['feature'],
                                orientation='h',
                                marker_color='#4CAF50'
                            ))
                            
                            fig_eng_importance.update_layout(
                                title=dict(
                                    text='Feature Importance for Engagement Prediction',
                                    x=0.5,
                                    y=0.95,
                                    xanchor='center',
                                    yanchor='top',
                                    font=dict(size=16, color='#1a1a1a')
                                ),
                                xaxis_title="Importance Score",
                                yaxis_title="Feature",
                                plot_bgcolor='white',
                                paper_bgcolor='white',
                                yaxis={'categoryorder': 'total ascending'},
                                margin=dict(l=20, r=20, t=40, b=20),
                                height=400
                            )
                            st.plotly_chart(fig_eng_importance, use_container_width=True)
                        
                        with col2:
                            # Model performance metrics
                            st.markdown("""
                                <div style="background-color: #f8f9fa; padding: 1.5rem; border-radius: 8px; margin-bottom: 1rem;">
                                    <h4 style="color: #1a1a1a; margin-bottom: 1rem;">Model Performance</h4>
                                    <div style="color: #666;">
                                    </div>
                                </div>
                            """, unsafe_allow_html=True)
                            
                            # Top predictive features
                            st.markdown("""
                                <div style="background-color: #f8f9fa; padding: 1.5rem; border-radius: 8px;">
                                    <h4 style="color: #1a1a1a; margin-bottom: 1rem;">Top Predictive Features</h4>
                                    <div style="color: #666;">
                            """, unsafe_allow_html=True)
                            
                            for _, row in engagement_model['feature_importance'].head(5).iterrows():
                                st.markdown(f"""
                                    <div style="margin-bottom: 0.5rem;">
                                        <div style="font-weight: 500;">{row['feature']}</div>
                                        <div style="color: #4CAF50;">Importance: {row['importance']:.3f}</div>
                                    </div>
                                """, unsafe_allow_html=True)
                            
                            st.markdown("</div></div>", unsafe_allow_html=True)
                    
                    # Optimization Recommendations
                    st.markdown("""
                        <div class="chart-container" style="margin-top: 2rem;">
                            <h3 style="color: #1E1E1E; margin-bottom: 1rem;">🎯 Content Optimization Recommendations</h3>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("""
                            <div style="background-color: #f8f9fa; padding: 1.5rem; border-radius: 8px;">
                                <h4 style="color: #1a1a1a; margin-bottom: 1rem;">Views Optimization</h4>
                                <ul style="color: #666; margin: 0; padding-left: 1.2rem;">
                        """, unsafe_allow_html=True)
                        
                        if view_model:
                            top_features = view_model['feature_importance'].head(3)
                            for _, feature in top_features.iterrows():
                                st.markdown(f"""
                                    <li style="margin-bottom: 0.5rem;">
                                        Focus on optimizing <strong>{feature['feature']}</strong> 
                                        (Impact Score: {feature['importance']:.3f})
                                    </li>
                                """, unsafe_allow_html=True)
                        
                        st.markdown("</ul></div>", unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown("""
                            <div style="background-color: #f8f9fa; padding: 1.5rem; border-radius: 8px;">
                                <h4 style="color: #1a1a1a; margin-bottom: 1rem;">Engagement Optimization</h4>
                                <ul style="color: #666; margin: 0; padding-left: 1.2rem;">
                        """, unsafe_allow_html=True)
                        
                        if engagement_model:
                            top_features = engagement_model['feature_importance'].head(3)
                            for _, feature in top_features.iterrows():
                                st.markdown(f"""
                                    <li style="margin-bottom: 0.5rem;">
                                        Focus on optimizing <strong>{feature['feature']}</strong> 
                                        (Impact Score: {feature['importance']:.3f})
                                    </li>
                                """, unsafe_allow_html=True)
                        
                        st.markdown("</ul></div>", unsafe_allow_html=True)
                    
                    # Additional Insights
                    # if view_model and engagement_model:
                    #     st.markdown("""
                    #         <div style="background-color: #f8f9fa; padding: 1.5rem; border-radius: 8px; margin-top: 1rem;">
                    #             <h4 style="color: #1a1a1a; margin-bottom: 1rem;">📊 Key Insights</h4>
                    #             <ul style="color: #666; margin: 0; padding-left: 1.2rem;">
                    #                 <li style="margin-bottom: 0.5rem;">
                    #                     The view prediction model explains {view_model['r2_score']*100:.1f}% of the variance in view counts
                    #                 </li>
                    #                 <li style="margin-bottom: 0.5rem;">
                    #                     The engagement prediction model explains {engagement_model['r2_score']*100:.1f}% of the variance in engagement rates
                    #                 </li>
                    #                 <li style="margin-bottom: 0.5rem;">
                    #                     Common important features between both models suggest focusing on content timing and quality
                    #                 </li>
                    #             </ul>
                    #         </div>
                    #     """, unsafe_allow_html=True)

                with tabs[5]:
                    st.markdown("""
                        <div class="chart-container">
                            <h2 style="color: #1E1E1E; margin-bottom: 1rem;">📈 Trend Analysis</h2>
                            <p style="color: #666; margin-bottom: 2rem;">Analyze current YouTube trends and optimize your content strategy</p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Region selection with improved UI
                    st.markdown("""
                        <div style="background-color: #f8f9fa; padding: 1.5rem; border-radius: 8px; margin-bottom: 2rem;">
                            <h4 style="color: #1a1a1a; margin-bottom: 1rem;">🌎 Select Region</h4>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    regions = {
                        'US': '🇺🇸 United States',
                        'GB': '🇬🇧 United Kingdom',
                        'CA': '🇨🇦 Canada',
                        'AU': '🇦🇺 Australia',
                        'IN': '🇮🇳 India',
                        'JP': '🇯🇵 Japan',
                        'KR': '🇰🇷 South Korea',
                        'DE': '🇩🇪 Germany',
                        'FR': '🇫🇷 France',
                        'BR': '🇧🇷 Brazil'
                    }
                    
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        selected_region = st.selectbox(
                            "Select Region",
                            options=list(regions.keys()),
                            format_func=lambda x: regions[x],
                            key="region_selector"
                        )
                    
                    # Fetch trending videos with loading state
                    with st.spinner("📊 Analyzing trending videos..."):
                        trending_videos = youtube_api.get_trending_videos(region_code=selected_region)
                        
                        if trending_videos:
                            trend_analysis = youtube_api.analyze_trends(trending_videos)
                            
                            if trend_analysis:
                                # Overview metrics
                                st.markdown("""
                                    <div class="chart-container">
                                        <h3 style="color: #1E1E1E; margin-bottom: 1rem;">📊 Trend Overview</h3>
                                    </div>
                                """, unsafe_allow_html=True)
                                
                                col1, col2, col3, col4 = st.columns(4)
                                metrics = [
                                    {
                                        "title": "Trending Videos",
                                        "value": trend_analysis['total_videos'],
                                        "format": "{:,}",
                                        "prefix": "📹"
                                    },
                                    {
                                        "title": "Average Views",
                                        "value": trend_analysis['avg_views'],
                                        "format": "{:,.0f}",
                                        "prefix": "👁️"
                                    },
                                    {
                                        "title": "Engagement Rate",
                                        "value": trend_analysis['avg_engagement'],
                                        "format": "{:.2f}%",
                                        "prefix": "❤️"
                                    },
                                    {
                                        "title": "Avg Duration",
                                        "value": trend_analysis['avg_duration'],
                                        "format": "{:.1f} min",
                                        "prefix": "⏱️"
                                    }
                                ]
                                
                                for metric, col in zip(metrics, [col1, col2, col3, col4]):
                                    with col:
                                        st.markdown(f"""
                                            <div style="background-color: #f8f9fa; padding: 1.5rem; border-radius: 8px; text-align: center;">
                                                <div style="font-size: 2em; margin-bottom: 0.5rem;">{metric['prefix']}</div>
                                                <div style="color: #666; font-size: 0.9em;">{metric['title']}</div>
                                                <div style="color: #FF0000; font-size: 1.5em; font-weight: 500; margin-top: 0.5rem;">
                                                    {metric['format'].format(metric['value'])}
                                                </div>
                                            </div>
                                        """, unsafe_allow_html=True)
                                
                                # Timing Analysis
                                st.markdown("""
                                    <div class="chart-container" style="margin-top: 2rem;">
                                        <h3 style="color: #1E1E1E; margin-bottom: 1rem;">⏰ Upload Timing Patterns</h3>
                                    </div>
                                """, unsafe_allow_html=True)
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.markdown("""
                                        <div style="background-color: #f8f9fa; padding: 1.5rem; border-radius: 8px;">
                                            <h4 style="color: #1a1a1a; margin-bottom: 1rem;">🕒 Popular Upload Hours</h4>
                                            <div style="color: #666;">
                                    """, unsafe_allow_html=True)
                                    
                                    for hour in trend_analysis['popular_hours']:
                                        st.markdown(f"""
                                            <div style="margin-bottom: 0.5rem; padding: 0.5rem; background-color: white; border-radius: 4px;">
                                                <span style="font-size: 1.2em;">🕐</span> {hour:02d}:00 - {(hour+1):02d}:00
                                            </div>
                                        """, unsafe_allow_html=True)
                                    
                                    st.markdown("</div></div>", unsafe_allow_html=True)
                                
                                with col2:
                                    st.markdown("""
                                        <div style="background-color: #f8f9fa; padding: 1.5rem; border-radius: 8px;">
                                            <h4 style="color: #1a1a1a; margin-bottom: 1rem;">📅 Popular Upload Days</h4>
                                            <div style="color: #666;">
                                    """, unsafe_allow_html=True)
                                    
                                    day_emojis = {
                                        'Monday': '1️⃣', 'Tuesday': '2️⃣', 'Wednesday': '3️⃣',
                                        'Thursday': '4️⃣', 'Friday': '5️⃣', 'Saturday': '6️⃣',
                                        'Sunday': '7️⃣'
                                    }
                                    
                                    for day in trend_analysis['popular_days']:
                                        st.markdown(f"""
                                            <div style="margin-bottom: 0.5rem; padding: 0.5rem; background-color: white; border-radius: 4px;">
                                                <span style="font-size: 1.2em;">{day_emojis.get(day, '📅')}</span> {day}
                                            </div>
                                        """, unsafe_allow_html=True)
                                    
                                    st.markdown("</div></div>", unsafe_allow_html=True)
                                
                                # Content Analysis
                                st.markdown("""
                                    <div class="chart-container" style="margin-top: 2rem;">
                                        <h3 style="color: #1E1E1E; margin-bottom: 1rem;">📝 Content Analysis</h3>
                                        <p style="color: #666;">Content analysis is currently disabled</p>
                                    </div>
                                """, unsafe_allow_html=True)
                                
                                # Channel Performance
                                st.markdown("""
                                    <div class="chart-container" style="margin-top: 2rem;">
                                        <h3 style="color: #1E1E1E; margin-bottom: 1rem;">🏆 Top Trending Channels</h3>
                                        <p style="color: #666;">Top trending channels analysis is currently disabled</p>
                                    </div>
                                """, unsafe_allow_html=True)
                                
                                # Category Performance
                                st.markdown("""
                                    <div class="chart-container" style="margin-top: 2rem;">
                                        <h3 style="color: #1E1E1E; margin-bottom: 1rem;">📊 Category Performance</h3>
                                        <p style="color: #666;">Category performance analysis is currently disabled</p>
                                    </div>
                                """, unsafe_allow_html=True)
                                
                                # Key Insights
                                st.markdown("""
                                    <div style="background-color: #f8f9fa; padding: 1.5rem; border-radius: 8px; margin-top: 2rem;">
                                        <h4 style="color: #1a1a1a; margin-bottom: 1rem;">🔍 Key Insights</h4>
                                        <div style="color: #666;">
                                """, unsafe_allow_html=True)

                                # Format upload time
                                upload_hour = trend_analysis['popular_hours'][0]
                                upload_time = f"{upload_hour:02d}:00-{(upload_hour+1):02d}:00"

                                # Format title words
                                title_words = list(trend_analysis['common_title_words'].keys())
                                top_title_words = f"'{title_words[0]}'" if title_words else "N/A"
                                if len(title_words) > 1:
                                    top_title_words += f" and '{title_words[1]}'"

                                insights = [
                                    {
                                        "icon": "⏰",
                                        "text": f"Peak Upload Time: {upload_time}",
                                        "detail": "Most trending videos are published during this timeframe"
                                    },
                                    {
                                        "icon": "📅",
                                        "text": f"Best Day: {trend_analysis['popular_days'][0]}",
                                        "detail": "Shows the highest potential for trending content"
                                    },
                                    {
                                        "icon": "⌛",
                                        "text": f"Optimal Duration: {trend_analysis['avg_duration']:.1f} minutes",
                                        "detail": "Average length of trending videos"
                                    },
                                    {
                                        "icon": "📝",
                                        "text": f"Title Keywords: {top_title_words}",
                                        "detail": "Most frequent words in trending video titles"
                                    },
                                    {
                                        "icon": "👥",
                                        "text": f"Engagement Rate: {trend_analysis['avg_engagement']:.2f}%",
                                        "detail": "Average engagement rate for trending videos"
                                    }
                                ]

                                for insight in insights:
                                    st.markdown(f"""
                                        <div style="margin-bottom: 1rem; padding: 1rem; background-color: white; border-radius: 8px;">
                                            <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                                                <span style="font-size: 1.2em; margin-right: 0.5rem;">{insight['icon']}</span>
                                                <span style="color: #1a1a1a; font-weight: 500;">{insight['text']}</span>
                                            </div>
                                            <div style="color: #666; font-size: 0.9em; margin-left: 2rem;">
                                                {insight['detail']}
                                            </div>
                                        </div>
                                    """, unsafe_allow_html=True)

                                st.markdown("</div></div>", unsafe_allow_html=True)
                        else:
                            st.error("Unable to fetch trending videos. Please try again later.")

                with tabs[6]:
                    st.markdown("""
                        <div class="chart-container">
                            <h2 style="color: #1E1E1E; margin-bottom: 1rem;">🏆 Top Performing Videos</h2>
                            <p style="color: #666; margin-bottom: 2rem;">Analyze your most successful content and understand what makes them perform well</p>
                        </div>
                    """, unsafe_allow_html=True)

                    # Overview metrics
                    if not df.empty:
                        # Convert published_at to timezone-naive datetime
                        df['published_at'] = pd.to_datetime(df['published_at']).dt.tz_localize(None)
                        
                        top_10_views = df.nlargest(10, 'views')
                        total_views = top_10_views['views'].sum()
                        avg_engagement = top_10_views['engagement_rate'].mean()
                        avg_duration = top_10_views['duration_minutes'].mean()
                        
                        col1, col2, col3 = st.columns(3)
                        metrics = [
                            {
                                "title": "Total Views",
                                "value": total_views,
                                "format": "{:,.0f}",
                                "prefix": "👁️"
                            },
                            {
                                "title": "Average Engagement",
                                "value": avg_engagement,
                                "format": "{:.2f}%",
                                "prefix": "❤️"
                            },
                            {
                                "title": "Average Duration",
                                "value": avg_duration,
                                "format": "{:.1f} min",
                                "prefix": "⏱️"
                            }
                        ]
                        
                        for metric, col in zip(metrics, [col1, col2, col3]):
                            with col:
                                st.markdown(f"""
                                    <div style="background-color: #f8f9fa; padding: 1.5rem; border-radius: 8px; text-align: center;">
                                        <div style="font-size: 2em; margin-bottom: 0.5rem;">{metric['prefix']}</div>
                                        <div style="color: #666; font-size: 0.9em;">{metric['title']}</div>
                                        <div style="color: #FF0000; font-size: 1.5em; font-weight: 500; margin-top: 0.5rem;">
                                            {metric['format'].format(metric['value'])}
                                        </div>
                                    </div>
                                """, unsafe_allow_html=True)

                        # Top videos list
                        st.markdown("""
                            <div style="margin-top: 2rem;">
                                <h3 style="color: #1E1E1E; margin-bottom: 1rem;">📈 Top 10 Videos</h3>
                            </div>
                        """, unsafe_allow_html=True)

                        for i, video in top_10_views.iterrows():
                            with st.expander(f"#{i + 1} {video['title']}", expanded=i == 0):
                                col1, col2 = st.columns([2, 1])
                                with col1:
                                    # Video embed with thumbnail preview
                                    video_url = youtube_api.get_video_url(video['video_id'])
                                    st.markdown(f"""
                                        <div style="position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden; border-radius: 8px; margin-bottom: 1rem;">
                                            <iframe 
                                                style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; border: none;"
                                                src="{video_url}"
                                                frameborder="0"
                                                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                                                allowfullscreen>
                                            </iframe>
                                        </div>
                                    """, unsafe_allow_html=True)
                                
                                with col2:
                                    # Video metrics
                                    metrics = [
                                        ("📅 Published", video['published_at'].strftime('%Y-%m-%d')),
                                        ("👁️ Views", f"{int(video['views']):,}"),
                                        ("❤️ Likes", f"{int(video['likes']):,}"),
                                        ("💬 Comments", f"{int(video['comments']):,}"),
                                        ("📊 Engagement", f"{video['engagement_rate']:.2f}%"),
                                        ("⏱️ Duration", f"{video['duration_minutes']:.1f} min")
                                    ]
                                    
                                    st.markdown("""
                                        <div style="background-color: #f8f9fa; padding: 1.5rem; border-radius: 8px;">
                                            <h4 style="color: #1a1a1a; margin-bottom: 1rem;">Video Performance</h4>
                                    """, unsafe_allow_html=True)
                                    
                                    for metric, value in metrics:
                                        st.markdown(f"""
                                            <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem; padding: 0.5rem; background-color: white; border-radius: 4px;">
                                                <span style="color: #666;">{metric}</span>
                                                <span style="color: #1a1a1a; font-weight: 500;">{value}</span>
                                            </div>
                                        """, unsafe_allow_html=True)
                                    
                                    # Sentiment indicator
                                    sentiment_color = '#4CAF50' if video['sentiment'] > 0 else ('#dc3545' if video['sentiment'] < 0 else '#ffc107')
                                    sentiment_label = '😊 Positive' if video['sentiment'] > 0 else ('😞 Negative' if video['sentiment'] < 0 else '😐 Neutral')
                                    
                                    st.markdown(f"""
                                        <div style="margin-top: 1rem;">
                                            <h4 style="color: #1a1a1a; margin-bottom: 0.5rem;">Sentiment Analysis</h4>
                                            <div style="background-color: white; padding: 1rem; border-radius: 4px; text-align: center;">
                                                <div style="font-size: 1.2em; margin-bottom: 0.5rem;">{sentiment_label}</div>
                                                <div style="color: {sentiment_color}; font-weight: 500;">
                                                    Score: {video['sentiment']:.2f}
                                                </div>
                                            </div>
                                        </div>
                                    """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                            <div style="margin-top: 2rem;">
                                <h3 style="color: #1E1E1E; margin-bottom: 1rem;">📈 Top 10 Videos</h3>
                                <p style="color: #666;">No video data available to display</p>
                            </div>
                        """, unsafe_allow_html=True)

                with tabs[7]:
                    st.markdown("""
                        <div class="chart-container">
                            <h2 style="color: #1E1E1E; margin-bottom: 1rem;">⚙️ MLflow Tracking</h2>
                            <p style="color: #666; margin-bottom: 2rem;">Track MLflow experiment runs and metrics</p>
                        </div>
                    """, unsafe_allow_html=True)

                    experiment_name = "YouTube_Analytics_Experiments"
                    latest_runs = get_latest_runs(experiment_name, n=10)

                    if not latest_runs.empty:
                        # Convert start_time to datetime without timezone for display
                        latest_runs['start_time'] = pd.to_datetime(latest_runs['start_time']).dt.tz_localize(None)
                        
                        # Overview metrics
                        latest_run = latest_runs.iloc[0]
                        metrics = latest_run.get('metrics', {})
                        
                        col1, col2, col3 = st.columns(3)
                        overview_metrics = [
                            {
                                "title": "Latest R² Score",
                                "value": metrics.get('r2', 0),
                                "format": "{:.3f}",
                                "prefix": "📊",
                                "color": "#4CAF50" if metrics.get('r2', 0) > 0.7 else "#FFC107"
                            },
                            {
                                "title": "RMSE",
                                "value": metrics.get('rmse', 0),
                                "format": "{:.2f}",
                                "prefix": "📉",
                                "color": "#4CAF50" if metrics.get('rmse', 0) < 1000 else "#FFC107"
                            },
                            {
                                "title": "Model Runs",
                                "value": len(latest_runs),
                                "format": "{}",
                                "prefix": "🔄",
                                "color": "#FF0000"
                            }
                        ]
                        
                        for metric, col in zip(overview_metrics, [col1, col2, col3]):
                            with col:
                                st.markdown(f"""
                                    <div style="background-color: #f8f9fa; padding: 1.5rem; border-radius: 8px; text-align: center;">
                                        <div style="font-size: 2em; margin-bottom: 0.5rem;">{metric['prefix']}</div>
                                        <div style="color: #666; font-size: 0.9em;">{metric['title']}</div>
                                        <div style="color: {metric['color']}; font-size: 1.5em; font-weight: 500; margin-top: 0.5rem;">
                                            {metric['format'].format(metric['value'])}
                                        </div>
                                    </div>
                                """, unsafe_allow_html=True)

                        # MLflow Runs Table
                        st.markdown("""
                            <div style="margin-top: 2rem;">
                                <h3 style="color: #1E1E1E; margin-bottom: 1rem;">🔍 MLflow Runs</h3>
                            </div>
                        """, unsafe_allow_html=True)

                        # Format the dataframe for display
                        display_df = latest_runs.copy()
                        display_df['start_time'] = display_df['start_time'].dt.strftime('%Y-%m-%d %H:%M:%S')
                        
                        # Style the dataframe
                        st.dataframe(
                            display_df[['run_id', 'start_time', 'metrics.r2', 'metrics.rmse']].rename(
                                columns={
                                    'run_id': 'Run ID',
                                    'start_time': 'Start Time',
                                    'metrics.r2': 'R² Score',
                                    'metrics.rmse': 'RMSE'
                                }
                            ),
                            use_container_width=True,
                            hide_index=True
                        )

                        # MLflow Run Details
                        st.markdown("""
                            <div style="margin-top: 2rem;">
                                <h3 style="color: #1E1E1E; margin-bottom: 1rem;">🔧 Run Details</h3>
                            </div>
                        """, unsafe_allow_html=True)

                        # Select run for details
                        selected_run = st.selectbox(
                            "Select Run",
                            options=latest_runs['run_id'].tolist(),
                            format_func=lambda x: f"Run {x} ({latest_runs[latest_runs['run_id']==x]['start_time'].iloc[0]})"
                        )

                        if selected_run:
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("""
                                    <div style="background-color: #f8f9fa; padding: 1.5rem; border-radius: 8px;">
                                        <h4 style="color: #1a1a1a; margin-bottom: 1rem;">⚙️ Parameters</h4>
                                """, unsafe_allow_html=True)
                                
                                params = get_run_params(selected_run)
                                if params:
                                    for param, value in params.items():
                                        st.markdown(f"""
                                            <div style="margin-bottom: 0.5rem; padding: 0.5rem; background-color: white; border-radius: 4px;">
                                                <div style="color: #666; font-size: 0.9em;">{param}</div>
                                                <div style="color: #1a1a1a; font-weight: 500;">{value}</div>
                                            </div>
                                        """, unsafe_allow_html=True)
                                else:
                                    st.info("No parameters found for this run")

                            with col2:
                                st.markdown("""
                                    <div style="background-color: #f8f9fa; padding: 1.5rem; border-radius: 8px;">
                                        <h4 style="color: #1a1a1a; margin-bottom: 1rem;">📊 Metrics</h4>
                                """, unsafe_allow_html=True)
                                
                                metrics = get_run_metrics(selected_run)
                                if metrics:
                                    for metric, value in metrics.items():
                                        st.markdown(f"""
                                            <div style="margin-bottom: 0.5rem; padding: 0.5rem; background-color: white; border-radius: 4px;">
                                                <div style="color: #666; font-size: 0.9em;">{metric}</div>
                                                <div style="color: #1a1a1a; font-weight: 500;">{value:.3f}</div>
                                            </div>
                                        """, unsafe_allow_html=True)
                                else:
                                    st.info("No metrics found for this run")
                    else:
                        st.info("No MLflow runs found for this experiment. Start training models to see tracking data.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        st.error("An unexpected error occurred. Please try again later.") 