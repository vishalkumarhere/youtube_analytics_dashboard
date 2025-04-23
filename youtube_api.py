from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
from logging_config import logger
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, classification_report
from textblob import TextBlob
import joblib
import re
import mlflow
import mlflow.sklearn
import json
import tempfile
import matplotlib.pyplot as plt
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import time
from functools import lru_cache
from typing import Dict, List, Optional, Tuple, Union
from model_monitoring_utils import get_latest_runs, get_run_params, get_run_metrics
from wordcloud import WordCloud

# Load environment variables
load_dotenv()

class YouTubeAPI:
    def __init__(self):
        mlflow.set_experiment("YouTube_Analytics_Experiments")

        self.api_key = os.getenv('YOUTUBE_API_KEY')
        if not self.api_key:
            raise ValueError("YouTube API key not found in environment variables")
        
        self.youtube = build('youtube', 'v3', developerKey=self.api_key)
        logger.info("YouTube API client initialized successfully")
        
        # Initialize ML models
        self.view_predictor = None
        self.engagement_predictor = None
        self.category_classifier = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        # Initialize caches
        self._channel_cache: Dict[str, Dict] = {}
        self._video_cache: Dict[str, Dict] = {}
        self._trending_cache: Dict[str, Dict] = {}
        self._cache_expiry = 3600  # 1 hour cache expiry
        
        # Rate limiting
        self._last_request_time = 0
        self._min_request_interval = 1.0  # Minimum 1 second between requests

    def _rate_limit(self):
        """Implement rate limiting to prevent API quota exhaustion"""
        current_time = time.time()
        time_since_last_request = current_time - self._last_request_time
        if time_since_last_request < self._min_request_interval:
            time.sleep(self._min_request_interval - time_since_last_request)
        self._last_request_time = time.time()

    def _get_cached_data(self, cache: Dict, key: str) -> Optional[Dict]:
        """Get data from cache if it exists and hasn't expired"""
        if key in cache:
            data, timestamp = cache[key]
            if time.time() - timestamp < self._cache_expiry:
                return data
            del cache[key]
        return None

    def _set_cached_data(self, cache: Dict, key: str, data: Dict):
        """Store data in cache with timestamp"""
        cache[key] = (data, time.time())

    @lru_cache(maxsize=100)
    def search_channels(self, query: str, max_results: int = 5) -> List[Dict]:
        """Search for channels with caching"""
        try:
            self._rate_limit()
            request = self.youtube.search().list(
                part="snippet",
                q=query,
                type="channel",
                maxResults=max_results
            )
            response = request.execute()
            
            if not response['items']:
                logger.info(f"No channels found for query: {query}")
                return []
            
            channels = []
            for item in response['items']:
                channels.append({
                    'name': item['snippet']['title'],
                    'channel_id': item['id']['channelId'],
                    'thumbnail': item['snippet']['thumbnails']['default']['url']
                })
            
            logger.info(f"Found {len(channels)} channels for query: {query}")
            return channels
        except HttpError as e:
            logger.error(f"An HTTP error occurred while searching for channels: {e.resp.status} {e.content}")
            raise

    @lru_cache(maxsize=100)
    def get_channel_id_by_name(self, channel_name: str) -> Optional[str]:
        """Get channel ID by name with caching"""
        try:
            self._rate_limit()
            request = self.youtube.search().list(
                part="snippet",
                q=channel_name,
                type="channel",
                maxResults=1
            )
            response = request.execute()
            
            if not response['items']:
                logger.error(f"No channel found with name: {channel_name}")
                return None
            
            channel_id = response['items'][0]['id']['channelId']
            logger.info(f"Found channel ID: {channel_id} for channel name: {channel_name}")
            return channel_id
        except HttpError as e:
            logger.error(f"An HTTP error occurred while searching for channel: {e.resp.status} {e.content}")
            raise

    def get_channel_stats(self, channel_id: str) -> Optional[Dict]:
        """Get channel statistics with caching"""
        cached_data = self._get_cached_data(self._channel_cache, channel_id)
        if cached_data:
            return cached_data

        try:
            self._rate_limit()
            request = self.youtube.channels().list(
                part="snippet,contentDetails,statistics",
                id=channel_id
            )
            response = request.execute()
            
            if not response['items']:
                logger.error(f"No channel found with ID: {channel_id}")
                return None
            
            channel_data = response['items'][0]
            logger.info(f"Successfully retrieved channel stats for channel ID: {channel_id}")
            
            stats = {
                'channel_name': channel_data['snippet']['title'],
                'subscribers': channel_data['statistics']['subscriberCount'],
                'views': channel_data['statistics']['viewCount'],
                'total_videos': channel_data['statistics']['videoCount'],
                'playlist_id': channel_data['contentDetails']['relatedPlaylists']['uploads']
            }
            
            self._set_cached_data(self._channel_cache, channel_id, stats)
            return stats
        except HttpError as e:
            logger.error(f"An HTTP error occurred: {e.resp.status} {e.content}")
            raise

    def get_video_stats(self, video_id: str) -> Optional[Dict]:
        """Get video statistics with caching"""
        cached_data = self._get_cached_data(self._video_cache, video_id)
        if cached_data:
            return cached_data

        try:
            self._rate_limit()
            request = self.youtube.videos().list(
                part="snippet,statistics,contentDetails",
                id=video_id
            )
            response = request.execute()
            
            if not response['items']:
                logger.error(f"No video found with ID: {video_id}")
                return None
            
            video_data = response['items'][0]
            duration = video_data['contentDetails']['duration']
            minutes = self._parse_duration(duration)
            
            stats = {
                'title': video_data['snippet']['title'],
                'published_at': video_data['snippet']['publishedAt'],
                'views': video_data['statistics']['viewCount'],
                'likes': video_data['statistics'].get('likeCount', 0),
                'comments': video_data['statistics'].get('commentCount', 0),
                'duration_minutes': minutes,
                'description': video_data['snippet']['description'],
                'tags': video_data['snippet'].get('tags', []),
                'category_id': video_data['snippet']['categoryId'],
                'video_id': video_id
            }
            
            self._set_cached_data(self._video_cache, video_id, stats)
            return stats
        except HttpError as e:
            logger.error(f"An HTTP error occurred while fetching video stats: {e.resp.status} {e.content}")
            raise

    def _parse_duration(self, duration):
        """Convert ISO 8601 duration to minutes"""
        hours = 0
        minutes = 0
        seconds = 0
        
        if 'H' in duration:
            hours = int(duration.split('H')[0].split('T')[1])
        if 'M' in duration:
            minutes = int(duration.split('M')[0].split('H')[-1].split('T')[-1])
        if 'S' in duration:
            seconds = int(duration.split('S')[0].split('M')[-1].split('H')[-1].split('T')[-1])
            
        return hours * 60 + minutes + seconds / 60

    def analyze_engagement(self, df):
        """Calculate engagement metrics"""
        df['engagement_rate'] = (df['likes'] + df['comments']) / df['views'] * 100
        df['like_rate'] = df['likes'] / df['views'] * 100
        df['comment_rate'] = df['comments'] / df['views'] * 100
        return df

    def analyze_content_clusters(self, df):
        """Perform content clustering using video titles and descriptions"""
        try:
            # Check if we have enough data
            if len(df) < 2:
                logger.warning("Not enough videos for clustering analysis")
                return np.zeros(len(df)), {'Cluster 1': ['Not enough data']}

            # Combine titles and descriptions for better clustering
            text_data = df['title'] + ' ' + df['description'].fillna('')
            
            # Create TF-IDF vectors with adjusted parameters for small datasets
            vectorizer = TfidfVectorizer(
                max_features=min(1000, len(df) * 10),  # Adjust features based on dataset size
                stop_words='english',
                token_pattern=r'[a-zA-Z]{2,}',  # Reduce minimum word length to 2
                min_df=1,  # Allow terms that appear in at least 1 document
                max_df=1.0  # Allow terms that appear in all documents
            )
            
            try:
                tfidf_matrix = vectorizer.fit_transform(text_data)
            except ValueError as e:
                logger.error(f"TF-IDF vectorization failed: {str(e)}")
                return np.zeros(len(df)), {'Cluster 1': ['Vectorization failed']}
            
            # Determine optimal number of clusters
            n_clusters = min(5, max(2, len(df) // 10))  # At least 2 clusters, at most 5
            
            # Perform K-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(tfidf_matrix)
            
            # Get cluster terms
            cluster_terms = {}
            feature_names = vectorizer.get_feature_names_out()
            
            for i in range(n_clusters):
                cluster_docs = tfidf_matrix[clusters == i]
                if cluster_docs.shape[0] > 0:
                    avg_tfidf = cluster_docs.mean(axis=0).A1
                    top_terms = [feature_names[j] for j in avg_tfidf.argsort()[-5:][::-1]]
                    cluster_terms[f'Cluster {i+1}'] = top_terms
            
            return clusters, cluster_terms
            
        except Exception as e:
            logger.error(f"Error in content clustering: {str(e)}")
            return np.zeros(len(df)), {'Cluster 1': ['Analysis failed']}

    def analyze_sentiment(self, df):
        """Analyze sentiment of video titles"""
        sentiments = []
        for title in df['title']:
            blob = TextBlob(title)
            sentiments.append(blob.sentiment.polarity)

        if mlflow.active_run():
            mlflow.end_run()

        with mlflow.start_run(run_name="Sentiment_Analysis"):
            avg_sentiment = np.mean(sentiments)
            mlflow.log_metric("average_sentiment", avg_sentiment)

            # Save plot
            plt.figure(figsize=(6, 4))
            plt.hist(sentiments, bins=20, color='skyblue')
            plt.xlabel("Sentiment Polarity")
            plt.ylabel("Frequency")
            plt.title("Title Sentiment Distribution")
            plot_path = tempfile.mktemp(suffix=".png")
            plt.savefig(plot_path)
            mlflow.log_artifact(plot_path, artifact_path="plots")
            plt.close()
        return sentiments

    def get_trending_topics(self, df):
        """Extract trending topics from video titles and descriptions"""
        try:
            # Check if we have enough data
            if len(df) < 2:
                logger.warning("Not enough videos for trending topics analysis")
                return ['Not enough data']

            # Combine all text data
            text_data = ' '.join(df['title'] + ' ' + df['description'].fillna(''))
            
            # Create TF-IDF vectors with adjusted parameters
            vectorizer = TfidfVectorizer(
                max_features=min(20, len(df) * 2),  # Adjust features based on dataset size
                stop_words='english',
                token_pattern=r'[a-zA-Z]{2,}',  # Reduce minimum word length to 2
                min_df=1,  # Allow terms that appear in at least 1 document
                max_df=1.0  # Allow terms that appear in all documents
            )
            
            try:
                tfidf_matrix = vectorizer.fit_transform([text_data])
            except ValueError as e:
                logger.error(f"TF-IDF vectorization failed: {str(e)}")
                return ['Analysis failed']
            
            # Get top terms
            feature_names = vectorizer.get_feature_names_out()
            tfidf_scores = tfidf_matrix.toarray()[0]
            top_terms = [feature_names[i] for i in tfidf_scores.argsort()[-10:][::-1]]
            
            return top_terms
            
        except Exception as e:
            logger.error(f"Error in trending topics analysis: {str(e)}")
            return ['Analysis failed']

    def get_video_url(self, video_id):
        """Get embeddable URL for a video"""
        return f"https://www.youtube.com/embed/{video_id}?autoplay=0"

    def analyze_upload_patterns(self, df):
        """Analyze video upload patterns"""
        df['hour'] = pd.to_datetime(df['published_at']).dt.hour
        df['day_of_week'] = pd.to_datetime(df['published_at']).dt.day_name()
        
        # Set correct day order
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        # Calculate best upload times with proper ordering and formatting
        best_hours = df.groupby('hour').agg({
            'views': ['mean', 'count'],
            'engagement_rate': 'mean'
        }).reset_index()
        
        best_hours.columns = ['hour', 'mean', 'count', 'engagement_rate']
        best_hours = best_hours[best_hours['count'] >= 2]  # Filter hours with at least 2 uploads
        best_hours = best_hours.sort_values('mean', ascending=False)
        
        # Format hour labels for better readability
        best_hours['hour_label'] = best_hours['hour'].apply(
            lambda x: f"{x:02d}:00-{(x+1):02d}:00"
        )
        
        # Calculate best days with proper ordering and formatting
        best_days = df.groupby('day_of_week').agg({
            'views': ['mean', 'count'],
            'engagement_rate': 'mean'
        }).reset_index()
        
        best_days.columns = ['day_of_week', 'mean', 'count', 'engagement_rate']
        best_days = best_days[best_days['count'] >= 2]  # Filter days with at least 2 uploads
        best_days['day_of_week'] = pd.Categorical(best_days['day_of_week'], categories=day_order, ordered=True)
        best_days = best_days.sort_values('mean', ascending=False)
        
        # Add percentages and rankings
        for df_stats in [best_hours, best_days]:
            total_views = df_stats['mean'].sum()
            df_stats['view_percentage'] = (df_stats['mean'] / total_views * 100).round(1)
            df_stats['rank'] = range(1, len(df_stats) + 1)
        
        return best_hours, best_days

    def prepare_features(self, df):
        """Prepare features for machine learning models"""
        try:
            # Extract time-based features
            df['hour'] = pd.to_datetime(df['published_at']).dt.hour
            df['day_of_week'] = pd.to_datetime(df['published_at']).dt.dayofweek
            df['month'] = pd.to_datetime(df['published_at']).dt.month
            
            # Extract title features
            df['title_length'] = df['title'].str.len()
            df['word_count'] = df['title'].str.split().str.len()
            
            # Create title complexity score (average word length)
            df['title_complexity'] = df['title'].apply(lambda x: np.mean([len(word) for word in x.split()]))
            
            # Extract description features
            df['description_length'] = df['description'].fillna('').str.len()
            df['has_links'] = df['description'].fillna('').str.contains('http').astype(int)
            df['hashtag_count'] = df['description'].fillna('').str.count('#')
            
            # Duration features - handle with care for small datasets
            if len(df) >= 5:  # Only create buckets if we have enough data
                try:
                    df['duration_bucket'] = pd.qcut(
                        df['duration_minutes'],
                        q=min(5, len(df)),
                        labels=['very_short', 'short', 'medium', 'long', 'very_long'],
                        duplicates='drop'
                    )
                except ValueError:
                    # If qcut fails, use simple binary categorization
                    median_duration = df['duration_minutes'].median()
                    df['duration_bucket'] = df['duration_minutes'].apply(
                        lambda x: 'long' if x > median_duration else 'short'
                    )
            else:
                # For very small datasets, use binary categorization
                df['duration_bucket'] = 'medium'
            
            # Engagement features with error handling
            df['likes_per_view'] = df.apply(
                lambda row: row['likes'] / row['views'] if row['views'] > 0 else 0,
                axis=1
            )
            df['comments_per_view'] = df.apply(
                lambda row: row['comments'] / row['views'] if row['views'] > 0 else 0,
                axis=1
            )
            
            return df
        except Exception as e:
            logger.error(f"Error in feature preparation: {str(e)}")
            raise

    def train_view_predictor(self, df):
        """Train a model to predict video views"""
        try:
            # Check if we have enough data
            if len(df) < 5:
                logger.warning("Not enough data to train view predictor")
                return {
                    'r2_score': 0,
                    'rmse': 0,
                    'feature_importance': pd.DataFrame({'feature': [], 'importance': []})
                }
            
            # Prepare features
            df = self.prepare_features(df)
            
            # Select features for view prediction
            feature_cols = [
                'hour', 'day_of_week', 'month', 'duration_minutes',
                'title_length', 'word_count', 'title_complexity',
                'description_length', 'has_links', 'hashtag_count'
            ]
            
            X = df[feature_cols]
            y = df['views']
            
            # Handle small datasets
            test_size = min(0.2, 1/len(df))
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model with adjusted parameters for small datasets
            self.view_predictor = RandomForestRegressor(
                n_estimators=min(100, max(10, len(df))),
                max_depth=min(10, len(df) // 2),
                min_samples_split=2,
                random_state=42
            )

            if mlflow.active_run():
                mlflow.end_run()

            
            with mlflow.start_run(run_name="View_Predictor"):
                self.view_predictor.fit(X_train_scaled, y_train)
                # Evaluate model
                y_pred = self.view_predictor.predict(X_test_scaled)
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mlflow.log_param("n_estimators", self.view_predictor.n_estimators)
                mlflow.log_param("max_depth", self.view_predictor.max_depth)
                mlflow.log_metric("r2_score", r2)
                mlflow.log_metric("rmse", rmse)
                mlflow.sklearn.log_model(self.view_predictor, "view_model")


            
            # Get feature importance
            feature_importance = pd.DataFrame({
                'feature': feature_cols,
                'importance': self.view_predictor.feature_importances_
            }).sort_values('importance', ascending=False)
            
            return {
                'r2_score': r2,
                'rmse': rmse,
                'feature_importance': feature_importance
            }
            
        except Exception as e:
            logger.error(f"Error training view predictor: {str(e)}")
            return None

    def predict_engagement(self, df):
        """Train a model to predict engagement rates"""
        try:
            # Check if we have enough data
            if len(df) < 5:
                logger.warning("Not enough data to train engagement predictor")
                return {
                    'r2_score': 0,
                    'rmse': 0,
                    'feature_importance': pd.DataFrame({'feature': [], 'importance': []})
                }
            
            # Prepare features
            df = self.prepare_features(df)
            
            # Calculate target variable (engagement rate) with error handling
            df['engagement_rate'] = df.apply(
                lambda row: (row['likes'] + row['comments']) / row['views'] * 100 if row['views'] > 0 else 0,
                axis=1
            )
            
            # Select features
            feature_cols = [
                'hour', 'day_of_week', 'month', 'duration_minutes',
                'title_length', 'word_count', 'title_complexity',
                'description_length', 'has_links', 'hashtag_count'
            ]
            
            X = df[feature_cols]
            y = df['engagement_rate']
            
            # Handle small datasets
            test_size = min(0.2, 1/len(df))
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model with adjusted parameters for small datasets
            self.engagement_predictor = RandomForestRegressor(
                n_estimators=min(100, max(10, len(df))),
                max_depth=min(10, len(df) // 2),
                min_samples_split=2,
                random_state=42
            )
            if mlflow.active_run():
                mlflow.end_run()

            with mlflow.start_run(run_name="Engagement_Predictor"):
                self.engagement_predictor.fit(X_train_scaled, y_train)
            
                # Evaluate model
                y_pred = self.engagement_predictor.predict(X_test_scaled)
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))

                mlflow.log_param("n_estimators", self.engagement_predictor.n_estimators)
                mlflow.log_param("max_depth", self.engagement_predictor.max_depth)
                mlflow.log_metric("r2_score", r2)
                mlflow.log_metric("rmse", rmse)
                mlflow.sklearn.log_model(self.engagement_predictor, "engagement_model")
            
            # Get feature importance
            feature_importance = pd.DataFrame({
                'feature': feature_cols,
                'importance': self.engagement_predictor.feature_importances_
            }).sort_values('importance', ascending=False)
            
            return {
                'r2_score': r2,
                'rmse': rmse,
                'feature_importance': feature_importance
            }
            
        except Exception as e:
            logger.error(f"Error training engagement predictor: {str(e)}")
            return None

    def analyze_video_categories(self, df):
        """Analyze and classify videos into categories"""
        try:
            # Prepare text data
            text_data = df['title'] + ' ' + df['description'].fillna('')
            
            # Create TF-IDF vectors
            vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                token_pattern=r'[a-zA-Z]{3,}'
            )
            X = vectorizer.fit_transform(text_data)
            
            # Perform clustering
            n_clusters = min(5, len(df))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(X)

            if mlflow.active_run():
                mlflow.end_run()
            
            with mlflow.start_run(run_name="Content_Clustering"):
                mlflow.log_param("n_clusters", n_clusters)
                cluster_stats = []
                cluster_distribution = []
                # Get cluster characteristics
                cluster_stats = []
                for i in range(n_clusters):
                    cluster_mask = clusters == i
                    cluster_data = df[cluster_mask]
                    
                    stats = {
                        'cluster_id': i,
                        'size': len(cluster_data),
                        'avg_views': cluster_data['views'].mean(),
                        'avg_engagement': (cluster_data['likes'] + cluster_data['comments']).mean() / cluster_data['views'].mean() * 100,
                        'avg_duration': cluster_data['duration_minutes'].mean(),
                        'top_terms': self._get_cluster_terms(vectorizer, kmeans, i)
                    }
                    cluster_stats.append(stats)
                    cluster_distribution.append(len(cluster_data))
            
            # Log cluster stats as JSON
            with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
                json.dump(cluster_stats, f, indent=2)
                mlflow.log_artifact(f.name, artifact_path="clustering")

            # Plot cluster sizes
            plt.figure(figsize=(6, 4))
            plt.bar(range(n_clusters), cluster_distribution)
            plt.xlabel("Cluster")
            plt.ylabel("Number of Videos")
            plt.title("Cluster Distribution")
            plot_path = tempfile.mktemp(suffix=".png")
            plt.savefig(plot_path)
            mlflow.log_artifact(plot_path, artifact_path="plots")
            plt.close()


            return pd.DataFrame(cluster_stats)
            
        except Exception as e:
            logger.error(f"Error in video categorization: {str(e)}")
            return None

    def _get_cluster_terms(self, vectorizer, kmeans, cluster_id):
        """Get top terms for a cluster"""
        order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
        terms = vectorizer.get_feature_names_out()
        return [terms[i] for i in order_centroids[cluster_id, :5]]

    def get_optimal_parameters(self, df):
        """Get optimal parameters for video success"""
        try:
            # Analyze successful videos (top 25% by views)
            threshold = df['views'].quantile(0.75)
            successful_videos = df[df['views'] >= threshold]
            
            optimal_params = {
                'duration': {
                    'mean': successful_videos['duration_minutes'].mean(),
                    'median': successful_videos['duration_minutes'].median(),
                    'range': (successful_videos['duration_minutes'].quantile(0.25),
                            successful_videos['duration_minutes'].quantile(0.75))
                },
                'title_length': {
                    'mean': successful_videos['title'].str.len().mean(),
                    'range': (successful_videos['title'].str.len().quantile(0.25),
                            successful_videos['title'].str.len().quantile(0.75))
                },
                'best_hours': successful_videos['hour'].mode().tolist()[:3],
                'best_days': successful_videos['day_of_week'].mode().tolist()[:3],
                'engagement_rate': {
                    'mean': (successful_videos['likes'] + successful_videos['comments']).sum() / successful_videos['views'].sum() * 100
                }
            }
            
            return optimal_params
            
        except Exception as e:
            logger.error(f"Error calculating optimal parameters: {str(e)}")
            return None

    def get_trending_videos(self, region_code: str = 'US', category_id: Optional[str] = None, max_results: int = 50) -> List[Dict]:
        """Get trending videos with caching"""
        cache_key = f"{region_code}_{category_id}_{max_results}"
        cached_data = self._get_cached_data(self._trending_cache, cache_key)
        if cached_data:
            return cached_data

        try:
            # Validate parameters
            if not isinstance(region_code, str) or len(region_code) != 2:
                logger.error("Invalid region code")
                return []
            
            if max_results < 1 or max_results > 50:
                logger.warning("Invalid max_results value, using default of 50")
                max_results = 50
            
            # Calculate date 7 days ago in RFC 3339 format with UTC timezone
            seven_days_ago = (datetime.utcnow().replace(tzinfo=None) - timedelta(days=7)).strftime('%Y-%m-%dT%H:%M:%SZ')
            
            # Build search parameters
            search_params = {
                'part': 'snippet',
                'type': 'video',
                'order': 'viewCount',
                'publishedAfter': seven_days_ago,
                'regionCode': region_code.upper(),
                'maxResults': max_results
            }
            
            if category_id and str(category_id).isdigit():
                search_params['videoCategoryId'] = str(category_id)
            
            self._rate_limit()
            request = self.youtube.search().list(**search_params)
            response = request.execute()
            
            if not response.get('items'):
                logger.warning(f"No trending videos found for region {region_code}")
                return []
            
            video_ids = [item['id']['videoId'] for item in response['items']]
            
            trending_videos = []
            for i in range(0, len(video_ids), 50):
                batch_ids = video_ids[i:i + 50]
                try:
                    self._rate_limit()
                    videos_request = self.youtube.videos().list(
                        part="snippet,contentDetails,statistics",
                        id=','.join(batch_ids)
                    )
                    videos_response = videos_request.execute()
                    
                    for item in videos_response.get('items', []):
                        try:
                            video_data = {
                                'title': item['snippet']['title'],
                                'published_at': item['snippet']['publishedAt'],
                                'views': int(item['statistics'].get('viewCount', 0)),
                                'likes': int(item['statistics'].get('likeCount', 0)),
                                'comments': int(item['statistics'].get('commentCount', 0)),
                                'duration_minutes': self._parse_duration(item['contentDetails']['duration']),
                                'description': item['snippet']['description'],
                                'tags': item['snippet'].get('tags', []),
                                'category_id': item['snippet']['categoryId'],
                                'video_id': item['id'],
                                'channel_title': item['snippet']['channelTitle']
                            }
                            trending_videos.append(video_data)
                        except (KeyError, ValueError) as e:
                            logger.warning(f"Skipping video due to missing/invalid data: {str(e)}")
                            continue
                        
                except HttpError as e:
                    logger.error(f"Error fetching video details: {str(e)}")
                    continue
            
            trending_videos.sort(key=lambda x: x['views'], reverse=True)
            seen = set()
            unique_trending_videos = []
            for video in trending_videos:
                if video['video_id'] not in seen:
                    seen.add(video['video_id'])
                    unique_trending_videos.append(video)
            
            logger.info(f"Successfully retrieved {len(unique_trending_videos)} trending videos for region {region_code}")
            result = unique_trending_videos[:max_results]
            self._set_cached_data(self._trending_cache, cache_key, result)
            return result
            
        except HttpError as e:
            logger.error(f"Error fetching trending videos: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error in get_trending_videos: {str(e)}")
            return []

    def analyze_trends(self, trending_videos):
        """Analyze trends in popular videos"""
        try:
            if not trending_videos:
                logger.warning("No trending videos to analyze")
                return None
            
            df = pd.DataFrame(trending_videos)
            # Convert to timezone-aware datetime and then strip timezone for consistency
            df['published_at'] = pd.to_datetime(df['published_at']).dt.tz_localize(None)
            
            # Calculate basic trend metrics
            trend_analysis = {
                'total_videos': len(df),
                'avg_views': df['views'].mean(),
                'avg_engagement': (
                    (df['likes'] + df['comments']).sum() / df['views'].sum() * 100
                    if df['views'].sum() > 0 else 0
                ),
                'avg_duration': df['duration_minutes'].mean(),
                
                # Time-based patterns
                'popular_hours': df['published_at'].dt.hour.value_counts().head(3).index.tolist(),
                'popular_days': df['published_at'].dt.day_name().value_counts().head(3).index.tolist(),
                
                # Title analysis
                'avg_title_length': df['title'].str.len().mean(),
                'common_title_words': self._get_common_words(df['title'], n=10),
                
                # Tag analysis
                'common_tags': self._get_common_tags(df['tags'], n=10),
                
                # Channel analysis
                'top_channels': df['channel_title'].value_counts().head(5).to_dict(),
                
                # Category performance
                'category_performance': self._analyze_category_performance(df)
            }

            with mlflow.start_run(run_name="Trend_Analysis", nested=True):
                mlflow.log_metric("total_videos", trend_analysis["total_videos"])
                mlflow.log_metric("avg_views", trend_analysis["avg_views"])
                mlflow.log_metric("avg_engagement", trend_analysis["avg_engagement"])

                with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
                    json.dump(trend_analysis, f, indent=2)
                    mlflow.log_artifact(f.name, artifact_path="trend_analysis")
                
            return trend_analysis
        except Exception as e:
            logger.error(f"Error analyzing trends: {str(e)}")
            return None

    def predict_trend_performance(self, df):
        """Predict performance metrics for trending videos"""
        try:
            # Prepare features
            df = self.prepare_features(df)
            
            # Add trend-specific features
            df['title_keywords'] = df['title'].apply(self._extract_keywords)
            df['description_keywords'] = df['description'].fillna('').apply(self._extract_keywords)
            df['total_keywords'] = df['title_keywords'] + df['description_keywords']
            
            # Calculate target variables
            df['trend_score'] = (
                df['views'] * 0.4 +
                df['likes'] * 0.3 +
                df['comments'] * 0.3
            ) / df['views'].max()
            
            # Select features for trend prediction
            feature_cols = [
                'hour', 'day_of_week', 'month',
                'duration_minutes', 'title_length',
                'word_count', 'title_complexity',
                'description_length', 'has_links',
                'hashtag_count', 'total_keywords'
            ]
            
            X = df[feature_cols]
            y = df['trend_score']
            
            # Split and scale data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model
            trend_predictor = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )

            with mlflow.start_run(run_name="Trend_Predictor",nested=True):
                trend_predictor.fit(X_train_scaled, y_train)
            
                # Evaluate model
                y_pred = trend_predictor.predict(X_test_scaled)
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))

                mlflow.log_param("n_estimators", trend_predictor.n_estimators)
                mlflow.log_param("max_depth", trend_predictor.max_depth)
                mlflow.log_metric("r2_score", r2)
                mlflow.log_metric("rmse", rmse)
                mlflow.sklearn.log_model(trend_predictor, "trend_model")
            
            # Get feature importance
            feature_importance = pd.DataFrame({
                'feature': feature_cols,
                'importance': trend_predictor.feature_importances_
            }).sort_values('importance', ascending=False)
            
            return {
                'r2_score': r2,
                'rmse': rmse,
                'feature_importance': feature_importance,
                'model': trend_predictor
            }
        except Exception as e:
            logger.error(f"Error in trend prediction: {str(e)}")
            return None

    def _get_common_words(self, series, n=10):
        """Extract common words from a series of text"""
        text = ' '.join(series)
        words = re.findall(r'\b\w+\b', text.lower())
        stop_words = set([
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
            'for', 'of', 'with', 'by'
        ])
        words = [word for word in words if word not in stop_words]
        return pd.Series(words).value_counts().head(n).to_dict()

    def _get_common_tags(self, tags_lists, n=10):
        """Get most common tags from a series of tag lists"""
        all_tags = []
        for tags in tags_lists:
            if isinstance(tags, list):
                all_tags.extend(tags)
        return pd.Series(all_tags).value_counts().head(n).to_dict()

    def _analyze_category_performance(self, df):
        """Analyze performance by video category"""
        category_stats = df.groupby('category_id').agg({
            'views': 'mean',
            'likes': 'mean',
            'comments': 'mean',
            'video_id': 'count'
        }).round(2)
        
        category_stats['engagement_rate'] = (
            (category_stats['likes'] + category_stats['comments']) / 
            category_stats['views'] * 100
        ).round(2)
        
        return category_stats.to_dict('index')

    def _extract_keywords(self, text):
        """Extract meaningful keywords from text"""
        # Remove special characters and convert to lowercase
        text = re.sub(r'[^\w\s]', '', text.lower())
        words = text.split()
        
        # Remove common stop words
        stop_words = set([
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
            'for', 'of', 'with', 'by', 'video', 'watch', 'how', 'what', 'why',
            'when', 'where', 'who', 'new'
        ])
        
        return len([w for w in words if w not in stop_words])

    def get_channel_videos(self, playlist_id, max_results=50):
        try:
            videos = []
            next_page_token = None
            
            while len(videos) < max_results:
                request = self.youtube.playlistItems().list(
                    part="snippet,contentDetails",
                    playlistId=playlist_id,
                    maxResults=min(50, max_results - len(videos)),
                    pageToken=next_page_token
                )
                response = request.execute()
                
                if not response.get('items'):
                    break
                
                for item in response['items']:
                    video_id = item['contentDetails']['videoId']
                    video_stats = self.get_video_stats(video_id)
                    if video_stats:
                        videos.append(video_stats)
                
                next_page_token = response.get('nextPageToken')
                if not next_page_token:
                    break
            
            logger.info(f"Successfully retrieved {len(videos)} videos from playlist: {playlist_id}")
            return videos
        except HttpError as e:
            logger.error(f"An HTTP error occurred while fetching videos: {e.resp.status} {e.content}")
            raise 