# YouTube Analytics Dashboard 📊

A powerful analytics dashboard for YouTube channels that provides deep insights into video performance, content analysis, and trend prediction using machine learning. The dashboard uses the YouTube Data API v3 and advanced analytics to help content creators optimize their channel performance.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
  - [Local Setup](#local-setup)
  - [Docker Setup](#docker-setup)
- [Usage Guide](#usage-guide)
- [Dashboard Sections](#dashboard-sections)
- [Metrics Guide](#metrics-guide)
- [Machine Learning Models](#machine-learning-models)
- [Troubleshooting](#troubleshooting)
- [Best Practices](#best-practices)

## Features

### 1. Channel Overview
- Real-time channel statistics
- Subscriber count and growth trends
- Total view count and video metrics
- Historical performance analysis

### 2. Content Analysis
- Video title sentiment analysis
- Duration optimization recommendations
- Content clustering and categorization
- Topic trend identification
- Keyword effectiveness analysis

### 3. Performance Analytics
- View count progression
- Engagement rate tracking
- Like and comment analysis
- Performance prediction models
- Growth trajectory analysis

### 4. Audience Insights
- Upload timing optimization
- Day-of-week performance
- Seasonal trend analysis
- Regional performance metrics
- Audience engagement patterns

### 5. ML-Powered Predictions
- View count forecasting
- Engagement rate prediction
- Content performance estimation
- Trend analysis and forecasting
- Category performance prediction

## Installation

### Local Setup

1. **Prerequisites**:
   - Python 3.9 or higher
   - pip package manager
   - Google Cloud project with YouTube Data API v3 enabled
   - YouTube API key

2. **Installation Steps**:
   ```bash
   # Clone the repository
   git clone <repository-url>
   cd youtube-analytics

   # Create and activate virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

   # Install dependencies
   pip install -r requirements.txt

   # Set up environment variables
   # Create .env file with:
   YOUTUBE_API_KEY=your_api_key_here
   STREAMLIT_THEME_BASE=light
   STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
   ```

### Docker Setup

1. **Prerequisites**:
   - Docker Engine (20.10.0+)
   - Docker Compose (2.0.0+)
   - YouTube API key
   - 2GB+ available memory
   - Port 8501 available

2. **Docker Installation**:
   ```bash
   # Build and start the container
   docker-compose up --build

   # For production deployment
   docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
   ```

For detailed Docker setup instructions, refer to [Docker.md](Docker.md).

## Usage Guide

### Getting Started

1. **Launch the Application**:
   ```bash
   # Local setup
   streamlit run app.py

   # Or access Docker deployment at:
   http://localhost:8501
   ```

2. **Channel Analysis**:
   - Enter a YouTube channel name in the search box
   - Select from the auto-suggested channels
   - Wait for the analysis to complete (~30-60 seconds)

### Dashboard Sections

#### 1. Channel Overview
- **Key Metrics Display**:
  - Total subscribers
  - Overall views
  - Video count
  - Average views per video
- **Growth Trends**:
  - Subscriber growth rate
  - View count progression
  - Content output frequency

#### 2. Views Trends Tab
- **Time Series Analysis**:
  - Historical view performance
  - 7-day moving average
  - 30-day moving average
  - Growth trajectory
- **Performance Indicators**:
  - Peak viewing periods
  - Viral content identification
  - Seasonal patterns
  - Growth acceleration/deceleration

#### 3. Engagement Analysis Tab
- **Interaction Metrics**:
  - Views vs. Engagement Rate
  - Like Rate Analysis
  - Comment Rate Trends
  - Overall Engagement Score
- **Correlation Analysis**:
  - Views-Engagement correlation
  - Like-Comment relationship
  - Audience interaction patterns

#### 4. Content Analysis Tab
- **Sentiment Analysis**:
  - Title sentiment distribution
  - Emotional impact scores
  - Keyword effectiveness
- **Duration Analysis**:
  - Optimal video length
  - Duration-performance correlation
  - Audience retention patterns
- **Topic Clustering**:
  - Content categorization
  - Theme performance
  - Topic trends

#### 5. Upload Patterns Tab
- **Timing Analysis**:
  - Best upload hours (24-hour format)
  - Optimal days of the week
  - Time zone considerations
- **Performance by Time**:
  - Views by upload hour
  - Engagement by day
  - Seasonal variations

#### 6. Predictions & Insights Tab
- **View Prediction Model**:
  - Expected view ranges
  - Confidence intervals
  - Feature importance
- **Engagement Forecasting**:
  - Predicted engagement rates
  - Performance factors
  - Optimization suggestions

#### 7. Trends Analysis Tab
- **Market Analysis**:
  - Regional performance
  - Category trends
  - Competition benchmarking
- **Content Strategy**:
  - Trending topics
  - Popular tags
  - Keyword opportunities

## Metrics Guide

### 1. Engagement Metrics

#### Engagement Rate
- **Formula**: (Likes + Comments) / Views × 100
- **Interpretation**:
  - Excellent: > 8%
  - Good: 5-8%
  - Average: 2-5%
  - Poor: < 2%

#### Like Rate
- **Formula**: Likes / Views × 100
- **Benchmarks**:
  - High: > 4%
  - Average: 2-4%
  - Low: < 2%

#### Comment Rate
- **Formula**: Comments / Views × 100
- **Benchmarks**:
  - High: > 1%
  - Average: 0.5-1%
  - Low: < 0.5%

### 2. Content Performance

#### View Velocity
- **Formula**: Views / Days Since Upload
- **Interpretation**:
  - Strong: > 1000 views/day
  - Good: 500-1000 views/day
  - Average: 100-500 views/day
  - Slow: < 100 views/day

#### Sentiment Score
- **Range**: -1.0 to 1.0
- **Interpretation**:
  - Positive: > 0.3
  - Neutral: -0.3 to 0.3
  - Negative: < -0.3

### 3. Statistical Measures

#### R² Score (Model Accuracy)
- **Range**: 0 to 1
- **Interpretation**:
  - Excellent: > 0.8
  - Good: 0.6-0.8
  - Fair: 0.4-0.6
  - Poor: < 0.4

#### Confidence Intervals
- **Standard**: 95% confidence level
- **Interpretation**:
  - Narrow range: High confidence
  - Wide range: Lower confidence

## Machine Learning Models

### 1. View Prediction Model
- **Type**: Random Forest Regressor
- **Features**:
  - Historical performance
  - Time-based features
  - Content characteristics
  - Engagement metrics

### 2. Engagement Predictor
- **Type**: Gradient Boosting
- **Features**:
  - Title sentiment
  - Upload timing
  - Content category
  - Historical engagement

### 3. Content Analyzer
- **Components**:
  - TF-IDF Vectorization
  - K-means Clustering
  - Sentiment Analysis (TextBlob)

## Troubleshooting

### Common Issues

1. **API Key Errors**:
   - Verify API key in .env file
   - Check API quota limits
   - Ensure API is enabled in Google Cloud Console

2. **Performance Issues**:
   - Clear browser cache
   - Reduce date range selection
   - Check internet connection
   - Verify system resources

3. **Data Loading Errors**:
   - Confirm channel exists and is public
   - Check for valid video IDs
   - Verify data permissions

4. **Model Errors**:
   - Ensure sufficient historical data
   - Check for missing values
   - Verify feature compatibility

## Best Practices

### 1. Content Strategy
- Use sentiment analysis for title optimization
- Follow recommended video durations
- Post during peak engagement hours
- Target high-performing content categories

### 2. Performance Optimization
- Monitor engagement trends
- Analyze successful video patterns
- Test different content types
- Track seasonal variations

### 3. Growth Tactics
- Focus on high-impact metrics
- Optimize upload timing
- Engage with audience comments
- Monitor competitive benchmarks

### 4. Data Analysis
- Regular performance reviews
- Track long-term trends
- Monitor prediction accuracy
- Update content strategy based on insights

## Support

For additional support:
- Check the [documentation](docs/)
- Submit issues on GitHub
- Contact the development team
- Join our community forum

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 