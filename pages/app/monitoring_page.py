"""
Admin Monitoring Page for Streamlit App

Provides a simplified monitoring view within Streamlit for quick checks.
For detailed monitoring, use the main dashboard at /report

Author: Goodreads Recommendation Team
Date: 2025
"""

import streamlit as st
import requests
import pandas as pd
from datetime import datetime
import os

API_BASE_URL = os.environ.get("API_BASE_URL", "https://recommendation-service-491512947755.us-central1.run.app")

st.set_page_config(
    page_title="ML Monitoring",
    page_icon="📊",
    layout="wide"
)

st.title("📊 ML Model & API Monitoring")
st.markdown("Quick monitoring overview - [Full Dashboard →]({}/report)".format(API_BASE_URL))

tab1, tab2, tab3 = st.tabs(["📈 API Performance", "🤖 Model Metrics", "🔍 Data Drift"])

with tab1:
    st.header("API Performance Metrics")
    
    try:
        response = requests.get(f"{API_BASE_URL}/metrics", timeout=10)
        if response.status_code == 200:
            data = response.json()
            summary = data.get('summary', {})
            endpoints = data.get('endpoints', [])
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Total Requests",
                    f"{summary.get('total_requests', 0):,}"
                )
            
            with col2:
                error_rate = summary.get('error_rate', 0)
                st.metric(
                    "Error Rate",
                    f"{error_rate:.2f}%",
                    delta=None,
                    delta_color="inverse"
                )
            
            with col3:
                p95 = summary.get('p95_latency_ms', 0)
                st.metric(
                    "P95 Latency",
                    f"{p95:.0f} ms"
                )
            
            with col4:
                rpm = summary.get('requests_per_minute', 0)
                st.metric(
                    "Requests/Min",
                    f"{rpm:.1f}"
                )
            
            st.markdown("---")
            st.subheader("Top Endpoints")
            
            if endpoints:
                df = pd.DataFrame(endpoints[:10])
                df = df[['endpoint', 'request_count', 'avg_latency_ms', 'error_rate', 'p95_latency_ms']]
                df.columns = ['Endpoint', 'Requests', 'Avg Latency (ms)', 'Error Rate (%)', 'P95 (ms)']
                st.dataframe(df, use_container_width=True)
        else:
            st.error("Failed to fetch API metrics")
    except Exception as e:
        st.error(f"Error connecting to monitoring API: {e}")

with tab2:
    st.header("Model Performance Metrics")
    
    st.info("For detailed model performance metrics, visit the [Full Monitoring Dashboard]({}/report)".format(API_BASE_URL))
    
    st.markdown("""
    **Model Metrics Tracked:**
    - RMSE (Root Mean Square Error)
    - MAE (Mean Absolute Error)
    - R² Score
    - Prediction Accuracy (±0.5 stars, ±1.0 stars)
    
    **Monitoring Checks:**
    - Model decay detection (performance degradation over time)
    - Automated alerts when metrics exceed thresholds
    - Historical trend analysis
    """)

with tab3:
    st.header("Data Drift Detection")
    
    st.info("For detailed drift analysis, visit the [Full Monitoring Dashboard]({}/report)".format(API_BASE_URL))
    
    st.markdown("""
    **Features Monitored:**
    - avg_rating
    - ratings_count
    - book_age_years
    - text_reviews_count
    
    **Statistical Tests:**
    - Kolmogorov-Smirnov (KS) Test
    - Population Stability Index (PSI)
    - Mean Shift Detection
    
    **Thresholds:**
    - KS p-value < 0.05 → Drift detected
    - PSI > 0.2 → Significant distribution shift
    - Mean shift > 2σ → Feature drift
    """)

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    if st.button("🔄 Refresh Metrics", type="primary", use_container_width=True):
        st.rerun()

with col2:
    if st.button("📊 Open Full Dashboard", use_container_width=True):
        st.markdown(f'<meta http-equiv="refresh" content="0;url={API_BASE_URL}/report" />', 
                    unsafe_allow_html=True)
        st.markdown(f"[Click here to open monitoring dashboard]({API_BASE_URL}/report)")

st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
