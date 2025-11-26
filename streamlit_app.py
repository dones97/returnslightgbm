"""
Enhanced Streamlit Application for Indian Stock Screening
ML-powered stock screener using LightGBM with comprehensive model analysis
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
import pickle
from data_collector_enhanced import EnhancedStockDataCollector, collect_data_for_universe
from model_trainer import ReturnDirectionModel, StockScorer
from nifty500_tickers import get_quality_stocks

# Page configuration
st.set_page_config(
    page_title="Indian Stock Screener - ML Powered",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .model-info-box {
        background-color: #e8f4f8;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .status-success {
        background-color: #d4edda;
        color: #155724;
        padding: 0.75rem;
        border-radius: 0.3rem;
        border-left: 4px solid #28a745;
    }
    .status-warning {
        background-color: #fff3cd;
        color: #856404;
        padding: 0.75rem;
        border-radius: 0.3rem;
        border-left: 4px solid #ffc107;
    }
    .quintile-q5 {
        background-color: #d4edda;
        padding: 0.5rem;
        border-radius: 0.3rem;
    }
    .quintile-q4 {
        background-color: #d1ecf1;
        padding: 0.5rem;
        border-radius: 0.3rem;
    }
    .quintile-q3 {
        background-color: #fff3cd;
        padding: 0.5rem;
        border-radius: 0.3rem;
    }
</style>
""", unsafe_allow_html=True)


class EnhancedStockScreener:
    """Enhanced application class with model persistence and detailed analysis"""

    def __init__(self):
        self.data_file = 'stock_data_cache.pkl'
        self.model_file = 'trained_model.pkl'
        self.results_file = 'screening_results.pkl'

    def load_universe_tickers(self) -> list:
        """Load stock tickers from NSE_Universe.csv"""
        try:
            df = pd.read_csv('NSE_Universe.csv')
            tickers = df['Ticker'].dropna().unique().tolist()
            tickers = [t if t.endswith('.NS') else f"{t}.NS" for t in tickers]
            return tickers
        except Exception as e:
            st.error(f"Error loading universe tickers: {str(e)}")
            return []

    def load_backup_tickers(self) -> list:
        """Load backup tickers from indian_stocks_tickers.csv"""
        try:
            df = pd.read_csv('indian_stocks_tickers.csv', encoding='utf-8-sig')
            tickers = df['Ticker'].dropna().unique().tolist()
            return tickers
        except Exception as e:
            st.error(f"Error loading backup tickers: {str(e)}")
            return []

    def display_model_info_box(self, model: ReturnDirectionModel):
        """Display model information in a highlighted box"""
        if not model.model_metadata:
            st.warning("Model metadata not available")
            return

        meta = model.model_metadata
        st.markdown('<div class="model-info-box">', unsafe_allow_html=True)
        st.markdown("### 🤖 Current Model Information")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Training Date:**")
            st.markdown(f"🕐 {meta.get('trained_at', 'Unknown')}")
            st.markdown("**Data Range:**")
            st.markdown(f"📅 {meta.get('data_date_range', 'Unknown')}")

        with col2:
            st.markdown("**Training Data:**")
            st.markdown(f"📊 {meta.get('n_stocks', 'Unknown')} stocks")
            st.markdown(f"📈 {meta.get('n_training_samples', 0):,} training samples")
            st.markdown(f"📉 {meta.get('n_test_samples', 0):,} test samples")

        with col3:
            metrics = meta.get('metrics', {})
            st.markdown("**Performance:**")
            st.markdown(f"✅ Accuracy: {metrics.get('accuracy', 0):.2%}")
            st.markdown(f"🎯 ROC AUC: {metrics.get('roc_auc', 0):.3f}")
            st.markdown(f"⚙️ Features: {meta.get('n_features', 'Unknown')}")

        st.markdown('</div>', unsafe_allow_html=True)

    def data_collection_page(self):
        """Enhanced page for collecting data with progress indicators"""
        st.markdown('<div class="main-header">📊 Step 1: Data Collection</div>', unsafe_allow_html=True)

        st.write("""
        This step fetches historical price data and fundamental information from Yahoo Finance.
        The process may take several minutes depending on the number of stocks.
        """)

        col1, col2 = st.columns(2)

        with col1:
            lookback_years = st.slider(
                "Historical data period (years)",
                min_value=2,
                max_value=10,
                value=5,
                help="5 years recommended for fundamental research - provides 2-3 years of quarterly fundamental data (40-60% coverage)"
            )

        with col2:
            max_stocks = st.number_input(
                "Maximum number of stocks to process",
                min_value=10,
                max_value=200,
                value=150,
                help="Pre-defined quality list has ~195 stocks. Recommended: 50-150 stocks"
            )

        # Show info about stock availability
        from nifty500_tickers import QUALITY_STOCKS
        st.info(f"ℹ️ Using pre-defined quality stock list ({len(QUALITY_STOCKS)} stocks available). "
                f"These are Nifty 50/100/200 companies sorted by market cap.")

        use_cached = st.checkbox("Use cached data if available", value=True)

        if st.button("Start Data Collection", type="primary"):
            # Check if cached data exists
            if use_cached and os.path.exists(self.data_file):
                st.markdown('<div class="status-success">✅ Loading cached data...</div>', unsafe_allow_html=True)
                with open(self.data_file, 'rb') as f:
                    data = pickle.load(f)
                st.session_state['data'] = data
                st.success(f"Loaded {len(data)} rows of cached data!")
                st.dataframe(data.head())
                return

            # Load tickers
            status_placeholder = st.empty()
            progress_bar = st.progress(0)

            with status_placeholder.container():
                st.info("🔍 Loading stock universe...")

            tickers = self.load_universe_tickers()

            if not tickers:
                tickers = self.load_backup_tickers()

            if not tickers:
                st.error("Failed to load stock tickers!")
                return

            progress_bar.progress(10)
            status_placeholder.info(f"✅ Found {len(tickers)} tickers in universe")

            # Use pre-defined quality stocks list (avoids rate limiting)
            status_placeholder.info(f"🎯 Selecting top {max_stocks} quality stocks from Nifty 500...")
            progress_bar.progress(15)

            # Get quality stocks from pre-defined list (no API calls needed!)
            selected_tickers = get_quality_stocks(n=max_stocks)
            if selected_tickers and len(selected_tickers) > 0:
                tickers = selected_tickers
                if len(tickers) < max_stocks:
                    st.warning(f"⚠️ Requested {max_stocks} stocks but only {len(tickers)} available in pre-defined list. Using all {len(tickers)} stocks.")
                status_placeholder.info(f"✅ Selected {len(tickers)} quality stocks (Nifty 50/100/200 companies)")
            else:
                # Fallback to first N if list is empty
                st.warning("Pre-defined list empty, using first stocks from universe")
                tickers = tickers[:max_stocks]

            # Collect data with progress updates
            status_placeholder.info(f"📥 Collecting data for {len(tickers)} stocks... This may take 15-30 minutes...")
            progress_bar.progress(20)

            data = collect_data_for_universe(tickers, lookback_years=lookback_years)

            progress_bar.progress(90)

            if data.empty:
                st.error("Failed to collect data!")
                return

            # Save data
            with open(self.data_file, 'wb') as f:
                pickle.dump(data, f)

            progress_bar.progress(100)
            status_placeholder.markdown('<div class="status-success">✅ Data collection complete!</div>', unsafe_allow_html=True)

            st.session_state['data'] = data

            st.success(f"Successfully collected data for {data['Ticker'].nunique()} stocks!")
            st.write(f"Total records: {len(data):,}")
            st.write(f"Date range: {data['Date'].min()} to {data['Date'].max()}")

            # Show sample
            st.subheader("Sample Data")
            st.dataframe(data.head(20))

            # Show statistics
            st.subheader("Data Statistics")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Stocks", data['Ticker'].nunique())
            with col2:
                st.metric("Total Records", f"{len(data):,}")
            with col3:
                st.metric("Features", len(data.columns))
            with col4:
                avg_months = len(data) / data['Ticker'].nunique()
                st.metric("Avg Months/Stock", f"{avg_months:.0f}")

    def model_training_page(self):
        """Enhanced page for training the model with detailed metrics"""
        st.markdown('<div class="main-header">🤖 Step 2: Model Training</div>', unsafe_allow_html=True)

        if 'data' not in st.session_state:
            # Try to load cached data
            if os.path.exists(self.data_file):
                with st.spinner("Loading cached data..."):
                    with open(self.data_file, 'rb') as f:
                        data = pickle.load(f)
                    st.session_state['data'] = data
                    st.success(f"Loaded cached data: {len(data):,} records")
            else:
                st.warning("⚠️ Please collect data first (Step 1)")
                return

        data = st.session_state['data']

        st.write(f"""
        Training LightGBM model on **{len(data):,}** records from **{data['Ticker'].nunique()}** stocks.
        The model will predict the direction of next month's returns.
        """)

        col1, col2 = st.columns(2)

        with col1:
            test_size = st.slider(
                "Test set size (%)",
                min_value=10,
                max_value=30,
                value=20
            ) / 100

        with col2:
            use_time_series_split = st.checkbox(
                "Use time-series split",
                value=True,
                help="Recommended to avoid look-ahead bias"
            )

        if st.button("Train Model", type="primary"):
            status_placeholder = st.empty()
            progress_bar = st.progress(0)

            status_placeholder.info("⚙️ Initializing LightGBM model...")
            progress_bar.progress(10)

            # Initialize model
            model = ReturnDirectionModel(test_size=test_size)

            status_placeholder.info("🔄 Training model... This may take 1-2 minutes...")
            progress_bar.progress(30)

            # Train
            metrics = model.train_model(data, use_time_series_split=use_time_series_split)

            progress_bar.progress(80)
            status_placeholder.info("💾 Saving model...")

            # Save model
            model.save_model(self.model_file)
            st.session_state['model'] = model

            progress_bar.progress(100)
            status_placeholder.markdown('<div class="status-success">✅ Model training complete!</div>', unsafe_allow_html=True)

            # Display comprehensive results
            st.success("🎉 Model training complete!")

            # Show model info box
            self.display_model_info_box(model)

            # Metrics
            st.subheader("📊 Model Performance Metrics")
            col1, col2, col3, col4, col5 = st.columns(5)

            with col1:
                st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
            with col2:
                st.metric("Precision", f"{metrics['precision']:.3f}")
            with col3:
                st.metric("Recall", f"{metrics['recall']:.3f}")
            with col4:
                st.metric("F1 Score", f"{metrics['f1']:.3f}")
            with col5:
                st.metric("ROC AUC", f"{metrics['roc_auc']:.3f}")

            # Confusion matrix
            st.subheader("🎯 Confusion Matrix")
            cm = metrics['confusion_matrix']
            fig = go.Figure(data=go.Heatmap(
                z=cm,
                x=['Predicted Negative', 'Predicted Positive'],
                y=['Actual Negative', 'Actual Positive'],
                colorscale='Blues',
                text=cm,
                texttemplate='%{text}',
                textfont={"size": 16}
            ))
            fig.update_layout(
                title="Confusion Matrix",
                xaxis_title="Predicted",
                yaxis_title="Actual",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

            # Feature importance
            st.subheader("⭐ Top 20 Most Important Features")
            st.write("These features have the strongest influence on predicting return direction:")

            feature_imp = model.get_feature_importance(top_n=20)

            fig = px.bar(
                feature_imp,
                x='importance',
                y='feature',
                orientation='h',
                title="Feature Importance (Gain)",
                color='importance',
                color_continuous_scale='Blues'
            )
            fig.update_layout(
                yaxis={'categoryorder': 'total ascending'},
                height=600
            )
            st.plotly_chart(fig, use_container_width=True)

            # Show top features as metrics
            st.subheader("🔝 Top 10 Features Breakdown")
            top_10 = feature_imp.head(10)
            cols = st.columns(5)
            for idx, (i, row) in enumerate(top_10.iterrows()):
                with cols[idx % 5]:
                    st.metric(
                        f"#{idx+1} {row['feature'][:15]}...",
                        f"{row['importance']:.0f}",
                        help=f"Full name: {row['feature']}"
                    )

    def stock_screening_page(self):
        """Enhanced page for screening stocks with model info and per-stock analysis"""
        st.markdown('<div class="main-header">🔍 Step 3: Stock Screening & Analysis</div>', unsafe_allow_html=True)

        # Load model and data
        if 'model' not in st.session_state:
            if os.path.exists(self.model_file):
                with st.spinner("Loading saved model..."):
                    model = ReturnDirectionModel()
                    model.load_model(self.model_file)
                    st.session_state['model'] = model
                    st.success("✅ Model loaded from disk")
            else:
                st.warning("⚠️ Please train the model first (Step 2)")
                return
        else:
            model = st.session_state['model']

        # Display model info
        self.display_model_info_box(model)

        # Check if model needs retraining
        if model.model_metadata:
            trained_date = model.model_metadata.get('trained_at', 'Unknown')
            try:
                trained_datetime = datetime.strptime(trained_date, '%Y-%m-%d %H:%M:%S')
                days_old = (datetime.now() - trained_datetime).days
                if days_old > 30:
                    st.markdown(f'<div class="status-warning">⚠️ Model was trained {days_old} days ago. Consider retraining with fresh data.</div>', unsafe_allow_html=True)
            except:
                pass

        st.write("""
        This page scores **all stocks** in the NSE universe using the trained model.
        The model was trained on high-quality stocks with historical data, but screening applies to the entire universe.
        Higher quintiles (Q4, Q5) indicate stocks predicted to have positive returns.
        """)

        # Option to collect fresh screening data
        with st.expander("⚙️ Screening Data Options"):
            st.info("""
            **Important:** Screening requires current fundamental data for all ~2,000 NSE stocks.

            - **Use Cached Data** (Recommended): Fast, uses pre-collected data
            - **Collect Fresh Data**: Slow (~30-60 min), may hit rate limits
            """)

            use_cached_screening = st.checkbox("Use cached screening data if available", value=True)

            if st.button("🔄 Collect Fresh Screening Data", help="Only use this if you need the absolute latest data"):
                status_placeholder = st.empty()
                progress_bar = st.progress(0)

                # Load ENTIRE universe for screening
                status_placeholder.info("🔍 Loading entire NSE stock universe...")
                progress_bar.progress(5)

                all_tickers = self.load_universe_tickers()
                if not all_tickers:
                    all_tickers = self.load_backup_tickers()

                if not all_tickers:
                    st.error("Failed to load stock tickers!")
                    return

                status_placeholder.info(f"✅ Found {len(all_tickers)} stocks in NSE universe")
                progress_bar.progress(10)

                # Collect CURRENT data for all stocks (not historical, just latest)
                status_placeholder.info(f"📥 Collecting current fundamental data for {len(all_tickers)} stocks...")
                status_placeholder.warning("⏱️ This will take 30-60 minutes and may hit rate limits. Please be patient...")
                progress_bar.progress(15)

                # Collect current data (last 6 months to get latest fundamentals)
                screening_data = collect_data_for_universe(all_tickers, lookback_years=0.5)

                if screening_data is None or len(screening_data) == 0:
                    st.error("Failed to collect screening data! Try again later if rate limited.")
                    return

                # Save screening data separately
                screening_cache_file = 'screening_data_cache.pkl'
                with open(screening_cache_file, 'wb') as f:
                    pickle.dump(screening_data, f)

                st.session_state['screening_data'] = screening_data

                progress_bar.progress(100)
                status_placeholder.markdown('<div class="status-success">✅ Screening data collected!</div>', unsafe_allow_html=True)
                st.success(f"Collected data for {screening_data['Ticker'].nunique()} stocks with sufficient data")

        if st.button("Run Screening", type="primary"):
            status_placeholder = st.empty()
            progress_bar = st.progress(0)

            # Load screening data (either cached or fresh)
            screening_cache_file = 'screening_data_cache.pkl'

            if 'screening_data' in st.session_state:
                screening_data = st.session_state['screening_data']
                status_placeholder.info("✅ Using screening data from this session")
            elif use_cached_screening and os.path.exists(screening_cache_file):
                status_placeholder.info("📂 Loading cached screening data...")
                with open(screening_cache_file, 'rb') as f:
                    screening_data = pickle.load(f)
                st.session_state['screening_data'] = screening_data
                status_placeholder.info(f"✅ Loaded screening data for {screening_data['Ticker'].nunique()} stocks")
            else:
                st.error("⚠️ No screening data available! Please collect fresh screening data first using the button above.")
                return

            progress_bar.progress(30)

            status_placeholder.info("📊 Scoring all stocks with the trained model...")
            progress_bar.progress(50)

            # Initialize scorer
            scorer = StockScorer(model)

            # Score entire universe
            results = scorer.score_current_universe(screening_data)

            progress_bar.progress(90)
            status_placeholder.info("💾 Saving results...")

            # Save results
            with open(self.results_file, 'wb') as f:
                pickle.dump(results, f)

            st.session_state['results'] = results

            progress_bar.progress(100)
            status_placeholder.markdown('<div class="status-success">✅ Screening complete!</div>', unsafe_allow_html=True)

            st.success(f"Analyzed {len(results)} stocks from the entire NSE universe!")

        # Display results if available
        if 'results' in st.session_state or os.path.exists(self.results_file):
            if 'results' not in st.session_state:
                with open(self.results_file, 'rb') as f:
                    results = pickle.load(f)
                st.session_state['results'] = results
            else:
                results = st.session_state['results']

            # Filter options
            st.subheader("🔎 Filter Results")
            col1, col2, col3 = st.columns(3)

            with col1:
                # Get available quintiles and set safe defaults
                available_quintiles = sorted(results['quality_quintile'].unique().tolist(), reverse=True)
                # Only use Q5 and Q4 as defaults if they exist
                default_quintiles = [q for q in ['Q5', 'Q4'] if q in available_quintiles]
                # If no Q5/Q4, use all available quintiles
                if not default_quintiles:
                    default_quintiles = available_quintiles

                quintile_filter = st.multiselect(
                    "Quality Quintile",
                    options=available_quintiles,
                    default=default_quintiles,
                    help="Q5 = Highest quality, Q1 = Lowest quality"
                )

            with col2:
                min_probability = st.slider(
                    "Minimum Probability",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.5,
                    step=0.05,
                    help="Minimum predicted probability of positive returns"
                )

            with col3:
                sort_by = st.selectbox(
                    "Sort By",
                    options=['predicted_probability', 'composite_score', 'Ticker'],
                    index=0
                )

            # Apply filters
            filtered = results[
                (results['quality_quintile'].isin(quintile_filter)) &
                (results['predicted_probability'] >= min_probability)
            ].sort_values(sort_by, ascending=False)

            st.write(f"**{len(filtered)}** stocks match your criteria")

            # Display filtered results
            st.subheader("📋 Screening Results")

            # Add more columns for comprehensive view
            display_cols = [
                'Ticker', 'quality_quintile', 'composite_score', 'predicted_probability',
                'trailing_pe', 'price_to_book', 'roe', 'revenue_growth',
                'debt_to_equity', 'dividend_yield', 'ROC_12M', 'RSI'
            ]

            # Filter to only columns that exist
            available_cols = [col for col in display_cols if col in filtered.columns]

            if len(available_cols) > 0:
                display_df = filtered[available_cols].copy()

                # Format numeric columns for display
                format_dict = {}
                if 'predicted_probability' in display_df.columns:
                    display_df['Probability'] = display_df['predicted_probability'].apply(lambda x: f"{x:.1%}")
                    format_dict['Probability'] = 'string'
                if 'composite_score' in display_df.columns:
                    display_df['Score'] = display_df['composite_score'].apply(lambda x: f"{x:.1f}")
                    format_dict['Score'] = 'string'
                if 'trailing_pe' in display_df.columns:
                    display_df['P/E'] = display_df['trailing_pe'].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "N/A")
                if 'roe' in display_df.columns:
                    display_df['ROE'] = display_df['roe'].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")
                if 'revenue_growth' in display_df.columns:
                    display_df['Rev Growth'] = display_df['revenue_growth'].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")

                # Reorder with formatted columns
                display_order = ['Ticker', 'quality_quintile', 'Score', 'Probability']
                if 'P/E' in display_df.columns:
                    display_order.append('P/E')
                if 'ROE' in display_df.columns:
                    display_order.append('ROE')
                if 'Rev Growth' in display_df.columns:
                    display_order.append('Rev Growth')

                # Use only columns that exist
                display_order = [col for col in display_order if col in display_df.columns]

                st.info(f"💡 **Tip:** Composite Score is calculated from all fundamental metrics. Higher score = Better quality stock.")

                st.dataframe(
                    display_df[display_order],
                    use_container_width=True,
                    height=400,
                    hide_index=True
                )

            # Download button
            csv = filtered.to_csv(index=False)
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name=f"stock_screening_results_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                type="primary"
            )

            # Per-stock deep dive analysis
            st.subheader("🔬 Deep Dive Stock Analysis")
            st.write("Select any stock for comprehensive analysis:")

            col_select, col_button = st.columns([3, 1])
            with col_select:
                selected_ticker = st.selectbox(
                    "Select Stock for Deep Dive",
                    options=filtered['Ticker'].tolist(),
                    index=0 if len(filtered) > 0 else None
                )

            with col_button:
                st.write("")  # Spacing
                analyze_button = st.button("🔍 Analyze", type="primary")

            if selected_ticker and analyze_button:
                with st.spinner(f"Performing deep dive on {selected_ticker}..."):
                    try:
                        # Get stock data
                        stock_data = filtered[filtered['Ticker'] == selected_ticker].iloc[0]

                        st.markdown(f"## 📈 {selected_ticker.replace('.NS', '')} - Complete Analysis")

                        # Key metrics at top
                        metric_cols = st.columns(5)
                        with metric_cols[0]:
                            st.metric("Quintile", stock_data['quality_quintile'])
                        with metric_cols[1]:
                            st.metric("Composite Score", f"{stock_data['composite_score']:.1f}")
                        with metric_cols[2]:
                            st.metric("Probability", f"{stock_data['predicted_probability']:.1%}")
                        with metric_cols[3]:
                            if 'trailing_pe' in stock_data and pd.notna(stock_data['trailing_pe']):
                                st.metric("P/E Ratio", f"{stock_data['trailing_pe']:.1f}")
                        with metric_cols[4]:
                            if 'roe' in stock_data and pd.notna(stock_data['roe']):
                                st.metric("ROE", f"{stock_data['roe']:.1%}")

                        st.markdown("---")

                        # Three column layout for comprehensive info
                        col1, col2, col3 = st.columns(3)

                        # Column 1: Valuation Metrics
                        with col1:
                            st.markdown("### 💰 Valuation")
                            valuation_metrics = {}
                            if 'trailing_pe' in stock_data:
                                valuation_metrics['P/E Ratio'] = f"{stock_data['trailing_pe']:.2f}" if pd.notna(stock_data['trailing_pe']) else "N/A"
                            if 'forward_pe' in stock_data:
                                valuation_metrics['Forward P/E'] = f"{stock_data['forward_pe']:.2f}" if pd.notna(stock_data['forward_pe']) else "N/A"
                            if 'price_to_book' in stock_data:
                                valuation_metrics['P/B Ratio'] = f"{stock_data['price_to_book']:.2f}" if pd.notna(stock_data['price_to_book']) else "N/A"
                            if 'enterprise_to_ebitda' in stock_data:
                                valuation_metrics['EV/EBITDA'] = f"{stock_data['enterprise_to_ebitda']:.2f}" if pd.notna(stock_data['enterprise_to_ebitda']) else "N/A"
                            if 'dividend_yield' in stock_data:
                                valuation_metrics['Dividend Yield'] = f"{stock_data['dividend_yield']:.2%}" if pd.notna(stock_data['dividend_yield']) else "N/A"

                            for metric, value in valuation_metrics.items():
                                st.text(f"{metric}: {value}")

                        # Column 2: Profitability & Growth
                        with col2:
                            st.markdown("### 📊 Profitability & Growth")
                            profitability_metrics = {}
                            if 'roe' in stock_data:
                                profitability_metrics['ROE'] = f"{stock_data['roe']:.2%}" if pd.notna(stock_data['roe']) else "N/A"
                            if 'roa' in stock_data:
                                profitability_metrics['ROA'] = f"{stock_data['roa']:.2%}" if pd.notna(stock_data['roa']) else "N/A"
                            if 'profit_margin' in stock_data:
                                profitability_metrics['Profit Margin'] = f"{stock_data['profit_margin']:.2%}" if pd.notna(stock_data['profit_margin']) else "N/A"
                            if 'revenue_growth' in stock_data:
                                profitability_metrics['Revenue Growth'] = f"{stock_data['revenue_growth']:.2%}" if pd.notna(stock_data['revenue_growth']) else "N/A"
                            if 'earnings_growth' in stock_data:
                                profitability_metrics['Earnings Growth'] = f"{stock_data['earnings_growth']:.2%}" if pd.notna(stock_data['earnings_growth']) else "N/A"

                            for metric, value in profitability_metrics.items():
                                st.text(f"{metric}: {value}")

                        # Column 3: Risk & Technical
                        with col3:
                            st.markdown("### ⚠️ Risk & Technical")
                            risk_metrics = {}
                            if 'debt_to_equity' in stock_data:
                                risk_metrics['Debt/Equity'] = f"{stock_data['debt_to_equity']:.2f}" if pd.notna(stock_data['debt_to_equity']) else "N/A"
                            if 'current_ratio' in stock_data:
                                risk_metrics['Current Ratio'] = f"{stock_data['current_ratio']:.2f}" if pd.notna(stock_data['current_ratio']) else "N/A"
                            if 'beta' in stock_data:
                                risk_metrics['Beta'] = f"{stock_data['beta']:.2f}" if pd.notna(stock_data['beta']) else "N/A"
                            if 'ROC_12M' in stock_data:
                                risk_metrics['12M Return'] = f"{stock_data['ROC_12M']:.2%}" if pd.notna(stock_data['ROC_12M']) else "N/A"
                            if 'RSI' in stock_data:
                                risk_metrics['RSI'] = f"{stock_data['RSI']:.1f}" if pd.notna(stock_data['RSI']) else "N/A"

                            for metric, value in risk_metrics.items():
                                st.text(f"{metric}: {value}")

                        st.markdown("---")

                        # Model prediction breakdown
                        st.markdown("### 🤖 Model Prediction Breakdown")
                        st.write("Understanding how the model arrived at its prediction:")

                        contrib_df, pred_proba = model.explain_prediction(results, ticker=selected_ticker)

                        # Show top positive and negative contributors
                        col_pos, col_neg = st.columns(2)

                        with col_pos:
                            st.markdown("#### ✅ Top Positive Drivers")
                            top_positive = contrib_df[contrib_df['contribution'] > 0].head(10)
                            if len(top_positive) > 0:
                                for idx, row in top_positive.iterrows():
                                    st.text(f"• {row['feature']}: +{row['contribution']:.2f}")
                            else:
                                st.text("No strong positive drivers")

                        with col_neg:
                            st.markdown("#### ❌ Top Negative Drivers")
                            top_negative = contrib_df[contrib_df['contribution'] < 0].head(10)
                            if len(top_negative) > 0:
                                for idx, row in top_negative.iterrows():
                                    st.text(f"• {row['feature']}: {row['contribution']:.2f}")
                            else:
                                st.text("No strong negative drivers")

                        # Feature contribution chart
                        st.markdown("#### 📊 Feature Contribution Chart")
                        top_contrib = contrib_df.head(20)

                        fig = px.bar(
                            top_contrib,
                            x='contribution',
                            y='feature',
                            orientation='h',
                            title=f"Top 20 Feature Contributions for {selected_ticker}",
                            color='contribution',
                            color_continuous_scale='RdYlGn',
                            labels={'contribution': 'Contribution Score', 'feature': 'Feature'}
                        )
                        fig.update_layout(
                            yaxis={'categoryorder': 'total ascending'},
                            height=600,
                            showlegend=False
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        # Detailed feature table
                        with st.expander("📋 View All Feature Details"):
                            display_contrib = contrib_df[['feature', 'value', 'importance', 'contribution']].copy()
                            display_contrib.columns = ['Feature', 'Current Value', 'Model Importance', 'Contribution']
                            display_contrib['Current Value'] = display_contrib['Current Value'].apply(lambda x: f"{x:.4f}" if not pd.isna(x) else "N/A")
                            display_contrib['Model Importance'] = display_contrib['Model Importance'].apply(lambda x: f"{x:.0f}")
                            display_contrib['Contribution'] = display_contrib['Contribution'].apply(lambda x: f"{x:.2f}")
                            st.dataframe(display_contrib, use_container_width=True, height=400)

                    except Exception as e:
                        st.error(f"Error analyzing stock: {str(e)}")
                        st.exception(e)

            # Quintile distribution
            st.subheader("📊 Quintile Distribution")
            quintile_counts = results['quality_quintile'].value_counts().sort_index(ascending=False)

            fig = px.bar(
                x=quintile_counts.index,
                y=quintile_counts.values,
                labels={'x': 'Quality Quintile', 'y': 'Number of Stocks'},
                title="Distribution of Stocks by Quality Quintile",
                color=quintile_counts.values,
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig, use_container_width=True)

    def run(self):
        """Run the application"""
        st.sidebar.title("📈 Stock Screener Navigation")

        page = st.sidebar.radio(
            "Choose a step:",
            ["📊 Data Collection", "🤖 Model Training", "🔍 Stock Screening"]
        )

        if page == "📊 Data Collection":
            self.data_collection_page()
        elif page == "🤖 Model Training":
            self.model_training_page()
        elif page == "🔍 Stock Screening":
            self.stock_screening_page()

        # Sidebar info
        st.sidebar.markdown("---")
        st.sidebar.markdown("### ℹ️ About")
        st.sidebar.info("""
        This application uses LightGBM to predict stock return direction based on:
        - 🔢 Fundamental metrics (80% weight)
        - 📈 Technical indicators (20% weight)

        Model predictions help identify high-quality stocks for further research.
        """)

        # Show current status
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📊 Current Status")

        if os.path.exists(self.data_file):
            st.sidebar.success("✅ Data collected")
        else:
            st.sidebar.warning("⚠️ No data collected")

        if os.path.exists(self.model_file):
            try:
                model = ReturnDirectionModel()
                model.load_model(self.model_file)
                meta = model.model_metadata
                trained_date = meta.get('trained_at', 'Unknown')
                st.sidebar.success(f"✅ Model trained\n\n📅 {trained_date}")
            except:
                st.sidebar.success("✅ Model file exists")
        else:
            st.sidebar.warning("⚠️ No trained model")

        if os.path.exists(self.results_file):
            st.sidebar.success("✅ Screening results available")
        else:
            st.sidebar.warning("⚠️ No screening results")


if __name__ == "__main__":
    app = EnhancedStockScreener()
    app.run()
