"""
Streamlit Application for Indian Stock Screening
ML-powered stock screener using LightGBM to predict return direction
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


class StreamlitStockScreener:
    """Main application class"""

    def __init__(self):
        self.data_file = 'stock_data_cache.pkl'
        self.model_file = 'trained_model.pkl'
        self.results_file = 'screening_results.pkl'

    def load_universe_tickers(self) -> list:
        """Load stock tickers from NSE_Universe.csv"""
        try:
            df = pd.read_csv('NSE_Universe.csv')
            # Get tickers and add .NS suffix for NSE stocks
            tickers = df['Ticker'].dropna().unique().tolist()
            # Add .NS suffix if not already present
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

    def data_collection_page(self):
        """Page for collecting data"""
        st.markdown('<div class="main-header">Step 1: Data Collection</div>', unsafe_allow_html=True)

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
                max_value=2000,
                value=200,
                help="Start with 200 for testing. Use 1500+ for full universe"
            )

        use_cached = st.checkbox("Use cached data if available", value=True)

        if st.button("Start Data Collection", type="primary"):
            # Check if cached data exists
            if use_cached and os.path.exists(self.data_file):
                st.info("Loading cached data...")
                with open(self.data_file, 'rb') as f:
                    data = pickle.load(f)
                st.session_state['data'] = data
                st.success(f"Loaded {len(data)} rows of cached data!")
                st.dataframe(data.head())
                return

            # Load tickers
            with st.spinner("Loading stock universe..."):
                tickers = self.load_universe_tickers()

                if not tickers:
                    tickers = self.load_backup_tickers()

                if not tickers:
                    st.error("Failed to load stock tickers!")
                    return

                st.info(f"Found {len(tickers)} tickers in universe")

                # Limit number of stocks
                tickers = tickers[:max_stocks]

            # Collect data
            with st.spinner(f"Collecting data for {len(tickers)} stocks... This may take a while."):
                progress_bar = st.progress(0)
                status_text = st.empty()

                # Use a simpler approach for progress tracking
                data = collect_data_for_universe(tickers, lookback_years=lookback_years)

                progress_bar.progress(100)

            if data.empty:
                st.error("Failed to collect data!")
                return

            # Save data
            with open(self.data_file, 'wb') as f:
                pickle.dump(data, f)

            st.session_state['data'] = data

            st.success(f"Successfully collected data for {data['Ticker'].nunique()} stocks!")
            st.write(f"Total records: {len(data)}")
            st.write(f"Date range: {data['Date'].min()} to {data['Date'].max()}")

            # Show sample
            st.subheader("Sample Data")
            st.dataframe(data.head(20))

            # Show statistics
            st.subheader("Data Statistics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Stocks", data['Ticker'].nunique())
            with col2:
                st.metric("Total Records", len(data))
            with col3:
                st.metric("Features", len(data.columns))

    def model_training_page(self):
        """Page for training the model"""
        st.markdown('<div class="main-header">Step 2: Model Training</div>', unsafe_allow_html=True)

        if 'data' not in st.session_state:
            st.warning("Please collect data first (Step 1)")
            return

        data = st.session_state['data']

        st.write(f"""
        Training LightGBM model on {len(data)} records from {data['Ticker'].nunique()} stocks.
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
            with st.spinner("Training LightGBM model..."):
                # Initialize model
                model = ReturnDirectionModel(test_size=test_size)

                # Train
                metrics = model.train_model(data, use_time_series_split=use_time_series_split)

                # Save model
                model.save_model(self.model_file)
                st.session_state['model'] = model

            # Display results
            st.success("Model training complete!")

            # Metrics
            st.subheader("Model Performance")
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
            st.subheader("Confusion Matrix")
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
                yaxis_title="Actual"
            )
            st.plotly_chart(fig, use_container_width=True)

            # Feature importance
            st.subheader("Top 20 Most Important Features")
            feature_imp = model.get_feature_importance(top_n=20)

            fig = px.bar(
                feature_imp,
                x='importance',
                y='feature',
                orientation='h',
                title="Feature Importance (Gain)"
            )
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)

    def stock_screening_page(self):
        """Page for screening stocks"""
        st.markdown('<div class="main-header">Step 3: Stock Screening</div>', unsafe_allow_html=True)

        # Load model and data
        if 'model' not in st.session_state:
            if os.path.exists(self.model_file):
                st.info("Loading saved model...")
                model = ReturnDirectionModel()
                model.load_model(self.model_file)
                st.session_state['model'] = model
            else:
                st.warning("Please train the model first (Step 2)")
                return

        if 'data' not in st.session_state:
            if os.path.exists(self.data_file):
                st.info("Loading cached data...")
                with open(self.data_file, 'rb') as f:
                    data = pickle.load(f)
                st.session_state['data'] = data
            else:
                st.warning("Please collect data first (Step 1)")
                return

        model = st.session_state['model']
        data = st.session_state['data']

        st.write("""
        This page scores all stocks based on their characteristics and assigns them to quality quintiles.
        """)

        if st.button("Run Screening", type="primary"):
            with st.spinner("Scoring stocks..."):
                # Initialize scorer
                scorer = StockScorer(model)

                # Score universe
                results = scorer.score_current_universe(data)

                # Save results
                with open(self.results_file, 'wb') as f:
                    pickle.dump(results, f)

                st.session_state['results'] = results

            st.success(f"Screening complete! Analyzed {len(results)} stocks.")

        # Display results if available
        if 'results' in st.session_state or os.path.exists(self.results_file):
            if 'results' not in st.session_state:
                with open(self.results_file, 'rb') as f:
                    results = pickle.load(f)
                st.session_state['results'] = results
            else:
                results = st.session_state['results']

            # Filter options
            st.subheader("Filter Results")
            col1, col2, col3 = st.columns(3)

            with col1:
                quintile_filter = st.multiselect(
                    "Quality Quintile",
                    options=results['quality_quintile'].unique().tolist(),
                    default=results['quality_quintile'].unique().tolist()
                )

            with col2:
                min_score = st.slider(
                    "Minimum Composite Score",
                    min_value=0.0,
                    max_value=100.0,
                    value=0.0
                )

            with col3:
                min_prob = st.slider(
                    "Minimum Return Probability",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.0,
                    step=0.05
                )

            # Apply filters
            filtered_results = results[
                (results['quality_quintile'].isin(quintile_filter)) &
                (results['composite_score'] >= min_score) &
                (results['return_probability'] >= min_prob)
            ].copy()

            # Sort by composite score
            filtered_results = filtered_results.sort_values('composite_score', ascending=False)

            # Display summary
            st.subheader("Screening Results")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Stocks", len(filtered_results))
            with col2:
                q5_count = len(filtered_results[filtered_results['quality_quintile'] == 'Q5 (Highest)'])
                st.metric("Q5 (Highest Quality)", q5_count)
            with col3:
                avg_score = filtered_results['composite_score'].mean()
                st.metric("Avg Composite Score", f"{avg_score:.1f}")
            with col4:
                avg_prob = filtered_results['return_probability'].mean()
                st.metric("Avg Return Probability", f"{avg_prob:.2%}")

            # Quintile distribution
            st.subheader("Distribution by Quintile")
            quintile_counts = filtered_results['quality_quintile'].value_counts().sort_index()
            fig = px.bar(
                x=quintile_counts.index,
                y=quintile_counts.values,
                labels={'x': 'Quality Quintile', 'y': 'Number of Stocks'},
                title="Stock Distribution by Quality Quintile"
            )
            st.plotly_chart(fig, use_container_width=True)

            # Display table
            st.subheader("Stock Rankings")

            # Select columns to display
            display_cols = [
                'Ticker', 'quality_quintile', 'composite_score', 'return_probability',
                'trailing_pe', 'price_to_book', 'roe', 'revenue_growth',
                'debt_to_equity', 'dividend_yield', 'beta', 'ROC_12M'
            ]

            # Filter to existing columns
            display_cols = [col for col in display_cols if col in filtered_results.columns]

            display_df = filtered_results[display_cols].copy()

            # Format numeric columns
            format_dict = {
                'composite_score': '{:.1f}',
                'return_probability': '{:.2%}',
                'trailing_pe': '{:.2f}',
                'price_to_book': '{:.2f}',
                'roe': '{:.2%}',
                'revenue_growth': '{:.2%}',
                'debt_to_equity': '{:.2f}',
                'dividend_yield': '{:.2%}',
                'beta': '{:.2f}',
                'ROC_12M': '{:.2%}'
            }

            # Apply formatting
            for col, fmt in format_dict.items():
                if col in display_df.columns:
                    if '%' in fmt:
                        display_df[col] = display_df[col].apply(lambda x: f"{x*100:.2f}%" if pd.notna(x) else "N/A")
                    else:
                        display_df[col] = display_df[col].apply(lambda x: fmt.format(x) if pd.notna(x) else "N/A")

            st.dataframe(
                display_df,
                use_container_width=True,
                height=600
            )

            # Download button
            csv = filtered_results.to_csv(index=False)
            st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name=f"stock_screening_results_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

    def run(self):
        """Main application runner"""
        st.sidebar.title("Navigation")
        page = st.sidebar.radio(
            "Select Page",
            ["Data Collection", "Model Training", "Stock Screening"]
        )

        st.sidebar.markdown("---")
        st.sidebar.subheader("About")
        st.sidebar.info("""
        This application uses machine learning (LightGBM) to screen Indian stocks.

        **Process:**
        1. Collect historical data
        2. Train model to predict return direction
        3. Score stocks and assign quality quintiles

        **Features:**
        - Technical indicators (RSI, MACD, etc.)
        - Fundamental metrics (P/E, ROE, etc.)
        - Momentum factors
        - Risk metrics
        """)

        # Route to appropriate page
        if page == "Data Collection":
            self.data_collection_page()
        elif page == "Model Training":
            self.model_training_page()
        elif page == "Stock Screening":
            self.stock_screening_page()


if __name__ == "__main__":
    app = StreamlitStockScreener()
    app.run()
