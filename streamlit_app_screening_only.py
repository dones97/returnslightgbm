"""
Stock Screening App - Screening Only

This version focuses solely on:
1. Model Insights - View trained model info and feature importance
2. Stock Screening - Score and filter stocks using the trained model

Training is handled via GitHub Actions workflows.
"""

import streamlit as st
import pandas as pd
import pickle
import os
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from model_trainer import ReturnDirectionModel, StockScorer
from data_collector_enhanced import collect_data_for_universe


# Page config
st.set_page_config(
    page_title="Stock Screener - ML Powered",
    page_icon="📈",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .status-success { padding: 1rem; background-color: #d4edda; border-radius: 0.5rem; color: #155724; }
    .status-warning { padding: 1rem; background-color: #fff3cd; border-radius: 0.5rem; color: #856404; }
    .status-error { padding: 1rem; background-color: #f8d7da; border-radius: 0.5rem; color: #721c24; }
</style>
""", unsafe_allow_html=True)


class StockScreenerApp:
    """Stock screening application"""

    def __init__(self):
        self.model_file = 'trained_model.pkl'
        self.screening_cache = 'screening_data_cache.pkl'
        self.results_file = 'screening_results.pkl'

    def load_model(self):
        """Load the trained model"""
        if not os.path.exists(self.model_file):
            return None

        model = ReturnDirectionModel()
        model.load_model(self.model_file)
        return model

    def model_insights_page(self):
        """Page 1: Model Insights"""
        st.title("📊 Model Insights")

        # Load model
        model = self.load_model()

        if model is None:
            st.error("⚠️ No trained model found!")
            st.info("""
            **Model Training**

            The model is trained automatically via GitHub Actions:
            - **Monthly:** Automated training on 1st of each month
            - **Manual:** Trigger via GitHub Actions → "Train Model" workflow

            Once trained, the model is committed to the repository and appears here.
            """)
            return

        # Display model metadata
        meta = model.model_metadata

        st.markdown('<div class="status-success">✅ Model loaded successfully</div>', unsafe_allow_html=True)

        # Top-level metrics
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            trained_date = meta.get('trained_at', 'Unknown')
            st.metric("Training Date", trained_date.split()[0] if ' ' in trained_date else trained_date)

        with col2:
            st.metric("Training Stocks", f"{meta.get('n_stocks', 'N/A')}")

        with col3:
            accuracy = meta.get('metrics', {}).get('accuracy', 0)
            st.metric("Accuracy", f"{accuracy:.1%}")

        with col4:
            roc_auc = meta.get('metrics', {}).get('roc_auc', 0)
            st.metric("ROC-AUC", f"{roc_auc:.3f}")

        with col5:
            n_features = meta.get('n_features', 'N/A')
            st.metric("Features", f"{n_features}")

        st.markdown("---")

        # Check if model needs retraining
        if 'trained_at' in meta:
            try:
                trained_datetime = datetime.strptime(meta['trained_at'], '%Y-%m-%d %H:%M:%S')
                days_old = (datetime.now() - trained_datetime).days

                if days_old > 30:
                    st.warning(f"⚠️ Model is {days_old} days old. Consider retraining via GitHub Actions.")
                elif days_old > 60:
                    st.error(f"❌ Model is {days_old} days old. Retrain via GitHub Actions recommended!")
                else:
                    st.success(f"✅ Model is {days_old} days old (fresh)")
            except:
                pass

        # Two-column layout
        col_left, col_right = st.columns(2)

        with col_left:
            # Model Performance
            st.subheader("📈 Model Performance")

            if 'metrics' in meta:
                metrics = meta['metrics']

                # Performance metrics table
                perf_data = {
                    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC'],
                    'Train/Test': [
                        f"{metrics.get('accuracy', 0):.1%}",
                        f"{metrics.get('precision', 0):.1%}",
                        f"{metrics.get('recall', 0):.1%}",
                        f"{metrics.get('f1', 0):.1%}",
                        f"{metrics.get('roc_auc', 0):.3f}"
                    ],
                    'Interpretation': [
                        'Overall correctness',
                        'When predicting UP, how often correct?',
                        'Of all UP stocks, how many caught?',
                        'Balance of precision and recall',
                        'Ranking quality (0.5=random, 1.0=perfect)'
                    ]
                }

                st.dataframe(pd.DataFrame(perf_data), use_container_width=True, hide_index=True)

                # Performance gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=metrics.get('accuracy', 0) * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Model Accuracy", 'font': {'size': 24}},
                    delta={'reference': 50, 'increasing': {'color': "green"}},
                    gauge={
                        'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                        'bar': {'color': "darkblue"},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "gray",
                        'steps': [
                            {'range': [0, 50], 'color': '#ffcccc'},
                            {'range': [50, 60], 'color': '#ffffcc'},
                            {'range': [60, 100], 'color': '#ccffcc'}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 70
                        }
                    }
                ))

                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)

        with col_right:
            # Dataset Information
            st.subheader("📦 Training Dataset")

            dataset_info = {
                'Property': ['Stocks', 'Training Samples', 'Test Samples', 'Features', 'Date Range', 'Split Method'],
                'Value': [
                    f"{meta.get('n_stocks', 'N/A')}",
                    f"{meta.get('n_training_samples', 'N/A'):,}",
                    f"{meta.get('n_test_samples', 'N/A'):,}",
                    f"{meta.get('n_features', 'N/A')}",
                    meta.get('data_date_range', 'N/A'),
                    'Time-series (80/20)' if meta.get('use_time_series_split') else 'Random'
                ]
            }

            st.dataframe(pd.DataFrame(dataset_info), use_container_width=True, hide_index=True)

            # Hyperparameters
            with st.expander("🔧 Model Hyperparameters"):
                if 'hyperparameters' in meta:
                    params = meta['hyperparameters']
                    param_df = pd.DataFrame([
                        {'Parameter': k, 'Value': str(v)}
                        for k, v in params.items()
                    ])
                    st.dataframe(param_df, use_container_width=True, hide_index=True)

        st.markdown("---")

        # Feature Importance
        st.subheader("🎯 Feature Importance")

        st.info("💡 **Feature Importance** shows which factors the model considers most important for predicting stock returns.")

        # Top 20 features
        top_features = model.get_feature_importance(top_n=20)

        fig = px.bar(
            top_features,
            x='importance',
            y='feature',
            orientation='h',
            title="Top 20 Most Important Features",
            labels={'importance': 'Importance Score', 'feature': 'Feature'},
            color='importance',
            color_continuous_scale='Blues'
        )

        fig.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            height=600,
            showlegend=False
        )

        st.plotly_chart(fig, use_container_width=True)

        # Feature importance table
        with st.expander("📋 View All Features"):
            all_features = model.feature_importance
            all_features['importance_pct'] = (all_features['importance'] / all_features['importance'].sum() * 100).apply(lambda x: f"{x:.1f}%")
            display_features = all_features[['feature', 'importance', 'importance_pct']].copy()
            display_features.columns = ['Feature', 'Importance Score', '% of Total']
            st.dataframe(display_features, use_container_width=True, height=400, hide_index=True)

    def screening_page(self):
        """Page 2: Stock Screening"""
        st.title("🔍 Stock Screening")

        # Load model
        model = self.load_model()

        if model is None:
            st.error("⚠️ No trained model found! Check Model Insights page for details.")
            return

        # Show model info
        meta = model.model_metadata
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Model Trained", meta.get('trained_at', 'Unknown').split()[0])
        with col2:
            accuracy = meta.get('metrics', {}).get('accuracy', 0)
            st.metric("Accuracy", f"{accuracy:.1%}")
        with col3:
            roc_auc = meta.get('metrics', {}).get('roc_auc', 0)
            st.metric("ROC-AUC", f"{roc_auc:.3f}")

        st.markdown("---")

        st.write("""
        This page scores **all stocks** in the NSE universe using the trained model.
        The model was trained on 195 quality stocks, but screening applies to the entire market.
        """)

        # Screening data options
        with st.expander("⚙️ Screening Data Options"):
            st.info("""
            **Important:** Screening requires current fundamental data for all ~2,000 NSE stocks.

            - **Use Cached Data** (Recommended): Fast, uses pre-collected data
            - **Collect Fresh Data**: Slow (~30-60 min), gets latest fundamentals (do weekly/monthly)
            """)

            use_cached_screening = st.checkbox("Use cached screening data if available", value=True)

            if st.button("🔄 Collect Fresh Screening Data", help="Only use this if you need the absolute latest data"):
                st.warning("This feature will be implemented. For now, screening data collection happens via GitHub Actions.")

        # Load screening results
        st.markdown("### 📋 Screening Results")

        # Check if screening results exist
        if not os.path.exists('screening_results.pkl'):
            st.warning("""
            ⚠️ **No screening results found!**

            Screening results are generated by GitHub Actions workflow.

            **To get results:**
            1. Go to GitHub → Actions → Weekly Stock Screening
            2. Run the workflow (or wait for weekly run)
            3. Download 'screening-results' artifact
            4. Extract `screening_results.pkl` to repository root
            5. Refresh this page

            Or run: `python scripts/automated_screening.py` locally
            """)
            return

        # Load results
        try:
            with open('screening_results.pkl', 'rb') as f:
                results = pickle.load(f)

            st.success(f"✅ Loaded screening results for {len(results)} stocks")

            # Filters
            st.markdown("### 🎯 Filter Results")

            col1, col2, col3 = st.columns(3)

            with col1:
                quintiles = st.multiselect(
                    "Quality Quintile",
                    options=['Q5 (Highest)', 'Q4 (High)', 'Q3 (Medium)', 'Q2 (Low)', 'Q1 (Lowest)'],
                    default=['Q5 (Highest)', 'Q4 (High)']
                )

            with col2:
                min_prob = st.slider("Min Probability", 0, 100, 50, help="Minimum predicted probability of positive return") / 100

            with col3:
                sort_by = st.selectbox(
                    "Sort by",
                    options=['composite_score', 'predicted_probability', 'ROE', 'ROC_12M'],
                    index=0
                )

            # Apply filters
            filtered = results.copy()

            if quintiles:
                filtered = filtered[filtered['quality_quintile'].isin(quintiles)]

            filtered = filtered[filtered['predicted_probability'] >= min_prob]

            # Sort
            filtered = filtered.sort_values(sort_by, ascending=False)

            st.markdown(f"### 📊 Filtered Results: {len(filtered)} stocks")

            # Display results
            display_cols = [
                'Ticker', 'quality_quintile', 'composite_score', 'predicted_probability',
                'ROE', 'ROC_12M', 'F_Score', 'Z_Score', 'PE_Ratio', 'Debt_to_Equity'
            ]

            # Check which columns exist
            display_cols = [col for col in display_cols if col in filtered.columns]

            # Format for display
            display_df = filtered[display_cols].copy()
            display_df['Ticker'] = display_df['Ticker'].str.replace('.NS', '')

            # Format numeric columns
            if 'predicted_probability' in display_df.columns:
                display_df['predicted_probability'] = display_df['predicted_probability'].apply(lambda x: f"{x:.1%}")
            if 'composite_score' in display_df.columns:
                display_df['composite_score'] = display_df['composite_score'].apply(lambda x: f"{x:.1f}")
            if 'ROE' in display_df.columns:
                display_df['ROE'] = display_df['ROE'].apply(lambda x: f"{x:.1%}" if pd.notna(x) else 'N/A')
            if 'ROC_12M' in display_df.columns:
                display_df['ROC_12M'] = display_df['ROC_12M'].apply(lambda x: f"{x:.1%}" if pd.notna(x) else 'N/A')

            # Display table
            st.dataframe(
                display_df.head(50),
                use_container_width=True,
                height=600
            )

            # Download button
            csv = filtered.to_csv(index=False)
            st.download_button(
                label="📥 Download Full Results (CSV)",
                data=csv,
                file_name=f"screening_results_filtered_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

            # Summary statistics
            st.markdown("### 📈 Summary Statistics")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Stocks", len(filtered))

            with col2:
                avg_prob = filtered['predicted_probability'].mean()
                st.metric("Avg Probability", f"{avg_prob:.1%}")

            with col3:
                if 'ROE' in filtered.columns:
                    avg_roe = filtered['ROE'].median()
                    st.metric("Median ROE", f"{avg_roe:.1%}" if pd.notna(avg_roe) else "N/A")

            with col4:
                if 'composite_score' in filtered.columns:
                    avg_score = filtered['composite_score'].mean()
                    st.metric("Avg Score", f"{avg_score:.1f}")

        except Exception as e:
            st.error(f"Error loading screening results: {str(e)}")
            st.exception(e)


    def run(self):
        """Run the application"""
        st.sidebar.title("📈 Stock Screener")
        st.sidebar.markdown("---")

        page = st.sidebar.radio(
            "Navigation",
            ["📊 Model Insights", "🔍 Stock Screening"]
        )

        st.sidebar.markdown("---")
        st.sidebar.markdown("""
        ### About

        **Model Training:**
        - Automated via GitHub Actions
        - Monthly refresh (1st of month)
        - 195 quality stocks, 5-year data

        **Screening:**
        - Uses trained model
        - Scores entire NSE universe
        - Weekly automated runs
        """)

        # Route to pages
        if page == "📊 Model Insights":
            self.model_insights_page()
        elif page == "🔍 Stock Screening":
            self.screening_page()


if __name__ == "__main__":
    app = StockScreenerApp()
    app.run()
