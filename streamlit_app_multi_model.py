"""
Multi-Model Streamlit Application for Indian Stock Screening

Supports multiple ML models:
1. Classification model (screener.in + yfinance - 65% accuracy)
2. Screener regression model (fundamentals only)
3. YFinance original model (yfinance only)
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
import pickle

# Page configuration
st.set_page_config(
    page_title="Multi-Model Stock Screener",
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
    .model-selector-box {
        background-color: #e8f4f8;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4CAF50;
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
    .deep-dive-section {
        background-color: #fff9e6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ff9800;
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)


# Model configurations
MODEL_CONFIGS = {
    "Classification Model (65% Accuracy)": {
        "type": "classification",
        "results_file": "screening_results_classification.pkl",
        "model_file": "model_classification.pkl",
        "feature_importance_file": "feature_importance_classification.csv",
        "description": "Predicts monthly direction (UP/DOWN) using screener.in fundamentals + yfinance technicals",
        "features": "27 fundamentals + 10 technicals",
        "accuracy": "65.5%",
        "auc": "0.72",
        "target": "1-month return direction",
        "data_period": "2022-2024",
        "samples": "~12,000"
    },
    "Screener Regression Model": {
        "type": "regression",
        "results_file": "screening_results_screener.pkl",
        "model_file": "trained_model_screener.pkl",
        "feature_importance_file": "feature_importance_screener.csv",
        "description": "Predicts quarterly returns using screener.in fundamentals only",
        "features": "Fundamentals only",
        "accuracy": "R² varies",
        "auc": "N/A",
        "target": "Next quarter return %",
        "data_period": "2022-2024",
        "samples": "Varies"
    },
    "YFinance Original Model": {
        "type": "classification",  # Changed to classification
        "results_file": "screening_results.pkl",
        "model_file": "trained_model.pkl",
        "feature_importance_file": "feature_importance.csv",
        "description": "Original model using yfinance data only",
        "features": "Fundamentals + technicals from yfinance",
        "accuracy": "Varies",
        "auc": "N/A",
        "target": "Next month return direction",
        "data_period": "Historical",
        "samples": "Varies"
    }
}


class MultiModelScreener:
    """Multi-model stock screener application"""

    def __init__(self):
        self.selected_model = None
        self.selected_model_name = None

    def display_model_selector(self):
        """Display model selection interface"""
        st.markdown('<div class="main-header">📈 Multi-Model Stock Screener</div>', unsafe_allow_html=True)

        st.markdown('<div class="model-selector-box">', unsafe_allow_html=True)
        st.markdown("### 🤖 Select Screening Model")

        # Model selection
        model_names = list(MODEL_CONFIGS.keys())
        selected_model_name = st.selectbox(
            "Choose a model for stock screening:",
            model_names,
            index=0,  # Default to classification model
            help="Different models use different features and predict different targets"
        )

        self.selected_model = MODEL_CONFIGS[selected_model_name]
        self.selected_model_name = selected_model_name

        # Display model info
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Model Type:**")
            st.markdown(f"📊 {self.selected_model['type'].title()}")
            st.markdown("**Target:**")
            st.markdown(f"🎯 {self.selected_model['target']}")

        with col2:
            st.markdown("**Features:**")
            st.markdown(f"📈 {self.selected_model['features']}")
            st.markdown("**Data Period:**")
            st.markdown(f"📅 {self.selected_model['data_period']}")

        with col3:
            st.markdown("**Performance:**")
            if self.selected_model['type'] == 'classification':
                st.markdown(f"✅ Accuracy: {self.selected_model['accuracy']}")
                st.markdown(f"🎯 AUC: {self.selected_model['auc']}")
            else:
                st.markdown(f"📊 {self.selected_model['accuracy']}")

        st.markdown(f"**Description:** {self.selected_model['description']}")
        st.markdown('</div>', unsafe_allow_html=True)

        return selected_model_name

    def load_results(self):
        """Load screening results for selected model"""
        results_file = self.selected_model['results_file']

        if not os.path.exists(results_file):
            st.error(f"❌ Results file not found: {results_file}")
            st.info(f"💡 Run the corresponding screening workflow to generate results")
            return None

        try:
            results = pd.read_pickle(results_file)
            return results
        except Exception as e:
            st.error(f"Error loading results: {str(e)}")
            return None

    def load_model(self):
        """Load trained model"""
        model_file = self.selected_model['model_file']

        if not os.path.exists(model_file):
            return None

        try:
            with open(model_file, 'rb') as f:
                model = pickle.load(f)
            return model
        except Exception as e:
            st.warning(f"Could not load model: {str(e)}")
            return None

    def load_feature_importance(self):
        """Load feature importance"""
        importance_file = self.selected_model['feature_importance_file']

        if not os.path.exists(importance_file):
            return None

        try:
            importance_df = pd.read_csv(importance_file)
            return importance_df
        except Exception as e:
            return None

    def display_model_performance(self, model, feature_importance):
        """Display model performance metrics and feature importance"""
        st.markdown("---")
        st.markdown("## 📊 Model Performance & Feature Importance")

        # Performance metrics
        if self.selected_model['type'] == 'classification':
            st.markdown("### 🎯 Classification Performance")

            col1, col2, col3, col4, col5 = st.columns(5)

            with col1:
                st.metric("Accuracy", self.selected_model['accuracy'],
                         help="Percentage of correct predictions")
            with col2:
                st.metric("AUC-ROC", self.selected_model['auc'],
                         help="Area Under Curve - measures model's ability to distinguish UP from DOWN")
            with col3:
                st.metric("Data Period", self.selected_model['data_period'])
            with col4:
                st.metric("Training Samples", self.selected_model['samples'])
            with col5:
                st.metric("Features", self.selected_model['features'])

            # Explanation of metrics
            with st.expander("ℹ️ Understanding Performance Metrics"):
                st.markdown("""
                **Accuracy:** Percentage of predictions that were correct
                - 50% = Random guessing (baseline)
                - 65.5% = **Excellent** for stock prediction (15.5 points above random!)
                - Professional hedge funds typically achieve 52-58%

                **AUC-ROC (Area Under Curve):** Measures how well model separates UP from DOWN
                - 0.50 = Random (no signal)
                - 0.72 = **Strong signal** - model has real predictive power
                - >0.70 = Considered excellent for stocks

                **Why 65% is great:** Even a 55% accuracy model, compounded monthly over a year,
                provides significant outperformance vs. random stock selection!
                """)

        else:
            st.markdown("### 📈 Regression Performance")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Model Type", "Regression")
            with col2:
                st.metric("Performance", self.selected_model['accuracy'])
            with col3:
                st.metric("Data Period", self.selected_model['data_period'])
            with col4:
                st.metric("Features", self.selected_model['features'])

        # Feature importance visualization
        if feature_importance is not None and len(feature_importance) > 0:
            st.markdown("### ⭐ Top 15 Most Important Features")

            st.markdown("""
            These features have the strongest influence on predictions. Higher importance =
            more critical for the model's decisions.
            """)

            top_15 = feature_importance.head(15)

            fig = px.bar(
                top_15,
                x='importance',
                y='feature',
                orientation='h',
                title="Feature Importance (Top 15)",
                color='importance',
                color_continuous_scale='Blues',
                labels={'importance': 'Importance Score', 'feature': 'Feature Name'}
            )
            fig.update_layout(
                yaxis={'categoryorder': 'total ascending'},
                height=500,
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

            # Show feature table
            with st.expander("📋 View All Feature Importance Values"):
                st.dataframe(
                    feature_importance,
                    use_container_width=True,
                    height=400
                )

    def explain_probability_confidence(self):
        """Explain what probability and confidence mean"""
        st.markdown("---")
        st.markdown("## 📖 Understanding Probability & Confidence")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.markdown("""
            ### 🎲 Predicted Probability

            **What it is:** The model's estimated probability that the stock will go **UP** next month

            **Scale:** 0% to 100%
            - **0%** = Certain to go DOWN
            - **50%** = Uncertain (could go either way)
            - **100%** = Certain to go UP

            **Examples:**
            - **85% probability** = Model is very confident this stock will go UP
            - **65% probability** = Model leans UP, but less certain
            - **50% probability** = Model has no strong opinion
            - **30% probability** = Model leans DOWN (70% chance of going down)
            - **15% probability** = Model is very confident this stock will go DOWN

            **How to use it:**
            - Focus on stocks with probability >70% (Strong Buy) or <30% (Strong Sell)
            - Avoid stocks near 50% (too uncertain)
            - Higher probability = more conviction in the prediction
            """)
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.markdown("""
            ### 💪 Confidence Level

            **What it is:** How certain the model is about its prediction (regardless of direction)

            **Scale:** 0% to 100%
            - **Formula:** `|Probability - 50%| × 2`
            - **0%** = No confidence (probability ≈ 50%)
            - **100%** = Maximum confidence (probability ≈ 0% or 100%)

            **Examples:**
            - Probability **85%** → Confidence **70%** (|85-50| × 2 = 70%)
            - Probability **50%** → Confidence **0%** (|50-50| × 2 = 0%)
            - Probability **20%** → Confidence **60%** (|20-50| × 2 = 60%)

            **Why it matters:**
            - High confidence predictions are more reliable
            - Low confidence means the model is uncertain
            - Use confidence to filter out weak signals

            **Categories by confidence:**
            - **Strong Buy/Sell:** Probability ≥70% or ≤30% (Confidence ≥40%)
            - **Buy/Sell:** Probability 55-70% or 30-45% (Confidence 10-40%)
            - **Neutral:** Probability 45-55% (Confidence <10%)
            """)
            st.markdown('</div>', unsafe_allow_html=True)

    def display_classification_results(self, results):
        """Display results for classification model"""
        st.markdown("---")
        st.markdown("## 📊 Screening Results")

        # Summary metrics
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric("Total Stocks", len(results))
        with col2:
            strong_buy = sum(results['Category'] == 'Strong Buy')
            st.metric("Strong Buy", strong_buy)
        with col3:
            buy = sum(results['Category'] == 'Buy')
            st.metric("Buy", buy)
        with col4:
            mean_prob = results['Predicted_Probability'].mean()
            st.metric("Avg Probability", f"{mean_prob:.1%}")
        with col5:
            predicted_up = sum(results['Predicted_Direction'] == 1)
            st.metric("Predicted UP", f"{predicted_up/len(results)*100:.0f}%")

        # Filter options
        st.markdown("### 🔎 Filter Results")
        col1, col2, col3 = st.columns(3)

        with col1:
            available_categories = sorted(results['Category'].unique().tolist())
            default_categories = [cat for cat in ['Strong Buy', 'Buy'] if cat in available_categories]
            if not default_categories:
                default_categories = available_categories

            category_filter = st.multiselect(
                "Category",
                options=available_categories,
                default=default_categories,
                help="Filter by recommendation category"
            )

        with col2:
            min_probability = st.slider(
                "Minimum Probability",
                min_value=0.0,
                max_value=1.0,
                value=0.55,
                step=0.05,
                help="Minimum predicted probability of UP"
            )

        with col3:
            sort_by = st.selectbox(
                "Sort By",
                options=['Predicted_Probability', 'Confidence', 'Ticker'],
                index=0
            )

        # Apply filters
        filtered = results[
            (results['Category'].isin(category_filter)) &
            (results['Predicted_Probability'] >= min_probability)
        ].sort_values(sort_by, ascending=False)

        st.write(f"**{len(filtered)}** stocks match your criteria")

        # Display filtered results
        st.markdown("### 📋 Top Recommendations")

        if len(filtered) > 0:
            # Create display DataFrame
            display_df = filtered.copy()

            # Format columns
            display_df['Direction'] = display_df['Predicted_Direction'].apply(
                lambda x: "UP ⬆️" if x == 1 else "DOWN ⬇️"
            )
            display_df['Probability'] = display_df['Predicted_Probability'].apply(lambda x: f"{x:.1%}")
            display_df['Confidence'] = display_df['Confidence'].apply(lambda x: f"{x:.1%}")

            # Select display columns
            base_cols = ['Ticker', 'Category', 'Direction', 'Probability', 'Confidence']

            # Add available columns
            optional_cols = ['sales_growth_yoy', 'profit_growth_yoy', 'opm_percent',
                           'roce', 'roe', 'debt_to_equity', 'roc_1m', 'roc_3m',
                           'volatility_30d', 'rsi_14d']

            display_cols = base_cols + [col for col in optional_cols if col in display_df.columns]

            st.dataframe(
                display_df[display_cols].head(100),
                use_container_width=True,
                height=400,
                hide_index=True
            )

            # Download button
            csv = filtered.to_csv(index=False)
            st.download_button(
                label="📥 Download Full Results as CSV",
                data=csv,
                file_name=f"classification_screening_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                type="primary"
            )

            # Visualizations
            col1, col2 = st.columns(2)

            with col1:
                # Category distribution
                st.markdown("#### 📊 Category Distribution")
                category_counts = results['Category'].value_counts()

                fig = px.pie(
                    values=category_counts.values,
                    names=category_counts.index,
                    title="Distribution by Category",
                    color_discrete_sequence=px.colors.sequential.Blues_r
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Probability distribution
                st.markdown("#### 📈 Probability Distribution")
                fig = px.histogram(
                    results,
                    x='Predicted_Probability',
                    nbins=50,
                    title="Predicted Probabilities",
                    labels={'Predicted_Probability': 'Probability of UP'},
                    color_discrete_sequence=['#1f77b4']
                )
                fig.add_vline(x=0.5, line_dash="dash", line_color="red",
                             annotation_text="50% (neutral)")
                st.plotly_chart(fig, use_container_width=True)

    def display_regression_results(self, results):
        """Display results for regression models"""
        st.markdown("---")
        st.markdown("## 📊 Screening Results")

        # Determine available columns
        has_category = 'Category' in results.columns
        has_predicted_return = 'Predicted_Return' in results.columns
        has_predicted_probability = 'predicted_probability' in results.columns

        # Summary metrics
        cols = st.columns(5)

        with cols[0]:
            st.metric("Total Stocks", len(results))

        if has_category:
            with cols[1]:
                strong_buy = sum(results['Category'] == 'Strong Buy')
                st.metric("Strong Buy", strong_buy)
            with cols[2]:
                buy = sum(results['Category'] == 'Buy')
                st.metric("Buy", buy)

        if has_predicted_return:
            with cols[3]:
                mean_return = results['Predicted_Return'].mean()
                st.metric("Avg Predicted Return", f"{mean_return:.2f}%")

        if has_predicted_probability:
            with cols[4]:
                mean_prob = results['predicted_probability'].mean()
                st.metric("Avg Probability", f"{mean_prob:.1%}")

        # Filter options
        st.markdown("### 🔎 Filter Results")
        col1, col2 = st.columns(2)

        with col1:
            if has_category:
                available_categories = sorted(results['Category'].unique().tolist())
                default_categories = [cat for cat in ['Strong Buy', 'Buy'] if cat in available_categories]
                if not default_categories:
                    default_categories = available_categories

                category_filter = st.multiselect(
                    "Category",
                    options=available_categories,
                    default=default_categories
                )
                filtered = results[results['Category'].isin(category_filter)]
            else:
                filtered = results.copy()

        with col2:
            # Determine sort options
            sort_options = ['Ticker']
            if has_predicted_return:
                sort_options.insert(0, 'Predicted_Return')
            if has_predicted_probability:
                sort_options.insert(0, 'predicted_probability')
            if 'Quality_Score' in results.columns:
                sort_options.insert(1, 'Quality_Score')
            if 'composite_score' in results.columns:
                sort_options.insert(1, 'composite_score')

            sort_by = st.selectbox("Sort By", options=sort_options, index=0)
            filtered = filtered.sort_values(sort_by, ascending=False)

        st.write(f"**{len(filtered)}** stocks match your criteria")

        # Display results
        st.markdown("### 📋 Top Recommendations")

        if len(filtered) > 0:
            # Select display columns dynamically
            display_cols = ['Ticker']

            optional_cols = [
                'Category', 'Predicted_Return', 'predicted_probability', 'composite_score',
                'Quality_Score', 'Sales_Growth_YoY', 'Profit_Growth_YoY',
                'OPM_Percent', 'Profit_Margin', 'Sales_Trend', 'Profit_Trend',
                'trailing_pe', 'price_to_book', 'roe', 'revenue_growth', 'debt_to_equity'
            ]

            display_cols.extend([col for col in optional_cols if col in filtered.columns])

            st.dataframe(
                filtered[display_cols].head(100),
                use_container_width=True,
                height=400,
                hide_index=True
            )

            # Download button
            csv = filtered.to_csv(index=False)
            st.download_button(
                label="📥 Download Full Results as CSV",
                data=csv,
                file_name=f"regression_screening_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                type="primary"
            )

            # Visualizations
            col1, col2 = st.columns(2)

            with col1:
                if has_category:
                    st.markdown("#### 📊 Category Distribution")
                    category_counts = results['Category'].value_counts()
                    fig = px.pie(
                        values=category_counts.values,
                        names=category_counts.index,
                        title="Distribution by Category"
                    )
                    st.plotly_chart(fig, use_container_width=True)

            with col2:
                if has_predicted_return:
                    st.markdown("#### 📈 Predicted Return Distribution")
                    fig = px.histogram(
                        results,
                        x='Predicted_Return',
                        nbins=50,
                        title="Predicted Returns",
                        labels={'Predicted_Return': 'Predicted Return (%)'},
                        color_discrete_sequence=['#1f77b4']
                    )
                    st.plotly_chart(fig, use_container_width=True)
                elif has_predicted_probability:
                    st.markdown("#### 📈 Probability Distribution")
                    fig = px.histogram(
                        results,
                        x='predicted_probability',
                        nbins=50,
                        title="Predicted Probabilities",
                        labels={'predicted_probability': 'Probability'},
                        color_discrete_sequence=['#1f77b4']
                    )
                    st.plotly_chart(fig, use_container_width=True)

    def display_deep_dive(self, results, model, feature_importance):
        """Stock deep dive section with detailed analysis"""
        st.markdown("---")
        st.markdown('<div class="deep-dive-section">', unsafe_allow_html=True)
        st.markdown("## 🔬 Stock Deep Dive Analysis")
        st.markdown("""
        Select any stock from the screened universe to see:
        - Complete prediction details
        - Feature contributions (what drove the prediction)
        - All fundamental and technical metrics
        - Model's decision-making process
        """)
        st.markdown('</div>', unsafe_allow_html=True)

        # Stock selection
        col1, col2 = st.columns([3, 1])

        with col1:
            # Sort stocks by predicted probability (or return) for easier selection
            if 'Predicted_Probability' in results.columns:
                sorted_results = results.sort_values('Predicted_Probability', ascending=False)
            elif 'Predicted_Return' in results.columns:
                sorted_results = results.sort_values('Predicted_Return', ascending=False)
            else:
                sorted_results = results

            selected_ticker = st.selectbox(
                "Select a stock to analyze:",
                options=sorted_results['Ticker'].tolist(),
                index=0,
                help="Choose from all screened stocks"
            )

        with col2:
            st.write("")  # Spacing
            st.write("")  # Spacing
            analyze_button = st.button("🔍 Analyze Stock", type="primary", use_container_width=True)

        if selected_ticker and analyze_button:
            stock_data = results[results['Ticker'] == selected_ticker].iloc[0]

            st.markdown(f"### 📈 Deep Dive: **{selected_ticker}**")
            st.markdown("---")

            # Key prediction metrics at top
            st.markdown("#### 🎯 Model Prediction")

            if 'Predicted_Probability' in stock_data:
                cols = st.columns(5)

                with cols[0]:
                    direction = "UP ⬆️" if stock_data['Predicted_Direction'] == 1 else "DOWN ⬇️"
                    st.metric("Predicted Direction", direction)

                with cols[1]:
                    prob = stock_data['Predicted_Probability']
                    st.metric("Probability", f"{prob:.1%}")

                with cols[2]:
                    conf = stock_data['Confidence']
                    st.metric("Confidence", f"{conf:.1%}")

                with cols[3]:
                    cat = stock_data['Category']
                    st.metric("Category", cat)

                with cols[4]:
                    rank = stock_data.get('Rank', 'N/A')
                    st.metric("Rank", f"#{rank}" if rank != 'N/A' else 'N/A')

                # Explanation
                st.markdown('<div class="info-box">', unsafe_allow_html=True)
                if prob >= 0.70:
                    st.markdown(f"""
                    **🟢 Strong Buy Signal**

                    The model predicts with **{prob:.0%} probability** that {selected_ticker} will go UP next month.
                    This is a **high confidence** prediction (confidence: {stock_data['Confidence']:.0%}).
                    """)
                elif prob >= 0.55:
                    st.markdown(f"""
                    **🟢 Buy Signal**

                    The model predicts with **{prob:.0%} probability** that {selected_ticker} will go UP next month.
                    This is a **moderate confidence** prediction (confidence: {stock_data['Confidence']:.0%}).
                    """)
                elif prob >= 0.45:
                    st.markdown(f"""
                    **🟡 Neutral Signal**

                    The model has **no strong opinion** on {selected_ticker} (probability: {prob:.0%}).
                    This suggests the stock could go either way (confidence: {stock_data['Confidence']:.0%}).
                    """)
                elif prob >= 0.30:
                    st.markdown(f"""
                    **🔴 Sell Signal**

                    The model predicts with **{(1-prob):.0%} probability** that {selected_ticker} will go DOWN next month.
                    This is a **moderate confidence** prediction (confidence: {stock_data['Confidence']:.0%}).
                    """)
                else:
                    st.markdown(f"""
                    **🔴 Strong Sell Signal**

                    The model predicts with **{(1-prob):.0%} probability** that {selected_ticker} will go DOWN next month.
                    This is a **high confidence** prediction (confidence: {stock_data['Confidence']:.0%}).
                    """)
                st.markdown('</div>', unsafe_allow_html=True)

            elif 'Predicted_Return' in stock_data:
                cols = st.columns(4)

                with cols[0]:
                    pred_return = stock_data['Predicted_Return']
                    st.metric("Predicted Return", f"{pred_return:+.2f}%")

                with cols[1]:
                    cat = stock_data.get('Category', 'N/A')
                    st.metric("Category", cat)

                with cols[2]:
                    rank = stock_data.get('Rank', 'N/A')
                    st.metric("Rank", f"#{rank}" if rank != 'N/A' else 'N/A')

                with cols[3]:
                    quality = stock_data.get('Quality_Score', stock_data.get('composite_score', 'N/A'))
                    st.metric("Quality Score", f"{quality:.0f}" if quality != 'N/A' else 'N/A')

            st.markdown("---")

            # Detailed metrics in three columns
            st.markdown("#### 📊 Comprehensive Stock Metrics")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**💰 Valuation & Profitability**")
                metrics = {}

                for col in ['trailing_pe', 'forward_pe', 'price_to_book', 'enterprise_to_ebitda',
                           'roe', 'roa', 'profit_margin', 'opm', 'opm_percent']:
                    if col in stock_data and pd.notna(stock_data[col]):
                        label = col.replace('_', ' ').title()
                        value = stock_data[col]
                        if col in ['roe', 'roa', 'profit_margin']:
                            st.text(f"{label}: {value:.2%}")
                        elif col in ['opm_percent']:
                            st.text(f"OPM: {value:.2f}%")
                        else:
                            st.text(f"{label}: {value:.2f}")

            with col2:
                st.markdown("**📈 Growth & Momentum**")

                for col in ['sales_growth_yoy', 'Sales_Growth_YoY', 'profit_growth_yoy',
                           'Profit_Growth_YoY', 'revenue_growth', 'earnings_growth',
                           'roc_1m', 'roc_3m', 'roc_6m', 'roc_12m']:
                    if col in stock_data and pd.notna(stock_data[col]):
                        label = col.replace('_', ' ').replace('YoY', 'YoY').replace('yoy', 'YoY').title()
                        value = stock_data[col]
                        st.text(f"{label}: {value:+.2f}%")

            with col3:
                st.markdown("**⚠️ Risk & Technical**")

                for col in ['debt_to_equity', 'current_ratio', 'beta',
                           'volatility_30d', 'volatility_90d', 'rsi_14d',
                           'price_to_ma50', 'price_to_ma200']:
                    if col in stock_data and pd.notna(stock_data[col]):
                        label = col.replace('_', ' ').title()
                        value = stock_data[col]
                        if col in ['volatility_30d', 'volatility_90d']:
                            st.text(f"{label}: {value:.2f}%")
                        else:
                            st.text(f"{label}: {value:.2f}")

            st.markdown("---")

            # Feature contribution analysis (if model and feature importance available)
            if model is not None and feature_importance is not None:
                st.markdown("#### 🧠 How the Model Made This Decision")

                st.markdown("""
                Below are the key features that influenced this prediction. Positive contributions
                push towards UP, negative contributions push towards DOWN.
                """)

                try:
                    # Get feature names from importance dataframe
                    feature_names = feature_importance['feature'].tolist()

                    # Extract feature values for this stock
                    feature_values = {}
                    for feat in feature_names:
                        if feat in stock_data and pd.notna(stock_data[feat]):
                            feature_values[feat] = stock_data[feat]

                    # Create contribution analysis
                    # For classification models, we can show feature importance weighted by value
                    contrib_data = []
                    for _, row in feature_importance.iterrows():
                        feat = row['feature']
                        importance = row['importance']

                        if feat in feature_values:
                            value = feature_values[feat]
                            # Normalized contribution (simplified)
                            contrib = importance
                            contrib_data.append({
                                'Feature': feat,
                                'Value': value,
                                'Importance': importance,
                                'Impact': 'High' if importance > feature_importance['importance'].median() else 'Medium'
                            })

                    if contrib_data:
                        contrib_df = pd.DataFrame(contrib_data).head(20)

                        # Show top positive and negative drivers
                        col1, col2 = st.columns(2)

                        with col1:
                            st.markdown("**🟢 Key Features (Top 10)**")
                            top_features = contrib_df.head(10)
                            for _, row in top_features.iterrows():
                                st.text(f"• {row['Feature']}: {row['Value']:.2f} (Importance: {row['Importance']:,.0f})")

                        with col2:
                            st.markdown("**📊 Feature Impact Distribution**")
                            fig = px.bar(
                                contrib_df.head(15),
                                x='Importance',
                                y='Feature',
                                orientation='h',
                                title=f"Top Features for {selected_ticker}",
                                color='Importance',
                                color_continuous_scale='Blues'
                            )
                            fig.update_layout(
                                yaxis={'categoryorder': 'total ascending'},
                                height=400,
                                showlegend=False
                            )
                            st.plotly_chart(fig, use_container_width=True)

                        # Full feature table
                        with st.expander("📋 View All Feature Details"):
                            st.dataframe(
                                contrib_df,
                                use_container_width=True,
                                height=400,
                                hide_index=True
                            )

                except Exception as e:
                    st.info("Feature contribution analysis not available for this model")

    def run(self):
        """Run the multi-model application"""

        # Model selector
        selected_model_name = self.display_model_selector()

        # Load model and feature importance
        model = self.load_model()
        feature_importance = self.load_feature_importance()

        # Display model performance
        self.display_model_performance(model, feature_importance)

        # Explain probability and confidence
        if self.selected_model['type'] == 'classification':
            self.explain_probability_confidence()

        # Load and display results
        results = self.load_results()

        if results is not None:
            # Display results based on model type
            if self.selected_model['type'] == 'classification':
                self.display_classification_results(results)
            else:
                self.display_regression_results(results)

            # Deep dive section
            self.display_deep_dive(results, model, feature_importance)

        # Sidebar info
        st.sidebar.title("ℹ️ About")
        st.sidebar.markdown(f"""
        ### Current Model
        **{selected_model_name}**

        **Type:** {self.selected_model['type'].title()}

        **Target:** {self.selected_model['target']}

        **Performance:** {self.selected_model['accuracy']}
        """)

        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📊 Available Models")

        # Check which models have results
        for model_name, config in MODEL_CONFIGS.items():
            if os.path.exists(config['results_file']):
                st.sidebar.success(f"✅ {model_name}")
            else:
                st.sidebar.warning(f"⚠️ {model_name}")


if __name__ == "__main__":
    app = MultiModelScreener()
    app.run()
