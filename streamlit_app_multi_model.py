"""
Multi-Model Streamlit Application for Indian Stock Screening

Supports multiple ML models:
1. YFinance-only model (regression - original)
2. Classification model (screener.in + yfinance - 65% accuracy)
3. Screener-only model (regression - quarterly returns)
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

# Custom CSS (reusing from original)
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
</style>
""", unsafe_allow_html=True)


# Model configurations
MODEL_CONFIGS = {
    "Classification Model (65% Accuracy)": {
        "type": "classification",
        "results_file": "screening_results_classification.pkl",
        "model_file": "model_classification.pkl",
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
        "description": "Predicts quarterly returns using screener.in fundamentals only",
        "features": "Fundamentals only",
        "accuracy": "R² varies",
        "auc": "N/A",
        "target": "Next quarter return %",
        "data_period": "2022-2024",
        "samples": "Varies"
    },
    "YFinance Original Model": {
        "type": "regression",
        "results_file": "screening_results.pkl",
        "model_file": "trained_model.pkl",
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

    def display_classification_results(self, results):
        """Display results for classification model"""
        st.subheader("📊 Classification Model Results")

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
        st.subheader("🔎 Filter Results")
        col1, col2, col3 = st.columns(3)

        with col1:
            available_categories = sorted(results['Category'].unique().tolist())
            default_categories = [cat for cat in ['Strong Buy', 'Buy'] if cat in available_categories]
            if not default_categories:
                default_categories = available_categories

            category_filter = st.multiselect(
                "Category",
                options=available_categories,
                default=default_categories
            )

        with col2:
            min_probability = st.slider(
                "Minimum Probability",
                min_value=0.0,
                max_value=1.0,
                value=0.55,
                step=0.05,
                help="Minimum predicted probability of positive returns"
            )

        with col3:
            sort_by = st.selectbox(
                "Sort By",
                options=['Predicted_Probability', 'Confidence', 'sales_growth_yoy', 'Ticker'],
                index=0
            )

        # Apply filters
        filtered = results[
            (results['Category'].isin(category_filter)) &
            (results['Predicted_Probability'] >= min_probability)
        ].sort_values(sort_by, ascending=False)

        st.write(f"**{len(filtered)}** stocks match your criteria")

        # Display filtered results
        st.subheader("📋 Top Recommendations")

        # Select columns to display
        display_cols = ['Ticker', 'Category', 'Predicted_Probability', 'Confidence', 'Predicted_Direction']

        # Add fundamental columns if available
        fundamental_cols = ['sales_growth_yoy', 'profit_growth_yoy', 'opm_percent', 'roce', 'roe', 'debt_to_equity']
        for col in fundamental_cols:
            if col in filtered.columns:
                display_cols.append(col)

        # Add technical columns if available
        technical_cols = ['roc_1m', 'roc_3m', 'volatility_30d', 'rsi_14d']
        for col in technical_cols:
            if col in filtered.columns:
                display_cols.append(col)

        # Create display DataFrame
        if len(filtered) > 0:
            display_df = filtered[display_cols].copy()

            # Format columns
            display_df['Direction'] = display_df['Predicted_Direction'].apply(lambda x: "UP ⬆️" if x == 1 else "DOWN ⬇️")
            display_df['Probability'] = display_df['Predicted_Probability'].apply(lambda x: f"{x:.1%}")
            display_df['Confidence'] = display_df['Confidence'].apply(lambda x: f"{x:.1%}")

            # Reorder columns
            final_cols = ['Ticker', 'Category', 'Direction', 'Probability', 'Confidence']
            for col in display_cols:
                if col not in ['Ticker', 'Category', 'Predicted_Direction', 'Predicted_Probability', 'Confidence'] and col in display_df.columns:
                    final_cols.append(col)

            st.dataframe(
                display_df[final_cols],
                use_container_width=True,
                height=400,
                hide_index=True
            )

            # Download button
            csv = filtered.to_csv(index=False)
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name=f"classification_screening_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                type="primary"
            )

            # Category distribution chart
            st.subheader("📊 Category Distribution")
            category_counts = results['Category'].value_counts()

            fig = px.pie(
                values=category_counts.values,
                names=category_counts.index,
                title="Distribution by Category",
                color_discrete_sequence=px.colors.sequential.Blues_r
            )
            st.plotly_chart(fig, use_container_width=True)

            # Probability distribution
            st.subheader("📈 Probability Distribution")
            fig = px.histogram(
                results,
                x='Predicted_Probability',
                nbins=50,
                title="Distribution of Predicted Probabilities",
                labels={'Predicted_Probability': 'Probability of UP', 'count': 'Number of Stocks'},
                color_discrete_sequence=['#1f77b4']
            )
            fig.add_vline(x=0.5, line_dash="dash", line_color="red", annotation_text="50% threshold")
            st.plotly_chart(fig, use_container_width=True)

    def display_regression_results(self, results):
        """Display results for regression models"""
        st.subheader("📊 Regression Model Results")

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Stocks", len(results))
        with col2:
            if 'Category' in results.columns:
                strong_buy = sum(results['Category'] == 'Strong Buy')
                st.metric("Strong Buy", strong_buy)
        with col3:
            if 'Predicted_Return' in results.columns:
                mean_return = results['Predicted_Return'].mean()
                st.metric("Avg Predicted Return", f"{mean_return:.2f}%")
        with col4:
            if 'Category' in results.columns:
                buy = sum(results['Category'] == 'Buy')
                st.metric("Buy", buy)

        # Filter options
        st.subheader("🔎 Filter Results")
        col1, col2 = st.columns(2)

        with col1:
            if 'Category' in results.columns:
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
            sort_by = st.selectbox(
                "Sort By",
                options=[col for col in ['Predicted_Return', 'Quality_Score', 'Ticker'] if col in results.columns],
                index=0
            )
            filtered = filtered.sort_values(sort_by, ascending=False)

        st.write(f"**{len(filtered)}** stocks match your criteria")

        # Display results
        st.subheader("📋 Top Recommendations")

        if len(filtered) > 0:
            # Select relevant columns
            display_cols = [col for col in filtered.columns if col in [
                'Ticker', 'Category', 'Predicted_Return', 'Quality_Score',
                'Sales_Growth_YoY', 'Profit_Growth_YoY', 'OPM_Percent',
                'Profit_Margin', 'Sales_Trend', 'Profit_Trend'
            ]]

            st.dataframe(
                filtered[display_cols].head(50),
                use_container_width=True,
                height=400,
                hide_index=True
            )

            # Download button
            csv = filtered.to_csv(index=False)
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name=f"regression_screening_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                type="primary"
            )

            # Distribution charts
            if 'Predicted_Return' in filtered.columns:
                st.subheader("📈 Predicted Return Distribution")
                fig = px.histogram(
                    results,
                    x='Predicted_Return',
                    nbins=50,
                    title="Distribution of Predicted Returns",
                    labels={'Predicted_Return': 'Predicted Return (%)', 'count': 'Number of Stocks'},
                    color_discrete_sequence=['#1f77b4']
                )
                st.plotly_chart(fig, use_container_width=True)

    def run(self):
        """Run the multi-model application"""

        # Model selector in main area
        selected_model_name = self.display_model_selector()

        # Load results
        st.markdown("---")
        results = self.load_results()

        if results is not None:
            # Display appropriate results view based on model type
            if self.selected_model['type'] == 'classification':
                self.display_classification_results(results)
            else:
                self.display_regression_results(results)

        # Sidebar info
        st.sidebar.title("ℹ️ About")
        st.sidebar.markdown(f"""
        ### Current Model: {selected_model_name}

        **Type:** {self.selected_model['type'].title()}

        **Features:** {self.selected_model['features']}

        **Target:** {self.selected_model['target']}

        **Performance:**
        - {self.selected_model['accuracy']}
        """)

        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📊 File Status")

        # Check which models have results
        for model_name, config in MODEL_CONFIGS.items():
            if os.path.exists(config['results_file']):
                st.sidebar.success(f"✅ {model_name}")
            else:
                st.sidebar.warning(f"⚠️ {model_name}")

        # Helpful links
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🔗 Quick Links")
        st.sidebar.markdown("""
        - [GitHub Repository](https://github.com/yourusername/yourrepo)
        - [Model Documentation](https://github.com/yourusername/yourrepo#readme)
        - [Feature Importance](./feature_importance_classification.csv)
        """)


if __name__ == "__main__":
    app = MultiModelScreener()
    app.run()
