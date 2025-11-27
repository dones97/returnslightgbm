"""
Price-Based Technical Indicators

Calculates momentum and technical indicators from price data
to supplement fundamental features.
"""

import pandas as pd
import numpy as np
from typing import Optional
from price_data_helper import PriceDataHelper


class TechnicalIndicators:
    """Calculate technical indicators from price data"""

    def __init__(self, use_cache: bool = False):
        """
        Initialize technical indicators calculator

        Args:
            use_cache: If True, use cached price data
        """
        self.price_helper = PriceDataHelper(use_cache=use_cache)

    def calculate_indicators(self, ticker: str, quarter_dates: pd.Series) -> pd.DataFrame:
        """
        Calculate technical indicators for each quarter date

        Args:
            ticker: Stock ticker (without .NS)
            quarter_dates: Series of quarter end dates

        Returns:
            DataFrame with technical indicators for each quarter
        """
        # Fetch price data
        prices = self.price_helper._fetch_price_data(ticker)

        if prices is None or prices.empty:
            # Return empty DataFrame with expected columns
            return pd.DataFrame({
                'roc_1m': [np.nan] * len(quarter_dates),
                'roc_3m': [np.nan] * len(quarter_dates),
                'roc_6m': [np.nan] * len(quarter_dates),
                'roc_12m': [np.nan] * len(quarter_dates),
                'volatility_30d': [np.nan] * len(quarter_dates),
                'volatility_90d': [np.nan] * len(quarter_dates),
                'rsi_14d': [np.nan] * len(quarter_dates),
                'price_to_ma50': [np.nan] * len(quarter_dates),
                'price_to_ma200': [np.nan] * len(quarter_dates),
                'momentum_score': [np.nan] * len(quarter_dates)
            }, index=quarter_dates.index)

        indicators_list = []

        for quarter_date in quarter_dates:
            indicators = self._calculate_at_date(prices, quarter_date)
            indicators_list.append(indicators)

        result = pd.DataFrame(indicators_list, index=quarter_dates.index)
        return result

    def _calculate_at_date(self, prices: pd.DataFrame, date: pd.Timestamp) -> dict:
        """Calculate all indicators at a specific date"""

        # Get historical prices up to this date
        hist = prices[prices.index <= date].copy()

        if len(hist) < 20:
            return self._empty_indicators()

        current_price = hist['Close'].iloc[-1]

        # 1. Rate of Change (ROC) - Momentum
        roc_1m = self._calculate_roc(hist, periods=21)  # ~1 month
        roc_3m = self._calculate_roc(hist, periods=63)  # ~3 months
        roc_6m = self._calculate_roc(hist, periods=126)  # ~6 months
        roc_12m = self._calculate_roc(hist, periods=252)  # ~12 months

        # 2. Volatility
        volatility_30d = self._calculate_volatility(hist, window=30)
        volatility_90d = self._calculate_volatility(hist, window=90)

        # 3. RSI (Relative Strength Index)
        rsi_14d = self._calculate_rsi(hist, period=14)

        # 4. Price vs Moving Averages
        ma50 = hist['Close'].tail(min(50, len(hist))).mean()
        ma200 = hist['Close'].tail(min(200, len(hist))).mean()

        price_to_ma50 = ((current_price - ma50) / ma50) * 100 if ma50 > 0 else 0
        price_to_ma200 = ((current_price - ma200) / ma200) * 100 if ma200 > 0 else 0

        # 5. Composite Momentum Score
        momentum_score = self._calculate_momentum_score(
            roc_1m, roc_3m, roc_6m, roc_12m, rsi_14d
        )

        return {
            'roc_1m': roc_1m,
            'roc_3m': roc_3m,
            'roc_6m': roc_6m,
            'roc_12m': roc_12m,
            'volatility_30d': volatility_30d,
            'volatility_90d': volatility_90d,
            'rsi_14d': rsi_14d,
            'price_to_ma50': price_to_ma50,
            'price_to_ma200': price_to_ma200,
            'momentum_score': momentum_score
        }

    def _calculate_roc(self, hist: pd.DataFrame, periods: int) -> float:
        """Rate of Change over periods"""
        if len(hist) < periods + 1:
            return np.nan

        current = hist['Close'].iloc[-1]
        past = hist['Close'].iloc[-(periods + 1)]

        if past == 0:
            return np.nan

        return ((current - past) / past) * 100

    def _calculate_volatility(self, hist: pd.DataFrame, window: int) -> float:
        """Calculate annualized volatility"""
        if len(hist) < window:
            return np.nan

        returns = hist['Close'].pct_change().tail(window)
        volatility = returns.std() * np.sqrt(252)  # Annualized

        return volatility * 100  # Return as percentage

    def _calculate_rsi(self, hist: pd.DataFrame, period: int = 14) -> float:
        """Relative Strength Index"""
        if len(hist) < period + 1:
            return np.nan

        # Calculate price changes
        delta = hist['Close'].diff()

        # Separate gains and losses
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        # Calculate average gain and loss
        avg_gain = gain.tail(period).mean()
        avg_loss = loss.tail(period).mean()

        if avg_loss == 0:
            return 100  # No losses = max RSI

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def _calculate_momentum_score(self, roc_1m, roc_3m, roc_6m, roc_12m, rsi) -> float:
        """
        Composite momentum score (0-100)

        Combines multiple momentum indicators
        """
        scores = []

        # ROC scores (positive = good)
        if not np.isnan(roc_1m):
            scores.append(50 + min(roc_1m, 50))  # Cap at 100
        if not np.isnan(roc_3m):
            scores.append(50 + min(roc_3m, 50))
        if not np.isnan(roc_6m):
            scores.append(50 + min(roc_6m, 50))
        if not np.isnan(roc_12m):
            scores.append(50 + min(roc_12m, 50))

        # RSI score (40-60 is neutral, >70 overbought, <30 oversold)
        if not np.isnan(rsi):
            scores.append(rsi)

        if not scores:
            return np.nan

        return np.mean(scores)

    def _empty_indicators(self) -> dict:
        """Return empty indicators when insufficient data"""
        return {
            'roc_1m': np.nan,
            'roc_3m': np.nan,
            'roc_6m': np.nan,
            'roc_12m': np.nan,
            'volatility_30d': np.nan,
            'volatility_90d': np.nan,
            'rsi_14d': np.nan,
            'price_to_ma50': np.nan,
            'price_to_ma200': np.nan,
            'momentum_score': np.nan
        }
