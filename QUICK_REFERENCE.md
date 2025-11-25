# Quick Reference Guide - Updated for Fundamental Focus

## What Changed?

✅ **5-year lookback** (was 10) → 40-60% fundamental coverage (was 15%)
✅ **Time-varying fundamentals** → ROE, margins, ratios change over time
✅ **40+ fundamental features** → F-Score, Z-Score, growth rates, ratios
✅ **Enhanced data collector** → Extracts quarterly financial statements
✅ **Focus on fundamentals** → 80% fundamentals, 20% technicals

---

## Quick Start (3 Steps)

### 1. Launch App
```bash
streamlit run streamlit_app.py
```

### 2. Collect Data
- Set years: **5** (default)
- Set stocks: **200** (test) or **1500** (full)
- Click "Start Data Collection"
- Wait 10-15 minutes

### 3. Train & Screen
- Train model (Step 2)
- Screen stocks (Step 3)
- Export results

---

## Key Features Now Available

### Time-Varying Fundamentals (2-3 years)
- ✅ ROE, ROA, ROIC (quarterly)
- ✅ Gross/Operating/Net Margins
- ✅ Debt/Equity, Current Ratio
- ✅ Revenue/Earnings Growth (YoY, QoQ)
- ✅ TTM metrics (trailing 12 months)

### Advanced Scores
- ✅ **Piotroski F-Score** (0-9) - Quality score
- ✅ **Altman Z-Score** - Bankruptcy predictor

### What This Means
You can now analyze **how fundamentals at a point in time predicted future returns!**

---

## Research Questions to Explore

After training, check feature importance to see:

1. **Which metrics predict returns?**
   - High F-Score → outperformance?
   - High ROE → better returns?
   - Low Debt/Equity → safer + higher returns?

2. **Do improving fundamentals matter?**
   - Rising margins → positive signal?
   - Accelerating revenue growth → better returns?
   - Improving Z-Score → reduced risk?

3. **What's the optimal combination?**
   - Best 3-5 fundamental factors?
   - How to weight them?

---

## File Structure

### Main Files (Use These)
- `streamlit_app.py` - Run this
- `data_collector_enhanced.py` - New collector (used automatically)
- `model_trainer.py` - Model training
- `config.py` - Settings

### Documentation
- `IMPLEMENTATION_SUMMARY.md` - Detailed changes
- `ALTERNATIVE_DATA_SOURCES.md` - For future phases
- `FUNDAMENTAL_DATA_SOLUTION.md` - Why 5 years vs 10

### Old Files (Still Work)
- `data_collector.py` - Original (not used anymore)
- `demo_workflow.py` - CLI demo (update import to use new collector)

---

## Interpreting Results

### Model Performance
- **Accuracy 55-60%**: Normal for market prediction
- **Focus on**: Feature importance, not accuracy
- **Look for**: Which fundamentals rank highest

### Feature Importance
**Top features should include**:
- F_Score, Z_Score
- ROE, ROA, ROIC
- Margin metrics
- Growth rates (YoY)
- Debt ratios

**If technical features dominate**: Model not finding fundamental signals (may need more data/stocks)

### Stock Rankings
- **Q5 (Highest)**: Best fundamental characteristics
- **Q1 (Lowest)**: Worst fundamental characteristics
- Export and analyze top quintile

---

## Troubleshooting

### "Not enough fundamental data"
- Some stocks don't have quarterly data
- Normal - model handles this
- Filter for large caps if needed

### "Model accuracy is low"
- 55-60% is expected!
- Market prediction is hard
- Focus on relative rankings

### "Training takes long"
- 200 stocks: ~10-15 min data collection
- 1500 stocks: ~60-90 min
- Use cached data for reruns

---

## Next Phase: Screener.in Scraping

If you need 10+ years of fundamentals:

1. Build web scraper for screener.in
2. Extract quarterly results (10+ years available)
3. Get true 10-year fundamental history
4. See `ALTERNATIVE_DATA_SOURCES.md`

For now, 5 years with yfinance is ready to use!

---

## Commands Cheat Sheet

```bash
# Run main app
streamlit run streamlit_app.py

# Test data availability
python test_quarterly_fundamentals.py

# Quick demo (need to update import first)
python demo_workflow.py

# Check 10-year data (for reference)
python check_10year_data.py
```

---

## Key Files to Edit

### To change defaults:
`config.py` - Lines 7-13

### To add features:
`data_collector_enhanced.py` - `extract_quarterly_fundamentals()` method

### To change model:
`model_trainer.py` - `ReturnDirectionModel` class

---

## Success Checklist

After running, you should have:
- ✅ ~100,000+ monthly observations
- ✅ ~1,680 stocks with data
- ✅ 40-60% with time-varying fundamentals
- ✅ Feature importance showing fundamental dominance
- ✅ Clear understanding of what predicts returns
- ✅ Actionable insights for stock selection

---

**Ready?** Run:
```bash
streamlit run streamlit_app.py
```

Good luck with your fundamental research! 📊
