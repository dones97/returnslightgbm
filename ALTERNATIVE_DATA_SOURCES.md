# Alternative Data Sources for Indian Stock Fundamentals

## Your Research Goal

**"Understand what fundamental characteristics lead to outperformance"**

This requires **time-varying fundamental data** over many years - you need to see how P/E, ROE, debt levels, margins, etc. **at each point in time** predicted subsequent returns.

---

## Comparison of Free Data Sources

### 1. **Yahoo Finance (yfinance)** - CURRENT

| Aspect | Details |
|--------|---------|
| **Price Data** | 10 years (excellent) |
| **Fundamental Data** | 1.5-2 years only (6-8 quarters) |
| **Coverage** | All NSE/BSE stocks |
| **Free** | Yes, unlimited |
| **Reliability** | Good |
| **API** | Yes (yfinance library) |
| **Suitable for your goal?** | **Partially** - limited fundamental history |

**Verdict**: Good for 2-3 year studies, not enough for 5-10 year fundamental research.

---

### 2. **Screener.in** - BEST FOR YOUR NEEDS ⭐

| Aspect | Details |
|--------|---------|
| **Price Data** | 10+ years |
| **Fundamental Data** | **10+ years quarterly** ✓ |
| **Coverage** | All NSE/BSE stocks |
| **Free** | Yes |
| **Reliability** | **Excellent** (from company filings) |
| **API** | No official API (web scraping required) |
| **Data Available** | Quarterly results, annual reports, ratios |

**What you can get**:
- Quarterly Revenue, Profit, EPS (10+ years)
- Quarterly Balance Sheet data (10+ years)
- Historical P/E, P/B, ROE, ROCE, Debt/Equity
- All ratios calculated over time
- Sourced directly from company filings

**Example**: https://www.screener.in/company/RELIANCE/consolidated/

**Implementation Options**:

**Option A: Manual Export** (Quick start)
- Visit screener.in for each stock
- Export quarterly results to Excel
- Combine manually
- **Pros**: No coding needed, guaranteed to work
- **Cons**: Tedious for many stocks, not scalable

**Option B: Web Scraping** (Recommended)
- Use Python (BeautifulSoup/Selenium/Scrapy)
- Automate data extraction
- **Pros**: Scalable, repeatable
- **Cons**: Requires coding, may break if site changes

**Option C: Existing Libraries**
- Check if someone has built a screener.in wrapper
- GitHub search: "screener.in python"
- **Pros**: If available, saves time
- **Cons**: May not exist or be maintained

**Verdict**: **BEST option for fundamental-focused research**. This gives you exactly what you need!

---

### 3. **Alpha Vantage** - GOOD ALTERNATIVE

| Aspect | Details |
|--------|---------|
| **Price Data** | 20+ years |
| **Fundamental Data** | Annual (5-10 years), Quarterly (2 years) |
| **Coverage** | Global, **some Indian stocks** |
| **Free** | 500 calls/day (enough for 500 stocks) |
| **Reliability** | Good |
| **API** | Yes (official REST API) |

**Free API Key**: https://www.alphavantage.co/support/#api-key

**What you can get**:
- Income Statement (annual & quarterly)
- Balance Sheet (annual & quarterly)
- Cash Flow (annual & quarterly)
- Earnings data

**Indian Stock Format**:
- NSE: `RELIANCE.NSE` or `RELIANCE.NS`
- BSE: `RELIANCE.BSE` or `RELIANCE.BO`

**Limitations**:
- 500 calls/day = process 500 stocks/day
- **Coverage for Indian stocks is variable** (need to test)
- May not have full 10 years for all stocks

**Implementation**:
```python
import requests

API_KEY = "your_free_api_key"
symbol = "RELIANCE.NSE"

# Get income statement
url = f"https://www.alphavantage.co/query?function=INCOME_STATEMENT&symbol={symbol}&apikey={API_KEY}"
response = requests.get(url)
data = response.json()
```

**Verdict**: Worth testing! May have 5-10 years of annual fundamentals.

---

### 4. **Financial Modeling Prep** - LIMITED

| Aspect | Details |
|--------|---------|
| **Free Tier** | 250 calls/day |
| **Fundamental Data** | Up to 5 years |
| **Indian Coverage** | **Very limited** |
| **Verdict** | Not ideal for Indian market |

---

### 5. **MoneyControl** - REQUIRES SCRAPING

| Aspect | Details |
|--------|---------|
| **Fundamental Data** | Good historical data |
| **Coverage** | All Indian stocks |
| **Free** | Yes |
| **API** | No (web scraping required) |
| **Reliability** | Good |

**Verdict**: Similar to Screener.in but less structured. Screener.in is better.

---

### 6. **BSE/NSE Official** - LIMITED FUNDAMENTALS

| Aspect | Details |
|--------|---------|
| **Price Data** | Excellent, complete |
| **Fundamental Data** | Limited availability |
| **Coverage** | Complete for their exchange |
| **Free** | Yes |
| **Verdict** | Good for prices, not for fundamentals |

---

### 7. **Paid Options** (For Reference)

**CMIE Prowess** (Best for academic research)
- 20+ years of comprehensive fundamental data
- Used by researchers, institutions
- Expensive (INR 50,000+ per year)
- **Gold standard for Indian equity research**

**Capitaline**
- Good fundamental data
- Moderate pricing
- Used by analysts

**Bloomberg/Refinitiv**
- Enterprise-level
- Very expensive
- Comprehensive but overkill for your needs

---

## My Recommendations for Your Research

### Recommended Approach: Three-Phase Strategy

### **Phase 1: Start with 5 Years (yfinance)** ✓ IMMEDIATE

**Rationale**:
- yfinance gives you ~2-3 years of quarterly fundamentals
- 5 years of price data
- Quick to implement (already done!)
- Tests your methodology

**What you get**:
- ~24-36 months of fundamental history per stock
- ~1,400 stocks with full data
- **~36,000-48,000 monthly observations**
- Enough to test if fundamentals predict returns

**Implementation**:
```python
# Already implemented! Just change config:
DATA_CONFIG = {
    'default_lookback_years': 5,  # Change from 10 to 5
}
```

**Research Questions You Can Answer**:
- Which fundamental ratios predict returns?
- Do low P/E stocks outperform?
- Does high ROE predict future returns?
- How important is debt/equity?
- What's the optimal combination of factors?

**Pros**:
- Ready to run TODAY
- No new data sources needed
- Tests your research approach
- 2-3 years may be sufficient for factor analysis

**Cons**:
- Limited to recent market conditions
- Shorter cycles captured
- Less robust for long-term patterns

---

### **Phase 2: Test Alpha Vantage** (1-2 days)

**If Phase 1 results are promising**, test Alpha Vantage:

1. Get free API key
2. Test 10-20 major stocks
3. Check fundamental data availability
4. If good coverage, build integration

**Potential**:
- May get 5-10 years of annual fundamentals
- 500 stocks/day = full universe in 3 days
- Still free

**Implementation effort**: Medium (1-2 days of coding)

---

### **Phase 3: Screener.in Scraping** (If needed for 10-year study)

**If you need full 10 years of quarterly fundamentals**:

1. Build web scraper for screener.in
2. Extract quarterly results for all stocks
3. Process into clean dataset
4. Run full 10-year analysis

**This gives you**:
- 10+ years of quarterly fundamentals
- Exactly what you need for comprehensive research
- Best fundamental data for Indian market

**Implementation effort**: High (3-5 days of coding + testing)

---

## Does 5 Years Give Better Perspective?

### YES - For Fundamental Research! ✓

**With 5 years lookback**:

| Aspect | 10 Years (current) | 5 Years (recommended) |
|--------|-------------------|---------------------|
| **Price Data** | 120 months | 60 months |
| **Fundamental Data** | ~6 quarters (last 1.5 years only!) | **~12 quarters (2-3 years)** |
| **Fundamental Coverage** | 12% of period | **40-60% of period** ✓ |
| **Training Records** | ~168,000 | ~84,000 (still plenty!) |
| **Data Quality** | Recent fundamentals better | **Much better fundamental coverage** |

**Key Insight**:
- With 10 years: Only 1.5/10 years have real fundamentals (15%)
- With 5 years: 2-3/5 years have real fundamentals (40-60%)
- **Better for understanding fundamental → return relationships!**

---

## Recommended Action Plan

### Step 1: Change to 5 Years (NOW - 5 minutes)

Update `config.py`:
```python
DATA_CONFIG = {
    'default_lookback_years': 5,  # Better fundamental coverage
}
```

### Step 2: Enhance Feature Engineering (NOW - add to code)

Focus on fundamental features, add more derived ratios:

**I can update the code to**:
1. Extract quarterly financials (what's available)
2. Calculate historical ratios month-by-month
3. Add more fundamental features:
   - Revenue growth (YoY, QoQ)
   - Margin trends
   - Asset turnover
   - Working capital changes
   - ROIC, ROCE
   - Piotroski F-Score
   - Altman Z-Score

### Step 3: Test Alpha Vantage (OPTIONAL - 1 day)

Create integration to test if it gives better data:
```python
# I can build this module
from alpha_vantage_collector import AlphaVantageCollector

collector = AlphaVantageCollector(api_key="YOUR_KEY")
fundamentals = collector.get_fundamentals("RELIANCE.NSE", years=10)
```

### Step 4: Build Screener.in Scraper (IF NEEDED - 3-5 days)

Only if you need full 10-year quarterly fundamentals:
```python
# I can build this module
from screener_scraper import ScreenerScraper

scraper = ScreenerScraper()
quarterly_data = scraper.get_quarterly_results("RELIANCE", years=10)
```

---

## What Should We Do NOW?

### RECOMMENDED: Update to 5 Years + Enhanced Fundamentals

**I can immediately**:

1. ✅ Change default to 5 years
2. ✅ Enhance data_collector.py to:
   - Extract quarterly fundamentals (all available)
   - Calculate time-varying ratios
   - Forward-fill to monthly
   - Add more fundamental features
3. ✅ Focus model on fundamental features
4. ✅ Add fundamental-specific feature importance analysis

**This gives you**:
- 2-3 years of true time-varying fundamentals
- Much better for your research goal
- Can run immediately
- Tests if approach works before investing in scraping

### THEN: Decide on Phase 2/3

After seeing 5-year results:
- **If sufficient**: Stick with yfinance (5 years)
- **If promising but need more history**: Test Alpha Vantage
- **If need full 10 years**: Build screener.in scraper

---

## My Recommendation

**START HERE**:

1. **Use 5 years with enhanced fundamental extraction** (I'll implement)
2. **This gives you 40-60% fundamental coverage vs 15%**
3. **Still ~84,000 training records (plenty!)**
4. **Much better for your "fundamentals → returns" research**

**Would you like me to**:
- ✅ Update to 5 years
- ✅ Enhance fundamental feature extraction (use quarterly data properly)
- ✅ Add more fundamental metrics
- ✅ Focus model on fundamental factors
- ⏸️ Keep Alpha Vantage integration as future option
- ⏸️ Keep Screener.in scraping as future option

This way you can **start testing your research questions immediately** with much better fundamental data than the current implementation!

Shall I proceed with updating the code for 5-year lookback with enhanced fundamental features?
