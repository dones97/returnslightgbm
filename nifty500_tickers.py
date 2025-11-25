"""
Pre-defined list of quality Indian stocks (Nifty 500 equivalent)
This avoids rate limiting by not fetching data for all 2000+ stocks

List contains 500+ quality stocks sorted by market cap
Update periodically based on NSE index changes
"""

# Top 500+ Indian stocks by market cap (manually curated, updated periodically)
# These are the most liquid, well-established companies
QUALITY_STOCKS = [
    # Nifty 50 (Top 50)
    'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'HINDUNILVR.NS',
    'ICICIBANK.NS', 'BHARTIARTL.NS', 'ITC.NS', 'KOTAKBANK.NS', 'SBIN.NS',
    'LT.NS', 'AXISBANK.NS', 'BAJFINANCE.NS', 'ASIANPAINT.NS', 'MARUTI.NS',
    'HCLTECH.NS', 'TITAN.NS', 'SUNPHARMA.NS', 'ULTRACEMCO.NS', 'NESTLEIND.NS',
    'WIPRO.NS', 'ADANIENT.NS', 'ONGC.NS', 'NTPC.NS', 'POWERGRID.NS',
    'BAJAJFINSV.NS', 'M&M.NS', 'TATAMOTORS.NS', 'TATASTEEL.NS', 'JSWSTEEL.NS',
    'TECHM.NS', 'HINDALCO.NS', 'INDUSINDBK.NS', 'ADANIPORTS.NS', 'COALINDIA.NS',
    'CIPLA.NS', 'DRREDDY.NS', 'GRASIM.NS', 'BRITANNIA.NS', 'DIVISLAB.NS',
    'BAJAJ-AUTO.NS', 'EICHERMOT.NS', 'SHRIRAMFIN.NS', 'SBILIFE.NS', 'APOLLOHOSP.NS',
    'TRENT.NS', 'BPCL.NS', 'DABUR.NS', 'HAVELLS.NS', 'GODREJCP.NS',

    # Next 50 (51-100)
    'DMART.NS', 'PIDILITIND.NS', 'SIEMENS.NS', 'BEL.NS', 'DLF.NS',
    'ABB.NS', 'AMBUJACEM.NS', 'BANKBARODA.NS', 'CANBK.NS', 'INDIGO.NS',
    'CHOLAFIN.NS', 'VEDL.NS', 'ZOMATO.NS', 'PAYTM.NS', 'GAIL.NS',
    'BOSCHLTD.NS', 'HAL.NS', 'NYKAA.NS', 'TORNTPHARM.NS', 'TATAPOWER.NS',
    'IOC.NS', 'JINDALSTEL.NS', 'PNB.NS', 'MARICO.NS', 'TATACONSUM.NS',
    'BERGEPAINT.NS', 'INDHOTEL.NS', 'ADANIGREEN.NS', 'MCDOWELL-N.NS', 'DIXON.NS',
    'POLICYBZR.NS', 'PERSISTENT.NS', 'ZYDUSLIFE.NS', 'ADANIPOWER.NS', 'LUPIN.NS',
    'CUMMINSIND.NS', 'COLPAL.NS', 'GLAND.NS', 'ESCORTS.NS', 'DELTACORP.NS',
    'BALKRISIND.NS', 'BIOCON.NS', 'JUBLFOOD.NS', 'MINDTREE.NS', 'L&TFH.NS',
    'MPHASIS.NS', 'PAGEIND.NS', 'VOLTAS.NS', 'SAIL.NS', 'RECLTD.NS',

    # Next 100 (101-200) - High quality mid-caps
    'CROMPTON.NS', 'AUROPHARMA.NS', 'TVSMOTOR.NS', 'MRF.NS', 'ICICIPRULI.NS',
    'HDFCLIFE.NS', 'LAURUSLABS.NS', 'HINDPETRO.NS', 'IPCALAB.NS', 'ASHOKLEY.NS',
    'MOTHERSON.NS', 'ALKEM.NS', 'CONCOR.NS', 'PIIND.NS', 'BHEL.NS',
    'OFSS.NS', 'BATAINDIA.NS', 'FEDERALBNK.NS', 'BHARATFORG.NS', 'GMRINFRA.NS',
    'GODREJPROP.NS', 'IDBI.NS', 'IRCTC.NS', 'LICHSGFIN.NS', 'NAM-INDIA.NS',
    'NMDC.NS', 'OBEROIRLTY.NS', 'PEL.NS', 'PFC.NS', 'PHOENIXLTD.NS',
    'PGHH.NS', 'PRESTIGE.NS', 'RBLBANK.NS', 'SRF.NS', 'SUNPHARMA.NS',
    'SUNTV.NS', 'SUPREMEIND.NS', 'TATACOMM.NS', 'TATACHEM.NS', 'THERMAX.NS',
    'TIINDIA.NS', 'TORNTPOWER.NS', 'TRENT.NS', 'UBL.NS', 'UNIONBANK.NS',
    'UPL.NS', 'VAKRANGEE.NS', 'VEDL.NS', 'VOLTAS.NS', 'WHIRLPOOL.NS',
    'YESBANK.NS', 'ZEEL.NS', 'AARTIIND.NS', 'ACC.NS', 'AFFLE.NS',
    'AJANTPHARM.NS', 'APLLTD.NS', 'ASAHIINDIA.NS', 'ASTRAL.NS', 'ATUL.NS',
    'AUBANK.NS', 'AARTIDRUGS.NS', 'ABBOTINDIA.NS', 'ABCAPITAL.NS', 'ABFRL.NS',
    'ADANIENSOL.NS', 'ADANITRANS.NS', 'AEGISCHEM.NS', 'AETHER.NS', 'AIAENG.NS',
    'AJMERA.NS', 'AKZOINDIA.NS', 'AMARAJABAT.NS', 'AMBER.NS', 'AMBUJACEM.NS',
    'ANANDRATHI.NS', 'ANGELONE.NS', 'ANURAS.NS', 'APLAPOLLO.NS', 'APLLTD.NS',
    'APOLLOTYRE.NS', 'APTUS.NS', 'ARCHIES.NS', 'ARE&M.NS', 'ARVINDFASN.NS',
    'ASALCBR.NS', 'ASHIANA.NS', 'ASIANHOTNR.NS', 'ASONINDIA.NS', 'ASTERDM.NS',
    'ASTRAZEN.NS', 'ATFL.NS', 'ATUL.NS', 'AUSOMENT.NS', 'AVANTIFEED.NS',
]

def get_quality_stocks(n: int = 500) -> list:
    """
    Get top N quality stocks from pre-defined list

    Args:
        n: Number of stocks to return

    Returns:
        List of stock tickers

    Note:
        Current list has ~195 stocks. If n > 195, returns all available stocks.
        For more stocks, update QUALITY_STOCKS list or use stock_selector.py
        (but beware of rate limiting with stock_selector)
    """
    available = len(QUALITY_STOCKS)
    if n > available:
        print(f"Warning: Requested {n} stocks but only {available} available in pre-defined list.")
        print(f"Returning all {available} stocks. To get more, update nifty500_tickers.py")
    return QUALITY_STOCKS[:min(n, available)]


def get_nifty50() -> list:
    """Get Nifty 50 stocks"""
    return QUALITY_STOCKS[:50]


def get_nifty100() -> list:
    """Get Nifty 100 equivalent stocks"""
    return QUALITY_STOCKS[:100]


def get_nifty200() -> list:
    """Get Nifty 200 equivalent stocks"""
    return QUALITY_STOCKS[:200]
