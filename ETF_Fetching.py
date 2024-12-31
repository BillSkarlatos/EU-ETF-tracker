
def get_top_etfs_by_volume(region="Europe", num_etfs=20):
    # Predefined list of popular ETFs for Europe
    return [
        "IWDA.AS", "EQQQ.AS", "SPY5.DE", "XMRV.DE", "XDWD.DE", "IS3N.DE", "EXXT.DE", "XD9U.DE", 
        "XLYE.DE", "SPYD.L", "VEUR.DE", "VGVE.DE", "IWMO.DE", "SXR8.DE", "XDJP.DE", "EUNL.DE", 
        "XRS2.DE", "IUSA.L", "VWRL.L", "UB43.L"
    ]

# Automatically fetch ETF tickers
def fetch_etf_tickers():
    try:
        tickers = get_top_etfs_by_volume()
        return tickers
    except Exception as e:
        print(f"Error fetching tickers: {e}")
        return []