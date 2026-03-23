import yfinance as yf
from utils.logger import get_logger

logger = get_logger("financial_data", "pipeline.log")


def get_financial_metrics(ticker: str) -> dict | None:
    """
    Fetches live financial metrics for a company
    using yFinance — completely free, no API key needed.

    Args:
        ticker: Stock ticker symbol (e.g. AAPL)

    Returns:
        Dictionary of financial metrics
    """
    logger.info(f"Fetching live financial metrics for {ticker}")

    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        metrics = {
            "ticker": ticker.upper(),
            "company_name": info.get("longName", "N/A"),
            "current_price": info.get("currentPrice", "N/A"),
            "market_cap": info.get("marketCap", "N/A"),
            "pe_ratio": info.get("trailingPE", "N/A"),
            "forward_pe": info.get("forwardPE", "N/A"),
            "eps": info.get("trailingEps", "N/A"),
            "revenue_ttm": info.get("totalRevenue", "N/A"),
            "gross_margin": info.get("grossMargins", "N/A"),
            "profit_margin": info.get("profitMargins", "N/A"),
            "debt_to_equity": info.get("debtToEquity", "N/A"),
            "free_cashflow": info.get("freeCashflow", "N/A"),
            "dividend_yield": info.get("dividendYield", "N/A"),
            "52_week_high": info.get("fiftyTwoWeekHigh", "N/A"),
            "52_week_low": info.get("fiftyTwoWeekLow", "N/A"),
            "analyst_target": info.get("targetMeanPrice", "N/A"),
            "recommendation": info.get("recommendationKey", "N/A"),
        }

        # Return None if no valid data found
        if (metrics["company_name"] == "N/A"
                and metrics["current_price"] == "N/A"):
            logger.warning(f"No valid data found for {ticker}")
            return None
        
        logger.info(f"Successfully fetched metrics for {ticker}")
        logger.debug(f"Metrics: {metrics}")
        return metrics

    except Exception as e:
        logger.error(f"Failed to fetch metrics for {ticker}: {e}")
        return None


def get_quarterly_financials(ticker: str) -> dict | None:
    """
    Fetches quarterly revenue and earnings history.
    Used for building trend charts in the frontend.

    Args:
        ticker: Stock ticker symbol

    Returns:
        Dictionary with quarterly revenue and earnings
    """
    logger.info(f"Fetching quarterly financials for {ticker}")

    try:
        stock = yf.Ticker(ticker)
        quarterly = stock.quarterly_income_stmt

        if quarterly is None or quarterly.empty:
            logger.warning(f"No quarterly data found for {ticker}")
            return None

        result = {
            "ticker": ticker.upper(),
            "quarters": [],
            "revenue": [],
            "net_income": [],
            "gross_profit": []
        }

        # Get last 8 quarters - columns are dates
        cols = list(quarterly.columns)[:8]

        for col in cols:
            # Format date as string
            quarter_date = str(col)[:10]
            result["quarters"].append(quarter_date)

            # Revenue - try different row names
            revenue = None
            for row_name in ["Total Revenue", "Revenue",
                             "totalRevenue"]:
                if row_name in quarterly.index:
                    val = quarterly.loc[row_name, col]
                    if val is not None:
                        try:
                            revenue = float(val)
                        except Exception:
                            revenue = None
                        break
            result["revenue"].append(revenue)

            # Net income
            net_income = None
            for row_name in ["Net Income", "NetIncome",
                             "netIncome"]:
                if row_name in quarterly.index:
                    val = quarterly.loc[row_name, col]
                    if val is not None:
                        try:
                            net_income = float(val)
                        except Exception:
                            net_income = None
                        break
            result["net_income"].append(net_income)

            # Gross profit
            gross_profit = None
            for row_name in ["Gross Profit", "GrossProfit",
                             "grossProfit"]:
                if row_name in quarterly.index:
                    val = quarterly.loc[row_name, col]
                    if val is not None:
                        try:
                            gross_profit = float(val)
                        except Exception:
                            gross_profit = None
                        break
            result["gross_profit"].append(gross_profit)

        logger.info(f"Fetched {len(result['quarters'])} quarters "
                   f"of data for {ticker}")
        return result

    except Exception as e:
        logger.error(f"Failed to fetch quarterly financials "
                    f"for {ticker}: {e}")
        return None


def format_metrics_for_llm(metrics: dict) -> str:
    """
    Formats financial metrics into readable text
    that can be passed to Claude as context.

    Args:
        metrics: Dictionary of financial metrics

    Returns:
        Formatted string
    """
    if not metrics:
        return "No financial metrics available."

    def fmt_number(val, prefix="$", suffix="", billions=False):
        if val == "N/A" or val is None:
            return "N/A"
        try:
            if billions:
                return f"{prefix}{float(val)/1e9:.2f}B{suffix}"
            return f"{prefix}{float(val):.2f}{suffix}"
        except Exception:
            return str(val)

    def fmt_percent(val, already_decimal=True):
        if val == "N/A" or val is None:
            return "N/A"
        try:
            if already_decimal:
                return f"{float(val)*100:.2f}%"
            return f"{float(val):.2f}%"
        except Exception:
            return str(val)

    text = f"""
LIVE FINANCIAL METRICS FOR {metrics['ticker']}
({metrics['company_name']}):

Price & Valuation:
- Current Price: {fmt_number(metrics['current_price'])}
- Market Cap: {fmt_number(metrics['market_cap'], billions=True)}
- P/E Ratio (TTM): {fmt_number(metrics['pe_ratio'], prefix='')}
- Forward P/E: {fmt_number(metrics['forward_pe'], prefix='')}
- EPS (TTM): {fmt_number(metrics['eps'])}
- Analyst Target: {fmt_number(metrics['analyst_target'])}
- Recommendation: {metrics['recommendation'].upper()
                   if metrics['recommendation'] != 'N/A'
                   else 'N/A'}

Revenue & Profitability:
- Revenue (TTM): {fmt_number(metrics['revenue_ttm'], billions=True)}
- Gross Margin: {fmt_percent(metrics['gross_margin'])}
- Profit Margin: {fmt_percent(metrics['profit_margin'])}
- Free Cash Flow: {fmt_number(metrics['free_cashflow'], billions=True)}

Financial Health:
- Debt to Equity: {fmt_number(metrics['debt_to_equity'], prefix='')}
- Dividend Yield: {fmt_number(metrics['dividend_yield'], prefix='', suffix='%')}
- 52 Week High: {fmt_number(metrics['52_week_high'])}
- 52 Week Low: {fmt_number(metrics['52_week_low'])}
"""
    return text.strip()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m pipeline.financial_data <TICKER>")
        print("Example: python -m pipeline.financial_data AAPL")
        sys.exit(1)

    ticker = sys.argv[1].upper()

    print(f"\nFetching live metrics for {ticker}...")
    metrics = get_financial_metrics(ticker)
    if metrics:
        print(format_metrics_for_llm(metrics))
    else:
        print(f"Could not fetch metrics for {ticker}")

    print(f"\nFetching quarterly financials for {ticker}...")
    quarterly = get_quarterly_financials(ticker)
    if quarterly:
        print(f"\nQuarterly Revenue ({ticker}):")
        for q, r in zip(quarterly["quarters"], quarterly["revenue"]):
            if r:
                print(f"  {q}: ${r/1e9:.2f}B")
