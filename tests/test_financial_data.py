import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.financial_data import get_financial_metrics  # noqa: E402


def test_get_financial_metrics_valid():
    """Test metrics fetch for valid ticker"""
    metrics = get_financial_metrics("AAPL")
    assert metrics is not None
    assert metrics["ticker"] == "AAPL"
    assert "current_price" in metrics
    assert "market_cap" in metrics


def test_get_financial_metrics_invalid():
    """Test metrics fetch for invalid ticker"""
    metrics = get_financial_metrics("INVALIDTICKER123")
    assert metrics is None


def test_get_financial_metrics_contains_required_fields():
    """Test all required fields present"""
    metrics = get_financial_metrics("MSFT")
    assert metrics is not None
    required_fields = [
        "ticker", "company_name", "current_price",
        "market_cap", "pe_ratio", "profit_margin"
    ]
    for field in required_fields:
        assert field in metrics
