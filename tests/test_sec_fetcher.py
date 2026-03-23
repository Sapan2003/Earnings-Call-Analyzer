import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion.sec_fetcher import get_cik  # noqa: E402


def test_get_cik_valid_ticker():
    """Test CIK lookup for valid ticker"""
    cik = get_cik("AAPL")
    assert cik is not None
    assert cik == "0000320193"


def test_get_cik_invalid_ticker():
    """Test CIK lookup for invalid ticker"""
    cik = get_cik("INVALIDTICKER123")
    assert cik is None


def test_get_cik_case_insensitive():
    """Test CIK lookup is case insensitive"""
    cik_upper = get_cik("AAPL")
    cik_lower = get_cik("aapl")
    assert cik_upper == cik_lower
