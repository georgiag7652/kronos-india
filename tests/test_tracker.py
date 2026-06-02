"""
Unit tests for tracker.py — _determine_outcome and _next_business_days.

Run with:  python -m pytest tests/
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import pytest
from datetime import datetime

from tracker import _determine_outcome, _next_business_days


# ── Helpers ───────────────────────────────────────────────────────────────────

def _candles(rows: list[dict]) -> pd.DataFrame:
    """
    Build a minimal OHLCV DataFrame from a list of dicts with keys
    open, high, low, close.  Index is IST-aware timestamps one hour apart.
    """
    base = pd.Timestamp("2025-06-02 09:15:00", tz="Asia/Kolkata")
    index = [base + pd.Timedelta(hours=i) for i in range(len(rows))]
    df = pd.DataFrame(rows, index=index)
    df["volume"] = 1000
    return df


# ── _determine_outcome — LONG ─────────────────────────────────────────────────

class TestDetermineOutcomeLong:

    def test_target_hit_first(self):
        # LONG entry=100, target=+5% (105), SL=-2.5% (97.5)
        # candle 1 doesn't touch either; candle 2 high reaches target
        candles = _candles([
            {"open": 100, "high": 102, "low": 99,  "close": 101},
            {"open": 101, "high": 106, "low": 100, "close": 105},
        ])
        outcome, pnl, hi, lo, cl = _determine_outcome("LONG", 100.0, 5.0, 2.5, candles)
        assert outcome == "WIN"
        assert abs(pnl - 5.0) < 0.01
        assert hi == 106.0

    def test_stoploss_hit_first(self):
        # SL candle appears before target candle
        candles = _candles([
            {"open": 100, "high": 101, "low": 97.0, "close": 97.5},
            {"open": 97,  "high": 105, "low": 96,   "close": 100},
        ])
        outcome, pnl, *_ = _determine_outcome("LONG", 100.0, 5.0, 2.5, candles)
        assert outcome == "LOSS"
        assert abs(pnl - (-2.5)) < 0.01

    def test_neither_hit_expired(self):
        # Price drifts but never reaches target or SL
        candles = _candles([
            {"open": 100, "high": 102, "low": 99, "close": 101},
            {"open": 101, "high": 103, "low": 100, "close": 102},
        ])
        outcome, pnl, *_ = _determine_outcome("LONG", 100.0, 5.0, 2.5, candles)
        assert outcome == "EXPIRED"
        assert abs(pnl - 2.0) < 0.01   # close=102, entry=100 → +2%

    def test_expired_uses_last_date_close(self):
        # Two candles on day 1, one on day 2. EXPIRED should use last close of day 2.
        base_d1 = pd.Timestamp("2025-06-02 09:15:00", tz="Asia/Kolkata")
        base_d2 = pd.Timestamp("2025-06-03 09:15:00", tz="Asia/Kolkata")
        rows = [
            {"open": 100, "high": 102, "low": 99, "close": 101},
            {"open": 101, "high": 103, "low": 100, "close": 102},
            {"open": 102, "high": 104, "low": 101, "close": 103},
        ]
        index = [base_d1, base_d1 + pd.Timedelta(hours=1), base_d2]
        df = pd.DataFrame(rows, index=index)
        df["volume"] = 1000
        outcome, pnl, hi, lo, cl = _determine_outcome("LONG", 100.0, 5.0, 2.5, df)
        assert outcome == "EXPIRED"
        assert cl == 103.0   # last close of day 2, not day 1's 102

    def test_exact_target_boundary(self):
        # High exactly equals adj_target — should be WIN
        candles = _candles([
            {"open": 100, "high": 105.0, "low": 99, "close": 104},
        ])
        outcome, *_ = _determine_outcome("LONG", 100.0, 5.0, 2.5, candles)
        assert outcome == "WIN"

    def test_exact_stoploss_boundary(self):
        # Low exactly equals adj_stoploss — should be LOSS
        candles = _candles([
            {"open": 100, "high": 101, "low": 97.5, "close": 98},
        ])
        outcome, *_ = _determine_outcome("LONG", 100.0, 5.0, 2.5, candles)
        assert outcome == "LOSS"


# ── _determine_outcome — SHORT ────────────────────────────────────────────────

class TestDetermineOutcomeShort:

    def test_target_hit_first(self):
        # SHORT entry=100, target=-5% (95), SL=+2.5% (102.5)
        candles = _candles([
            {"open": 100, "high": 101, "low": 99,  "close": 100},
            {"open": 99,  "high": 100, "low": 94,  "close": 95},
        ])
        outcome, pnl, *_ = _determine_outcome("SHORT", 100.0, 5.0, 2.5, candles)
        assert outcome == "WIN"
        assert abs(pnl - 5.0) < 0.01

    def test_stoploss_hit_first(self):
        candles = _candles([
            {"open": 100, "high": 103.0, "low": 99, "close": 102},
        ])
        outcome, pnl, *_ = _determine_outcome("SHORT", 100.0, 5.0, 2.5, candles)
        assert outcome == "LOSS"
        assert abs(pnl - (-2.5)) < 0.01

    def test_neither_hit_expired(self):
        candles = _candles([
            {"open": 100, "high": 101, "low": 98, "close": 98},
        ])
        outcome, pnl, *_ = _determine_outcome("SHORT", 100.0, 5.0, 2.5, candles)
        assert outcome == "EXPIRED"
        assert abs(pnl - 2.0) < 0.01   # entry=100, close=98 → +2% for SHORT


# ── _next_business_days ───────────────────────────────────────────────────────

class TestNextBusinessDays:

    def test_zero_days_returns_same_date(self):
        d = datetime(2025, 6, 2)  # Monday
        assert _next_business_days(d, 0) == d

    def test_skips_weekend(self):
        # Friday 30 May 2025 + 1 business day should be Monday 2 Jun 2025
        friday = datetime(2025, 5, 30)
        result = _next_business_days(friday, 1)
        assert result.weekday() < 5
        assert result == datetime(2025, 6, 2)

    def test_skips_nse_holiday(self):
        # 13 Mar 2025 (Thursday) + 1 business day
        # 14 Mar 2025 is Holi (NSE holiday) → should land on 17 Mar 2025 (Monday)
        thursday = datetime(2025, 3, 13)
        result = _next_business_days(thursday, 1)
        assert result == datetime(2025, 3, 17)

    def test_three_business_days(self):
        # Monday 2 Jun 2025 + 3 = Thu 5 Jun 2025 (no holidays in that week)
        monday = datetime(2025, 6, 2)
        result = _next_business_days(monday, 3)
        assert result == datetime(2025, 6, 5)

    def test_skips_multiple_holidays_in_row(self):
        # 21 Oct 2025 (Diwali Laxmi Puja) and 22 Oct 2025 (Balipratipada) are both holidays
        # 20 Oct 2025 (Monday) + 1 business day should skip both and land on 23 Oct 2025 (Thursday)
        monday = datetime(2025, 10, 20)
        result = _next_business_days(monday, 1)
        assert result == datetime(2025, 10, 23)
