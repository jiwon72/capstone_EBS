from datetime import datetime
from typing import List
from fastapi.testclient import TestClient
from agents.technical.api import app
from agents.technical.models import MarketData, TimeFrame, TrendDirection
import pytest

client = TestClient(app)

def generate_test_market_data() -> List[MarketData]:
    """테스트용 시장 데이터 생성"""
    return [
        MarketData(
            ticker="TEST",
            timestamp=datetime.now(),
            open=100.0,
            high=105.0,
            low=95.0,
            close=102.0,
            volume=1000000
        )
    ]

def test_technical_analysis_endpoint():
    """기술적 분석 엔드포인트 테스트"""
    market_data = generate_test_market_data()
    request_data = {
        "market_data": [data.model_dump(mode='json') for data in market_data],
        "indicators": ["sma", "ema", "rsi", "macd"]
    }

    response = client.post("/analyze", json=request_data)
    assert response.status_code == 200

    data = response.json()
    assert "indicators" in data

def test_moving_averages():
    """이동평균 계산 테스트"""
    market_data = generate_test_market_data()
    request_data = {
        "market_data": [data.model_dump(mode='json') for data in market_data],
        "window_sizes": [20, 50],
        "ma_types": ["sma", "ema"]
    }

    response = client.post("/indicators/moving_averages", json=request_data)
    assert response.status_code == 200

    data = response.json()
    assert "sma_values" in data
    assert "ema_values" in data
    assert "trend_direction" in data
    assert data["trend_direction"] in [t.value for t in TrendDirection]

def test_support_resistance():
    """지지/저항 레벨 테스트"""
    market_data = generate_test_market_data()
    request_data = {
        "market_data": [data.model_dump(mode='json') for data in market_data],
        "lookback_period": 20,
        "min_touches": 2
    }

    response = client.post("/analyze/support_resistance", json=request_data)
    assert response.status_code == 200

    data = response.json()
    assert "support_levels" in data
    assert "resistance_levels" in data

def test_pattern_recognition():
    """차트 패턴 인식 테스트"""
    market_data = generate_test_market_data()
    request_data = {
        "market_data": [data.model_dump(mode='json') for data in market_data],
        "pattern_types": ["double_top", "double_bottom", "head_shoulders"],
        "min_confidence": 0.7
    }

    response = client.post("/analyze/patterns", json=request_data)
    assert response.status_code == 200

    data = response.json()
    assert "patterns" in data
    assert "timestamp" in data
    assert "error_message" in data
    assert data["error_message"] is None

def test_trading_signals():
    """트레이딩 신호 생성 테스트"""
    market_data = generate_test_market_data()
    request_data = {
        "market_data": [data.model_dump(mode='json') for data in market_data],
        "signal_types": ["trend_following", "mean_reversion", "breakout"]
    }

    response = client.post("/signals", json=request_data)
    assert response.status_code == 200

    data = response.json()
    assert "signals" in data
    assert "timestamp" in data
    assert "error_message" in data
    assert data["error_message"] is None 