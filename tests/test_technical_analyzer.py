import pytest
from agents.technical.technical_analyzer import TechnicalAnalyzer
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

@pytest.fixture
def technical_analyzer():
    return TechnicalAnalyzer()

@pytest.fixture
def sample_data():
    # 샘플 주가 데이터 생성
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    data = pd.DataFrame({
        'Open': np.random.normal(100, 2, len(dates)),
        'High': np.random.normal(102, 2, len(dates)),
        'Low': np.random.normal(98, 2, len(dates)),
        'Close': np.random.normal(100, 2, len(dates)),
        'Volume': np.random.normal(1000000, 200000, len(dates))
    }, index=dates)
    
    # High와 Low가 Open과 Close를 포함하도록 조정
    data['High'] = data[['Open', 'Close', 'High']].max(axis=1)
    data['Low'] = data[['Open', 'Close', 'Low']].min(axis=1)
    
    return data

def test_analyze_technical_indicators(technical_analyzer):
    """기술적 지표 분석 테스트"""
    result = technical_analyzer.analyze_technical_indicators('005930.KS')  # 삼성전자
    
    # 기본 구조 확인
    assert 'symbol' in result
    assert 'technical_indicators' in result
    assert 'chart_patterns' in result
    assert 'trend_analysis' in result
    assert 'volume_analysis' in result
    assert 'price_forecast' in result
    assert 'analysis_summary' in result
    
    # 기술적 지표 확인
    indicators = result['technical_indicators']
    assert 'sma_20' in indicators
    assert 'sma_50' in indicators
    assert 'sma_200' in indicators
    assert 'rsi' in indicators
    assert 'macd' in indicators
    assert 'bb_upper' in indicators
    assert 'stoch_k' in indicators

def test_calculate_technical_indicators(technical_analyzer, sample_data):
    """기술적 지표 계산 테스트"""
    indicators = technical_analyzer._calculate_technical_indicators(sample_data)
    
    # 이동평균선 확인
    assert 'sma_20' in indicators
    assert 'sma_50' in indicators
    assert 'sma_200' in indicators
    assert not indicators['sma_20'].isna().all()
    
    # RSI 확인
    assert 'rsi' in indicators
    assert not indicators['rsi'].isna().all()
    assert all(0 <= x <= 100 for x in indicators['rsi'].dropna())
    
    # MACD 확인
    assert 'macd' in indicators
    assert 'macd_signal' in indicators
    assert 'macd_hist' in indicators
    
    # 볼린저 밴드 확인
    assert 'bb_upper' in indicators
    assert 'bb_middle' in indicators
    assert 'bb_lower' in indicators

def test_analyze_chart_patterns(technical_analyzer, sample_data):
    """차트 패턴 분석 테스트"""
    patterns = technical_analyzer._analyze_chart_patterns(sample_data)
    
    assert isinstance(patterns, list)
    for pattern in patterns:
        assert 'pattern' in pattern
        assert 'direction' in pattern
        assert 'strength' in pattern
        assert pattern['direction'] in ['BULLISH', 'BEARISH']

def test_analyze_trend(technical_analyzer, sample_data):
    """추세 분석 테스트"""
    trend_analysis = technical_analyzer._analyze_trend(sample_data)
    
    assert 'short_term_trend' in trend_analysis
    assert 'mid_term_trend' in trend_analysis
    assert 'long_term_trend' in trend_analysis
    assert 'trend_strength' in trend_analysis
    
    assert trend_analysis['short_term_trend'] in ['UP', 'DOWN']
    assert trend_analysis['mid_term_trend'] in ['UP', 'DOWN']
    assert trend_analysis['long_term_trend'] in ['UP', 'DOWN']
    assert 0 <= trend_analysis['trend_strength'] <= 100

def test_analyze_volume(technical_analyzer, sample_data):
    """거래량 분석 테스트"""
    volume_analysis = technical_analyzer._analyze_volume(sample_data)
    
    assert 'volume_trend' in volume_analysis
    assert 'volume_sma' in volume_analysis
    assert 'current_volume' in volume_analysis
    assert 'volume_ratio' in volume_analysis
    assert 'vwap' in volume_analysis
    
    assert volume_analysis['volume_trend'] in ['UP', 'DOWN']
    assert volume_analysis['volume_ratio'] > 0
    assert volume_analysis['vwap'] > 0

def test_forecast_price(technical_analyzer, sample_data):
    """가격 예측 테스트"""
    forecast = technical_analyzer._forecast_price(sample_data)
    
    assert 'forecast_prices' in forecast
    assert 'forecast_dates' in forecast
    
    assert len(forecast['forecast_prices']) == 5  # 기본값 5일
    assert len(forecast['forecast_dates']) == 5
    assert all(isinstance(price, (int, float)) for price in forecast['forecast_prices'])
    assert all(isinstance(date, datetime) for date in forecast['forecast_dates'])

def test_generate_analysis_summary(technical_analyzer, sample_data):
    """분석 요약 생성 테스트"""
    # 샘플 데이터 준비
    indicators = technical_analyzer._calculate_technical_indicators(sample_data)
    patterns = technical_analyzer._analyze_chart_patterns(sample_data)
    trend_analysis = technical_analyzer._analyze_trend(sample_data)
    volume_analysis = technical_analyzer._analyze_volume(sample_data)
    price_forecast = technical_analyzer._forecast_price(sample_data)
    
    summary = technical_analyzer._generate_analysis_summary(
        indicators, patterns, trend_analysis, volume_analysis, price_forecast
    )
    
    assert 'trading_signals' in summary
    assert 'confidence_score' in summary
    assert 'key_findings' in summary
    
    assert isinstance(summary['trading_signals'], list)
    assert 0 <= summary['confidence_score'] <= 1
    assert isinstance(summary['key_findings'], list)

def test_generate_trading_signals(technical_analyzer, sample_data):
    """매매 신호 생성 테스트"""
    indicators = technical_analyzer._calculate_technical_indicators(sample_data)
    patterns = technical_analyzer._analyze_chart_patterns(sample_data)
    trend_analysis = technical_analyzer._analyze_trend(sample_data)
    
    signals = technical_analyzer._generate_trading_signals(
        indicators, patterns, trend_analysis
    )
    
    assert isinstance(signals, list)
    for signal in signals:
        assert 'type' in signal
        assert 'indicator' in signal
        assert 'strength' in signal
        assert 'reason' in signal
        
        assert signal['type'] in ['BUY', 'SELL']
        assert signal['indicator'] in ['RSI', 'MACD', 'TREND']
        assert signal['strength'] in ['STRONG', 'MODERATE', 'WEAK']

def test_calculate_confidence_score(technical_analyzer, sample_data):
    """신뢰도 점수 계산 테스트"""
    indicators = technical_analyzer._calculate_technical_indicators(sample_data)
    patterns = technical_analyzer._analyze_chart_patterns(sample_data)
    trend_analysis = technical_analyzer._analyze_trend(sample_data)
    volume_analysis = technical_analyzer._analyze_volume(sample_data)
    
    confidence_score = technical_analyzer._calculate_confidence_score(
        indicators, patterns, trend_analysis, volume_analysis
    )
    
    assert 0 <= confidence_score <= 1 