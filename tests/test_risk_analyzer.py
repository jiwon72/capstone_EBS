import pytest
from agents.risk.risk_analyzer import RiskAnalyzer
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

@pytest.fixture
def risk_analyzer():
    return RiskAnalyzer()

@pytest.fixture
def sample_strategy_output():
    return {
        'portfolio': {
            '005930.KS': 0.3,  # 삼성전자
            '000660.KS': 0.2,  # SK하이닉스
            '035720.KS': 0.2,  # 카카오
            '035420.KS': 0.2,  # NAVER
            '051910.KS': 0.1   # LG화학
        },
        'strategy_type': 'trend_following',
        'risk_tolerance': 0.6
    }

def test_analyze_portfolio_risk(risk_analyzer, sample_strategy_output):
    """포트폴리오 리스크 분석 테스트"""
    result = risk_analyzer.analyze_portfolio_risk(sample_strategy_output)
    
    # 기본 구조 확인
    assert 'basic_risk_metrics' in result
    assert 'var_metrics' in result
    assert 'correlation_analysis' in result
    assert 'stress_test_results' in result
    assert 'risk_assessment' in result
    
    # 기본 리스크 지표 확인
    risk_metrics = result['basic_risk_metrics']
    assert 'volatility' in risk_metrics
    assert 'max_drawdown' in risk_metrics
    assert 'sharpe_ratio' in risk_metrics
    assert 'annual_return' in risk_metrics
    
    # VaR 지표 확인
    var_metrics = result['var_metrics']
    assert 'historical_var' in var_metrics
    assert 'parametric_var' in var_metrics
    assert 'confidence_level' in var_metrics
    
    # 상관관계 분석 확인
    correlation_analysis = result['correlation_analysis']
    assert 'correlation_matrix' in correlation_analysis
    assert 'mean_correlation' in correlation_analysis
    assert 'high_correlation_pairs' in correlation_analysis
    
    # 스트레스 테스트 결과 확인
    stress_test = result['stress_test_results']
    assert 'worst_5_days' in stress_test
    assert 'stress_volatility' in stress_test
    assert 'stress_correlation' in stress_test
    
    # 리스크 평가 확인
    risk_assessment = result['risk_assessment']
    assert 'risk_level' in risk_assessment
    assert 'risk_factors' in risk_assessment
    assert 'risk_mitigation' in risk_assessment

def test_calculate_basic_risk_metrics(risk_analyzer):
    """기본 리스크 지표 계산 테스트"""
    # 샘플 수익률 데이터 생성
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    returns = pd.DataFrame({
        'Stock1': np.random.normal(0.0005, 0.02, len(dates)),
        'Stock2': np.random.normal(0.0003, 0.015, len(dates))
    }, index=dates)
    
    metrics = risk_analyzer._calculate_basic_risk_metrics(returns)
    
    assert 'volatility' in metrics
    assert 'max_drawdown' in metrics
    assert 'sharpe_ratio' in metrics
    assert 'annual_return' in metrics
    
    # 값의 범위 확인
    assert 0 <= metrics['volatility'] <= 1
    assert -1 <= metrics['max_drawdown'] <= 0
    assert metrics['annual_return'] > -1

def test_calculate_var(risk_analyzer):
    """VaR 계산 테스트"""
    # 샘플 수익률 데이터 생성
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    returns = pd.DataFrame({
        'Stock1': np.random.normal(0.0005, 0.02, len(dates)),
        'Stock2': np.random.normal(0.0003, 0.015, len(dates))
    }, index=dates)
    
    var_metrics = risk_analyzer._calculate_var(returns)
    
    assert 'historical_var' in var_metrics
    assert 'parametric_var' in var_metrics
    assert 'confidence_level' in var_metrics
    
    # VaR 값이 음수인지 확인 (손실 위험)
    assert var_metrics['historical_var'] < 0
    assert var_metrics['parametric_var'] < 0

def test_analyze_correlations(risk_analyzer):
    """상관관계 분석 테스트"""
    # 샘플 수익률 데이터 생성
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    returns = pd.DataFrame({
        'Stock1': np.random.normal(0.0005, 0.02, len(dates)),
        'Stock2': np.random.normal(0.0003, 0.015, len(dates)),
        'Stock3': np.random.normal(0.0004, 0.018, len(dates))
    }, index=dates)
    
    correlation_analysis = risk_analyzer._analyze_correlations(returns)
    
    assert 'correlation_matrix' in correlation_analysis
    assert 'mean_correlation' in correlation_analysis
    assert 'high_correlation_pairs' in correlation_analysis
    
    # 상관관계 값의 범위 확인
    assert -1 <= correlation_analysis['mean_correlation'] <= 1

def test_perform_stress_test(risk_analyzer):
    """스트레스 테스트 테스트"""
    # 샘플 수익률 데이터 생성
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    returns = pd.DataFrame({
        'Stock1': np.random.normal(0.0005, 0.02, len(dates)),
        'Stock2': np.random.normal(0.0003, 0.015, len(dates))
    }, index=dates)
    
    stress_test_results = risk_analyzer._perform_stress_test(returns)
    
    assert 'worst_5_days' in stress_test_results
    assert 'stress_volatility' in stress_test_results
    assert 'stress_correlation' in stress_test_results
    
    # 스트레스 테스트 결과의 유효성 확인
    assert len(stress_test_results['worst_5_days']) == 5
    assert stress_test_results['stress_volatility'] > 0
    assert stress_test_results['stress_correlation'] == 1.0

def test_generate_risk_assessment(risk_analyzer):
    """리스크 평가 생성 테스트"""
    # 샘플 데이터 생성
    risk_metrics = {
        'volatility': 0.25,
        'max_drawdown': -0.15,
        'sharpe_ratio': 1.5,
        'annual_return': 0.12
    }
    
    var_metrics = {
        'historical_var': -0.02,
        'parametric_var': -0.025,
        'confidence_level': 0.95
    }
    
    correlation_analysis = {
        'mean_correlation': 0.6,
        'high_correlation_pairs': []
    }
    
    stress_test_results = {
        'worst_5_days': pd.Series([-0.02, -0.018, -0.015, -0.012, -0.01]),
        'stress_volatility': 0.4,
        'stress_correlation': 1.0
    }
    
    assessment = risk_analyzer._generate_risk_assessment(
        risk_metrics, var_metrics, correlation_analysis, stress_test_results
    )
    
    assert 'risk_level' in assessment
    assert 'risk_factors' in assessment
    assert 'risk_mitigation' in assessment
    
    # 리스크 레벨이 유효한 값인지 확인
    assert assessment['risk_level'] in ['LOW', 'MODERATE', 'HIGH', 'VERY HIGH'] 