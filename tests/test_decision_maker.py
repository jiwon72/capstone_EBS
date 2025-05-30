import pytest
from datetime import datetime
from agents.decision.decision_maker import DecisionMaker
from agents.decision.models import (
    DecisionType, DecisionConfidence, RiskLevel, PositionSize,
    AgentAnalysis, MarketContext, RiskParameters, DecisionRequest
)

@pytest.fixture
def decision_maker():
    return DecisionMaker()

@pytest.fixture
def sample_market_context():
    return MarketContext(
        market_condition="TRENDING",
        volatility_level=0.3,
        trend_strength=0.8,
        volume_profile={
            "trend_aligned": True,
            "volume_increasing": True
        },
        market_sentiment="BULLISH"
    )

@pytest.fixture
def sample_risk_parameters():
    return RiskParameters(
        max_position_size=100000.0,
        stop_loss_percentage=0.02,
        take_profit_percentage=0.04,
        max_drawdown=0.1,
        risk_reward_ratio=2.0
    )

@pytest.fixture
def sample_agent_analyses():
    return [
        AgentAnalysis(
            agent_name="technical_analyzer",
            analysis_type="technical",
            confidence_score=0.85,
            recommendation={
                "signal": "BUY",
                "strength": "STRONG",
                "reasoning": ["RSI 과매도", "MACD 골든크로스"]
            },
            timestamp=datetime.now()
        ),
        AgentAnalysis(
            agent_name="fundamental_analyzer",
            analysis_type="fundamental",
            confidence_score=0.75,
            recommendation={
                "signal": "BUY",
                "strength": "MODERATE",
                "reasoning": ["PER 저평가", "실적 성장"]
            },
            timestamp=datetime.now()
        ),
        AgentAnalysis(
            agent_name="sentiment_analyzer",
            analysis_type="sentiment",
            confidence_score=0.65,
            recommendation={
                "signal": "HOLD",
                "strength": "WEAK",
                "reasoning": ["중립적 시장 심리"]
            },
            timestamp=datetime.now()
        )
    ]

def test_make_decision(decision_maker, sample_market_context, sample_risk_parameters, sample_agent_analyses):
    """최종 결정 생성 테스트"""
    request = DecisionRequest(
        symbol="005930.KS",
        timeframe="1d",
        current_price=70000.0,
        market_context=sample_market_context,
        risk_parameters=sample_risk_parameters,
        agent_analyses=sample_agent_analyses
    )
    
    response = decision_maker.make_decision(request)
    
    # 기본 구조 확인
    assert response.decision is not None
    assert response.error_message is None
    
    # 결정 타입 확인
    assert response.decision.decision_type in [DecisionType.BUY, DecisionType.SELL, DecisionType.HOLD, DecisionType.WAIT]
    
    # 신뢰도 확인
    assert response.decision.confidence in [DecisionConfidence.HIGH, DecisionConfidence.MEDIUM, DecisionConfidence.LOW]
    
    # 리스크 레벨 확인
    assert response.decision.risk_level in [RiskLevel.HIGH, RiskLevel.MODERATE, RiskLevel.LOW]
    
    # 포지션 크기 확인
    assert response.decision.position_size in [PositionSize.FULL, PositionSize.HALF, PositionSize.QUARTER, PositionSize.MINIMAL]
    
    # 가격 정보 확인
    assert response.decision.entry_price == 70000.0
    assert response.decision.stop_loss is not None
    assert response.decision.take_profit is not None
    
    # 이유 확인
    assert len(response.decision.reasoning) > 0

def test_analyze_agent_consensus(decision_maker, sample_agent_analyses):
    """에이전트 합의 분석 테스트"""
    consensus = decision_maker._analyze_agent_consensus(sample_agent_analyses)
    
    assert consensus['buy_signals'] == 2
    assert consensus['hold_signals'] == 1
    assert consensus['sell_signals'] == 0
    assert consensus['wait_signals'] == 0
    assert 0.6 <= consensus['average_confidence'] <= 0.9
    assert len(consensus['key_reasons']) > 0

def test_analyze_market_context(decision_maker, sample_market_context):
    """시장 상황 분석 테스트"""
    analysis = decision_maker._analyze_market_context(sample_market_context)
    
    assert analysis['is_trending'] is True
    assert analysis['is_volatile'] is False
    assert analysis['volume_support'] is True
    assert analysis['market_sentiment'] == "BULLISH"
    assert analysis['risk_level'] in ['HIGH', 'MODERATE', 'LOW']

def test_assess_risk(decision_maker, sample_market_context, sample_risk_parameters, sample_agent_analyses):
    """리스크 평가 테스트"""
    consensus = decision_maker._analyze_agent_consensus(sample_agent_analyses)
    risk_assessment = decision_maker._assess_risk(
        sample_risk_parameters,
        sample_market_context,
        consensus
    )
    
    assert risk_assessment['position_size'] in [PositionSize.FULL, PositionSize.HALF, PositionSize.QUARTER, PositionSize.MINIMAL]
    assert risk_assessment['stop_loss'] == 0.02
    assert risk_assessment['take_profit'] == 0.04
    assert risk_assessment['risk_reward_ratio'] == 2.0
    assert risk_assessment['max_drawdown'] == 0.1

def test_determine_decision_type(decision_maker):
    """결정 타입 결정 테스트"""
    # 강한 매수 신호
    consensus_strong_buy = {
        'buy_signals': 3,
        'sell_signals': 0,
        'hold_signals': 0,
        'wait_signals': 0,
        'average_confidence': 0.8,
        'key_reasons': []
    }
    market_analysis_trending = {
        'is_trending': True,
        'is_volatile': False,
        'volume_support': True,
        'market_sentiment': "BULLISH",
        'risk_level': "LOW"
    }
    
    decision_type = decision_maker._determine_decision_type(
        consensus_strong_buy,
        market_analysis_trending
    )
    assert decision_type == DecisionType.BUY
    
    # 변동성 높은 시장
    market_analysis_volatile = {
        'is_trending': False,
        'is_volatile': True,
        'volume_support': False,
        'market_sentiment': "NEUTRAL",
        'risk_level': "HIGH"
    }
    
    decision_type = decision_maker._determine_decision_type(
        consensus_strong_buy,
        market_analysis_volatile
    )
    assert decision_type == DecisionType.WAIT

def test_determine_confidence(decision_maker):
    """신뢰도 결정 테스트"""
    # 높은 신뢰도 조건
    consensus_high = {
        'buy_signals': 3,
        'sell_signals': 0,
        'hold_signals': 0,
        'wait_signals': 0,
        'average_confidence': 0.85,
        'key_reasons': []
    }
    market_analysis_strong = {
        'is_trending': True,
        'is_volatile': False,
        'volume_support': True,
        'market_sentiment': "BULLISH",
        'risk_level': "LOW"
    }
    
    confidence = decision_maker._determine_confidence(
        consensus_high,
        market_analysis_strong
    )
    assert confidence == DecisionConfidence.HIGH
    
    # 중간 신뢰도 조건
    consensus_medium = {
        'buy_signals': 2,
        'sell_signals': 1,
        'hold_signals': 0,
        'wait_signals': 0,
        'average_confidence': 0.65,
        'key_reasons': []
    }
    
    confidence = decision_maker._determine_confidence(
        consensus_medium,
        market_analysis_strong
    )
    assert confidence == DecisionConfidence.MEDIUM

def test_determine_risk_level(decision_maker):
    """리스크 레벨 결정 테스트"""
    # 높은 리스크 조건
    market_analysis_high_risk = {
        'is_trending': False,
        'is_volatile': True,
        'volume_support': False,
        'market_sentiment': "BEARISH",
        'risk_level': "HIGH"
    }
    risk_assessment_high = {
        'position_size': PositionSize.MINIMAL,
        'stop_loss': 0.02,
        'take_profit': 0.03,
        'risk_reward_ratio': 1.5,
        'max_drawdown': 0.1
    }
    
    risk_level = decision_maker._determine_risk_level(
        market_analysis_high_risk,
        risk_assessment_high
    )
    assert risk_level == RiskLevel.HIGH
    
    # 낮은 리스크 조건
    market_analysis_low_risk = {
        'is_trending': True,
        'is_volatile': False,
        'volume_support': True,
        'market_sentiment': "BULLISH",
        'risk_level': "LOW"
    }
    risk_assessment_low = {
        'position_size': PositionSize.FULL,
        'stop_loss': 0.02,
        'take_profit': 0.04,
        'risk_reward_ratio': 2.0,
        'max_drawdown': 0.1
    }
    
    risk_level = decision_maker._determine_risk_level(
        market_analysis_low_risk,
        risk_assessment_low
    )
    assert risk_level == RiskLevel.LOW 