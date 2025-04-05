import openai
from typing import List, Dict, Optional
import uuid
from datetime import datetime
import pandas as pd
import numpy as np
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from .models import (
    StrategyType, TimeHorizon, MarketCondition,
    EntryCondition, ExitCondition, RiskParameters,
    TechnicalIndicator, StrategyResponse
)

class StrategyGenerator:
    def __init__(self):
        self.openai_client = openai.OpenAI()
        
    def _analyze_user_input(self, user_input: str) -> Dict:
        """사용자 입력을 분석하여 전략 파라미터 추출"""
        prompt = f"""
        다음 트레이딩 전략 요청을 분석하여 주요 파라미터를 추출해주세요:
        {user_input}
        
        다음 형식으로 응답해주세요:
        - 전략 유형:
        - 시간 프레임:
        - 리스크 수준:
        - 주요 지표:
        - 대상 자산:
        """
        
        response = self.openai_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        
        # GPT 응답 파싱 및 파라미터 추출 로직
        analysis = self._parse_gpt_response(response.choices[0].message.content)
        return analysis
    
    def _parse_gpt_response(self, response: str) -> Dict:
        """GPT 응답을 파싱하여 구조화된 데이터로 변환"""
        lines = response.strip().split('\n')
        parsed = {}
        
        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip('- ').lower().replace(' ', '_')
                parsed[key] = value.strip()
                
        return parsed
    
    def _generate_technical_indicators(self, strategy_type: StrategyType) -> List[TechnicalIndicator]:
        """전략 유형에 따른 기술적 지표 생성"""
        indicators = []
        
        if strategy_type == StrategyType.MOMENTUM:
            indicators.extend([
                TechnicalIndicator(
                    name="RSI",
                    value=0.0,  # 실제 값은 실시간 데이터로 업데이트
                    signal="neutral",
                    parameters={"period": 14}
                ),
                TechnicalIndicator(
                    name="MACD",
                    value=0.0,
                    signal="neutral",
                    parameters={
                        "fast_period": 12,
                        "slow_period": 26,
                        "signal_period": 9
                    }
                )
            ])
        elif strategy_type == StrategyType.MEAN_REVERSION:
            indicators.extend([
                TechnicalIndicator(
                    name="Bollinger_Bands",
                    value=0.0,
                    signal="neutral",
                    parameters={
                        "period": 20,
                        "std_dev": 2
                    }
                )
            ])
            
        return indicators
    
    def _generate_entry_conditions(
        self,
        strategy_type: StrategyType,
        risk_tolerance: float
    ) -> List[EntryCondition]:
        """진입 조건 생성"""
        conditions = []
        
        if strategy_type == StrategyType.MOMENTUM:
            conditions.extend([
                EntryCondition(
                    indicator="RSI",
                    condition="less_than",
                    threshold=30.0,
                    additional_params={"lookback_period": 14}
                ),
                EntryCondition(
                    indicator="MACD",
                    condition="crosses_above",
                    threshold=0.0,
                    additional_params={
                        "fast_period": 12,
                        "slow_period": 26
                    }
                )
            ])
            
        return conditions
    
    def _generate_exit_conditions(
        self,
        strategy_type: StrategyType,
        risk_tolerance: float
    ) -> List[ExitCondition]:
        """청산 조건 생성"""
        conditions = []
        
        if strategy_type == StrategyType.MOMENTUM:
            conditions.extend([
                ExitCondition(
                    indicator="RSI",
                    condition="greater_than",
                    threshold=70.0,
                    additional_params={"lookback_period": 14}
                ),
                ExitCondition(
                    indicator="Stop_Loss",
                    condition="less_than",
                    threshold=-0.02 * (1 + risk_tolerance)
                )
            ])
            
        return conditions
    
    def _generate_risk_parameters(
        self,
        risk_tolerance: float,
        strategy_type: StrategyType
    ) -> RiskParameters:
        """리스크 파라미터 생성"""
        base_position_size = 0.1
        adjusted_position_size = base_position_size * (1 + risk_tolerance)
        
        return RiskParameters(
            max_position_size=min(adjusted_position_size, 1.0),
            stop_loss=0.02 * (1 + risk_tolerance),
            take_profit=0.04 * (1 + risk_tolerance),
            max_drawdown=0.1 * (1 + risk_tolerance),
            risk_reward_ratio=2.0,
            max_correlation=0.7
        )
    
    def generate_strategy(
        self,
        user_input: str,
        market_conditions: Optional[MarketCondition] = None,
        risk_tolerance: Optional[float] = 0.5,
        time_horizon: Optional[TimeHorizon] = TimeHorizon.MEDIUM_TERM
    ) -> StrategyResponse:
        """전략 생성 메인 로직"""
        # 사용자 입력 분석
        analysis = self._analyze_user_input(user_input)
        
        # 전략 유형 결정
        strategy_type = StrategyType.MOMENTUM  # 기본값, 실제로는 분석 결과에 따라 결정
        
        # 기술적 지표 생성
        technical_indicators = self._generate_technical_indicators(strategy_type)
        
        # 진입/청산 조건 생성
        entry_conditions = self._generate_entry_conditions(strategy_type, risk_tolerance)
        exit_conditions = self._generate_exit_conditions(strategy_type, risk_tolerance)
        
        # 리스크 파라미터 생성
        risk_parameters = self._generate_risk_parameters(risk_tolerance, strategy_type)
        
        # 전략 응답 생성
        strategy = StrategyResponse(
            strategy_id=f"STRAT_{uuid.uuid4().hex[:8]}",
            strategy_type=strategy_type,
            entry_conditions=entry_conditions,
            exit_conditions=exit_conditions,
            position_size=risk_parameters.max_position_size,
            risk_parameters=risk_parameters,
            technical_indicators=technical_indicators,
            target_assets=["AAPL", "MSFT"],  # 예시, 실제로는 분석 결과에 따라 결정
            time_horizon=time_horizon,
            explanation=f"Generated {strategy_type.value} strategy based on user input: {user_input}"
        )
        
        return strategy 