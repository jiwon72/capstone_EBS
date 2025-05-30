import logging
from datetime import datetime
from typing import List, Dict, Optional
from .models import (
    DecisionType, DecisionConfidence, RiskLevel, PositionSize,
    AgentAnalysis, MarketContext, RiskParameters, FinalDecision,
    DecisionRequest, DecisionResponse
)

class DecisionMaker:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # 로그 포맷 설정
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # 파일 핸들러 추가
        fh = logging.FileHandler(f'logs/decision_maker_{datetime.now().strftime("%Y%m%d")}.log')
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        
        # 콘솔 핸들러 추가
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

    def make_decision(self, request: DecisionRequest) -> DecisionResponse:
        """
        최종 투자 결정을 내립니다.
        """
        try:
            self.logger.info("최종 투자 결정 시작")
            
            # 1. 기술적 분석 결과 요약
            self.logger.info("=== 기술적 분석 결과 ===")
            tech_analysis = request.technical_analysis or {}
            self.logger.info(f"현재가: {float(tech_analysis.get('current_price', 0)):,.0f}원")
            self.logger.info(f"이동평균선: {float(tech_analysis.get('ma', 0)):,.0f}원")
            self.logger.info(f"RSI: {float(tech_analysis.get('rsi', 0)):.1f}")
            self.logger.info(f"MACD: {float(tech_analysis.get('macd', 0)):.1f}")
            self.logger.info(f"볼린저밴드: {float(tech_analysis.get('bollinger', 0)):,.0f}원")
            self.logger.info(f"거래량: {float(tech_analysis.get('volume', 0)):,.0f}")
            self.logger.info(f"변동성: {float(tech_analysis.get('volatility', 0)):.1f}")
            self.logger.info(f"추세강도: {float(tech_analysis.get('trend_strength', 0)):.1f}")
            self.logger.info(f"지지선: {float(tech_analysis.get('support', 0)):,.0f}원")
            self.logger.info(f"저항선: {float(tech_analysis.get('resistance', 0)):,.0f}원")
            
            # 2. 뉴스 분석 결과 요약
            self.logger.info("=== 뉴스 분석 결과 ===")
            news_analysis = request.news_analysis or {}
            self.logger.info(f"감성 점수: {news_analysis.get('sentiment_score', 0):.2f}")
            self.logger.info(f"시장 영향도: {news_analysis.get('market_impact', 'UNKNOWN')}")
            self.logger.info(f"주요 키워드: {', '.join(news_analysis.get('key_phrases', []))}")
            
            # 3. 전략 분석 결과 요약
            self.logger.info("=== 전략 분석 결과 ===")
            strategy = request.strategy or {}
            self.logger.info(f"추천 전략: {strategy.get('recommended_strategy', 'UNKNOWN')}")
            self.logger.info(f"목표가: {strategy.get('target_price', 0):,.0f}원")
            self.logger.info(f"손절가: {strategy.get('stop_loss', 0):,.0f}원")
            self.logger.info(f"예상 수익률: {strategy.get('expected_return', 0):.1f}%")
            
            # 4. 리스크 분석 결과 요약
            self.logger.info("=== 리스크 분석 결과 ===")
            risk = request.risk_assessment or {}
            self.logger.info(f"변동성: {risk.get('volatility', 0):.2f}")
            self.logger.info(f"리스크 점수: {risk.get('risk_score', 0):.2f}")
            self.logger.info(f"최대 손실 가능성: {risk.get('max_loss_potential', 0):.1f}%")
            
            # 5. 최종 결정 생성
            decision = self._generate_decision(
                request.technical_analysis,
                request.news_analysis,
                request.strategy,
                request.risk_assessment
            )
            
            # 6. 최종 결정 상세 정보
            self.logger.info("=== 최종 투자 결정 ===")
            self.logger.info(f"결정: {decision['decision']}")
            self.logger.info(f"신뢰도: {decision['confidence']:.2f}")
            self.logger.info(f"포지션 크기: {decision['position_size']:.1f}%")
            self.logger.info(f"진입가: {decision['entry_price']:,.0f}원")
            self.logger.info(f"목표가: {decision['target_price']:,.0f}원")
            self.logger.info(f"손절가: {decision['stop_loss']:,.0f}원")
            self.logger.info(f"예상 수익률: {decision['expected_return']:.1f}%")
            self.logger.info(f"최대 손실률: {decision['max_loss']:.1f}%")
            
            # 7. 결정 근거
            self.logger.info("=== 결정 근거 ===")
            for reason in decision['reasons']:
                self.logger.info(f"- {reason}")
            
            self.logger.info("최종 투자 결정 완료")
            return DecisionResponse(**decision)
            
        except Exception as e:
            self.logger.error(f"투자 결정 중 오류 발생: {str(e)}")
            return DecisionResponse(
                decision=DecisionType.HOLD,
                confidence=0.0,
                position_size=0.0,
                entry_price=0.0,
                target_price=0.0,
                stop_loss=0.0,
                expected_return=0.0,
                max_loss=0.0,
                reasons=["오류로 인해 보수적인 HOLD 결정"],
                error_message=str(e)
            )

    def _analyze_agent_consensus(self, agent_analyses: List[AgentAnalysis]) -> Dict:
        """
        각 에이전트의 분석 결과를 종합하여 합의를 도출합니다.
        """
        consensus = {
            'buy_signals': 0,
            'sell_signals': 0,
            'hold_signals': 0,
            'wait_signals': 0,
            'average_confidence': 0.0,
            'key_reasons': []
        }
        
        total_confidence = 0.0
        
        for analysis in agent_analyses:
            # 신호 카운트
            if analysis.recommendation.get('signal') == 'BUY':
                consensus['buy_signals'] += 1
            elif analysis.recommendation.get('signal') == 'SELL':
                consensus['sell_signals'] += 1
            elif analysis.recommendation.get('signal') == 'HOLD':
                consensus['hold_signals'] += 1
            else:
                consensus['wait_signals'] += 1
            
            # 신뢰도 평균 계산
            total_confidence += analysis.confidence_score
            
            # 주요 이유 수집
            if 'reasoning' in analysis.recommendation:
                consensus['key_reasons'].extend(analysis.recommendation['reasoning'])
        
        consensus['average_confidence'] = total_confidence / len(agent_analyses)
        
        return consensus

    def _analyze_market_context(self, market_context: MarketContext) -> Dict:
        """
        시장 상황을 분석합니다.
        """
        analysis = {
            'is_trending': market_context.trend_strength > 0.6,
            'is_volatile': market_context.volatility_level > 0.7,
            'volume_support': market_context.volume_profile.get('trend_aligned', False),
            'market_sentiment': market_context.market_sentiment,
            'risk_level': 'HIGH' if market_context.volatility_level > 0.7 else 
                         'MODERATE' if market_context.volatility_level > 0.4 else 'LOW'
        }
        
        return analysis

    def _assess_risk(
        self,
        risk_parameters: RiskParameters,
        market_context: MarketContext,
        agent_consensus: Dict
    ) -> Dict:
        """
        리스크를 평가합니다.
        """
        risk_assessment = {
            'position_size': self._determine_position_size(
                risk_parameters,
                market_context,
                agent_consensus
            ),
            'stop_loss': risk_parameters.stop_loss_percentage,
            'take_profit': risk_parameters.take_profit_percentage,
            'risk_reward_ratio': risk_parameters.risk_reward_ratio,
            'max_drawdown': risk_parameters.max_drawdown
        }
        
        return risk_assessment

    def _determine_position_size(
        self,
        risk_parameters: RiskParameters,
        market_context: MarketContext,
        agent_consensus: Dict
    ) -> PositionSize:
        """
        적절한 포지션 크기를 결정합니다.
        """
        # 시장 상황과 에이전트 합의를 기반으로 포지션 크기 결정
        if market_context.volatility_level > 0.7:
            return PositionSize.MINIMAL
        
        if agent_consensus['average_confidence'] > 0.8 and \
           market_context.trend_strength > 0.7 and \
           market_context.volume_profile.get('trend_aligned', False):
            return PositionSize.FULL
        
        if agent_consensus['average_confidence'] > 0.6 and \
           market_context.trend_strength > 0.5:
            return PositionSize.HALF
        
        return PositionSize.QUARTER

    def _generate_final_decision(
        self,
        request: DecisionRequest,
        agent_consensus: Dict,
        market_analysis: Dict,
        risk_assessment: Dict
    ) -> FinalDecision:
        """
        최종 투자 결정을 생성합니다.
        """
        # 결정 타입 결정
        decision_type = self._determine_decision_type(
            agent_consensus,
            market_analysis
        )
        
        # 신뢰도 결정
        confidence = self._determine_confidence(
            agent_consensus,
            market_analysis
        )
        
        # 리스크 레벨 결정
        risk_level = self._determine_risk_level(
            market_analysis,
            risk_assessment
        )
        
        # 진입가, 손절가, 목표가 계산
        entry_price = request.current_price
        stop_loss = entry_price * (1 - risk_assessment['stop_loss'])
        take_profit = entry_price * (1 + risk_assessment['take_profit'])
        
        # 결정 이유 수집
        reasoning = self._collect_decision_reasons(
            agent_consensus,
            market_analysis,
            risk_assessment
        )
        
        return FinalDecision(
            decision_type=decision_type,
            confidence=confidence,
            risk_level=risk_level,
            position_size=risk_assessment['position_size'],
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reasoning=reasoning,
            timestamp=datetime.now(),
            market_context=request.market_context,
            risk_parameters=request.risk_parameters,
            agent_analyses=request.agent_analyses
        )

    def _determine_decision_type(
        self,
        agent_consensus: Dict,
        market_analysis: Dict
    ) -> DecisionType:
        """
        최종 결정 타입을 결정합니다.
        """
        # 매수/매도 신호의 차이 계산
        signal_diff = agent_consensus['buy_signals'] - agent_consensus['sell_signals']
        
        # 시장 상황이 트렌딩이고 거래량이 지지하는 경우
        if market_analysis['is_trending'] and market_analysis['volume_support']:
            if signal_diff >= 2:  # 강한 매수 신호
                return DecisionType.BUY
            elif signal_diff <= -2:  # 강한 매도 신호
                return DecisionType.SELL
        
        # 시장이 변동성이 높은 경우
        if market_analysis['is_volatile']:
            return DecisionType.WAIT
        
        # 중립적인 상황
        if abs(signal_diff) < 2:
            return DecisionType.HOLD
        
        # 약한 신호
        return DecisionType.BUY if signal_diff > 0 else DecisionType.SELL

    def _determine_confidence(
        self,
        agent_consensus: Dict,
        market_analysis: Dict
    ) -> DecisionConfidence:
        """
        결정의 신뢰도를 결정합니다.
        """
        if agent_consensus['average_confidence'] > 0.8 and \
           market_analysis['is_trending'] and \
           market_analysis['volume_support']:
            return DecisionConfidence.HIGH
        
        if agent_consensus['average_confidence'] > 0.6 and \
           (market_analysis['is_trending'] or market_analysis['volume_support']):
            return DecisionConfidence.MEDIUM
        
        return DecisionConfidence.LOW

    def _determine_risk_level(
        self,
        market_analysis: Dict,
        risk_assessment: Dict
    ) -> RiskLevel:
        """
        리스크 레벨을 결정합니다.
        """
        if market_analysis['is_volatile'] or \
           risk_assessment['risk_reward_ratio'] < 1.5:
            return RiskLevel.HIGH
        
        if market_analysis['is_trending'] and \
           risk_assessment['risk_reward_ratio'] >= 2.0:
            return RiskLevel.LOW
        
        return RiskLevel.MODERATE

    def _collect_decision_reasons(
        self,
        agent_consensus: Dict,
        market_analysis: Dict,
        risk_assessment: Dict
    ) -> List[str]:
        """
        결정의 이유를 수집합니다.
        """
        reasons = []
        
        # 에이전트 합의 관련 이유
        if agent_consensus['buy_signals'] > agent_consensus['sell_signals']:
            reasons.append(f"매수 신호가 매도 신호보다 {agent_consensus['buy_signals'] - agent_consensus['sell_signals']}개 더 많습니다.")
        elif agent_consensus['sell_signals'] > agent_consensus['buy_signals']:
            reasons.append(f"매도 신호가 매수 신호보다 {agent_consensus['sell_signals'] - agent_consensus['buy_signals']}개 더 많습니다.")
        
        # 시장 상황 관련 이유
        if market_analysis['is_trending']:
            reasons.append("현재 강한 추세가 형성되어 있습니다.")
        if market_analysis['volume_support']:
            reasons.append("거래량이 추세를 지지하고 있습니다.")
        if market_analysis['is_volatile']:
            reasons.append("시장 변동성이 높습니다.")
        
        # 리스크 관련 이유
        if risk_assessment['risk_reward_ratio'] >= 2.0:
            reasons.append(f"리스크 대비 수익 비율이 {risk_assessment['risk_reward_ratio']:.1f}로 유리합니다.")
        
        return reasons

    def _generate_decision(
        self,
        technical_analysis: Optional[Dict],
        news_analysis: Optional[Dict],
        strategy: Optional[Dict],
        risk_assessment: Optional[Dict]
    ) -> Dict:
        """
        분석 결과를 종합하여 최종 결정을 생성합니다.
        """
        try:
            # 기본값 설정
            decision = {
                'decision': DecisionType.HOLD,
                'confidence': 0.5,
                'position_size': 0.0,
                'entry_price': 0.0,
                'target_price': 0.0,
                'stop_loss': 0.0,
                'expected_return': 0.0,
                'max_loss': 0.0,
                'reasons': []
            }

            # 기술적 분석 기반 결정
            if technical_analysis:
                if technical_analysis.get('trend_analysis', {}).get('short_term_trend') == 'UPTREND':
                    decision['decision'] = DecisionType.BUY
                    decision['confidence'] += 0.2
                    decision['reasons'].append("상승 추세가 형성되어 있습니다.")
                elif technical_analysis.get('trend_analysis', {}).get('short_term_trend') == 'DOWNTREND':
                    decision['decision'] = DecisionType.SELL
                    decision['confidence'] += 0.2
                    decision['reasons'].append("하락 추세가 형성되어 있습니다.")

            # 뉴스 분석 기반 결정
            if news_analysis:
                sentiment_score = news_analysis.get('sentiment_score', 0)
                if sentiment_score > 0.6:
                    decision['confidence'] += 0.1
                    decision['reasons'].append("긍정적인 뉴스 감성이 우세합니다.")
                elif sentiment_score < 0.4:
                    decision['confidence'] -= 0.1
                    decision['reasons'].append("부정적인 뉴스 감성이 우세합니다.")

            # 전략 분석 기반 결정
            if strategy:
                decision['target_price'] = strategy.get('target_price', 0.0)
                decision['stop_loss'] = strategy.get('stop_loss', 0.0)
                decision['expected_return'] = strategy.get('expected_return', 0.0)
                if strategy.get('expected_return', 0) > 5.0:
                    decision['confidence'] += 0.2
                    decision['reasons'].append(f"예상 수익률이 {strategy.get('expected_return', 0):.1f}%로 유리합니다.")

            # 리스크 분석 기반 결정
            if risk_assessment:
                decision['max_loss'] = risk_assessment.get('max_loss_potential', 0.0)
                if risk_assessment.get('risk_score', 0) > 70:
                    decision['position_size'] = 0.0
                    decision['reasons'].append("리스크가 높아 포지션을 취하지 않습니다.")
                elif risk_assessment.get('risk_score', 0) > 50:
                    decision['position_size'] = 0.5
                    decision['reasons'].append("리스크가 중간 수준이므로 포지션 크기를 제한합니다.")
                else:
                    decision['position_size'] = 1.0
                    decision['reasons'].append("리스크가 낮아 전체 포지션을 취합니다.")

            # 신뢰도 범위 조정 (0-1)
            decision['confidence'] = max(0.0, min(1.0, decision['confidence']))

            return decision

        except Exception as e:
            self.logger.error(f"결정 생성 중 오류 발생: {str(e)}")
            return {
                'decision': DecisionType.HOLD,
                'confidence': 0.0,
                'position_size': 0.0,
                'entry_price': 0.0,
                'target_price': 0.0,
                'stop_loss': 0.0,
                'expected_return': 0.0,
                'max_loss': 0.0,
                'reasons': ["오류로 인해 보수적인 HOLD 결정"]
            } 