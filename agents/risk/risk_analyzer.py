import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Union, Tuple
from datetime import datetime, timedelta
import yfinance as yf
from scipy import stats
from .models import (
    Position, PortfolioStats, RiskMetrics, RiskAlert,
    PositionAdjustment, RiskLevel, RiskAssessmentResponse,
    RiskAssessmentRequest
)
import logging
from agents.utils.call_openai_api import call_openai_api

class RiskAnalyzer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # 로그 포맷 설정
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # 파일 핸들러 추가
        fh = logging.FileHandler(f'logs/risk_analyzer_{datetime.now().strftime("%Y%m%d")}.log')
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        
        # 콘솔 핸들러 추가
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        self.risk_free_rate = 0.02  # 연간 2% 가정

    def _fetch_historical_data(self, symbols: List[str], period: str = "1y") -> pd.DataFrame:
        """자산의 역사적 가격 데이터 조회"""
        data = pd.DataFrame()
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period=period)
                if hist.empty:
                    self.logger.warning(f"{symbol}의 히스토리컬 데이터가 없습니다.")
                    continue
                data[symbol] = hist['Close']
            except Exception as e:
                self.logger.error(f"{symbol} 데이터 조회 중 오류 발생: {str(e)}")
                continue
        
        if data.empty:
            self.logger.warning("모든 심볼에 대한 데이터 조회에 실패했습니다.")
            # 기본 데이터프레임 반환
            return pd.DataFrame(columns=symbols)
            
        return data

    def _calculate_portfolio_stats(
        self,
        positions: List[Position],
        portfolio_value: float
    ) -> PortfolioStats:
        """포트폴리오 기본 통계 계산"""
        if not positions:
            # 포지션이 없는 경우 기본값 반환
            return PortfolioStats(
                total_value=portfolio_value,
                cash_balance=portfolio_value,
                equity_value=0.0,
                daily_pnl=0.0,
                total_pnl=0.0,
                number_of_positions=0,
                largest_position_size=0.0,
                portfolio_beta=0.0
            )
            
        equity_value = sum(pos.quantity * pos.current_price for pos in positions)
        cash_balance = portfolio_value - equity_value
        
        daily_pnl = sum(
            pos.quantity * (pos.current_price - pos.entry_price)
            for pos in positions
        )
        
        return PortfolioStats(
            total_value=portfolio_value,
            cash_balance=cash_balance,
            equity_value=equity_value,
            daily_pnl=daily_pnl,
            total_pnl=daily_pnl,  # 간단한 예시, 실제로는 누적 PnL 계산 필요
            number_of_positions=len(positions),
            largest_position_size=max(pos.position_weight for pos in positions) if positions else 0.0,
            portfolio_beta=1.0  # 실제로는 시장 대비 베타 계산 필요
        )

    def _calculate_risk_metrics(
        self,
        positions: List[Position],
        historical_data: pd.DataFrame
    ) -> RiskMetrics:
        """리스크 지표 계산"""
        if not positions or historical_data.empty:
            return RiskMetrics(
                value_at_risk_95=0.0,
                value_at_risk_99=0.0,
                expected_shortfall=0.0,
                volatility=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                beta=0.0,
                correlation_matrix={},
                concentration_risk=0.0,
                liquidity_risk=0.0
            )
        try:
            returns = historical_data.pct_change().dropna()
            if returns.empty:
                self.logger.warning("수익률 계산 결과가 없습니다.")
                return self._get_default_risk_metrics()

            weights = np.array([pos.position_weight for pos in positions])
            if not np.isclose(sum(weights), 1.0, atol=1e-6):
                self.logger.warning("포트폴리오 가중치의 합이 1이 아닙니다. 정규화를 수행합니다.")
                weights = weights / sum(weights)

            portfolio_returns = returns.dot(weights)
            var_95 = np.percentile(portfolio_returns, 5)
            var_99 = np.percentile(portfolio_returns, 1)

            es = portfolio_returns[portfolio_returns <= var_95].mean()
            if pd.isna(es):
                es = var_95

            volatility = portfolio_returns.std() * np.sqrt(252)
            if pd.isna(volatility):
                volatility = 0.0

            excess_returns = portfolio_returns - self.risk_free_rate/252
            if excess_returns.std() == 0:
                sharpe_ratio = 0.0
            else:
                sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std()

            cumulative_returns = (1 + portfolio_returns).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdowns = cumulative_returns / rolling_max - 1
            max_drawdown = drawdowns.min()
            if pd.isna(max_drawdown):
                max_drawdown = 0.0

            correlation_matrix = returns.corr().to_dict()
            concentration_risk = np.sum(weights ** 2)

            return RiskMetrics(
                value_at_risk_95=abs(var_95),
                value_at_risk_99=abs(var_99),
                expected_shortfall=abs(es),
                volatility=volatility,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=abs(max_drawdown),
                beta=1.0,
                correlation_matrix=correlation_matrix,
                concentration_risk=concentration_risk,
                liquidity_risk=0.5
            )
        except Exception as e:
            self.logger.error(f"리스크 지표 계산 중 오류 발생: {str(e)}")
            return self._get_default_risk_metrics()

    def _get_default_risk_metrics(self) -> RiskMetrics:
        """기본 리스크 지표를 반환합니다."""
        return RiskMetrics(
            value_at_risk_95=0.0,
            value_at_risk_99=0.0,
            expected_shortfall=0.0,
            volatility=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            beta=0.0,
            correlation_matrix={},
            concentration_risk=0.0,
            liquidity_risk=0.0
        )

    def _generate_risk_alerts(
        self,
        portfolio_stats: PortfolioStats,
        risk_metrics: RiskMetrics,
        risk_tolerance: float
    ) -> List[RiskAlert]:
        """리스크 알림 생성"""
        alerts = []
        
        # 변동성 알림
        if risk_metrics.volatility > 0.3:  # 30% 이상의 연간 변동성
            alerts.append(RiskAlert(
                alert_type="HIGH_VOLATILITY",
                severity=RiskLevel.HIGH,
                message=f"Portfolio volatility ({risk_metrics.volatility:.1%}) is above threshold",
                affected_assets=["PORTFOLIO"],
                recommended_actions=["Consider reducing position sizes", "Add hedging positions"]
            ))
        
        # 집중도 리스크 알림
        if risk_metrics.concentration_risk > 0.3:
            alerts.append(RiskAlert(
                alert_type="CONCENTRATION_RISK",
                severity=RiskLevel.MODERATE,
                message="Portfolio concentration is too high",
                affected_assets=["PORTFOLIO"],
                recommended_actions=["Diversify holdings", "Reduce largest positions"]
            ))
        
        # Sharpe Ratio 알림
        if risk_metrics.sharpe_ratio < 0.5:
            alerts.append(RiskAlert(
                alert_type="LOW_SHARPE_RATIO",
                severity=RiskLevel.MODERATE,
                message="Risk-adjusted returns are below target",
                affected_assets=["PORTFOLIO"],
                recommended_actions=["Review strategy performance", "Adjust position sizing"]
            ))
        
        return alerts

    def _calculate_position_adjustments(
        self,
        positions: List[Position],
        risk_metrics: RiskMetrics,
        risk_tolerance: float
    ) -> List[PositionAdjustment]:
        """포지션 조정 제안"""
        try:
            adjustments = []

            if not positions:
                return adjustments

            for pos in positions:
                try:
                    if pos.position_weight > 0.2:
                        adjustments.append(PositionAdjustment(
                            asset_id=pos.asset_id,
                            current_weight=pos.position_weight,
                            target_weight=0.2,
                            action="DECREASE",
                            reason="Position size exceeds concentration limit"
                        ))
                except Exception as e:
                    self.logger.error(f"포지션 {pos.asset_id} 조정 계산 중 오류 발생: {str(e)}")
                    continue

            return adjustments

        except Exception as e:
            self.logger.error(f"포지션 조정 계산 중 오류 발생: {str(e)}")
            return []

    def assess_risk(self, request: RiskAssessmentRequest) -> RiskAssessmentResponse:
        """
        리스크를 평가합니다.
        """
        try:
            self.logger.info("리스크 평가 시작")
            
            # 1. 변동성 분석
            volatility = self._calculate_volatility(request.market_data)
            self.logger.info(f"변동성: {volatility:.2f}")
            
            # 2. 리스크 점수 계산
            risk_score = self._calculate_risk_score(
                volatility,
                request.market_data,
                request.technical_indicators
            )
            self.logger.info(f"리스크 점수: {risk_score:.2f}")
            
            # 4. 리스크 레벨 결정
            risk_level = self._determine_risk_level(risk_score)
            self.logger.info(f"리스크 레벨: {risk_level}")
            
            # 5. 포지션 크기 제안
            position_size = self._suggest_position_size(risk_score, risk_level)
            self.logger.info(f"제안 포지션 크기: {position_size:.1f}%")
            
            # 6. 손절가/목표가 제안
            stop_loss, target_price = self._suggest_price_levels(
                request.market_data,
                request.technical_indicators,
                risk_level
            )
            self.logger.info(f"제안 손절가: {stop_loss:,.0f}원")
            self.logger.info(f"제안 목표가: {target_price:,.0f}원")
            
            # 7. 리스크 관리 전략 제안
            risk_management = self._suggest_risk_management(
                risk_level,
                volatility,
                request.technical_indicators
            )
            self.logger.info("리스크 관리 전략:")
            for strategy in risk_management:
                self.logger.info(f"- {strategy}")
            
            response = RiskAssessmentResponse(
                risk_score=risk_score,
                volatility=volatility,
                risk_level=risk_level,
                max_loss_potential=0.0,  # 최대 손실 가능성은 사용하지 않음
                suggested_position_size=position_size,
                stop_loss=stop_loss,
                target_price=target_price,
                risk_management_strategies=risk_management,
                timestamp=datetime.now().isoformat()
            )
            
            self.logger.info("리스크 평가 완료")
            return response
            
        except Exception as e:
            self.logger.error(f"리스크 평가 중 오류 발생: {str(e)}")
            return None

    def _determine_risk_level(self, risk_score: float) -> RiskLevel:
        """리스크 레벨 결정"""
        if risk_score >= 70:
            return RiskLevel.EXTREME
        elif risk_score >= 50:
            return RiskLevel.HIGH
        elif risk_score >= 30:
            return RiskLevel.MODERATE
        else:
            return RiskLevel.LOW

    def _suggest_position_size(self, risk_score: float, risk_level: RiskLevel) -> float:
        """포지션 크기 제안"""
        if risk_level == RiskLevel.EXTREME or getattr(risk_level, 'value', None) == 'extreme':
            return 0.1  # 최악의 경우 10% 포지션 크기 제안
        elif risk_level == RiskLevel.HIGH or getattr(risk_level, 'value', None) == 'high':
            return 0.2  # 높은 리스크 경우 20% 포지션 크기 제안
        elif risk_level == RiskLevel.MODERATE or getattr(risk_level, 'value', None) == 'moderate':
            return 0.3  # 중간 리스크 경우 30% 포지션 크기 제안
        else:
            return 0.4  # 낮은 리스크 경우 40% 포지션 크기 제안

    def _suggest_price_levels(self, market_data: pd.DataFrame, technical_indicators: Dict, risk_level: RiskLevel) -> Tuple[float, float]:
        """손절가와 목표가 제안"""
        # market_data가 비어 있거나 컬럼이 없으면 0.0 반환
        if market_data is None or market_data.empty or 'Low' not in market_data.columns or 'High' not in market_data.columns:
            return (0.0, 0.0)
        if risk_level == RiskLevel.EXTREME or getattr(risk_level, 'value', None) == 'extreme':
            stop_loss = market_data['Low'].min() * 0.9  # 최악의 경우 10% 손절
            target_price = market_data['High'].max() * 1.1  # 최악의 경우 10% 목표가
        elif risk_level == RiskLevel.HIGH or getattr(risk_level, 'value', None) == 'high':
            stop_loss = market_data['Low'].min() * 0.95  # 높은 리스크 경우 5% 손절
            target_price = market_data['High'].max() * 1.05  # 높은 리스크 경우 5% 목표가
        elif risk_level == RiskLevel.MODERATE or getattr(risk_level, 'value', None) == 'moderate':
            stop_loss = market_data['Low'].min() * 0.9  # 중간 리스크 경우 10% 손절
            target_price = market_data['High'].max() * 1.1  # 중간 리스크 경우 10% 목표가
        else:
            stop_loss = market_data['Low'].min() * 0.95  # 낮은 리스크 경우 5% 손절
            target_price = market_data['High'].max() * 1.05  # 낮은 리스크 경우 5% 목표가
        return (stop_loss, target_price)

    def _suggest_risk_management(self, risk_level: RiskLevel, volatility: float, technical_indicators: Dict) -> List[str]:
        """리스크 관리 전략 제안"""
        strategies = []
        if risk_level == RiskLevel.EXTREME or getattr(risk_level, 'value', None) == 'extreme':
            strategies.append("최악의 경우를 고려한 손절 전략 강화")
            strategies.append("최악의 경우를 고려한 헤지 전략 도입")
        elif risk_level == RiskLevel.HIGH or getattr(risk_level, 'value', None) == 'high':
            strategies.append("높은 리스크를 고려한 손절 전략 강화")
            strategies.append("높은 리스크를 고려한 헤지 전략 도입")
        elif risk_level == RiskLevel.MODERATE or getattr(risk_level, 'value', None) == 'moderate':
            strategies.append("중간 리스크를 고려한 손절 전략 강화")
            strategies.append("중간 리스크를 고려한 헤지 전략 도입")
        else:
            strategies.append("낮은 리스크를 고려한 손절 전략 강화")
            strategies.append("낮은 리스크를 고려한 헤지 전략 도입")
        
        return strategies

    def analyze_portfolio_risk(self, strategy_output: Dict) -> Dict:
        """
        포트폴리오의 전반적인 리스크를 분석합니다.
        
        Args:
            strategy_output (Dict): Strategy Agent의 출력 결과
            
        Returns:
            Dict: 리스크 분석 결과
        """
        try:
            # 1. 수익률 데이터 수집
            returns_data = self._collect_returns_data(strategy_output['portfolio'])
            
            # 2. 기본 리스크 지표 계산
            risk_metrics = self._calculate_basic_risk_metrics(returns_data)
            
            # 3. VaR 계산
            var_metrics = self._calculate_var(returns_data)
            
            # 4. 상관관계 분석
            correlation_analysis = self._analyze_correlations(returns_data)
            
            # 5. 스트레스 테스트
            stress_test_results = self._perform_stress_test(returns_data)
            
            return {
                'basic_risk_metrics': risk_metrics,
                'var_metrics': var_metrics,
                'correlation_analysis': correlation_analysis,
                'stress_test_results': stress_test_results,
                'risk_assessment': self._generate_risk_assessment(
                    risk_metrics, var_metrics, correlation_analysis, stress_test_results
                )
            }
            
        except Exception as e:
            self.logger.error(f"포트폴리오 리스크 분석 중 오류 발생: {str(e)}")
            raise

    def _collect_returns_data(self, portfolio: Dict) -> pd.DataFrame:
        """
        포트폴리오 종목들의 수익률 데이터를 수집합니다.
        """
        returns_data = {}
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)  # 1년치 데이터
        
        for stock, weight in portfolio.items():
            try:
                # yfinance를 사용하여 주가 데이터 수집
                stock_data = yf.download(stock, start=start_date, end=end_date)
                returns_data[stock] = stock_data['Adj Close'].pct_change().dropna()
            except Exception as e:
                self.logger.warning(f"{stock} 데이터 수집 실패: {str(e)}")
                
        return pd.DataFrame(returns_data)

    def _calculate_basic_risk_metrics(self, returns_data: pd.DataFrame) -> Dict:
        """
        기본적인 리스크 지표들을 계산합니다.
        """
        metrics = {}
        
        # 전체 포트폴리오 수익률
        portfolio_returns = returns_data.mean(axis=1)
        
        # 변동성 (연간화된 표준편차)
        volatility = portfolio_returns.std() * np.sqrt(252)
        
        # 최대 낙폭 (Maximum Drawdown)
        cumulative_returns = (1 + portfolio_returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = cumulative_returns / rolling_max - 1
        max_drawdown = drawdowns.min()
        
        # 샤프 비율 (무위험 수익률 2% 가정)
        risk_free_rate = 0.02
        excess_returns = portfolio_returns.mean() * 252 - risk_free_rate
        sharpe_ratio = excess_returns / volatility
        
        metrics.update({
            'volatility': volatility,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'annual_return': portfolio_returns.mean() * 252
        })
        
        return metrics

    def _calculate_var(self, returns_data: pd.DataFrame, confidence_level: float = 0.95) -> Dict:
        """
        Value at Risk (VaR)를 계산합니다.
        """
        portfolio_returns = returns_data.mean(axis=1)
        
        # Historical VaR
        historical_var = np.percentile(portfolio_returns, (1 - confidence_level) * 100)
        
        # Parametric VaR (정규분포 가정)
        mean_return = portfolio_returns.mean()
        std_return = portfolio_returns.std()
        parametric_var = mean_return + stats.norm.ppf(1 - confidence_level) * std_return
        
        return {
            'historical_var': historical_var,
            'parametric_var': parametric_var,
            'confidence_level': confidence_level
        }

    def _analyze_correlations(self, returns_data: pd.DataFrame) -> Dict:
        """
        종목 간 상관관계를 분석합니다.
        """
        correlation_matrix = returns_data.corr()
        
        # 평균 상관관계
        mean_correlation = correlation_matrix.values[np.triu_indices_from(correlation_matrix, k=1)].mean()
        
        # 높은 상관관계를 가진 종목 쌍 찾기
        high_corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                if abs(correlation_matrix.iloc[i,j]) > 0.7:  # 상관계수 0.7 이상
                    high_corr_pairs.append({
                        'pair': (correlation_matrix.columns[i], correlation_matrix.columns[j]),
                        'correlation': correlation_matrix.iloc[i,j]
                    })
        
        return {
            'correlation_matrix': correlation_matrix,
            'mean_correlation': mean_correlation,
            'high_correlation_pairs': high_corr_pairs
        }

    def _perform_stress_test(self, returns_data: pd.DataFrame) -> Dict:
        """
        스트레스 테스트를 수행합니다.
        """
        portfolio_returns = returns_data.mean(axis=1)
        
        # 최악의 시나리오 (최악의 5일 수익률)
        worst_5_days = portfolio_returns.nsmallest(5)
        
        # 변동성 스트레스 (표준편차의 2배)
        stress_volatility = portfolio_returns.std() * 2
        
        # 상관관계 스트레스 (모든 상관관계를 1로 가정)
        stress_correlation = 1.0
        
        return {
            'worst_5_days': worst_5_days,
            'stress_volatility': stress_volatility,
            'stress_correlation': stress_correlation
        }

    def _generate_risk_assessment(self, risk_metrics: Dict, var_metrics: Dict, 
                                correlation_analysis: Dict, stress_test_results: Dict) -> Dict:
        """
        종합적인 리스크 평가를 생성합니다.
        """
        try:
            # 리스크 등급 결정
            risk_level = self._determine_risk_level(risk_metrics, var_metrics)
            
            # 주요 리스크 요인 식별
            risk_factors = self._identify_risk_factors(risk_metrics, correlation_analysis)
            
            # 리스크 완화 제안
            risk_mitigation = self._suggest_risk_mitigation(risk_factors, stress_test_results)
            
            return {
                'risk_level': risk_level.value,  # RiskLevel 열거형의 value 속성 사용
                'risk_factors': risk_factors,
                'risk_mitigation': risk_mitigation
            }
        except Exception as e:
            self.logger.error(f"리스크 평가 생성 중 오류 발생: {str(e)}")
            return {
                'risk_level': RiskLevel.MODERATE.value,
                'risk_factors': [],
                'risk_mitigation': []
            }

    def _identify_risk_factors(self, risk_metrics: Dict, correlation_analysis: Dict) -> List[Dict]:
        """
        주요 리스크 요인을 식별합니다.
        """
        try:
            risk_factors = []
            
            # 변동성 리스크
            volatility = risk_metrics.get('volatility', 0)
            if volatility > 0.25:
                risk_factors.append({
                    'type': 'VOLATILITY',
                    'severity': 'HIGH',
                    'description': f'높은 변동성({volatility:.1%})으로 인한 리스크'
                })
            
            # 상관관계 리스크
            mean_correlation = correlation_analysis.get('mean_correlation', 0)
            if mean_correlation > 0.7:
                risk_factors.append({
                    'type': 'CORRELATION',
                    'severity': 'HIGH',
                    'description': f'높은 종목 간 상관관계({mean_correlation:.2f})로 인한 분산효과 감소'
                })
            
            # 최대낙폭 리스크
            max_drawdown = risk_metrics.get('max_drawdown', 0)
            if max_drawdown < -0.2:
                risk_factors.append({
                    'type': 'DRAWDOWN',
                    'severity': 'HIGH',
                    'description': f'큰 최대낙폭({max_drawdown:.1%})으로 인한 리스크'
                })
            
            return risk_factors
        except Exception as e:
            self.logger.error(f"리스크 요인 식별 중 오류 발생: {str(e)}")
            return []

    def _suggest_risk_mitigation(self, risk_factors: List[Dict], 
                               stress_test_results: Dict) -> List[Dict]:
        """
        리스크 완화 방안을 제안합니다.
        """
        try:
            mitigation_suggestions = []
            
            for factor in risk_factors:
                if factor.get('type') == 'VOLATILITY':
                    mitigation_suggestions.append({
                        'risk_factor': 'VOLATILITY',
                        'suggestion': '변동성 헤지 전략 도입 고려',
                        'priority': 'HIGH'
                    })
                elif factor.get('type') == 'CORRELATION':
                    mitigation_suggestions.append({
                        'risk_factor': 'CORRELATION',
                        'suggestion': '상관관계가 낮은 자산 추가 고려',
                        'priority': 'MEDIUM'
                    })
                elif factor.get('type') == 'DRAWDOWN':
                    mitigation_suggestions.append({
                        'risk_factor': 'DRAWDOWN',
                        'suggestion': '손절 전략 강화 및 포지션 크기 축소 고려',
                        'priority': 'HIGH'
                    })
            
            return mitigation_suggestions
        except Exception as e:
            self.logger.error(f"리스크 완화 방안 제안 중 오류 발생: {str(e)}")
            return []

    def _calculate_volatility(self, market_data: pd.DataFrame) -> float:
        """변동성 계산"""
        try:
            if market_data.empty:
                return 0.0
            returns = market_data['Close'].pct_change().dropna()
            return returns.std() * np.sqrt(252)  # 연간화된 변동성
        except Exception as e:
            self.logger.error(f"변동성 계산 중 오류 발생: {str(e)}")
            return 0.0

    def _calculate_liquidity_score(self, market_data: pd.DataFrame) -> float:
        """
        유동성 점수 계산 (0~100)
        - 일평균 거래대금, 스프레드 등 활용
        """
        try:
            if market_data is None or market_data.empty or 'Volume' not in market_data.columns or 'Close' not in market_data.columns:
                return 50.0
            avg_volume = market_data['Volume'].tail(20).mean()
            avg_price = market_data['Close'].tail(20).mean()
            trading_value = avg_volume * avg_price
            # 1억 미만: 0점, 100억 이상: 100점
            score = (trading_value - 1e7) / (1e10 - 1e7) * 100
            return max(0, min(score, 100))
        except Exception as e:
            self.logger.error(f"유동성 점수 계산 오류: {str(e)}")
            return 50.0

    def _calculate_financial_score(self, financials: dict) -> float:
        """
        재무 점수 계산 (0~100)
        - 부채비율, 이자보상배율 등 활용
        """
        try:
            if not financials:
                return 50.0
            debt_ratio = financials.get('debt_ratio', 100)
            interest_coverage = financials.get('interest_coverage', 5)
            # 부채비율 50% 이하: 100점, 300% 이상: 0점
            debt_score = max(0, min((300 - debt_ratio) / 2.5, 100))
            # 이자보상배율 10 이상: 100점, 1 이하: 0점
            ic_score = max(0, min((interest_coverage - 1) / 9 * 100, 100))
            score = 0.6 * debt_score + 0.4 * ic_score
            return max(0, min(score, 100))
        except Exception as e:
            self.logger.error(f"재무 점수 계산 오류: {str(e)}")
            return 50.0

    def _calculate_market_score(self, market_data: pd.DataFrame, benchmark_returns: pd.Series = None) -> float:
        """
        시장 점수 계산 (0~100)
        - 베타, 시장 상관계수 등 활용
        """
        try:
            if market_data is None or market_data.empty or 'Close' not in market_data.columns:
                return 50.0
            returns = market_data['Close'].pct_change().dropna()
            if benchmark_returns is not None and not benchmark_returns.empty:
                # 베타 계산
                beta = np.cov(returns, benchmark_returns[-len(returns):])[0,1] / np.var(benchmark_returns[-len(returns):])
                # 베타 0.8~1.2: 100점, 2.0 이상: 0점
                beta_score = max(0, min((2.0 - abs(beta - 1.0)) / 1.2 * 100, 100))
            else:
                beta_score = 50.0
            # 시장 상관계수 (없으면 0.5)
            corr = returns.corr(benchmark_returns) if benchmark_returns is not None and not benchmark_returns.empty else 0.5
            # 상관계수 0.3~0.7: 100점, 1.0: 0점
            corr_score = max(0, min((1.0 - abs(corr - 0.5)) / 0.5 * 100, 100))
            score = 0.7 * beta_score + 0.3 * corr_score
            return max(0, min(score, 100))
        except Exception as e:
            self.logger.error(f"시장 점수 계산 오류: {str(e)}")
            return 50.0

    def _calculate_risk_score(
        self,
        volatility: float,
        market_data: pd.DataFrame,
        technical_indicators: Dict,
        strategy_type: str = None,
        liquidity_score: float = None,
        financial_score: float = None,
        market_score: float = None
    ) -> float:
        """
        전략별 가중치를 적용한 리스크 점수 계산
        """
        # 전략별 가중치
        weights = {
            'momentum': {'volatility': 0.4, 'liquidity': 0.3, 'financial': 0.15, 'market': 0.15},
            'growth':   {'volatility': 0.3, 'liquidity': 0.2, 'financial': 0.25, 'market': 0.25},
            'value':    {'volatility': 0.2, 'liquidity': 0.25, 'financial': 0.3, 'market': 0.25},
        }
        stype = (strategy_type or 'momentum').lower()
        w = weights.get(stype, weights['momentum'])
        # 각 점수 산출 (없으면 0)
        v = min(volatility * 100, 100) if volatility is not None else 0
        l = min(liquidity_score, 100) if liquidity_score is not None else 0
        f = min(financial_score, 100) if financial_score is not None else 0
        m = min(market_score, 100) if market_score is not None else 0
        base_score = v * w['volatility'] + l * w['liquidity'] + f * w['financial'] + m * w['market']
        # 기술적 지표 기반 조정
        if technical_indicators:
            if 'rsi' in technical_indicators:
                rsi = technical_indicators['rsi']
                if isinstance(rsi, pd.Series):
                    rsi = rsi.iloc[-1]
                if rsi > 70:
                    base_score += 10
                elif rsi < 30:
                    base_score -= 10
            if 'macd' in technical_indicators and 'macd_signal' in technical_indicators:
                macd = technical_indicators['macd']
                macd_signal = technical_indicators['macd_signal']
                if isinstance(macd, pd.Series):
                    macd = macd.iloc[-1]
                if isinstance(macd_signal, pd.Series):
                    macd_signal = macd_signal.iloc[-1]
                if macd > macd_signal:
                    base_score -= 5
                else:
                    base_score += 5
        return max(0, min(base_score, 100))

    def propose(self, context):
        """
        자신의 리스크 분석 결과를 의견으로 제시합니다.
        """
        symbol = context.get('symbol', '005930')
        technical_indicators = context.get('technical_indicators', {})
        market_data = context.get('market_data', None)
        risk_score = 50.0
        risk_level = 'MODERATE'
        volatility = 0.0
        if market_data is not None and hasattr(self, '_calculate_volatility'):
            volatility = self._calculate_volatility(market_data)
            risk_score = self._calculate_risk_score(volatility, market_data, technical_indicators)
            risk_level_enum = self._determine_risk_level(risk_score)
            risk_level = risk_level_enum.value if hasattr(risk_level_enum, 'value') else str(risk_level_enum)
        decision = 'HOLD'
        confidence = 0.5
        reasons = []
        if risk_score >= 70:
            decision = 'SELL'
            confidence = min(1.0, (risk_score-60)/40)
            reasons.append(f'리스크 점수 {risk_score:.1f} (레벨: {risk_level}) - 리스크 과다, 변동성 {volatility:.2f}')
        elif risk_score <= 30:
            decision = 'BUY'
            confidence = min(1.0, (40-risk_score)/40)
            reasons.append(f'리스크 점수 {risk_score:.1f} (레벨: {risk_level}) - 리스크 낮음, 변동성 {volatility:.2f}')
        else:
            decision = 'HOLD'
            confidence = 0.5
            reasons.append(f'리스크 점수 {risk_score:.1f} (레벨: {risk_level}) - 중립, 변동성 {volatility:.2f}')
        reason = ', '.join(reasons) if reasons else '리스크 분석 결과'
        return {
            'agent': 'risk_analyzer',
            'decision': decision,
            'confidence': confidence,
            'reason': reason
        }

    def debate(self, context, others_opinions, my_opinion_1st_round=None):
        symbol = context.get('symbol', '005930')
        technical_indicators = context.get('technical_indicators', {})
        market_data = context.get('market_data', None)
        strategy_type = context.get('strategy_type', None)
        # 유동성/재무/시장 점수 산출(예시, 실제 적용시 context에서 받아야 함)
        liquidity_score = context.get('liquidity_score', 50.0)
        financial_score = context.get('financial_score', 50.0)
        market_score = context.get('market_score', 50.0)
        risk_score = 50.0
        risk_level = 'MODERATE'
        volatility = 0.0
        if market_data is not None and hasattr(self, '_calculate_volatility'):
            volatility = self._calculate_volatility(market_data)
            risk_score = self._calculate_risk_score(
                volatility, market_data, technical_indicators,
                strategy_type=strategy_type,
                liquidity_score=liquidity_score,
                financial_score=financial_score,
                market_score=market_score
            )
            risk_level_enum = self._determine_risk_level(risk_score)
            risk_level = risk_level_enum.value if hasattr(risk_level_enum, 'value') else str(risk_level_enum)
        핵심지표 = {"리스크점수": risk_score, "변동성": volatility, "리스크레벨": risk_level}
        주장 = f"리스크 점수 {risk_score:.1f}, 변동성 {volatility:.2f}, 리스크레벨 {risk_level}. "
        if risk_score >= 70:
            추천 = "SELL"
            신뢰도 = min(1.0, (risk_score-60)/40)
            주장 += "리스크가 과다하므로 매도 추천."
        elif risk_score <= 30:
            추천 = "BUY"
            신뢰도 = min(1.0, (40-risk_score)/40)
            주장 += "리스크가 낮으므로 매수 추천."
        else:
            추천 = "HOLD"
            신뢰도 = 0.5
            주장 += "중립적."
        prompt = f"""너는 리스크 관리 전문가야. 아래 수치를 바탕으로 투자자에게 논리적으로 설명해줘.\n리스크점수: {risk_score:.1f}, 변동성: {volatility:.2f}, 리스크레벨: {risk_level}\n이 수치가 의미하는 바와 투자 판단에 미치는 영향, 추천 의견을 전문가답게 3~4문장으로 써줘."""
        전문가설명 = call_openai_api(prompt)
        return {
            "agent": "risk_analyzer",
            "분야": "리스크",
            "핵심지표": 핵심지표,
            "주장": 주장,
            "추천": 추천,
            "신뢰도": 신뢰도,
            "전문가설명": 전문가설명
        } 