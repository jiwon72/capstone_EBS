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
            
            # 3. 최대 손실 가능성 계산 (로그/보고서에서 제외)
            max_loss = self._calculate_max_loss(
                request.market_data,
                request.technical_indicators
            )
            
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
                max_loss_potential=0.0,  # 또는 None, 실제 계산값이 필요하면 대입
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

    def _calculate_risk_score(
        self,
        volatility: float,
        market_data: pd.DataFrame,
        technical_indicators: Dict
    ) -> float:
        """리스크 점수 계산"""
        try:
            # 기본 리스크 점수 (0-100)
            base_score = min(volatility * 100, 100)
            # 기술적 지표 기반 조정
            if technical_indicators:
                if 'rsi' in technical_indicators:
                    rsi = technical_indicators['rsi']
                    if isinstance(rsi, pd.Series):
                        rsi = rsi.iloc[-1]
                    if rsi > 70:  # 과매수
                        base_score += 10
                    elif rsi < 30:  # 과매도
                        base_score -= 10
                if 'macd' in technical_indicators and 'macd_signal' in technical_indicators:
                    macd = technical_indicators['macd']
                    macd_signal = technical_indicators['macd_signal']
                    if isinstance(macd, pd.Series):
                        macd = macd.iloc[-1]
                    if isinstance(macd_signal, pd.Series):
                        macd_signal = macd_signal.iloc[-1]
                    if macd > macd_signal:
                        base_score -= 5  # 상승 추세
                    else:
                        base_score += 5  # 하락 추세
            return max(0, min(base_score, 100))  # 0-100 범위로 제한
        except Exception as e:
            self.logger.error(f"리스크 점수 계산 중 오류 발생: {str(e)}")
            return 50.0  # 기본값

    def _calculate_max_loss(
        self,
        market_data: pd.DataFrame,
        technical_indicators: Dict
    ) -> float:
        """최대 손실 가능성 계산"""
        try:
            if market_data is None or market_data.empty:
                return None  # 데이터 없으면 None 반환
            # 최근 20일 최대 낙폭
            returns = market_data['Close'].pct_change().dropna()
            if returns.empty:
                return None
            rolling_max = returns.expanding().max()
            drawdowns = returns / rolling_max - 1
            max_drawdown = abs(drawdowns.min()) * 100
            # 기술적 지표 기반 조정
            if technical_indicators:
                if 'bb_lower' in technical_indicators and 'bb_middle' in technical_indicators:
                    bb_range = (technical_indicators['bb_middle'] - technical_indicators['bb_lower']) / technical_indicators['bb_middle']
                    max_drawdown = max(max_drawdown, bb_range * 100)
            return min(max_drawdown, 100.0)  # 최대 100%로 제한
        except Exception as e:
            self.logger.error(f"최대 손실 가능성 계산 중 오류 발생: {str(e)}")
            return None

    def propose(self, context):
        """
        자신의 리스크 분석 결과를 의견으로 제시합니다.
        """
        # 예시: context에서 symbol을 받아 리스크 분석
        symbol = context.get('symbol', '005930')
        # 실제 구현에서는 포트폴리오, 전략 등 더 많은 context 활용 가능
        # 여기서는 간단히 기본값 반환
        return {
            'agent': 'risk_analyzer',
            'decision': 'HOLD',
            'confidence': 0.5,
            'reason': '리스크 분석 결과(예시)'
        }

    def debate(self, context, others_opinions):
        """
        타 에이전트 의견을 참고해 자신의 의견을 보완/수정합니다.
        """
        my_opinion = self.propose(context)
        # 예시: 타 에이전트가 모두 BUY면 본인도 BUY로 보정
        if all(op['decision'] == 'BUY' for op in others_opinions):
            my_opinion['decision'] = 'BUY'
            my_opinion['reason'] += ' (타 에이전트 의견 반영)'
        return my_opinion 