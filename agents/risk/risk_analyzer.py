import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import yfinance as yf
from scipy import stats
from .models import (
    Position, PortfolioStats, RiskMetrics, RiskAlert,
    PositionAdjustment, RiskLevel, RiskAssessmentResponse
)

class RiskAnalyzer:
    def __init__(self):
        self.risk_free_rate = 0.02  # 연간 2% 가정

    def _fetch_historical_data(self, symbols: List[str], period: str = "1y") -> pd.DataFrame:
        """자산의 역사적 가격 데이터 조회"""
        data = pd.DataFrame()
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period=period)
                data[symbol] = hist['Close']
            except Exception as e:
                print(f"Error fetching data for {symbol}: {e}")
        return data

    def _calculate_portfolio_stats(
        self,
        positions: List[Position],
        portfolio_value: float
    ) -> PortfolioStats:
        """포트폴리오 기본 통계 계산"""
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
            largest_position_size=max(pos.position_weight for pos in positions),
            portfolio_beta=1.0  # 실제로는 시장 대비 베타 계산 필요
        )

    def _calculate_risk_metrics(
        self,
        positions: List[Position],
        historical_data: pd.DataFrame
    ) -> RiskMetrics:
        """리스크 지표 계산"""
        # 일간 수익률 계산
        returns = historical_data.pct_change().dropna()
        
        # 포트폴리오 가중치
        weights = np.array([pos.position_weight for pos in positions])
        
        # 포트폴리오 수익률
        portfolio_returns = returns.dot(weights)
        
        # VaR 계산
        var_95 = np.percentile(portfolio_returns, 5)
        var_99 = np.percentile(portfolio_returns, 1)
        
        # Expected Shortfall
        es = portfolio_returns[portfolio_returns <= var_95].mean()
        
        # 변동성
        volatility = portfolio_returns.std() * np.sqrt(252)  # 연간화
        
        # Sharpe Ratio
        excess_returns = portfolio_returns - self.risk_free_rate/252
        sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std()
        
        # Maximum Drawdown
        cumulative_returns = (1 + portfolio_returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = cumulative_returns / rolling_max - 1
        max_drawdown = drawdowns.min()
        
        # 상관관계 행렬
        correlation_matrix = returns.corr().to_dict()
        
        # 집중도 리스크 (HHI: Herfindahl-Hirschman Index)
        concentration_risk = np.sum(weights ** 2)
        
        return RiskMetrics(
            value_at_risk_95=abs(var_95),
            value_at_risk_99=abs(var_99),
            expected_shortfall=abs(es),
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=abs(max_drawdown),
            beta=1.0,  # 실제로는 시장 대비 베타 계산 필요
            correlation_matrix=correlation_matrix,
            concentration_risk=concentration_risk,
            liquidity_risk=0.5  # 실제로는 거래량 기반 유동성 리스크 계산 필요
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
        adjustments = []
        
        # 변동성 기반 포지션 조정
        for pos in positions:
            if pos.position_weight > 0.2:  # 단일 포지션 20% 초과
                adjustments.append(PositionAdjustment(
                    asset_id=pos.asset_id,
                    current_weight=pos.position_weight,
                    target_weight=0.2,
                    action="DECREASE",
                    reason="Position size exceeds concentration limit"
                ))
        
        # 상관관계 기반 조정
        # 실제 구현에서는 상관관계가 높은 자산들 간의 조정 로직 추가
        
        return adjustments

    def assess_risk(
        self,
        positions: List[Position],
        portfolio_value: float,
        risk_tolerance: float,
        market_conditions: Optional[Dict] = None
    ) -> RiskAssessmentResponse:
        """리스크 평가 메인 로직"""
        # 1. 포트폴리오 통계 계산
        portfolio_stats = self._calculate_portfolio_stats(positions, portfolio_value)
        
        # 2. 히스토리컬 데이터 조회
        symbols = [pos.asset_id for pos in positions]
        historical_data = self._fetch_historical_data(symbols)
        
        # 3. 리스크 지표 계산
        risk_metrics = self._calculate_risk_metrics(positions, historical_data)
        
        # 4. 리스크 알림 생성
        risk_alerts = self._generate_risk_alerts(portfolio_stats, risk_metrics, risk_tolerance)
        
        # 5. 포지션 조정 제안
        position_adjustments = self._calculate_position_adjustments(
            positions, risk_metrics, risk_tolerance
        )
        
        # 6. 전반적인 리스크 레벨 결정
        risk_level = self._determine_risk_level(risk_metrics, risk_tolerance)
        
        return RiskAssessmentResponse(
            risk_level=risk_level,
            portfolio_stats=portfolio_stats,
            risk_metrics=risk_metrics,
            risk_alerts=risk_alerts,
            position_adjustments=position_adjustments,
            explanation=self._generate_risk_explanation(
                risk_level, risk_metrics, risk_alerts
            )
        )

    def _determine_risk_level(
        self,
        risk_metrics: RiskMetrics,
        risk_tolerance: float
    ) -> RiskLevel:
        """전반적인 리스크 레벨 결정"""
        # 리스크 점수 계산 (0-100)
        risk_score = (
            0.3 * (risk_metrics.volatility * 100) +
            0.2 * (risk_metrics.value_at_risk_95 * 100) +
            0.2 * (risk_metrics.concentration_risk * 100) +
            0.15 * (risk_metrics.liquidity_risk * 100) +
            0.15 * (abs(risk_metrics.beta - 1) * 100)
        )
        
        # 리스크 허용도에 따른 임계값 조정
        threshold_high = 70 - (risk_tolerance * 20)
        threshold_moderate = 50 - (risk_tolerance * 15)
        threshold_low = 30 - (risk_tolerance * 10)
        
        if risk_score >= threshold_high:
            return RiskLevel.EXTREME
        elif risk_score >= threshold_moderate:
            return RiskLevel.HIGH
        elif risk_score >= threshold_low:
            return RiskLevel.MODERATE
        else:
            return RiskLevel.LOW

    def _generate_risk_explanation(
        self,
        risk_level: RiskLevel,
        risk_metrics: RiskMetrics,
        risk_alerts: List[RiskAlert]
    ) -> str:
        """리스크 평가 설명 생성"""
        explanation = f"Portfolio Risk Level: {risk_level.value.upper()}\n\n"
        
        explanation += "Key Risk Metrics:\n"
        explanation += f"- Volatility: {risk_metrics.volatility:.1%}\n"
        explanation += f"- 95% VaR: {risk_metrics.value_at_risk_95:.1%}\n"
        explanation += f"- Sharpe Ratio: {risk_metrics.sharpe_ratio:.2f}\n"
        explanation += f"- Maximum Drawdown: {risk_metrics.max_drawdown:.1%}\n"
        
        if risk_alerts:
            explanation += "\nRisk Alerts:\n"
            for alert in risk_alerts:
                explanation += f"- {alert.message}\n"
        
        return explanation 