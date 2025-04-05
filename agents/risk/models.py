from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from enum import Enum
from datetime import datetime

class RiskLevel(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    EXTREME = "extreme"

class Position(BaseModel):
    asset_id: str
    quantity: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    position_weight: float

class PortfolioStats(BaseModel):
    total_value: float
    cash_balance: float
    equity_value: float
    daily_pnl: float
    total_pnl: float
    number_of_positions: int
    largest_position_size: float
    portfolio_beta: float

class RiskMetrics(BaseModel):
    value_at_risk_95: float = Field(..., description="95% Value at Risk")
    value_at_risk_99: float = Field(..., description="99% Value at Risk")
    expected_shortfall: float = Field(..., description="Expected Shortfall (CVaR)")
    volatility: float = Field(..., description="Portfolio Volatility")
    sharpe_ratio: float = Field(..., description="Sharpe Ratio")
    max_drawdown: float = Field(..., description="Maximum Drawdown")
    beta: float = Field(..., description="Portfolio Beta")
    correlation_matrix: Dict[str, Dict[str, float]] = Field(..., description="Asset Correlation Matrix")
    concentration_risk: float = Field(..., description="Portfolio Concentration Risk")
    liquidity_risk: float = Field(..., description="Portfolio Liquidity Risk Score")

class RiskAlert(BaseModel):
    alert_type: str
    severity: RiskLevel
    message: str
    affected_assets: List[str]
    recommended_actions: List[str]
    timestamp: datetime = Field(default_factory=datetime.now)

class PositionAdjustment(BaseModel):
    asset_id: str
    current_weight: float
    target_weight: float
    action: str  # "INCREASE", "DECREASE", "HOLD"
    reason: str

class RiskAssessmentRequest(BaseModel):
    strategy_id: str
    portfolio_value: float
    positions: List[Position]
    risk_tolerance: float = Field(..., ge=0, le=1)
    market_conditions: Optional[Dict] = None
    target_beta: Optional[float] = None
    max_position_size: Optional[float] = Field(None, gt=0, le=1)
    max_sector_exposure: Optional[float] = Field(None, gt=0, le=1)

class RiskAssessmentResponse(BaseModel):
    risk_level: RiskLevel
    portfolio_stats: PortfolioStats
    risk_metrics: RiskMetrics
    risk_alerts: List[RiskAlert]
    position_adjustments: List[PositionAdjustment]
    timestamp: datetime = Field(default_factory=datetime.now)
    explanation: str 