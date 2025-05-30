from enum import Enum
from typing import List, Dict, Optional
from datetime import datetime
from pydantic import BaseModel, Field

class DecisionType(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    WAIT = "WAIT"

class DecisionConfidence(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"

class RiskLevel(str, Enum):
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    EXTREME = "EXTREME"

class PositionSize(str, Enum):
    FULL = "FULL"
    HALF = "HALF"
    QUARTER = "QUARTER"
    MINIMAL = "MINIMAL"

class AgentAnalysis(BaseModel):
    agent_name: str
    analysis_type: str
    confidence_score: float
    recommendation: Dict
    timestamp: datetime

class MarketContext(BaseModel):
    market_condition: str
    volatility_level: float
    trend_strength: float
    volume_profile: Dict
    market_sentiment: str

class RiskParameters(BaseModel):
    max_position_size: float
    stop_loss_percentage: float
    take_profit_percentage: float
    max_drawdown: float
    risk_reward_ratio: float

class FinalDecision(BaseModel):
    decision_type: DecisionType
    confidence: DecisionConfidence
    risk_level: RiskLevel
    position_size: PositionSize
    entry_price: Optional[float]
    stop_loss: Optional[float]
    take_profit: Optional[float]
    reasoning: List[str]
    timestamp: datetime
    market_context: MarketContext
    risk_parameters: RiskParameters
    agent_analyses: List[AgentAnalysis]

class DecisionRequest(BaseModel):
    symbol: str
    timeframe: str
    current_price: float
    market_context: MarketContext
    risk_parameters: RiskParameters
    agent_analyses: List[AgentAnalysis]
    technical_analysis: Optional[Dict] = None
    news_analysis: Optional[Dict] = None
    strategy: Optional[Dict] = None
    risk_assessment: Optional[Dict] = None

class DecisionResponse(BaseModel):
    decision: DecisionType
    confidence: float
    position_size: float
    entry_price: float
    target_price: float
    stop_loss: float
    expected_return: float
    max_loss: float
    reasons: List[str]
    error_message: Optional[str] = None 