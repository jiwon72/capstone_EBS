from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from enum import Enum
from datetime import datetime

class TimeFrame(str, Enum):
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"
    W1 = "1w"
    MN1 = "1M"

class SignalStrength(str, Enum):
    STRONG = "STRONG"
    MODERATE = "MODERATE"
    WEAK = "WEAK"

class SignalType(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    NEUTRAL = "NEUTRAL"

class TrendDirection(str, Enum):
    UP = "up"
    DOWN = "down"
    SIDEWAYS = "sideways"

class Signal(BaseModel):
    signal: SignalType
    strength: SignalStrength

class MarketData(BaseModel):
    ticker: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    adjusted_close: Optional[float] = None

class MovingAverages(BaseModel):
    sma_20: float
    sma_50: float
    sma_200: float
    ema_12: float
    ema_26: float
    trend_direction: TrendDirection
    golden_cross: Optional[bool] = None
    death_cross: Optional[bool] = None

class Oscillators(BaseModel):
    rsi_14: float
    stoch_k: float
    stoch_d: float
    macd_line: float
    macd_signal: float
    macd_histogram: float
    cci_20: float
    atr_14: float
    bollinger_upper: float
    bollinger_middle: float
    bollinger_lower: float

class VolumeIndicators(BaseModel):
    obv: float
    volume_sma_20: float
    volume_trend: TrendDirection
    volume_price_trend: float
    money_flow_index: float

class SupportResistance(BaseModel):
    support_levels: List[float]
    resistance_levels: List[float]
    pivot_point: float
    r1_level: float
    r2_level: float
    s1_level: float
    s2_level: float

class PatternSignal(BaseModel):
    pattern_name: str
    confidence: float = Field(..., ge=0, le=1)
    signal: SignalType
    price_target: Optional[float] = None
    stop_loss: Optional[float] = None

class TechnicalIndicators(BaseModel):
    moving_averages: MovingAverages
    oscillators: Oscillators
    volume_indicators: VolumeIndicators
    support_resistance: SupportResistance
    patterns: List[PatternSignal]
    timestamp: datetime = Field(default_factory=datetime.now)

class TradingSignal(BaseModel):
    ticker: str
    timeframe: TimeFrame
    signal: SignalType
    confidence: float = Field(..., ge=0, le=1)
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_reward_ratio: float
    indicators_used: List[str]
    reasoning: str

class TechnicalAnalysisRequest(BaseModel):
    market_data: List[MarketData]
    indicators: List[str]
    parameters: Optional[Dict] = None

class TechnicalAnalysisResponse(BaseModel):
    ticker: str = Field(default="UNKNOWN")
    timestamp: datetime = Field(default_factory=datetime.now)
    indicators: List[str] = Field(default_factory=list)
    signals: Dict[str, Dict[str, str]] = Field(default_factory=dict)
    moving_averages: Dict[str, float] = Field(default_factory=dict)
    oscillators: Dict[str, float] = Field(default_factory=dict)
    volume_indicators: Dict[str, float] = Field(default_factory=dict)
    support_resistance: Dict[str, float] = Field(default_factory=dict)
    error_message: Optional[str] = None

class MovingAverageRequest(BaseModel):
    market_data: List[MarketData]
    window_sizes: List[int]
    ma_types: List[str]

class MovingAverageResponse(BaseModel):
    sma_values: Dict[str, float]
    ema_values: Dict[str, float]
    trend_direction: TrendDirection
    error_message: Optional[str] = None

class SupportResistanceRequest(BaseModel):
    market_data: List[MarketData]
    lookback_period: int
    min_touches: int = 2

class SupportResistanceResponse(BaseModel):
    support_levels: List[float]
    resistance_levels: List[float]
    pivot_points: Dict[str, float]
    timestamp: datetime = Field(default_factory=datetime.now)
    error_message: Optional[str] = None

class PatternRequest(BaseModel):
    market_data: List[MarketData]
    pattern_types: List[str]
    min_confidence: float = 0.7

class PatternResponse(BaseModel):
    patterns: Dict[str, PatternSignal]
    timestamp: datetime = Field(default_factory=datetime.now)
    error_message: Optional[str] = None

class SignalRequest(BaseModel):
    market_data: List[MarketData]
    signal_types: List[str]
    parameters: Optional[Dict] = None

class SignalResponse(BaseModel):
    signals: Dict[str, Signal]
    timestamp: datetime = Field(default_factory=datetime.now)
    error_message: Optional[str] = None

class Pattern(BaseModel):
    type: str
    confidence: float
    timestamp: str

class PatternRequest(BaseModel):
    market_data: List[MarketData]
    pattern_types: List[str]
    min_confidence: float

class PatternResponse(BaseModel):
    patterns: Dict[str, Pattern]
    timestamp: str
    error_message: Optional[str] = None 