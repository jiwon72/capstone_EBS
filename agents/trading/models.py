from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Union, Any
from datetime import datetime
from enum import Enum
from decimal import Decimal

class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"

class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"

class OrderStatus(str, Enum):
    PENDING = "PENDING"
    OPEN = "OPEN"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"

class TimeInForce(str, Enum):
    GTC = "GTC"  # Good Till Cancelled
    IOC = "IOC"  # Immediate Or Cancel
    FOK = "FOK"  # Fill Or Kill
    DAY = "DAY"  # Day Order

class PositionSide(str, Enum):
    LONG = "long"
    SHORT = "short"

class PositionStatus(str, Enum):
    OPEN = "open"
    CLOSED = "closed"

class Order(BaseModel):
    order_id: Optional[str] = None
    ticker: str
    quantity: Decimal
    side: OrderSide
    order_type: OrderType
    price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    timestamp: datetime = datetime.now()

class Position(BaseModel):
    position_id: str
    ticker: str
    quantity: Decimal
    entry_price: Decimal
    current_price: Decimal
    unrealized_pnl: Decimal = Decimal('0')
    realized_pnl: Decimal = Decimal('0')
    side: OrderSide
    status: str = "open"
    timestamp: datetime = datetime.now()

class RiskLimits(BaseModel):
    max_position_size: Decimal
    max_total_exposure: Decimal
    max_drawdown: Decimal
    var_limit: Decimal

class PositionInfo(BaseModel):
    position_id: str
    ticker: str
    quantity: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    side: str
    status: str
    timestamp: str

class RiskMetrics(BaseModel):
    var: float
    drawdown: float
    exposure: float
    leverage: float

class ExecutionResponse(BaseModel):
    status: OrderStatus
    message: Optional[str] = None
    order_id: Optional[str] = None
    ticker: Optional[str] = None
    quantity: Optional[float] = None
    price: Optional[float] = None
    execution_price: Optional[float] = None
    position_info: Optional[PositionInfo] = None
    timestamp: str = datetime.now().isoformat()

class PortfolioStatus(BaseModel):
    portfolio_id: str
    cash_balance: float
    total_value: float
    positions: List[PositionInfo]
    risk_metrics: Dict[str, float]
    timestamp: str = datetime.now().isoformat()

# Custom Exceptions
class OrderValidationError(Exception):
    pass

class RiskLimitExceededError(Exception):
    pass

class OrderExecutionError(Exception):
    pass

class PortfolioError(Exception):
    pass

class ExecutionRequest(BaseModel):
    ticker: str
    quantity: float
    price: Optional[float] = None
    order_type: str = "MARKET"
    side: str = "BUY"
    time_in_force: str = "GTC"
    stop_price: Optional[float] = None
    limit_price: Optional[float] = None
    metadata: Optional[Dict] = None

class PositionUpdateRequest(BaseModel):
    position_id: str
    action: str  # modify, close
    updates: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None

class PositionUpdateResponse(BaseModel):
    request_id: str
    timestamp: datetime = Field(default_factory=datetime.now)
    position: Optional[Position] = None
    status: str
    message: str
    metadata: Optional[Dict[str, Any]] = None

class OrderCancelRequest(BaseModel):
    order_id: str
    metadata: Optional[Dict[str, Any]] = None

class OrderCancelResponse(BaseModel):
    order_id: str
    status: OrderStatus
    timestamp: datetime = Field(default_factory=datetime.now)
    message: str

class OrderUpdate(BaseModel):
    ticker: str
    quantity: float
    price: float
    timestamp: datetime = Field(default_factory=datetime.now)
    order_id: Optional[str] = None
    position_id: Optional[str] = None
    metadata: Optional[Dict] = None

class OrderStatusInfo(BaseModel):
    order_id: str
    ticker: str
    quantity: float
    price: float
    status: str
    timestamp: datetime
    position_info: Dict

class PositionInfo(BaseModel):
    position_id: str
    ticker: str
    quantity: int
    average_price: float
    current_price: float
    unrealized_pnl: float
    timestamp: str 