from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from datetime import datetime
from .models import (
    ExecutionRequest, ExecutionResponse,
    PositionUpdateRequest, PositionUpdateResponse,
    OrderCancelRequest, OrderCancelResponse,
    PortfolioStatus, OrderUpdate, OrderStatus,
    Position, Order, RiskMetrics
)
from .trading_executor import TradingExecutor
import uuid
from typing import Dict, List, Any
from decimal import Decimal

app = FastAPI(title="Trading Executor API")
executor = TradingExecutor()

@app.post("/execute", response_model=Dict[str, Any])
async def execute_order(request: ExecutionRequest):
    """주문 실행 엔드포인트"""
    try:
        order = Order(
            order_id=str(uuid.uuid4()),
            ticker=request.ticker,
            quantity=Decimal(str(request.quantity)),
            side=request.side,
            order_type=request.order_type,
            price=Decimal(str(request.price)) if request.price is not None else None,
            stop_price=Decimal(str(request.stop_price)) if request.stop_price is not None else None,
            timestamp=datetime.now()
        )
        
        # 리스크 한도 검증
        current_price = executor._get_current_price(order.ticker)
        valid_risk, risk_message = executor._validate_risk_limits(order, current_price)
        if not valid_risk:
            raise HTTPException(
                status_code=400,
                detail=risk_message
            )
            
        # 주문 실행
        result = await executor.execute_order(order)
        
        # 주문이 체결된 경우
        if result.status == OrderStatus.FILLED:
            position_info = result.position_info
            if position_info is None:
                position_info = {
                    "position_id": str(uuid.uuid4()),
                    "ticker": order.ticker,
                    "quantity": float(order.quantity),
                    "entry_price": float(result.execution_price),
                    "current_price": float(result.execution_price),
                    "unrealized_pnl": 0.0,
                    "realized_pnl": 0.0,
                    "side": order.side,
                    "status": "open",
                    "timestamp": datetime.now().isoformat()
                }
            
            return {
                "status": result.status.value,
                "message": result.message,
                "order_id": result.order_id,
                "ticker": order.ticker,
                "quantity": float(order.quantity),
                "price": float(order.price) if order.price else None,
                "execution_price": float(result.execution_price),
                "position_info": position_info,
                "timestamp": datetime.now().isoformat()
            }
        else:
            # 주문이 거부된 경우
            if result.message and ("Risk limits exceeded" in result.message or "Position size" in result.message):
                raise HTTPException(
                    status_code=400,
                    detail=result.message
                )
            
            return {
                "status": result.status.value,
                "message": result.message,
                "order_id": result.order_id,
                "ticker": order.ticker,
                "quantity": float(order.quantity),
                "price": float(order.price) if order.price else None,
                "execution_price": None,
                "position_info": None,
                "timestamp": datetime.now().isoformat()
            }
            
    except HTTPException:
        raise
    except Exception as e:
        return {
            "status": OrderStatus.REJECTED.value,
            "message": str(e),
            "order_id": None,
            "ticker": request.ticker,
            "quantity": float(request.quantity),
            "price": float(request.price) if request.price else None,
            "execution_price": None,
            "position_info": None,
            "timestamp": datetime.now().isoformat()
        }

@app.post("/position/update", response_model=PositionUpdateResponse)
async def update_position(request: PositionUpdateRequest):
    """포지션을 업데이트합니다."""
    try:
        result = await executor.update_position(request)
        if not isinstance(result, PositionUpdateResponse):
            result = PositionUpdateResponse(
                request_id=str(uuid.uuid4()),
                status="success",
                message="Position updated successfully",
                position=Position(
                    position_id=request.position_id,
                    ticker=request.updates.get("ticker", ""),
                    quantity=Decimal(str(request.updates.get("quantity", 0))),
                    entry_price=Decimal(str(request.updates.get("price", 0))),
                    current_price=Decimal(str(request.updates.get("price", 0))),
                    side=request.updates.get("side", "BUY"),
                    status="open",
                    timestamp=datetime.now()
                )
            )
        return jsonable_encoder(result)
    except Exception as e:
        return PositionUpdateResponse(
            request_id=str(uuid.uuid4()),
            status="error",
            message=str(e),
            position=None
        )

@app.post("/order/cancel/{order_id}", response_model=Dict[str, Any])
async def cancel_order(order_id: str):
    """주문 취소 엔드포인트"""
    try:
        result = await executor.cancel_order(order_id)
        return {
            "status": "success",
            "message": f"Order {order_id} cancelled successfully",
            "order_id": order_id,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "order_id": order_id,
            "timestamp": datetime.now().isoformat()
        }

@app.get("/portfolio", response_model=Dict[str, Any])
async def get_portfolio_status():
    """포트폴리오 상태 조회"""
    try:
        status = await executor.get_portfolio_status()
        return jsonable_encoder(status)
    except Exception as e:
        print(f"Error in portfolio endpoint: {str(e)}")  # 디버깅을 위한 로그 추가
        return {
            "error_message": str(e),
            "portfolio_id": str(uuid.uuid4()),
            "total_value": 0.0,
            "cash_balance": 0.0,
            "positions": [],
            "risk_metrics": {},
            "timestamp": datetime.now().isoformat()
        }

@app.get("/health")
async def health_check():
    """서비스 상태 확인"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    } 