from decimal import Decimal
from fastapi import APIRouter, HTTPException
from typing import Dict, Any

from .models import (
    ExecutionRequest, ExecutionResponse,
    PositionUpdateRequest, PositionUpdateResponse,
    OrderCancelRequest, OrderCancelResponse,
    PortfolioStatus, Order, OrderSide, OrderType
)
from .trading_executor import TradingExecutor

router = APIRouter()
executor = TradingExecutor()

@router.post("/execute", response_model=ExecutionResponse)
async def execute_order(request: ExecutionRequest):
    """주문 실행 엔드포인트"""
    try:
        # 주문 객체 생성
        order = Order(
            ticker=request.ticker,
            quantity=Decimal(str(request.quantity)),
            side=OrderSide(request.side),
            order_type=OrderType(request.order_type),
            price=Decimal(str(request.price)) if request.price is not None else None,
            stop_price=Decimal(str(request.stop_price)) if request.stop_price is not None else None,
            time_in_force=request.time_in_force if hasattr(request, 'time_in_force') else None
        )
        
        # 주문 실행
        response = await executor.execute_order(order)
        return response
        
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/position/update", response_model=PositionUpdateResponse)
async def update_position(request: PositionUpdateRequest):
    """포지션 업데이트 엔드포인트"""
    try:
        response = await executor.update_position(request)
        if not response.success:
            error_message = str(response.message)  # 문자열로 변환
            if "not found" in error_message.lower():
                raise HTTPException(status_code=404, detail=error_message)
            else:
                raise HTTPException(status_code=400, detail=error_message)
        return response
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/order/cancel", response_model=OrderCancelResponse)
async def cancel_order(request: OrderCancelRequest):
    """주문 취소 엔드포인트"""
    try:
        response = await executor.cancel_order(request.order_id)
        if response["status"] == "ERROR":
            raise HTTPException(status_code=400, detail=response["message"])
        return response
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/portfolio", response_model=PortfolioStatus)
async def get_portfolio_status():
    """포트폴리오 상태 조회 엔드포인트"""
    try:
        return await executor.get_portfolio_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 