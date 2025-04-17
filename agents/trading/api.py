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

app = FastAPI(
    title="Trading Executor API",
    description="""
    Trading Executor API는 주문 실행, 포지션 관리, 포트폴리오 상태 조회 등의 기능을 제공합니다.
    
    ## 주요 기능
    * 주문 실행 (시장가, 지정가, 스탑 주문)
    * 포지션 업데이트
    * 주문 취소
    * 포트폴리오 상태 조회
    * 리스크 관리
    
    ## 사용 방법
    각 엔드포인트의 자세한 사용 방법은 아래 API 문서를 참고하세요.
    """,
    version="1.0.0",
    contact={
        "name": "Trading System Team",
        "email": "trading@example.com"
    }
)

executor = TradingExecutor()

@app.post(
    "/execute", 
    response_model=Dict[str, Any],
    summary="주문 실행",
    description="""
    새로운 주문을 실행합니다. 시장가, 지정가, 스탑 주문을 지원합니다.
    
    - 시장가 주문: 현재 시장 가격으로 즉시 실행
    - 지정가 주문: 지정된 가격 이하(매수) 또는 이상(매도)일 때 실행
    - 스탑 주문: 지정된 가격에 도달하면 시장가 주문으로 전환
    
    리스크 한도를 초과하는 주문은 거부됩니다.
    """,
    response_description="실행된 주문의 상세 정보",
    responses={
        200: {
            "description": "주문 실행 성공",
            "content": {
                "application/json": {
                    "example": {
                        "order_id": "123e4567-e89b-12d3-a456-426614174000",
                        "status": "FILLED",
                        "execution_price": 150.0,
                        "filled_quantity": 100,
                        "timestamp": "2024-01-01T10:00:00"
                    }
                }
            }
        },
        400: {
            "description": "잘못된 요청 또는 리스크 한도 초과",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Maximum position size exceeded"
                    }
                }
            }
        },
        500: {
            "description": "서버 오류",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Internal server error"
                    }
                }
            }
        }
    }
)
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

@app.post(
    "/position/update",
    response_model=PositionUpdateResponse,
    summary="포지션 업데이트",
    description="""
    기존 포지션을 업데이트합니다. 수량 변경, 손절/이익실현 가격 조정 등을 지원합니다.
    """,
    response_description="업데이트된 포지션 정보",
    responses={
        200: {
            "description": "포지션 업데이트 성공",
            "content": {
                "application/json": {
                    "example": {
                        "position_id": "123e4567-e89b-12d3-a456-426614174000",
                        "ticker": "AAPL",
                        "quantity": 100,
                        "entry_price": 150.0,
                        "current_price": 155.0,
                        "unrealized_pnl": 500.0,
                        "status": "OPEN"
                    }
                }
            }
        }
    }
)
async def update_position(request: PositionUpdateRequest):
    """포지션 업데이트 엔드포인트"""
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

@app.post(
    "/order/cancel",
    response_model=OrderCancelResponse,
    summary="주문 취소",
    description="대기 중인 주문을 취소합니다.",
    response_description="취소된 주문 정보",
    responses={
        200: {
            "description": "주문 취소 성공",
            "content": {
                "application/json": {
                    "example": {
                        "order_id": "123e4567-e89b-12d3-a456-426614174000",
                        "status": "CANCELLED",
                        "timestamp": "2024-01-01T10:00:00",
                        "message": "Order cancelled successfully"
                    }
                }
            }
        }
    }
)
async def cancel_order(request: OrderCancelRequest):
    """주문 취소 엔드포인트"""
    try:
        result = await executor.cancel_order(request.order_id)
        return OrderCancelResponse(
            order_id=request.order_id,
            status=OrderStatus.CANCELLED,
            message="Order cancelled successfully",
            timestamp=datetime.now()
        )
    except ValueError as e:
        return OrderCancelResponse(
            order_id=request.order_id,
            status=OrderStatus.REJECTED,
            message=str(e),
            timestamp=datetime.now()
        )

@app.get(
    "/portfolio/status",
    response_model=PortfolioStatus,
    summary="포트폴리오 상태 조회",
    description="""
    현재 포트폴리오의 상태를 조회합니다. 
    보유 포지션, 미체결 주문, 손익 현황, 리스크 메트릭스 등의 정보를 제공합니다.
    """,
    response_description="포트폴리오 상태 정보",
    responses={
        200: {
            "description": "포트폴리오 상태 조회 성공",
            "content": {
                "application/json": {
                    "example": {
                        "portfolio_id": "123e4567-e89b-12d3-a456-426614174000",
                        "total_value": 100000.0,
                        "cash_balance": 50000.0,
                        "positions": [
                            {
                                "position_id": "123e4567-e89b-12d3-a456-426614174001",
                                "ticker": "AAPL",
                                "quantity": 100,
                                "entry_price": 150.0,
                                "current_price": 155.0,
                                "unrealized_pnl": 500.0,
                                "status": "OPEN"
                            }
                        ],
                        "risk_metrics": {
                            "var": 1000.0,
                            "max_drawdown": 2000.0,
                            "sharpe_ratio": 1.5
                        }
                    }
                }
            }
        }
    }
)
async def get_portfolio_status():
    """포트폴리오 상태 조회 엔드포인트"""
    try:
        result = await executor.get_portfolio_status()
        return jsonable_encoder(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get(
    "/health",
    summary="서비스 상태 확인",
    description="Trading Executor 서비스의 상태를 확인합니다.",
    response_description="서비스 상태 정보",
    responses={
        200: {
            "description": "서비스 정상 작동 중",
            "content": {
                "application/json": {
                    "example": {
                        "status": "healthy",
                        "timestamp": "2024-01-01T10:00:00",
                        "version": "1.0.0"
                    }
                }
            }
        }
    }
)
async def health_check():
    """서비스 상태 확인"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    } 