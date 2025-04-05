from datetime import datetime
import uuid
from typing import Dict, List, Optional
from .models import ExecutionRequest, OrderUpdate, PortfolioStatus, Position, RiskMetrics

class TradingExecutor:
    def __init__(self):
        self.positions = {}  # ticker -> Position
        self.orders = {}     # order_id -> Order
        self.portfolio = {
            "total_value": 1000000.0,
            "cash_balance": 500000.0,
            "positions": [],
            "risk_metrics": {
                "total_exposure": 0.5,
                "var_95": 10000.0,
                "sharpe_ratio": 1.5
            }
        }

    def generate_order_id(self) -> str:
        """고유한 주문 ID 생성"""
        return str(uuid.uuid4())

    def get_position_info(self, ticker: str) -> Dict:
        """특정 종목의 포지션 정보 조회"""
        if ticker in self.positions:
            position = self.positions[ticker]
            return {
                "position_id": position.position_id,
                "ticker": position.ticker,
                "quantity": position.quantity,
                "entry_price": position.entry_price,
                "current_price": position.current_price,
                "unrealized_pnl": position.unrealized_pnl
            }
        else:
            return {
                "position_id": str(uuid.uuid4()),
                "ticker": ticker,
                "quantity": 0,
                "entry_price": 0.0,
                "current_price": 0.0,
                "unrealized_pnl": 0.0
            }

    def get_portfolio_status(self) -> PortfolioStatus:
        """포트폴리오 상태 조회"""
        positions = []
        for ticker, position in self.positions.items():
            positions.append({
                "position_id": position.position_id,
                "ticker": position.ticker,
                "quantity": position.quantity,
                "entry_price": position.entry_price,
                "current_price": position.current_price,
                "unrealized_pnl": position.unrealized_pnl
            })

        return PortfolioStatus(
            portfolio_id="test_portfolio",
            total_value=self.portfolio["total_value"],
            cash_balance=self.portfolio["cash_balance"],
            positions=positions,
            risk_metrics=RiskMetrics(
                total_exposure=self.portfolio["risk_metrics"]["total_exposure"],
                var_95=self.portfolio["risk_metrics"]["var_95"],
                sharpe_ratio=self.portfolio["risk_metrics"]["sharpe_ratio"]
            )
        )

    def update_position(self, update: OrderUpdate) -> None:
        """포지션 업데이트"""
        if update.ticker not in self.positions:
            self.positions[update.ticker] = Position(
                position_id=str(uuid.uuid4()),
                ticker=update.ticker,
                quantity=update.quantity,
                entry_price=update.price,
                current_price=update.price,
                unrealized_pnl=0.0
            )
        else:
            position = self.positions[update.ticker]
            position.quantity += update.quantity
            position.entry_price = update.price
            position.current_price = update.price
            position.unrealized_pnl = (position.current_price - position.entry_price) * position.quantity

    def cancel_order(self, order_id: str) -> None:
        """주문 취소"""
        if order_id in self.orders:
            del self.orders[order_id]
        else:
            raise ValueError(f"Order {order_id} not found") 