from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import httpx
import json
from typing import Dict, Any
import asyncio
from datetime import datetime

app = FastAPI(title="Trading System Dashboard")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 에이전트 엔드포인트 설정
AGENT_ENDPOINTS = {
    "news": "http://localhost:8003",
    "technical": "http://localhost:8004",
    "xai": "http://localhost:8006",
    "trading": "http://localhost:8005"
}

# WebSocket 연결을 저장할 딕셔너리
websocket_connections: Dict[int, WebSocket] = {}

async def fetch_agent_data(url: str, method: str = "GET", data: Dict = None) -> Dict:
    """에이전트 데이터 조회"""
    try:
        async with httpx.AsyncClient() as client:
            if method == "GET":
                response = await client.get(url)
            else:  # POST
                response = await client.post(url, json=data)
            return response.json()
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/dashboard")
async def get_dashboard_data():
    """대시보드 데이터 조회"""
    try:
        # 포트폴리오 상태 조회
        portfolio_data = await fetch_agent_data(f"{AGENT_ENDPOINTS['trading']}/portfolio/status")
        
        # 뉴스 분석 결과 조회
        news_data = await fetch_agent_data(f"{AGENT_ENDPOINTS['news']}/analyze")
        
        # 기술적 분석 결과 조회
        technical_data = await fetch_agent_data(
            f"{AGENT_ENDPOINTS['technical']}/analyze",
            method="POST",
            data={
                "market_data": [],  # 실제 시장 데이터로 대체 필요
                "indicators": ["sma", "ema", "rsi", "macd"],
                "parameters": {
                    "sma_window": 20,
                    "ema_window": 20,
                    "rsi_window": 14
                }
            }
        )
        
        # XAI 분석 결과 조회
        xai_data = await fetch_agent_data(
            f"{AGENT_ENDPOINTS['xai']}/explain",
            method="POST",
            data={
                "model_id": "trading_model_001",
                "features": {
                    "price_momentum": 0.5,
                    "volume_trend": 0.3,
                    "sentiment_score": 0.7
                },
                "explanation_method": "shap",
                "num_samples": 1000
            }
        )
        
        return {
            "portfolio": portfolio_data,
            "news": news_data,
            "technical": technical_data,
            "xai": xai_data,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {"error": str(e)}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """실시간 업데이트를 위한 WebSocket 엔드포인트"""
    await websocket.accept()
    client_id = id(websocket)
    websocket_connections[client_id] = websocket
    
    try:
        while True:
            # 클라이언트로부터 메시지 수신
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # 요청된 데이터 처리
            if message.get("type") == "subscribe":
                # 구독 로직 구현
                pass
            
            # 실시간 업데이트 전송
            await websocket.send_json({
                "type": "update",
                "data": await get_dashboard_data()
            })
            
            await asyncio.sleep(1)  # 1초 대기
            
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        # 연결 종료 시 정리
        if client_id in websocket_connections:
            del websocket_connections[client_id]

@app.on_event("startup")
async def startup_event():
    """애플리케이션 시작 시 실행되는 이벤트"""
    print("Dashboard backend started")

@app.on_event("shutdown")
async def shutdown_event():
    """애플리케이션 종료 시 실행되는 이벤트"""
    print("Dashboard backend shutting down") 