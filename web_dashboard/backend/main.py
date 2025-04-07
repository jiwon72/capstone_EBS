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
    "xai": "http://localhost:8005",
    "trading": "http://localhost:8006"
}

# WebSocket 연결을 저장할 딕셔너리
websocket_connections: Dict[int, WebSocket] = {}

async def fetch_agent_data(client: httpx.AsyncClient, agent: str, endpoint: str) -> Dict[str, Any]:
    """에이전트로부터 데이터를 가져오는 함수"""
    try:
        response = await client.get(f"{AGENT_ENDPOINTS[agent]}/{endpoint}")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/dashboard")
async def get_dashboard_data():
    """대시보드 데이터를 가져오는 엔드포인트"""
    async with httpx.AsyncClient() as client:
        # 각 에이전트로부터 데이터 수집
        tasks = [
            fetch_agent_data(client, "trading", "portfolio"),  # 포트폴리오 상태
            fetch_agent_data(client, "news", "latest"),       # 최신 뉴스
            fetch_agent_data(client, "technical", "signals"), # 기술적 분석 신호
            fetch_agent_data(client, "xai", "summary")       # XAI 요약
        ]
        results = await asyncio.gather(*tasks)
        
        return {
            "portfolio": results[0],
            "news": results[1],
            "technical": results[2],
            "xai": results[3],
            "timestamp": datetime.now().isoformat()
        }

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