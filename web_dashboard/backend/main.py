import httpx
from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
import glob
import json
from typing import Dict, Any, List
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

# 정적 파일 마운트
app.mount("/dashboard1", StaticFiles(directory="../dashboard1", html=True), name="dashboard1")
app.mount("/dashboard_debate", StaticFiles(directory="../dashboard_debate", html=True), name="dashboard_debate")
app.mount("/system", StaticFiles(directory="../../system"), name="system")  # 정적 파일 경로 마운트

# 에이전트 엔드포인트 설정
AGENT_ENDPOINTS = {
    "news": "http://localhost:8003",
    "technical": "http://localhost:8004", 
    "xai": "http://localhost:8005",
    "trading": "http://localhost:8006"
}

# WebSocket 연결을 저장할 딕셔너리
websocket_connections: Dict[int, WebSocket] = {}

# 데이터 캐시
forecast_cache = None
dashboard_cache = None

def load_json_file(file_path: str) -> Dict[str, Any]:
    """JSON 파일을 로드하는 헬퍼 함수"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail=f"Invalid JSON format: {file_path}")

def get_forecast_data() -> Dict[str, Any]:
    """forecast.json 데이터를 캐시와 함께 가져오기"""
    global forecast_cache
    if forecast_cache is None:
        forecast_path = os.path.abspath("../../system/forecasts/forecast.json")
        if not os.path.exists(forecast_path):
            # 다른 경로 시도
            forecast_path = os.path.abspath("../../system/forecast.json")
        forecast_cache = load_json_file(forecast_path)
    return forecast_cache

def get_dashboard_data() -> Dict[str, Any]:
    """dashboard.json 데이터를 캐시와 함께 가져오기"""
    global dashboard_cache
    if dashboard_cache is None:
        dashboard_path = os.path.abspath("../../system/dashboard.json")
        dashboard_cache = load_json_file(dashboard_path)
    return dashboard_cache

# 루트 경로
@app.get("/")
def read_root():
    return FileResponse("../dashboard1/index.html")

# JSON 파일 서빙 (기존)
@app.get("/system/dashboard.json")
def get_dashboard_json():
    json_path = os.path.abspath("../../system/dashboard.json")
    if os.path.exists(json_path):
        return FileResponse(json_path)
    raise HTTPException(status_code=404, detail="Dashboard data not found")

@app.get("/system/forecast.json")
def get_forecast_json():
    json_path = os.path.abspath("../../system/forecast.json")
    if os.path.exists(json_path):
        return FileResponse(json_path)
    raise HTTPException(status_code=404, detail="Forecast data not found")

@app.get("/system/debate_logs/latest")
def get_latest_debate_log():
    try:
        pattern = os.path.abspath("../../system/debate_logs/debate_log_*.json")
        files = glob.glob(pattern)
        if not files:
            raise HTTPException(status_code=404, detail="No debate logs found")
        
        latest_file = max(files)
        return FileResponse(latest_file)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 새로운 API 엔드포인트들

@app.get("/api/stocks")
def get_stock_list():
    """종목 리스트 반환"""
    try:
        forecast_data = get_forecast_data()
        dashboard_data = get_dashboard_data()
        
        stocks = []
        for stock_name, stock_info in forecast_data.items():
            # dashboard에 해당 종목 데이터가 있는지 확인
            has_dashboard_data = any(
                item.get('종목') == stock_name for item in dashboard_data.get('strategy', [])
            )
            
            stocks.append({
                "name": stock_name,
                "code": stock_info.get("stock_code"),
                "has_full_data": has_dashboard_data
            })
        
        return {
            "stocks": stocks,
            "total_count": len(stocks),
            "with_full_data": len([s for s in stocks if s["has_full_data"]])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stocks/{stock_name}")
def get_stock_data(stock_name: str):
    """특정 종목의 모든 데이터 반환"""
    try:
        forecast_data = get_forecast_data()
        dashboard_data = get_dashboard_data()
        
        # forecast 데이터 확인
        if stock_name not in forecast_data:
            raise HTTPException(status_code=404, detail=f"Stock {stock_name} not found in forecast data")
        
        stock_forecast = forecast_data[stock_name]
        
        # dashboard 데이터 수집
        strategy = next((item for item in dashboard_data.get('strategy', []) if item.get('종목') == stock_name), None)
        risk = next((item for item in dashboard_data.get('risk', []) if item.get('종목') == stock_name), None)
        tech = next((item for item in dashboard_data.get('tech', []) if item.get('종목') == stock_name), None)
        news = dashboard_data.get('news', {}).get(stock_name)
        portfolio = next((item for item in dashboard_data.get('portfolio', []) if item.get('종목명') == stock_name), None)
        
        return {
            "stock_name": stock_name,
            "stock_code": stock_forecast.get("stock_code"),
            "forecast": {
                "dates": stock_forecast.get("dates", []),
                "prices": stock_forecast.get("prices", [])
            },
            "strategy": strategy,
            "risk": risk,
            "technical": tech,
            "news": news,
            "portfolio": portfolio,
            "has_complete_data": all([strategy, risk, tech, news, portfolio])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/portfolio/summary")
def get_portfolio_summary():
    """포트폴리오 전체 요약 정보"""
    try:
        dashboard_data = get_dashboard_data()
        portfolio_data = dashboard_data.get('portfolio', [])
        
        total_investment = 0
        total_current_value = 0
        stocks_data = []
        
        for item in portfolio_data:
            if item.get('종목명') == '현금':
                total_investment += item.get('투자금', 0)
                total_current_value += item.get('투자금', 0)
            else:
                investment = item.get('투자금', 0)
                shares = item.get('구매개수', 0)
                current_price = item.get('현재가', 0)
                current_value = shares * current_price
                
                total_investment += investment
                total_current_value += current_value
                
                stocks_data.append({
                    "name": item.get('종목명'),
                    "shares": shares,
                    "investment": investment,
                    "current_price": current_price,
                    "current_value": current_value,
                    "profit_loss": current_value - investment,
                    "profit_rate": (current_value - investment) / investment * 100 if investment > 0 else 0
                })
        
        total_profit = total_current_value - total_investment
        total_profit_rate = (total_profit / total_investment * 100) if total_investment > 0 else 0
        
        return {
            "total_investment": total_investment,
            "total_current_value": total_current_value,
            "total_profit": total_profit,
            "total_profit_rate": total_profit_rate,
            "stocks": stocks_data,
            "summary": dashboard_data.get('summary', [])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/cache/refresh")
def refresh_cache():
    """캐시 새로고침"""
    global forecast_cache, dashboard_cache
    forecast_cache = None
    dashboard_cache = None
    
    try:
        # 데이터 다시 로드하여 캐시 갱신
        get_forecast_data()
        get_dashboard_data()
        return {"message": "Cache refreshed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to refresh cache: {str(e)}")

# 나머지 기존 코드들...
async def fetch_agent_data(client: httpx.AsyncClient, agent: str, endpoint: str) -> Dict[str, Any]:
    """에이전트로부터 데이터를 가져오는 함수"""
    try:
        response = await client.get(f"{AGENT_ENDPOINTS[agent]}/{endpoint}")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/dashboard")
async def get_dashboard_data_legacy():
    """대시보드 데이터를 가져오는 엔드포인트 (기존 함수명 변경)"""
    async with httpx.AsyncClient() as client:
        tasks = [
            fetch_agent_data(client, "trading", "portfolio"),
            fetch_agent_data(client, "news", "latest"),
            fetch_agent_data(client, "technical", "signals"),
            fetch_agent_data(client, "xai", "summary")
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
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "subscribe":
                pass
            
            await websocket.send_json({
                "type": "update",
                "data": await get_dashboard_data_legacy()
            })
            
            await asyncio.sleep(1)
            
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        if client_id in websocket_connections:
            del websocket_connections[client_id]

@app.on_event("startup")
async def startup_event():
    """애플리케이션 시작 시 실행되는 이벤트"""
    print("Dashboard backend started")
    
    # 시작할 때 데이터 검증
    try:
        forecast_data = get_forecast_data()
        dashboard_data = get_dashboard_data()
        
        print(f"✅ Forecast data loaded: {len(forecast_data)} stocks")
        print(f"✅ Dashboard data loaded: strategy({len(dashboard_data.get('strategy', []))}), "
              f"risk({len(dashboard_data.get('risk', []))}), "
              f"tech({len(dashboard_data.get('tech', []))})")
        
        # 데이터 일관성 체크
        forecast_stocks = set(forecast_data.keys())
        strategy_stocks = set(item['종목'] for item in dashboard_data.get('strategy', []))
        missing_in_dashboard = forecast_stocks - strategy_stocks
        
        if missing_in_dashboard:
            print(f"⚠️ Missing in dashboard: {missing_in_dashboard}")
        else:
            print("✅ All forecast stocks have dashboard data")
            
    except Exception as e:
        print(f"❌ Startup data validation failed: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """애플리케이션 종료 시 실행되는 이벤트"""
    print("Dashboard backend shutting down")