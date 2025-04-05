from fastapi import FastAPI, HTTPException
import uvicorn
import os
from dotenv import load_dotenv
from .models import StrategyRequest, StrategyResponse
from .strategy_generator import StrategyGenerator

# 환경 변수 로드
load_dotenv()

app = FastAPI(title="Strategy Agent", description="Trading strategy formulation agent")
strategy_generator = StrategyGenerator()

@app.post("/strategy", response_model=StrategyResponse)
async def create_strategy(request: StrategyRequest):
    try:
        strategy = strategy_generator.generate_strategy(
            user_input=request.user_input,
            market_conditions=request.market_conditions,
            risk_tolerance=request.risk_tolerance,
            time_horizon=request.time_horizon
        )
        return strategy
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    port = int(os.getenv("STRATEGY_AGENT_PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port) 