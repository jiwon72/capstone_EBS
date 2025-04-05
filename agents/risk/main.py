from fastapi import FastAPI, HTTPException
import uvicorn
import os
from dotenv import load_dotenv
from .models import RiskAssessmentRequest, RiskAssessmentResponse
from .risk_analyzer import RiskAnalyzer

# 환경 변수 로드
load_dotenv()

app = FastAPI(title="Risk Management Agent", description="Trading risk management agent")
risk_analyzer = RiskAnalyzer()

@app.post("/risk", response_model=RiskAssessmentResponse)
async def assess_risk(request: RiskAssessmentRequest):
    try:
        risk_assessment = risk_analyzer.assess_risk(
            positions=request.positions,
            portfolio_value=request.portfolio_value,
            risk_tolerance=request.risk_tolerance,
            market_conditions=request.market_conditions
        )
        return risk_assessment
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    port = int(os.getenv("RISK_AGENT_PORT", 8002))
    uvicorn.run(app, host="0.0.0.0", port=port) 