from fastapi import FastAPI
import uvicorn
import os
from datetime import datetime
from .models import NewsAnalysisRequest, NewsAnalysisResponse, NewsToStrategyOutput
from .news_analyzer import NewsAnalyzer
import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)

from agents.utils.data_pipeline import DataPipeline
from agents.utils.logger import AgentLogger

app = FastAPI(title="News Analysis Agent")
news_analyzer = NewsAnalyzer()
pipeline = DataPipeline()
logger = AgentLogger("news_agent")

@app.post("/news/analyze")
async def analyze_news():
    try:
        logger.info("Starting news analysis")
        
        # 뉴스 분석 실행
        df_result, json_result = news_analyzer.run_sentiment_analysis()
        
        # Strategy Agent를 위한 출력 형식으로 변환
        strategy_input = NewsToStrategyOutput(
            timestamp=datetime.now(),
            stocks=json_result,
            market_conditions={
                "market_trend": "neutral",  # 실제 구현 필요
                "volatility_level": "medium",  # 실제 구현 필요
                "trading_volume": 0.0,  # 실제 구현 필요
                "sector_performance": {},  # 실제 구현 필요
                "major_events": []  # 실제 구현 필요
            },
            sector_sentiment={}  # 실제 구현 필요
        )
        
        # 파이프라인에 저장
        pipeline.save_agent_output("news", strategy_input.dict())
        logger.info("News analysis completed and saved to pipeline")
        
        return {"status": "success", "message": "News analysis completed"}
        
    except Exception as e:
        logger.error(f"Error in news analysis: {str(e)}")
        raise

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    port = int(os.getenv("NEWS_AGENT_PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port) 