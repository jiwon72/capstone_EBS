from fastapi import FastAPI, HTTPException
import uvicorn
import os
from dotenv import load_dotenv
from .models import NewsAnalysisRequest, NewsAnalysisResponse
from .news_analyzer import NewsAnalyzer

# 환경 변수 로드
load_dotenv()

app = FastAPI(title="News Analysis Agent", description="Market news analysis agent")
news_analyzer = NewsAnalyzer()

@app.post("/news", response_model=NewsAnalysisResponse)
async def analyze_news(request: NewsAnalysisRequest):
    try:
        news_analysis = news_analyzer.analyze_news(request)
        return news_analysis
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    port = int(os.getenv("NEWS_AGENT_PORT", 8003))
    uvicorn.run(app, host="0.0.0.0", port=port) 