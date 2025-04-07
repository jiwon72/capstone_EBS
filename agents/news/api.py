from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from datetime import datetime
from .models import NewsAnalysisRequest, NewsAnalysisResponse
from .news_analyzer import NewsAnalyzer
import os

app = FastAPI(title="News Analysis Agent API")
# 테스트 환경 여부 확인
is_test = os.getenv("TESTING", "false").lower() == "true"
analyzer = NewsAnalyzer(is_test=is_test)

@app.post("/news", response_model=NewsAnalysisResponse)
async def analyze_news(request: NewsAnalysisRequest):
    """뉴스 분석을 수행하고 결과를 반환합니다."""
    try:
        result = await analyzer.analyze(request)
        return jsonable_encoder(result)  # datetime 객체를 JSON으로 직렬화
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 