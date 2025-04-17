from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from datetime import datetime
from models import NewsAnalysisRequest, NewsAnalysisResponse
from news_analyzer import NewsAnalyzer
import os

app = FastAPI(title="News Analysis Agent API")

# CORS 설정 업데이트
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 개발 환경에서는 모든 origin 허용
    allow_credentials=True,
    allow_methods=["*"],  # 모든 HTTP 메서드 허용
    allow_headers=["*"],  # 모든 헤더 허용
    expose_headers=["*"]  # 모든 헤더 노출
)

# 테스트 환경 여부 확인
is_test = os.getenv("TESTING", "false").lower() == "true"
analyzer = NewsAnalyzer(is_test=is_test)

@app.post("/news", response_model=NewsAnalysisResponse)
async def analyze_news(request: NewsAnalysisRequest):
    """뉴스 분석을 수행하고 결과를 반환합니다."""
    try:
        result = await analyzer.analyze(request)
        return jsonable_encoder(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/news")
async def get_news_analysis():
    """최근 뉴스 분석 결과를 반환합니다."""
    try:
        request = NewsAnalysisRequest(
            time_range="1d",
            tickers=["005930.KS", "000660.KS"],  # 기본값으로 삼성전자와 SK하이닉스 설정
            min_relevance_score=0.1
        )
        
        result = await analyzer.analyze(request)
        return jsonable_encoder(result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 
