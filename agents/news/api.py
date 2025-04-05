from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from datetime import datetime
from .models import NewsAnalysisRequest, NewsAnalysisResponse
from .news_analyzer import NewsAnalyzer
import os

app = FastAPI(
    title="News Analysis Agent API",
    description="""
    News Analysis Agent API는 시장 관련 뉴스를 분석하여 투자 의사결정에 도움이 되는 인사이트를 제공합니다.
    
    ## 주요 기능
    * 뉴스 감성 분석
    * 뉴스 관련성 평가
    * 시장 영향도 분석
    * 키워드 추출
    * 뉴스 요약
    
    ## 분석 방법
    * 자연어 처리(NLP) 기반 감성 분석
    * 머신러닝 기반 관련성 평가
    * 과거 데이터 기반 영향도 분석
    * 통계적 키워드 추출
    """,
    version="1.0.0",
    contact={
        "name": "Trading System Team",
        "email": "trading@example.com"
    }
)

# 테스트 환경 여부 확인
is_test = os.getenv("TESTING", "false").lower() == "true"
analyzer = NewsAnalyzer(is_test=is_test)

@app.post(
    "/news",
    response_model=NewsAnalysisResponse,
    summary="뉴스 분석",
    description="""
    주어진 뉴스 데이터를 분석하여 투자 관련 인사이트를 제공합니다.
    
    분석 항목:
    - 뉴스 감성 (긍정/부정/중립)
    - 시장 관련성
    - 예상 시장 영향도
    - 주요 키워드
    - 뉴스 요약
    """,
    response_description="뉴스 분석 결과",
    responses={
        200: {
            "description": "분석 성공",
            "content": {
                "application/json": {
                    "example": {
                        "sentiment": {
                            "score": 0.75,
                            "label": "POSITIVE",
                            "confidence": 0.85
                        },
                        "relevance": {
                            "score": 0.9,
                            "topics": ["earnings", "growth", "market_share"]
                        },
                        "market_impact": {
                            "direction": "POSITIVE",
                            "magnitude": "HIGH",
                            "duration": "SHORT_TERM",
                            "confidence": 0.8
                        },
                        "keywords": [
                            {
                                "text": "revenue growth",
                                "importance": 0.9,
                                "sentiment": "positive"
                            },
                            {
                                "text": "market expansion",
                                "importance": 0.85,
                                "sentiment": "positive"
                            }
                        ],
                        "summary": "Company reported strong Q4 results with significant revenue growth and market share expansion.",
                        "timestamp": "2024-01-01T10:00:00"
                    }
                }
            }
        },
        400: {
            "description": "잘못된 요청",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Invalid news source specified"
                    }
                }
            }
        },
        500: {
            "description": "서버 오류",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Internal server error"
                    }
                }
            }
        }
    }
)
async def analyze_news(request: NewsAnalysisRequest):
    """뉴스 분석을 수행하고 결과를 반환합니다."""
    try:
        result = await analyzer.analyze(request)
        return jsonable_encoder(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get(
    "/health",
    summary="서비스 상태 확인",
    description="News Analysis Agent 서비스의 상태를 확인합니다.",
    response_description="서비스 상태 정보",
    responses={
        200: {
            "description": "서비스 정상 작동 중",
            "content": {
                "application/json": {
                    "example": {
                        "status": "healthy",
                        "timestamp": "2024-01-01T10:00:00"
                    }
                }
            }
        }
    }
)
async def health_check():
    """서비스 상태 확인"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    } 