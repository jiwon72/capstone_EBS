from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from datetime import datetime
from .models import (
    TechnicalAnalysisRequest, TechnicalAnalysisResponse,
    MovingAverageRequest, MovingAverageResponse,
    SupportResistanceRequest, SupportResistanceResponse,
    PatternRequest, PatternResponse,
    SignalRequest, SignalResponse,
    Signal, SignalStrength, TrendDirection
)
from .technical_analyzer import TechnicalAnalyzer
from typing import Dict, Any

app = FastAPI(
    title="Technical Analysis Agent API",
    description="""
    Technical Analysis Agent API는 시장 데이터에 대한 기술적 분석을 제공합니다.
    
    ## 주요 기능
    * 이동평균선 분석
    * 지지/저항 레벨 분석
    * 기술적 패턴 인식
    * 거래 신호 생성
    * 오실레이터 분석
    * 거래량 지표 분석
    
    ## 분석 방법
    * 다양한 시간대(1분~1개월)의 데이터 분석
    * 여러 기술적 지표의 조합을 통한 종합적 분석
    * 패턴 매칭을 통한 차트 패턴 인식
    """,
    version="1.0.0",
    contact={
        "name": "Trading System Team",
        "email": "trading@example.com"
    }
)

analyzer = TechnicalAnalyzer()

@app.post(
    "/analyze",
    response_model=TechnicalAnalysisResponse,
    summary="종합 기술적 분석",
    description="""
    주어진 시장 데이터에 대한 종합적인 기술적 분석을 수행합니다.
    
    분석 항목:
    - 이동평균선 (단순, 지수, 가중)
    - 오실레이터 (RSI, MACD, Stochastic)
    - 거래량 지표 (OBV, Volume Profile)
    - 지지/저항 레벨
    """,
    response_description="기술적 분석 결과",
    responses={
        200: {
            "description": "분석 성공",
            "content": {
                "application/json": {
                    "example": {
                        "moving_averages": {
                            "sma_20": 150.5,
                            "ema_50": 148.3,
                            "trend": "UPWARD"
                        },
                        "oscillators": {
                            "rsi_14": 65.5,
                            "macd": {
                                "macd_line": 2.5,
                                "signal_line": 1.8,
                                "histogram": 0.7
                            }
                        },
                        "volume_indicators": {
                            "obv": 1000000,
                            "volume_trend": "INCREASING"
                        },
                        "support_resistance": {
                            "support_levels": [145.0, 142.5],
                            "resistance_levels": [152.0, 155.5]
                        }
                    }
                }
            }
        },
        400: {
            "description": "잘못된 요청",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Invalid time frame specified"
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
async def analyze_technical(request: TechnicalAnalysisRequest):
    """기술적 분석을 수행하고 결과를 반환합니다."""
    try:
        result = analyzer.analyze(request)
        if not isinstance(result, TechnicalAnalysisResponse):
            result = TechnicalAnalysisResponse(
                moving_averages=result.get("moving_averages", {}),
                oscillators=result.get("oscillators", {}),
                volume_indicators=result.get("volume_indicators", {}),
                support_resistance=result.get("support_resistance", {}),
                error_message=None
            )
        return jsonable_encoder(result)
    except Exception as e:
        return TechnicalAnalysisResponse(
            moving_averages={},
            oscillators={},
            volume_indicators={},
            support_resistance={},
            error_message=str(e)
        )

@app.post(
    "/moving-averages",
    response_model=MovingAverageResponse,
    summary="이동평균선 분석",
    description="""
    다양한 기간의 이동평균선을 계산하고 추세를 분석합니다.
    
    지원하는 이동평균선:
    - 단순 이동평균 (SMA)
    - 지수 이동평균 (EMA)
    - 가중 이동평균 (WMA)
    """,
    response_description="이동평균선 분석 결과",
    responses={
        200: {
            "description": "분석 성공",
            "content": {
                "application/json": {
                    "example": {
                        "sma": {
                            "20": 150.5,
                            "50": 148.3,
                            "200": 145.0
                        },
                        "ema": {
                            "20": 151.2,
                            "50": 149.1
                        },
                        "trend": "UPWARD",
                        "crossovers": [
                            {
                                "type": "GOLDEN_CROSS",
                                "price": 150.0,
                                "timestamp": "2024-01-01T10:00:00"
                            }
                        ]
                    }
                }
            }
        }
    }
)
async def analyze_moving_averages(request: MovingAverageRequest):
    """이동평균선 분석을 수행합니다."""
    try:
        result = analyzer.analyze_moving_averages(request)
        return jsonable_encoder(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post(
    "/support-resistance",
    response_model=SupportResistanceResponse,
    summary="지지/저항 레벨 분석",
    description="""
    가격 데이터를 분석하여 주요 지지선과 저항선을 식별합니다.
    
    분석 방법:
    - 피봇 포인트
    - 가격 스윙 고점/저점
    - 거래량 프로파일
    """,
    response_description="지지/저항 레벨 분석 결과",
    responses={
        200: {
            "description": "분석 성공",
            "content": {
                "application/json": {
                    "example": {
                        "support_levels": [145.0, 142.5, 140.0],
                        "resistance_levels": [152.0, 155.5, 158.0],
                        "pivot_points": {
                            "current": 150.0,
                            "r1": 153.0,
                            "s1": 147.0
                        },
                        "strength": {
                            "support": "STRONG",
                            "resistance": "MODERATE"
                        }
                    }
                }
            }
        }
    }
)
async def analyze_support_resistance(request: SupportResistanceRequest):
    """지지/저항 레벨을 분석합니다."""
    try:
        result = analyzer.analyze_support_resistance(request)
        return jsonable_encoder(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post(
    "/patterns",
    response_model=PatternResponse,
    summary="차트 패턴 분석",
    description="""
    가격 차트에서 기술적 패턴을 식별합니다.
    
    지원하는 패턴:
    - 헤드앤숄더
    - 이중 바닥/천장
    - 삼각형 패턴
    - 채널
    - 플래그/페넌트
    """,
    response_description="차트 패턴 분석 결과",
    responses={
        200: {
            "description": "분석 성공",
            "content": {
                "application/json": {
                    "example": {
                        "patterns": [
                            {
                                "type": "DOUBLE_BOTTOM",
                                "confidence": 0.85,
                                "start_price": 140.0,
                                "end_price": 150.0,
                                "target_price": 160.0
                            }
                        ],
                        "trend": "UPWARD",
                        "reliability": "HIGH"
                    }
                }
            }
        }
    }
)
async def analyze_patterns(request: PatternRequest):
    """차트 패턴을 분석합니다."""
    try:
        result = analyzer.analyze_patterns(request)
        return jsonable_encoder(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post(
    "/signals",
    response_model=SignalResponse,
    summary="거래 신호 생성",
    description="""
    기술적 분석을 기반으로 거래 신호를 생성합니다.
    
    신호 유형:
    - 매수/매도 시그널
    - 추세 전환 신호
    - 돌파 신호
    - 과매수/과매도 신호
    """,
    response_description="거래 신호 분석 결과",
    responses={
        200: {
            "description": "신호 생성 성공",
            "content": {
                "application/json": {
                    "example": {
                        "signals": [
                            {
                                "type": "BUY",
                                "strength": "STRONG",
                                "price": 150.0,
                                "confidence": 0.85,
                                "indicators": ["RSI_OVERSOLD", "MACD_CROSS"]
                            }
                        ],
                        "trend": "UPWARD",
                        "timeframe": "1h"
                    }
                }
            }
        }
    }
)
async def generate_signals(request: SignalRequest):
    """거래 신호를 생성합니다."""
    try:
        result = analyzer.generate_signals(request)
        return jsonable_encoder(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get(
    "/health",
    summary="서비스 상태 확인",
    description="Technical Analysis Agent 서비스의 상태를 확인합니다.",
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