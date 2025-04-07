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

app = FastAPI(title="Technical Analysis Agent API")
analyzer = TechnicalAnalyzer()

@app.post("/analyze", response_model=TechnicalAnalysisResponse)
def analyze_technical(request: TechnicalAnalysisRequest):
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

@app.get("/health")
def health_check():
    """서비스 상태 확인"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/indicators/moving_averages", response_model=MovingAverageResponse)
def calculate_moving_averages(request: MovingAverageRequest):
    """이동평균 계산 엔드포인트"""
    try:
        result = analyzer.calculate_moving_averages(request)
        return result
    except Exception as e:
        return MovingAverageResponse(
            sma_values={},
            ema_values={},
            trend_direction=TrendDirection.SIDEWAYS,
            error_message=str(e)
        )

@app.post("/analyze/support_resistance", response_model=SupportResistanceResponse)
def analyze_support_resistance(request: SupportResistanceRequest):
    """지지/저항 레벨 분석 엔드포인트"""
    try:
        result = analyzer.analyze_support_resistance(request)
        if not isinstance(result, SupportResistanceResponse):
            result = SupportResistanceResponse(
                support_levels=result.get("support_levels", []),
                resistance_levels=result.get("resistance_levels", []),
                pivot_points=result.get("pivot_points", {}),
                error_message=None
            )
        return jsonable_encoder(result)
    except Exception as e:
        return SupportResistanceResponse(
            support_levels=[],
            resistance_levels=[],
            pivot_points={},
            error_message=str(e)
        )

@app.post("/analyze/patterns", response_model=PatternResponse)
def analyze_patterns(request: PatternRequest):
    """차트 패턴 분석 엔드포인트"""
    try:
        result = analyzer.analyze_patterns(request)
        return result
    except Exception as e:
        return PatternResponse(
            patterns={},
            timestamp=datetime.now().isoformat(),
            error_message=str(e)
        )

@app.post("/signals", response_model=SignalResponse)
def generate_signals(request: SignalRequest):
    """트레이딩 신호 생성 엔드포인트"""
    try:
        result = analyzer.generate_signals(request)
        return result
    except Exception as e:
        return SignalResponse(
            signals={},
            timestamp=datetime.now().isoformat(),
            error_message=str(e)
        ) 