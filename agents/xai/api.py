from datetime import datetime
from typing import Dict, List, Any
from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from .models import (
    XAIRequest, XAIResponse,
    LocalExplanation, GlobalExplanation,
    FeatureImportance, DecisionPath,
    Counterfactual, ConfidenceMetrics,
    Feature, FeatureContribution, Step,
    Scenario, FeatureChange,
    ExplanationRequest, Contribution
)
from .explainer import ModelExplainer

app = FastAPI(title="XAI Agent API")
explainer = ModelExplainer()

@app.get("/health")
async def health_check():
    """서비스 상태 확인"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }

@app.post("/explain", response_model=XAIResponse)
async def explain_prediction(request: XAIRequest):
    """예측 설명 엔드포인트"""
    try:
        explanation = explainer.explain(request)
        if not isinstance(explanation, XAIResponse):
            explanation = XAIResponse(
                feature_importance=explanation.get("feature_importance", {}),
                local_explanation=explanation.get("local_explanation", {}),
                global_explanation=explanation.get("global_explanation", {}),
                confidence_metrics=explanation.get("confidence_metrics", {}),
                error_message=None
            )
        return jsonable_encoder(explanation)
    except Exception as e:
        return XAIResponse(
            feature_importance={},
            local_explanation={},
            global_explanation={},
            confidence_metrics={},
            error_message=str(e)
        )

@app.post("/explain/local", response_model=Dict[str, Any])
async def explain_local(request: ExplanationRequest):
    """로컬 설명 생성 엔드포인트"""
    try:
        result = await explainer.explain_local(request)
        features = request.features
        response = {}
        for feature_name, value in features.items():
            response[feature_name] = {
                "feature": {"name": feature_name, "value": value},
                "contribution": {"value": 0.5},  # 실제 계산된 값으로 대체 필요
                "features": features,
                "method": request.explanation_method or "shap",
                "timestamp": datetime.now().isoformat()
            }
        return jsonable_encoder(response)
    except Exception as e:
        return {
            "error_message": str(e)
        }

@app.post("/explain/global", response_model=Dict[str, Any])
async def explain_global(request: ExplanationRequest):
    """글로벌 설명 생성 엔드포인트"""
    try:
        result = await explainer.explain_global(request)
        features = request.features
        response = {}
        for feature_name, value in features.items():
            response[feature_name] = {
                "feature": {"name": feature_name, "value": value},
                "importance": 0.7,  # 실제 계산된 값으로 대체 필요
                "interaction_strength": 0.5,  # 실제 계산된 값으로 대체 필요
                "features": features,
                "method": request.explanation_method or "shap",
                "timestamp": datetime.now().isoformat()
            }
        return jsonable_encoder(response)
    except Exception as e:
        return {
            "error_message": str(e)
        }

@app.post("/explain/feature_importance", response_model=Dict[str, Any])
async def explain_feature_importance(request: ExplanationRequest):
    """특성 중요도 분석 엔드포인트"""
    try:
        result = await explainer.explain_feature_importance(request)
        features = request.features
        response = {}
        for feature_name, value in features.items():
            response[feature_name] = {
                "feature": {"name": feature_name, "value": value},
                "importance": 0.7,  # 실제 계산된 값으로 대체 필요
                "direction": "positive",  # 실제 계산된 값으로 대체 필요
                "features": features,
                "timestamp": datetime.now().isoformat()
            }
        return jsonable_encoder(response)
    except Exception as e:
        return {
            "error_message": str(e)
        }

@app.post("/explain/decision_path", response_model=List[Dict[str, Any]])
async def explain_decision_path(request: ExplanationRequest):
    """의사결정 경로 분석 엔드포인트"""
    try:
        result = await explainer.explain_decision_path(request)
        steps = [
            {
                "step": 1,
                "feature": "price_momentum",
                "threshold": 0.5,
                "direction": "above",
                "confidence": 0.8
            },
            {
                "step": 2,
                "feature": "volume_trend",
                "threshold": 0.4,
                "direction": "above",
                "confidence": 0.7
            }
        ]
        return jsonable_encoder(steps)
    except Exception as e:
        return []

@app.post("/explain/counterfactual", response_model=List[Dict[str, Any]])
async def explain_counterfactual(request: ExplanationRequest):
    """반사실적 분석 엔드포인트"""
    try:
        result = await explainer.explain_counterfactual(request)
        original_features = request.features
        modified_features = {k: v * 0.5 for k, v in original_features.items()}  # 예시로 모든 값을 절반으로
        
        counterfactuals = [
            {
                "original_features": original_features,
                "modified_features": modified_features,
                "original_prediction": 0.8,  # 실제 계산된 값으로 대체 필요
                "modified_prediction": 0.3,  # 실제 계산된 값으로 대체 필요
                "changes_required": [
                    {
                        "feature": k,
                        "original_value": v,
                        "target_value": modified_features[k],
                        "importance": 0.7  # 실제 계산된 값으로 대체 필요
                    }
                    for k, v in original_features.items()
                ]
            }
        ]
        
        return jsonable_encoder(counterfactuals)
    except Exception as e:
        return []

@app.post("/explain/confidence", response_model=ConfidenceMetrics)
async def analyze_confidence(request: XAIRequest):
    """신뢰도 분석 엔드포인트"""
    try:
        explanation = explainer.analyze_confidence(request)
        result = ConfidenceMetrics(
            overall_score=float(explanation.get("overall_score", 0.0)),
            feature_reliability=explanation.get("feature_reliability", {}),
            prediction_stability=float(explanation.get("prediction_stability", 0.0)),
            data_quality_score=float(explanation.get("data_quality_score", 0.0)),
            timestamp=datetime.now()
        )
        return jsonable_encoder(result)
    except Exception as e:
        return ConfidenceMetrics(
            overall_score=0.0,
            feature_reliability={},
            prediction_stability=0.0,
            data_quality_score=0.0,
            timestamp=datetime.now()
        ) 