from datetime import datetime
from fastapi.testclient import TestClient
from agents.xai.api import app
from agents.xai.models import XAIRequest

client = TestClient(app)

def generate_test_request() -> XAIRequest:
    """테스트용 XAI 요청 생성"""
    return XAIRequest(
        model_id="test_model",
        features={
            "price_momentum": 0.8,
            "volume_trend": 0.6,
            "rsi_14": 70.5
        },
        target=1.0,
        explanation_method="shap",
        feature_subset=["price_momentum", "volume_trend"],
        num_samples=1000,
        metadata={}
    )

def test_explain_prediction_endpoint():
    """예측 설명 엔드포인트 테스트"""
    request = generate_test_request()
    response = client.post("/explain", json=request.model_dump(mode='json'))
    assert response.status_code == 200

    data = response.json()
    assert "feature_importance" in data
    assert "local_explanation" in data
    assert "global_explanation" in data

def test_local_explanation():
    """로컬 설명 테스트"""
    request = generate_test_request()
    response = client.post("/explain/local", json=request.model_dump(mode='json'))
    assert response.status_code == 200

    data = response.json()
    assert "price_momentum" in data
    assert "volume_trend" in data
    assert "rsi_14" in data

def test_global_explanation():
    """글로벌 설명 테스트"""
    request = generate_test_request()
    response = client.post("/explain/global", json=request.model_dump(mode='json'))
    assert response.status_code == 200

    data = response.json()
    assert "price_momentum" in data
    assert "volume_trend" in data
    assert "rsi_14" in data

def test_feature_subset():
    """특성 부분집합 테스트"""
    request = generate_test_request()
    response = client.post("/explain", json=request.model_dump(mode='json'))
    assert response.status_code == 200

    data = response.json()
    assert "feature_importance" in data
    assert all(f in request.feature_subset for f in data["feature_importance"])

def test_explanation_method():
    """설명 방법 테스트"""
    request = generate_test_request()
    request.explanation_method = "lime"
    response = client.post("/explain", json=request.model_dump(mode='json'))
    assert response.status_code == 200

    data = response.json()
    assert "feature_importance" in data
    assert "local_explanation" in data

def test_feature_importance_analysis():
    """특성 중요도 분석 테스트"""
    request = generate_test_request()
    response = client.post("/explain/feature_importance", json=request.model_dump(mode='json'))
    assert response.status_code == 200

    data = response.json()
    assert "price_momentum" in data
    assert "volume_trend" in data
    assert "rsi_14" in data

def test_decision_path_analysis():
    """의사결정 경로 분석 테스트"""
    request = generate_test_request()
    response = client.post("/explain/decision_path", json=request.model_dump(mode='json'))
    assert response.status_code == 200

    data = response.json()
    assert isinstance(data, list)
    for step in data:
        assert "step" in step
        assert "feature" in step
        assert "threshold" in step
        assert "direction" in step
        assert "confidence" in step

def test_counterfactual_analysis():
    """반사실적 분석 테스트"""
    request = generate_test_request()
    response = client.post("/explain/counterfactual", json=request.model_dump(mode='json'))
    assert response.status_code == 200

    data = response.json()
    assert isinstance(data, list)
    for cf in data:
        assert "original_features" in cf
        assert "modified_features" in cf
        assert "original_prediction" in cf
        assert "modified_prediction" in cf
        assert "changes_required" in cf

def test_confidence_analysis():
    """신뢰도 분석 테스트"""
    request = generate_test_request()
    response = client.post("/explain/confidence", json=request.model_dump(mode='json'))
    assert response.status_code == 200

    data = response.json()
    assert "overall_score" in data
    assert "feature_reliability" in data
    assert "prediction_stability" in data
    assert "data_quality_score" in data 