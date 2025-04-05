from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Union, Any
from datetime import datetime
from enum import Enum

class ExplanationMethod(str, Enum):
    SHAP = "shap"
    LIME = "lime"
    ELI5 = "eli5"
    ANCHORS = "anchors"

class Feature(BaseModel):
    name: str
    value: float

class FeatureContribution(BaseModel):
    value: float

class FeatureChange(BaseModel):
    feature: str
    change: float

class Step(BaseModel):
    step: int
    feature: str
    threshold: float
    direction: str
    confidence: float

class Scenario(BaseModel):
    original_features: Dict[str, float]
    modified_features: Dict[str, float]
    original_prediction: float
    modified_prediction: float
    changes_required: List[FeatureChange]

class FeatureImportance(BaseModel):
    feature: Feature
    importance: float
    direction: str
    features: Dict[str, float]
    method: str
    timestamp: datetime = Field(default_factory=datetime.now)

class DecisionPath(BaseModel):
    step: int
    feature: Feature
    threshold: float
    direction: str
    confidence: float
    steps: List[Step]
    method: str
    timestamp: datetime = Field(default_factory=datetime.now)

class LocalExplanation(BaseModel):
    feature: Feature
    contribution: FeatureContribution
    features: Dict[str, float]
    method: str
    timestamp: datetime = Field(default_factory=datetime.now)
    error_message: Optional[str] = None

class GlobalExplanation(BaseModel):
    feature: Feature
    importance: float
    interaction_strength: float
    features: Dict[str, float]
    method: str
    timestamp: datetime = Field(default_factory=datetime.now)

class Counterfactual(BaseModel):
    original_features: Dict[str, float]
    modified_features: Dict[str, float]
    original_prediction: float
    modified_prediction: float
    changes_required: List[FeatureChange]
    scenarios: List[Scenario]
    method: str
    timestamp: datetime = Field(default_factory=datetime.now)

class ConfidenceMetrics(BaseModel):
    overall_score: float
    feature_reliability: Dict[str, float]
    prediction_stability: float
    data_quality_score: float
    timestamp: datetime = Field(default_factory=datetime.now)

class SignalSource(str, Enum):
    TECHNICAL = "technical"
    NEWS = "news"
    FUNDAMENTAL = "fundamental"
    MARKET_SENTIMENT = "market_sentiment"

class SignalExplanation(BaseModel):
    signal_id: str
    timestamp: datetime = Field(default_factory=datetime.now)
    source: SignalSource
    signal_strength: float = Field(..., ge=0, le=1)
    confidence_score: float = Field(..., ge=0, le=1)
    key_drivers: List[str]  # 주요 결과 요인
    risk_factors: List[str]  # 잠재적 리스크 요소
    alternative_scenarios: List[Dict[str, str]]  # 대안 시나리오
    local_explanation: LocalExplanation
    supporting_evidence: Dict[str, str]  # 근거 데이터

class XAIRequest(BaseModel):
    model_id: str
    features: Dict[str, float]
    target: Optional[float] = None
    explanation_method: str = "shap"
    feature_subset: Optional[List[str]] = None
    num_samples: int = 1000
    metadata: Optional[Dict] = None

class XAIResponse(BaseModel):
    feature_importance: Dict[str, float]
    local_explanation: Dict[str, float]
    global_explanation: Dict[str, float]
    confidence_metrics: Dict[str, float]
    error_message: Optional[str] = None

class Contribution(BaseModel):
    value: float

class ExplanationRequest(BaseModel):
    features: Dict[str, float]
    explanation_method: Optional[str] = None
    feature_subset: Optional[List[str]] = None

class GlobalExplanation(BaseModel):
    feature: Feature
    importance: float
    features: Dict[str, float]
    interaction_strength: float
    timestamp: str
    error_message: Optional[str] = None

class FeatureImportance(BaseModel):
    feature: Feature
    importance: float
    direction: str
    features: Dict[str, float]
    timestamp: str
    error_message: Optional[str] = None

class Step(BaseModel):
    step: int
    feature: str
    threshold: float
    direction: str
    confidence: float

class DecisionPath(BaseModel):
    steps: List[Step]
    confidence: float
    timestamp: str
    error_message: Optional[str] = None

class FeatureChange(BaseModel):
    feature: str
    original_value: float
    target_value: float
    importance: float

class Scenario(BaseModel):
    original_features: Dict[str, float]
    modified_features: Dict[str, float]
    original_prediction: float
    modified_prediction: float
    changes_required: List[FeatureChange]

class Counterfactual(BaseModel):
    scenarios: List[Scenario]
    confidence: float
    timestamp: str
    error_message: Optional[str] = None 