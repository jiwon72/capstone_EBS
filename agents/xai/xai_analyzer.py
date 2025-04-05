from decimal import Decimal
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
import uuid
import shap
from sklearn.ensemble import RandomForestClassifier
from ..trading.models import OrderType, OrderSide, Order
from .models import (
    FeatureImportance, DecisionPath, LocalExplanation,
    GlobalExplanation, SignalSource, SignalExplanation,
    XAIRequest, XAIResponse, ExplanationMethod,
    Feature, FeatureContribution, Step, Scenario, Counterfactual
)

class XAIAnalyzer:
    def __init__(self):
        """XAI 분석기 초기화"""
        self.feature_descriptions = {
            "price_momentum": "Strong positive momentum in recent price movements",
            "volume_trend": "Above average trading volume",
            "rsi_14": "Relative Strength Index indicates overbought condition",
            "macd": "Moving Average Convergence Divergence shows bullish signal",
            "bollinger_bands": "Price near upper Bollinger Band",
            "sentiment_score": "Positive market sentiment from news analysis",
            "volatility": "High market volatility",
            "market_trend": "Upward market trend"
        }

    def _generate_feature_importance(self, feature_values: Dict[str, float], feature_subset: Optional[List[str]] = None) -> List[FeatureImportance]:
        """특성 중요도 생성"""
        features = []
        for feature_name, value in feature_values.items():
            if feature_subset and feature_name not in feature_subset:
                continue
            importance_score = abs(value) if isinstance(value, (int, float)) else 0.5
            features.append(FeatureImportance(
                feature_name=feature_name,
                importance_score=importance_score,
                description=self.feature_descriptions.get(feature_name, "Feature description not available")
            ))
        return sorted(features, key=lambda x: x.importance_score, reverse=True)

    def _generate_decision_path(self, feature_values: Dict[str, float]) -> List[Dict[str, Any]]:
        """의사결정 경로 생성"""
        decision_path = []
        
        # RSI 조건
        if "rsi_14" in feature_values:
            decision_path.append({
                "condition": "RSI > 70",
                "value": feature_values["rsi_14"] > 70,
                "description": "Overbought condition check"
            })

        # 가격 모멘텀 조건
        if "price_momentum" in feature_values:
            decision_path.append({
                "condition": "Price Momentum > 0.5",
                "value": feature_values["price_momentum"] > 0.5,
                "description": "Strong upward momentum check"
            })

        # 거래량 트렌드 조건
        if "volume_trend" in feature_values:
            decision_path.append({
                "condition": "Volume Trend > 0.5",
                "value": feature_values["volume_trend"] > 0.5,
                "description": "High volume confirmation"
            })

        return decision_path

    def _generate_counterfactuals(self, feature_values: Dict[str, float], target_outcome: str, max_changes: int) -> List[Dict[str, Any]]:
        """반사실적 시나리오 생성"""
        counterfactuals = []
        
        # RSI 기반 시나리오
        if "rsi_14" in feature_values:
            cf = {
                "changes": [{
                    "feature": "rsi_14",
                    "original_value": feature_values["rsi_14"],
                    "new_value": 30 if target_outcome == "BUY" else 80,
                    "explanation": "Change in RSI indicates different market condition"
                }],
                "predicted_outcome": target_outcome,
                "confidence": 0.85
            }
            counterfactuals.append(cf)

        # 가격 모멘텀 기반 시나리오
        if "price_momentum" in feature_values:
            cf = {
                "changes": [{
                    "feature": "price_momentum",
                    "original_value": feature_values["price_momentum"],
                    "new_value": -0.5 if target_outcome == "BUY" else 0.8,
                    "explanation": "Reversal in price momentum"
                }],
                "predicted_outcome": target_outcome,
                "confidence": 0.75
            }
            counterfactuals.append(cf)

        # 거래량 트렌드 기반 시나리오
        if "volume_trend" in feature_values and len(counterfactuals) < max_changes:
            cf = {
                "changes": [{
                    "feature": "volume_trend",
                    "original_value": feature_values["volume_trend"],
                    "new_value": 0.2 if target_outcome == "BUY" else 0.9,
                    "explanation": "Change in volume trend"
                }],
                "predicted_outcome": target_outcome,
                "confidence": 0.7
            }
            counterfactuals.append(cf)

        return counterfactuals[:max_changes]

    def _analyze_feature_interactions(
        self,
        model: RandomForestClassifier,
        feature_names: List[str]
    ) -> List[Dict[str, float]]:
        """특징 간 상호작용 분석"""
        interactions = []
        
        # 모든 특징 쌍에 대해 상호작용 점수 계산
        for i in range(len(feature_names)):
            for j in range(i + 1, len(feature_names)):
                interaction_score = np.random.random()  # 실제로는 더 복잡한 계산 필요
                
                interactions.append({
                    "feature1": feature_names[i],
                    "feature2": feature_names[j],
                    "interaction_score": interaction_score
                })
                
        return sorted(
            interactions,
            key=lambda x: x["interaction_score"],
            reverse=True
        )
    
    def _generate_local_explanation(
        self,
        features: Dict[str, float],
        model: RandomForestClassifier,
        feature_names: List[str]
    ) -> LocalExplanation:
        """로컬 설명 생성"""
        # 가장 중요한 특성 선택
        importance_scores = self._generate_feature_importance(features)
        most_important_feature = importance_scores[0] if importance_scores else None
        
        if not most_important_feature:
            return LocalExplanation(
                feature=None,
                contribution=None,
                features={},
                method="shap",
                timestamp=datetime.now().isoformat(),
                error_message="No features available for explanation"
            )
            
        return LocalExplanation(
            feature=Feature(
                name=most_important_feature.feature_name,
                value=features.get(most_important_feature.feature_name, 0.0)
            ),
            contribution=FeatureContribution(
                value=most_important_feature.importance_score
            ),
            features=features,
            method="shap",
            timestamp=datetime.now().isoformat()
        )
    
    def _generate_global_explanation(
        self,
        model: RandomForestClassifier,
        feature_names: List[str]
    ) -> GlobalExplanation:
        """전역 설명 생성"""
        # 전체 모델의 특징 중요도
        importances = model.feature_importances_
        
        # 가장 중요한 특성 선택
        max_importance_idx = np.argmax(importances)
        most_important_feature = feature_names[max_importance_idx]
        
        return GlobalExplanation(
            feature=Feature(
                name=most_important_feature,
                value=float(importances[max_importance_idx])
            ),
            importance=float(importances[max_importance_idx]),
            interaction_strength=0.8,  # 예시 값
            features=dict(zip(feature_names, map(float, importances))),
            method="shap",
            timestamp=datetime.now().isoformat()
        )
    
    def _generate_signal_explanation(
        self,
        request: XAIRequest,
        local_explanation: LocalExplanation,
        features: Dict[str, float]
    ) -> SignalExplanation:
        """시그널 설명 생성"""
        # 주요 결정 요인 추출
        key_drivers = [
            f"{impact.feature_name}: {impact.contribution}"
            for impact in local_explanation.feature_impacts[:3]
        ]
        
        # 리스크 요소 식별
        risk_factors = [
            "시장 변동성 증가",
            "거래량 감소",
            "기술적 지표 약화"
        ]
        
        # 대안 시나리오
        alternative_scenarios = [
            {
                "scenario": "강세 시나리오",
                "conditions": "거래량 증가 및 모멘텀 강화"
            },
            {
                "scenario": "약세 시나리오",
                "conditions": "지지선 붕괴 및 매도 압력 증가"
            }
        ]
        
        return SignalExplanation(
            signal_id=request.signal_id,
            source=request.source,
            signal_strength=0.75,
            confidence_score=0.8,
            key_drivers=key_drivers,
            risk_factors=risk_factors,
            alternative_scenarios=alternative_scenarios,
            local_explanation=local_explanation,
            supporting_evidence={
                "technical_indicators": "주요 기술적 지표 상태",
                "market_conditions": "현재 시장 상황",
                "historical_patterns": "유사 패턴 분석 결과"
            }
        )
    
    def _generate_decision_path_analysis(
        self,
        feature_values: Dict[str, float]
    ) -> DecisionPath:
        """의사결정 경로 분석 생성"""
        steps = []
        for i, (feature, value) in enumerate(feature_values.items(), 1):
            threshold = 0.5  # 예시 임계값
            direction = "above" if value > threshold else "below"
            steps.append(Step(
                step=i,
                feature=feature,
                threshold=threshold,
                direction=direction,
                confidence=0.8  # 예시 신뢰도
            ))
        
        return DecisionPath(
            steps=steps,
            confidence=0.85,  # 예시 전체 신뢰도
            timestamp=datetime.now().isoformat()
        )

    def _generate_counterfactual_analysis(
        self,
        feature_values: Dict[str, float]
    ) -> Counterfactual:
        """반사실적 분석 생성"""
        scenarios = []
        modified_features = feature_values.copy()
        
        # 각 특성에 대해 하나의 시나리오 생성
        for feature, value in feature_values.items():
            modified_features[feature] = value * 1.2  # 20% 증가
            
            changes = [FeatureChange(
                feature=feature,
                original_value=value,
                target_value=modified_features[feature],
                importance=0.8  # 예시 중요도
            )]
            
            scenarios.append(Scenario(
                original_features=feature_values,
                modified_features=modified_features.copy(),
                original_prediction=0.6,  # 예시 예측값
                modified_prediction=0.7,  # 예시 수정된 예측값
                changes_required=changes
            ))
            
            # 원래 값으로 복구
            modified_features[feature] = value
        
        return Counterfactual(
            scenarios=scenarios,
            confidence=0.85,
            timestamp=datetime.now().isoformat()
        )

    async def explain(self, request: XAIRequest) -> XAIResponse:
        """XAI 설명 생성"""
        try:
            # 임시 모델 생성 (실제로는 저장된 모델을 로드해야 함)
            model = RandomForestClassifier()
            feature_names = list(request.features.keys())
            
            # 각 설명 생성
            local_exp = self._generate_local_explanation(
                request.features, model, feature_names
            )
            global_exp = self._generate_global_explanation(model, feature_names)
            decision_path = self._generate_decision_path_analysis(request.features)
            counterfactual = self._generate_counterfactual_analysis(request.features)
            
            # 응답 생성
            return XAIResponse(
                feature_importance={
                    "feature_name": global_exp.feature.name,
                    "importance_score": global_exp.importance
                },
                local_explanation={
                    "feature_name": local_exp.feature.name,
                    "contribution": local_exp.contribution.value
                },
                global_explanation={
                    "feature_name": global_exp.feature.name,
                    "importance": global_exp.importance,
                    "interaction_strength": global_exp.interaction_strength
                },
                confidence_metrics={
                    "overall_confidence": 0.85,
                    "feature_reliability": 0.8,
                    "prediction_stability": 0.9
                }
            )
            
        except Exception as e:
            return XAIResponse(
                feature_importance={},
                local_explanation={},
                global_explanation={},
                confidence_metrics={},
                error_message=str(e)
            ) 

    def _calculate_execution_price(self, order: Order, current_price: Decimal) -> Optional[Decimal]:
        if order.order_type == OrderType.MARKET:
            return current_price
            
        elif order.order_type == OrderType.LIMIT:
            if order.side == OrderSide.BUY:
                # 매수: 현재가가 지정가보다 낮거나 같으면 체결
                return current_price if current_price <= order.price else None
            else:
                # 매도: 현재가가 지정가보다 높거나 같으면 체결
                return current_price if current_price >= order.price else None
                
        elif order.order_type == OrderType.STOP:
            if order.side == OrderSide.BUY:
                # 매수: 현재가가 스탑가보다 높거나 같으면 체결
                return current_price if current_price >= order.stop_price else None
            else:
                # 매도: 현재가가 스탑가보다 낮거나 같으면 체결
                return current_price if current_price <= order.stop_price else None
                
        return None 