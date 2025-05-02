import sys
import os
import json
from datetime import datetime

# 상위 디렉토리를 파이썬 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.strategy.strategy_generator import StrategyGenerator
from agents.strategy.models import TimeHorizon

def test_strategy_generation():
    """전략 생성 테스트"""
    print("\n=== 전략 생성 테스트 시작 ===\n")
    
    # StrategyGenerator 인스턴스 생성
    generator = StrategyGenerator()
    
    # 테스트용 입력 파라미터
    user_input = "삼성전자와 SK하이닉스에 대한 투자 전략을 분석해주세요."
    risk_tolerance = 0.6
    time_horizon = TimeHorizon.MEDIUM_TERM
    
    try:
        # 전략 생성
        strategy = generator.generate_strategy(
            user_input=user_input,
            risk_tolerance=risk_tolerance,
            time_horizon=time_horizon
        )
        
        # 결과 출력
        print("📊 생성된 전략 정보:")
        print(f"전략 ID: {strategy.strategy_id}")
        print(f"전략 유형: {strategy.strategy_type.value}")
        print(f"\n투자 대상 종목:")
        for asset in strategy.target_assets:
            print(f"- {asset}")
            
        print(f"\n섹터별 투자 비중:")
        for sector, weight in strategy.sector_allocation.items():
            print(f"- {sector}: {weight:.2%}")
            
        print(f"\n리스크 파라미터:")
        print(f"- 최대 포지션 크기: {strategy.risk_parameters.max_position_size:.2%}")
        print(f"- 손절: {strategy.risk_parameters.stop_loss:.2%}")
        print(f"- 익절: {strategy.risk_parameters.take_profit:.2%}")
        
        print("\n📝 전략 설명:")
        print(strategy.explanation)
        
        print("\n✅ 테스트 성공!")
        
    except Exception as e:
        print(f"\n❌ 테스트 실패: {str(e)}")
        raise e

if __name__ == "__main__":
    test_strategy_generation() 