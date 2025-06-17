import pandas as pd
import json
import os
import numpy as np
from datetime import datetime

def convert_csv_to_json():
    """CSV 파일을 JSON으로 변환하는 함수 (종목코드 포함)"""
    
    # CSV 파일 경로 (가장 최신 파일 찾기)
    forecasts_dir = "system/forecasts"
    csv_files = [f for f in os.listdir(forecasts_dir) if f.endswith('.csv')]
    
    if not csv_files:
        print("CSV 파일을 찾을 수 없습니다.")
        return
    
    # 가장 최신 파일 선택 (파일명에 날짜가 있다고 가정)
    latest_csv = max(csv_files)
    csv_path = os.path.join(forecasts_dir, latest_csv)
    
    print(f"변환 중: {csv_path}")
    
    try:
        # CSV 파일 읽기
        df = pd.read_csv(csv_path)
        
        # 날짜 포맷 변환 (시간 제거)
        df["예측일"] = pd.to_datetime(df["예측일"]).dt.strftime("%Y-%m-%d")
        
        # 종목별로 데이터 그룹화 (종목코드도 함께 저장)
        result = {}
        for stock_name in df["종목명"].unique():
            stock_data = df[df["종목명"] == stock_name].sort_values("예측일")
            
            # 해당 종목의 종목코드 가져오기 (첫 번째 행의 종목코드 사용)
            stock_code = stock_data["종목코드"].iloc[0]
            
            # numpy/pandas 타입을 Python 기본 타입으로 변환
            result[stock_name] = {
                "stock_code": str(stock_code),  # 문자열로 변환
                "dates": stock_data["예측일"].tolist(),
                "prices": [float(price) for price in stock_data["예측가격"].round(2)]  # float로 명시적 변환
            }
        
        # JSON 파일로 저장 (custom encoder 함수 추가)
        json_path = "system/forecast.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2, default=convert_numpy_types)
        
        print(f"✅ 변환 완료: {json_path}")
        print(f"📊 총 {len(result)}개 종목 데이터 변환됨")
        
        # 변환된 종목 목록 출력 (종목코드도 함께)
        for stock_name, data in result.items():
            print(f"   - {stock_name} ({data['stock_code']})")
            
    except Exception as e:
        print(f"❌ 변환 실패: {e}")
        import traceback
        traceback.print_exc()  # 상세 오류 정보 출력

def convert_numpy_types(obj):
    """numpy/pandas 타입을 JSON 직렬화 가능한 타입으로 변환"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (pd.Timestamp, pd.NaType)):
        return str(obj)
    elif hasattr(obj, 'item'):  # numpy scalar types
        return obj.item()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

def watch_and_convert():
    """CSV 파일 변경 감지 및 자동 변환 (개발용)"""
    import time
    
    forecasts_dir = "system/forecasts"
    last_modified = 0
    
    print("🔄 CSV 파일 변경 감지 시작...")
    
    while True:
        try:
            csv_files = [f for f in os.listdir(forecasts_dir) if f.endswith('.csv')]
            if csv_files:
                latest_csv = max(csv_files)
                csv_path = os.path.join(forecasts_dir, latest_csv)
                current_modified = os.path.getmtime(csv_path)
                
                if current_modified > last_modified:
                    print(f"📁 파일 변경 감지: {latest_csv}")
                    convert_csv_to_json()
                    last_modified = current_modified
                    
        except Exception as e:
            print(f"감지 오류: {e}")
            
        time.sleep(5)  # 5초마다 확인

if __name__ == "__main__":
    # 즉시 변환 실행
    convert_csv_to_json()
    
    # 자동 감지 모드 실행 여부 선택
    auto_watch = input("\n자동 갱신 모드를 실행하시겠습니까? (y/n): ").lower()
    if auto_watch == 'y':
        watch_and_convert()