import json
import re
import os

def parse_markdown_to_dashboard(md_path):
    """마크다운 파일을 dashboard.json 구조로 파싱"""
    
    with open(md_path, 'r', encoding='utf-8') as f:
        content = f.read()

    data = {
        "strategy": [],
        "news": {},
        "risk": [],
        "tech": [],
        "portfolio": [],
        "summary": []
    }

    # 섹션별로 내용 분리 (## 기준)
    sections = re.split(r'\n#+\s+', content)
    
    for section in sections:
        lines = section.split('\n')
        section_lower = section.lower()
        
        # 1. 전략 분석 파싱
        if '전략 분석' in section:
            table_started = False
            for line in lines:
                if '|' in line and not line.startswith('|---'):
                    parts = [p.strip() for p in line.split('|') if p.strip()]
                    if len(parts) >= 3:
                        # 헤더 행 건너뛰기
                        if parts[0] in ['종목', 'Stock', '---'] or '추천전략' in parts[1]:
                            continue
                        data["strategy"].append({
                            "종목": parts[0],
                            "추천전략": parts[1],
                            "설명": parts[2]
                        })

        # 2. 뉴스/이슈 분석 파싱
        elif '뉴스' in section or '이슈' in section:
            # 테이블 형태 파싱
            table_started = False
            for line in lines:
                if '|' in line and not line.startswith('|---'):
                    parts = [p.strip() for p in line.split('|') if p.strip()]
                    if len(parts) >= 4:
                        # 헤더 행 건너뛰기
                        if parts[0] in ['종목', 'Stock', '---'] or '감성점수' in parts[1]:
                            continue
                        data["news"][parts[0]] = {
                            "감성점수": parts[1],
                            "시장영향도": parts[2],
                            "키워드": parts[3]
                        }
            
            # 리스트 형태 파싱 (- **종목명**: 내용)
            for line in lines:
                if line.startswith('- **') and ':' in line:
                    match = re.match(r'- \*\*(.+?)\*\*:\s*(.+)', line)
                    if match:
                        stock_name = match.group(1).strip()
                        analysis = match.group(2).strip()
                        if stock_name not in data["news"]:
                            data["news"][stock_name] = analysis

        # 3. 리스크 분석 파싱
        elif '리스크' in section:
            table_started = False
            for line in lines:
                if '|' in line and not line.startswith('|---'):
                    parts = [p.strip() for p in line.split('|') if p.strip()]
                    if len(parts) >= 4:
                        # 헤더 행 건너뛰기
                        if parts[0] in ['종목', 'Stock', '---'] or '리스크점수' in parts[1]:
                            continue
                        
                        # 숫자 값 변환
                        risk_score = parts[1]
                        volatility = parts[2]
                        risk_level = parts[3]
                        
                        try:
                            risk_score = float(risk_score)
                        except:
                            pass
                        try:
                            volatility = float(volatility)
                        except:
                            pass
                            
                        data["risk"].append({
                            "종목": parts[0],
                            "리스크점수": risk_score,
                            "변동성": volatility,
                            "리스크레벨": risk_level
                        })

        # 4. 기술적 분석 파싱
        elif '기술적' in section:
            table_started = False
            for line in lines:
                if '|' in line and not line.startswith('|---'):
                    parts = [p.strip() for p in line.split('|') if p.strip()]
                    if len(parts) >= 6:
                        # 헤더 행 건너뛰기
                        if parts[0] in ['종목', 'Stock', '---'] or 'RSI' in parts[1]:
                            continue
                        
                        # 숫자 값 변환
                        try:
                            rsi = float(parts[1]) if parts[1] != '-' else '-'
                        except:
                            rsi = parts[1]
                        try:
                            macd = float(parts[2]) if parts[2] != '-' else '-'
                        except:
                            macd = parts[2]
                        try:
                            bb_upper = float(parts[3]) if parts[3] != '-' else '-'
                        except:
                            bb_upper = parts[3]
                        try:
                            bb_lower = float(parts[4]) if parts[4] != '-' else '-'
                        except:
                            bb_lower = parts[4]
                        try:
                            close_price = float(parts[5]) if parts[5] != '-' else '-'
                        except:
                            close_price = parts[5]
                            
                        data["tech"].append({
                            "종목": parts[0],
                            "RSI": rsi,
                            "MACD": macd,
                            "볼린저밴드상단": bb_upper,
                            "볼린저밴드하단": bb_lower,
                            "최근종가": close_price
                        })

        # 5. 최종 포트폴리오 구성 파싱
        elif '최종 포트폴리오' in section or ('포트폴리오' in section and '구성' in section):
            table_started = False
            for line in lines:
                if '|' in line and not line.startswith('|---'):
                    parts = [p.strip() for p in line.split('|') if p.strip()]
                    if len(parts) >= 4:
                        # 헤더 행 건너뛰기
                        if parts[0] in ['종목명', 'Stock', '---'] or '구매개수' in parts[1]:
                            continue
                        
                        # 숫자 값 변환
                        try:
                            quantity = int(parts[1].replace(',', '')) if parts[1] != '-' else 0
                        except:
                            quantity = 0
                        try:
                            investment = parts[2].replace(',', '')
                            investment = float(investment) if investment != '-' else 0
                        except:
                            investment = 0
                        try:
                            current_price = parts[3].replace(',', '')
                            current_price = float(current_price) if current_price != '-' else 0
                        except:
                            current_price = 0
                            
                        data["portfolio"].append({
                            "종목명": parts[0],
                            "구매개수": quantity,
                            "투자금": investment,
                            "현재가": current_price
                        })

        # 6. 포트폴리오 구성 이유 및 기대효과 파싱
        elif '포트폴리오 구성 이유' in section or '기대효과' in section:
            summary_text = ""
            for line in lines:
                if line.strip() and not line.startswith('#') and not line.startswith('|'):
                    if line.startswith('-'):
                        summary_text += line[1:].strip() + " "
                    else:
                        summary_text += line.strip() + " "
            
            if summary_text.strip():
                data["summary"].append(summary_text.strip())

    return data

def convert_markdown_to_json(md_path, json_path):
    """마크다운을 dashboard.json으로 변환"""
    
    try:
        # 마크다운 파싱
        parsed_data = parse_markdown_to_dashboard(md_path)
        
        # JSON 파일로 저장
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(parsed_data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 변환 완료: {json_path}")
        print(f"📊 파싱된 데이터:")
        print(f"   - 전략 분석: {len(parsed_data['strategy'])}개 종목")
        print(f"   - 뉴스 분석: {len(parsed_data['news'])}개 종목")
        print(f"   - 리스크 분석: {len(parsed_data['risk'])}개 종목")
        print(f"   - 기술적 분석: {len(parsed_data['tech'])}개 종목")
        print(f"   - 포트폴리오: {len(parsed_data['portfolio'])}개 항목")
        print(f"   - 종합 의견: {len(parsed_data['summary'])}개 항목")
        
    except Exception as e:
        print(f"❌ 변환 실패: {e}")
        import traceback
        traceback.print_exc()

def convert_markdown_to_dashboard_json(md_path, json_path):
    """호환성을 위한 함수명"""
    return convert_markdown_to_json(md_path, json_path)

def watch_and_convert_dashboard():
    """마크다운 파일 변경 감지 및 자동 변환"""
    import time
    
    reports_dir = "investment_reports"
    last_modified = 0
    
    print("🔄 마크다운 파일 변경 감지 시작...")
    
    while True:
        try:
            md_files = [f for f in os.listdir(reports_dir) if f.endswith('.md')]
            if md_files:
                latest_md = max(md_files)
                md_path = os.path.join(reports_dir, latest_md)
                current_modified = os.path.getmtime(md_path)
                
                if current_modified > last_modified:
                    print(f"📁 파일 변경 감지: {latest_md}")
                    convert_markdown_to_json(md_path, "system/dashboard.json")
                    last_modified = current_modified
                    
        except Exception as e:
            print(f"감지 오류: {e}")
            
        time.sleep(5)  # 5초마다 확인

if __name__ == "__main__":
    # 가장 최신 마크다운 파일 찾기
    reports_dir = "investment_reports"
    try:
        md_files = [f for f in os.listdir(reports_dir) if f.endswith('.md')]
        if md_files:
            latest_md = max(md_files)
            md_path = os.path.join(reports_dir, latest_md)
            print(f"📄 변환 대상: {latest_md}")
            convert_markdown_to_json(md_path, "system/dashboard.json")
        else:
            print("❌ 마크다운 파일을 찾을 수 없습니다.")
    except Exception as e:
        print(f"❌ 오류: {e}")
    
    # 자동 감지 모드 실행 여부 선택
    auto_watch = input("\n자동 갱신 모드를 실행하시겠습니까? (y/n): ").lower()
    if auto_watch == 'y':
        watch_and_convert_dashboard()