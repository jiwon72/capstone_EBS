import os
import re
import glob
import json
import csv
from flask import Flask, render_template

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPORT_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../investment_reports"))
HOLDINGS_PATH = os.path.abspath(os.path.join(BASE_DIR, "../../holdings.json"))
FORECAST_GLOB = os.path.abspath(os.path.join(BASE_DIR, "../../forecasts/forecast_*.csv"))

# 최신 보고서 파일 찾기
def get_latest_report():
    files = glob.glob(os.path.join(REPORT_DIR, "investment_report_portfolio_*.md"))
    if not files:
        return None
    return max(files, key=os.path.getmtime)

# holdings.json 불러오기
def load_holdings():
    if os.path.exists(HOLDINGS_PATH):
        with open(HOLDINGS_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}

# forecast_YYYYMMDD.csv 불러오기
def load_forecast():
    files = glob.glob(FORECAST_GLOB)
    if not files:
        return []
    latest = max(files, key=os.path.getmtime)
    rows = []
    with open(latest, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows

# 종목명에서 (코드) 패턴 robust 분리 함수
def extract_name_code(raw_name):
    m = re.match(r'(.*?)\s*\((\d{6})\)', raw_name)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return raw_name.strip(), None

# 최신 보고서에서 종목별 분석 파싱
def parse_md_report(md_path):
    with open(md_path, encoding="utf-8") as f:
        content = f.read()
    # 전략 분석 표 파싱 (전략 분석 섹션 내에서만)
    strategy_dict = {}
    name_to_code = {}
    code_to_name = {}
    strategy_section = re.search(r'# 1\. 전략 분석(.*?)(# 2\.|2\.|# 2\.|2\.|\Z)', content, re.DOTALL)
    if strategy_section:
        strategy_rows = re.findall(r'\|([^|\n]+)\|([^|\n]+)\|([^|\n]+)\|', strategy_section.group(1))
        for name_raw, strat, desc in strategy_rows:
            name, code = extract_name_code(name_raw)
            if code:
                name_to_code[name] = code
                code_to_name[code] = name
            if name in ["종목", "종목명", "현금", "-----------", "# 4. 기술적 분석", "종목코드", "구매개수", "투자금", "현재가", "", "-"] or name.startswith("-"):
                continue
            strategy_dict[name] = {"name": name, "strategy": strat.strip(), "desc": desc.strip()}
    # 리스크 분석 표 파싱 (리스크 분석 섹션 내에서만)
    risk_dict = {}
    risk_section = re.search(r'# 3\. 리스크 분석(.*?)(# 4\.|4\.|# 4\.|4\.|\Z)', content, re.DOTALL)
    if risk_section:
        risk_rows = re.findall(r'\|([^|\n]+)\|([^|\n]+)\|([^|\n]+)\|([^|\n]+)\|', risk_section.group(1))
        for name_raw, score, vol, level in risk_rows:
            name, code = extract_name_code(name_raw)
            if code:
                name_to_code[name] = code
                code_to_name[code] = name
            if name in ["종목", "종목명", "현금", "-----------", "# 4. 기술적 분석", "종목코드", "구매개수", "투자금", "현재가", "", "-"] or name.startswith("-"):
                continue
            risk_dict[name] = {"name": name, "score": score.strip(), "volatility": vol.strip(), "level": level.strip()}
    # 기술적 분석 표 파싱 (기술적 분석 섹션 내에서만)
    tech_dict = {}
    tech_section = re.search(r'# 4\. 기술적 분석(.*?)(# 5\.|5\.|# 5\.|5\.|\Z)', content, re.DOTALL)
    if tech_section:
        tech_rows = re.findall(r'\|([^|\n]+)\|([^|\n]+)\|([^|\n]+)\|([^|\n]+)\|([^|\n]+)\|([^|\n]+)\|', tech_section.group(1))
        for name_raw, rsi, macd, upper, lower, close in tech_rows:
            name, code = extract_name_code(name_raw)
            if code:
                name_to_code[name] = code
                code_to_name[code] = name
            if name in ["종목", "종목명", "현금", "-----------", "# 4. 기술적 분석", "종목코드", "구매개수", "투자금", "현재가", "", "-"] or name.startswith("-"):
                continue
            tech_dict[name] = {"name": name, "rsi": rsi.strip(), "macd": macd.strip(), "bb_upper": upper.strip(), "bb_lower": lower.strip(), "close": close.strip()}
    # 뉴스/이슈 분석 robust 파싱 (한 줄 요약/키워드/시장영향도 모두 추출)
    news_dict = {}
    news_section = re.search(r'(?:#\s*2\.|2\.)\s*\*?뉴스/이슈 분석\*?(.*?)(?:#\s*3\.|3\.|#\s*3\.|3\.|\\Z)', content, re.DOTALL)
    if news_section:
        lines = news_section.group(1).splitlines()
        for line in lines:
            m = re.match(r'-+\s*\*+\s*(.*?)\s*\((\d{6})\)\*+\s*:?\s*(.*)', line.strip())
            if m:
                name, code, rest = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
                # 요약, 키워드, 시장영향도 추출
                summary = rest
                keywords = '-'
                impact = '-'
                # 주요 키워드: ...
                kw_match = re.search(r'주요 키워드[:：]?\s*([^,]+(?:,\s*[^,]+)*)', rest)
                if kw_match:
                    keywords = kw_match.group(1).strip()
                # 시장영향도: ...
                imp_match = re.search(r'시장영향도[:：]?\s*([\d.\-]+)', rest)
                if imp_match:
                    impact = imp_match.group(1)
                # 요약: 키워드 앞까지
                if '주요 키워드' in rest:
                    summary = rest.split('주요 키워드')[0].strip(' :：-')
                news_dict[name] = {
                    "name": name,
                    "code": code,
                    "news_summary": summary if summary else '-',
                    "sentiment": '-',
                    "impact": impact,
                    "keywords": keywords
                }
    print('DEBUG news_dict:', news_dict)  # 디버깅용
    # news_dict 기반 name_to_code, code_to_name 보강
    for v in news_dict.values():
        n, c = v.get('name'), v.get('code')
        if n and c:
            name_to_code[n] = c
            code_to_name[c] = n
    print('DEBUG (보강 후) name_to_code:', name_to_code)
    print('DEBUG (보강 후) code_to_name:', code_to_name)
    return strategy_dict, news_dict, risk_dict, tech_dict, name_to_code, code_to_name

# forecast에서 종목별 예측 데이터 추출
def get_forecast_by_code(forecast_rows):
    forecast_map = {}
    for row in forecast_rows:
        code = row['종목코드']
        if code not in forecast_map:
            forecast_map[code] = []
        forecast_map[code].append({"date": row['예측일'], "price": row['예측가격']})
    return forecast_map

@app.route('/')
@app.route('/<stock_code>')
def dashboard(stock_code=None):
    holdings = load_holdings()
    forecast_rows = load_forecast()
    forecast_map = get_forecast_by_code(forecast_rows)
    md_path = get_latest_report()
    if not md_path:
        return "보고서 파일이 없습니다.", 404
    strategy_dict, news_dict, risk_dict, tech_dict, name_to_code, code_to_name = parse_md_report(md_path)
    # holdings가 종목코드 기반이면 종목명으로 변환
    holdings_named = {}
    for k, v in holdings.items():
        if k in code_to_name:
            holdings_named[code_to_name[k]] = v
        else:
            holdings_named[k] = v
    # 실제 종목명만 필터링 (사이드바 종목명만 표시)
    all_names = set(list(holdings_named.keys()) + list(strategy_dict.keys()) + list(risk_dict.keys()) + list(tech_dict.keys()) + list(news_dict.keys()))
    filtered_names = [n for n in all_names if n not in ["종목", "종목명", "현금", "-----------", "# 4. 기술적 분석", "종목코드", "구매개수", "투자금", "현재가", "", "-"] and not n.startswith("-")]
    print("DEBUG filtered_names:", filtered_names)
    print("DEBUG name_to_code:", name_to_code)
    print("DEBUG code_to_name:", code_to_name)
    stocks = []
    for name in filtered_names:
        holding_qty = holdings_named.get(name, 0)
        code = name_to_code.get(name)
        # name이 '신원 (009270)' 형태면 분리
        if not code:
            n, c = extract_name_code(name)
            if c:
                name, code = n, c
                name_to_code[name] = code
                code_to_name[code] = name
        print(f"DEBUG name: {name}, code: {code}")
        if not code or code == '-' or code == '':
            continue
        strategy = strategy_dict.get(name, {})
        risk = risk_dict.get(name, {})
        tech = tech_dict.get(name, {})
        news = news_dict.get(name, {})
        if not news:
            news = news_dict.get(code_to_name.get(code, ''), {})
        if not strategy:
            strategy = strategy_dict.get(code_to_name.get(code, ''), {})
        if not risk:
            risk = risk_dict.get(code_to_name.get(code, ''), {})
        if not tech:
            tech = tech_dict.get(code_to_name.get(code, ''), {})
        forecast = forecast_map.get(name, [])
        if not forecast:
            forecast = forecast_map.get(code, [])
        stock = {
            "code": code,
            "name": name,
            "holding_qty": holding_qty,
            "strategy": strategy,
            "news": news,
            "risk": risk,
            "tech": tech,
            "forecast": forecast
        }
        stocks.append(stock)
    # 종목 선택
    selected_stock = next((s for s in stocks if s["code"] == stock_code), stocks[0] if stocks else None)
    return render_template("dashboard_test.html",
        stocks=stocks,
        selected_stock=selected_stock,
        holdings=holdings
    )

if __name__ == "__main__":
    app.run(debug=True)
