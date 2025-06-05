import os
import re
import markdown
from flask import Flask, render_template

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MD_PATH = os.path.join(BASE_DIR, "../../investment_reports/investment_report_portfolio_20250601_202224.md")

def load_holdings_from_txt(path="../../data/stock_state.txt"):
    holdings = []
    reading = False
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("보유 종목:"):
                reading = True
                continue
            if reading and line:
                try:
                    name_code_part = line.split(" - ")[0]
                    name = name_code_part.split(" (")[0]
                    code = name_code_part.split("(")[1].rstrip(")")
                    holdings.append({"code": code, "name": name})
                except:
                    continue
    return holdings

def parse_md_report(md_path=MD_PATH):
    with open(md_path, encoding="utf-8") as f:
        content = f.read()

    report_sections = {}

    # 1. 전략 분석 섹션 파싱
    strategy_section = content.split("2. **뉴스/이슈 분석**")[0]
    rows = re.findall(r'\|\s*(.*?)\s*\(([A-Z0-9]+)\)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|', strategy_section)

    for name, code, strategy, description in rows[1:]:
        report_sections[code] = {
            "name": name.strip(),
            "code": code.strip(),
            "strategy": strategy.strip(),
            "description": description.strip()
        }

    return report_sections

@app.route('/')
@app.route('/<stock_code>')

def dashboard(stock_code=None):
    holdings = load_holdings_from_txt()
    md_data = parse_md_report()
    username = load_username_from_txt()

    holding_codes = set(s["code"] for s in holdings)
    extra_stocks = [{"code": k, "name": v["name"]} for k, v in md_data.items() if k not in holding_codes]

    all_stocks = holdings + extra_stocks
    selected_stock = next((s for s in all_stocks if s["code"] == stock_code), holdings[0] if holdings else None)

    if selected_stock is None:
        return "선택한 종목을 찾을 수 없습니다", 404

    strategy = strategy_analysis(selected_stock)
    issues = issues_analysis(selected_stock)
    risk = risk_analysis(selected_stock)
    technical = technical_analysis(selected_stock)

    return render_template("dashboard_test.html",
                       username=username,
                       stocks=holdings,
                       extra_stocks=extra_stocks,
                       selected_stock=selected_stock,
                       strategy_analysis=strategy,
                       issues_analysis = issues,
                       risk_analysis = risk,
                       technical_analysis = technical,
                       timeframe_analysis = None,
                       total_opinion = None,

                       analysis={
                           "strategy": "정보 없음",
                           "news": {"title": "-", "sensitivity": 0, "sentiment": 0},
                           "risk": {"level": "-", "score": 0, "volatility": 0},
                           "technical": {
                               "rsi": "-", "macd": "-", "bollinger_upper": "-",
                               "bollinger_lower": "-", "recent_price": "-"
                           },
                           "opinion": []
                       })

def load_username_from_txt(path="../../data/stock_state.txt"):
    with open(path, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()
        if first_line.startswith("사용자 이름:"):
            return first_line.split(":", 1)[1].strip()
    return "이름없음"

def strategy_analysis(md_data, selected_stock):
    code = selected_stock["code"]
    strategy_info = md_data.get(code)

    if strategy_info:
        return f"""
        <p><strong>전략:</strong> {strategy_info['strategy']}</p>
        <p><strong>설명:</strong> {strategy_info['description']}</p>
        """
    else:
        return "<p><em>정보 없음</em></p>"
    
def strategy_analysis(selected_stock):
    code = selected_stock["code"]
    with open(MD_PATH, encoding="utf-8") as f:
        content = f.read()

    strategy_section = content.split("2. **뉴스/이슈 분석**")[0]
    rows = re.findall(r'\|\s*(.*?)\s*\(([A-Z0-9]+)\)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|', strategy_section)

    for name, found_code, strategy, description in rows[1:]:
        if found_code == code:
            return f"""
            <p><strong>전략:</strong> {strategy.strip()}</p>
            <p><strong>설명:</strong> {description.strip()}</p>
            """
    return "<p><em>정보 없음</em></p>"

def issues_analysis(selected_stock):
    code = selected_stock["code"]
    with open(MD_PATH, encoding="utf-8") as f:
        content = f.read()

    pattern = r'\|\s*(.*?)\s*\(([A-Z0-9]+)\)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*(.*?)\s*\|'
    rows = re.findall(pattern, content)

    for name, found_code, sentiment, impact, keywords in rows:
        if found_code == code:
            return f"""
            <p><strong>뉴스 제목:</strong> {keywords}</p>
            <p><strong>감성점수:</strong> {sentiment}, <strong>시장영향도:</strong> {impact}</p>
            """
    return "<p><em>정보 없음</em></p>"


def risk_analysis(selected_stock):
    code = selected_stock["code"]
    with open(MD_PATH, encoding="utf-8") as f:
        content = f.read()

    pattern = r"\|\s*(.*?)\s*\(([A-Z0-9]+)\)\s*\| (\d+) \| ([0-9.]+%) \| (\S+) \|"
    rows = re.findall(pattern, content)

    for name, found_code, score, volatility, level in rows:
        if found_code == code:
            return f"""
            <p><strong>리스크 점수:</strong> {score}</p>
            <p><strong>변동성:</strong> {volatility}, <strong>레벨:</strong> {level}</p>
            """
    return "<p><em>정보 없음</em></p>"


def technical_analysis(selected_stock):
    code = selected_stock["code"]
    with open(MD_PATH, encoding="utf-8") as f:
        content = f.read()

    pattern = r"\|\s*(.*?)\s*\(([A-Z0-9]+)\)\s*\| (\d+) \| ([0-9.]+) \| (\d+) \| (\d+) \| (\d+) \|"
    rows = re.findall(pattern, content)

    for name, found_code, rsi, macd, upper, lower, close in rows:
        if found_code == code:
            return f"""
            <p><strong>RSI:</strong> {rsi}, <strong>MACD:</strong> {macd}</p>
            <p><strong>볼린저밴드:</strong> 상단 {upper}, 하단 {lower}</p>
            <p><strong>최근 가격:</strong> {close}</p>
            """
    return "<p><em>정보 없음</em></p>"



if __name__ == "__main__":
    app.run(debug=True)
