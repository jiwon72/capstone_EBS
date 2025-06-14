# Debate 기반 멀티에이전트 자동매매 시스템

## 프로젝트 개요
OpenAI 기반 LLM과 뉴스/전략/리스크/기술적 분석 등 다양한 에이전트가 debate(토론)하여, 종목별 핵심지표·추천·신뢰도·전문가 설명을 종합해 자동으로 포트폴리오를 결정하고, 실시간 대시보드로 시각화하는 자동매매 파이프라인입니다.

## 시스템 구조
```
capstone_EBS_project/
├── agents/                # 각종 분석 에이전트 (뉴스, 전략, 리스크, 기술적 등)
├── system_manager.py      # debate 결과 종합 및 포트폴리오 결정
├── web_dashboard/        # 대시보드 (backend: Flask/FastAPI, frontend: React)
├── artifacts/            # debate_logs, forecasts 등 자동 생성 폴더
├── holdings.json         # 전일 보유 종목/수량 자동 관리
├── requirements.txt      # 의존성 관리
└── investment_report_*.md # 일별 투자 리포트 (자동 생성)
```

## 주요 기능
- debate 기반 멀티에이전트(뉴스/전략/리스크/기술적) 종목 분석
- OpenAI API 기반 전문가 설명 및 한글 답변 유도
- holdings.json, forecast_YYYYMMDD.csv, investment_report_*.md 등 아티팩트 자동 관리
- robust 종목명-코드 매칭 및 파싱, NameError 등 버그 자동 방지
- 실시간 대시보드에서 종목별 분석/예측/보유수량/카드 시각화
- 자동매매/트레이딩 실전 구조 (보유종목+상위종목 동시분석, 자동 리밸런싱)

## 설치 및 실행
1. **의존성 설치**
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install --upgrade pip wheel
pip install -r requirements.txt
```
- FinanceDataReader 설치 오류 시:
  - pip 최신화, wheel 설치, 또는
  - `pip install git+https://github.com/FinanceData/FinanceDataReader.git`

2. **환경 변수 설정**
```bash
cp .env.example .env
# .env 파일에 OPENAI_API_KEY 등 필수 항목 입력
```

3. **시스템 실행**
```bash
# 메인 시스템 (debate 및 자동매매)
python system_manager.py

# (에이전트별 별도 실행 필요 시)
python -m agents.strategy.main
python -m agents.risk.main
# ... 기타 에이전트 실행
```

4. **대시보드 실행**
```bash
cd web_dashboard/backend
python app_test.py  # 또는 uvicorn main:app --reload
cd ../frontend
npm install && npm start
```

## 아티팩트/데이터 관리
- **debate_logs/** : debate 결과 자동 저장
- **forecasts/** : forecast_YYYYMMDD.csv (시계열 예측)
- **holdings.json** : 전일 보유 종목/수량 자동 관리
- **investment_report_*.md** : 일별 투자 리포트 (분석/포트폴리오 표 포함)
- 폴더/파일은 자동 생성 및 관리됨

## robust 종목명-코드 매칭
- 종목명/코드 혼재 표기, 마크다운 파싱, 대시보드 매칭 등에서 robust하게 동작
- NameError, 데이터 미표시 등 버그 자동 방지

## 대시보드 주요 화면
- 종목별 보유수량, 전략/뉴스/리스크/기술적 분석, 시계열 예측을 카드/테이블로 시각화
- 종목명만/종목명+코드 혼재 표기 모두 지원
- "표시할 종목 데이터가 없습니다." 등 안내 메시지 안전 처리

## 예시: debate 기반 자동 포트폴리오 결정
```python
from system_manager import SystemManager
sm = SystemManager()
sm.run_debate_and_decide_portfolio()
# holdings.json, investment_report_*.md, forecasts/ 등 자동 생성
```

## 참고/주의사항
- requirements.txt, pandas, FinanceDataReader 등 의존성 최신화 필요
- OPENAI_API_KEY 등 환경변수 필수
- artifacts/ 폴더는 자동 생성되며, 별도 관리 필요 없음
- 대시보드 연동 시 최신 report/forecast/holdings 파일 자동 반영

## 개발/테스트 가이드
- [에이전트 개발 가이드](docs/agent_development.md)
- [API 통신 프로토콜](docs/mcp_protocol.md)
- [테스트 가이드](docs/testing.md)

