# MCP-based Trading System

## 프로젝트 개요
MCP(Machine Conversation Protocol) 기반의 자동화된 트레이딩 시스템입니다. LLM을 활용하여 다양한 에이전트들과 통신하며 트레이딩 전략을 수립하고 실행합니다.

## 시스템 구조
```
trading_system/
├── agents/                     # 각 에이전트 서버
│   ├── strategy/              # 전략 수립 에이전트
│   ├── risk/                  # 리스크 관리 에이전트
│   ├── news/                  # 뉴스 분석 에이전트
│   ├── technical/            # 기술적 분석 에이전트
│   ├── xai/                  # Explainable AI 에이전트
│   └── trading/             # 거래 실행 에이전트
├── web_dashboard/            # 웹 인터페이스
│   ├── frontend/            # React 기반 프론트엔드
│   └── backend/             # FastAPI 기반 백엔드
├── config/                  # 설정 파일
├── utils/                   # 공통 유틸리티
├── tests/                   # 테스트 코드
└── docs/                    # 문서
```

## 주요 기능
1. 자연어 기반 트레이딩 전략 수립
2. 실시간 리스크 관리
3. 뉴스 및 시장 분석
4. 기술적 분석
5. 자동화된 거래 실행
6. XAI 기반 의사결정 설명
7. 실시간 모니터링 대시보드

## 설치 및 실행
1. 환경 설정
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. 환경 변수 설정
```bash
cp .env.example .env
# .env 파일에 필요한 설정 추가
```

3. 에이전트 서버 실행
```bash
python -m agents.strategy.main
python -m agents.risk.main
# ... 기타 에이전트 실행
```

4. 웹 대시보드 실행
```bash
cd web_dashboard/backend
uvicorn main:app --reload
cd ../frontend
npm start
```

## API 문서
- 각 에이전트의 API 문서는 `http://localhost:PORT/docs`에서 확인 가능
- Swagger UI를 통한 API 테스트 지원

## 개발 가이드
- [에이전트 개발 가이드](docs/agent_development.md)
- [API 통신 프로토콜](docs/mcp_protocol.md)
- [테스트 가이드](docs/testing.md)

## 라이선스
MIT License 

## Agents

### 1. News Analysis Agent
- Collects and analyzes news articles from various sources
- Performs sentiment analysis and impact assessment
- Generates trading signals based on news events
- Endpoint: `http://localhost:8003/news`

### 2. Technical Analysis Agent
- Analyzes market data using various technical indicators
- Provides comprehensive technical analysis including:
  - Moving Averages (SMA, EMA)
  - Oscillators (RSI, MACD, Stochastic)
  - Volume Indicators (OBV, MFI)
  - Support/Resistance Levels
  - Chart Pattern Detection
- Generates trading signals with entry/exit points
- Endpoint: `http://localhost:8004/analyze`

### 3. XAI (Explainable AI) Agent
- Provides detailed explanations for trading decisions
- Features include:
  - Feature importance analysis
  - Decision path visualization
  - Local and global explanations
  - Counterfactual analysis
  - Risk factor identification
- Supports multiple signal sources:
  - Technical analysis signals
  - News-based signals
  - Fundamental analysis signals
  - Market sentiment signals
- Endpoint: `http://localhost:8005/explain`

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Install TA-Lib:
```bash
# macOS
brew install ta-lib

# Ubuntu
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
sudo make install
```

3. Start the services:
```bash
# News Analysis Agent
uvicorn agents.news.api:app --host 0.0.0.0 --port 8003

# Technical Analysis Agent
uvicorn agents.technical.api:app --host 0.0.0.0 --port 8004

# XAI Agent
uvicorn agents.xai.api:app --host 0.0.0.0 --port 8005
```

## Usage Examples

### XAI Analysis
```python
import requests

# XAI analysis request
response = requests.post(
    "http://localhost:8005/explain",
    json={
        "signal_id": "tech_signal_001",
        "source": "technical",
        "analysis_type": "comprehensive",
        "additional_context": {
            "ticker": "AAPL",
            "timeframe": "1d"
        }
    }
)

# Parse response
result = response.json()
print(result['summary'])  # Print analysis summary
print(result['recommendations'])  # Print recommendations
```

## API Documentation

### XAI Agent

#### POST /explain
Generates explanations for trading signals.

Request Body:
```json
{
    "signal_id": "tech_signal_001",
    "source": "technical",
    "analysis_type": "comprehensive",
    "additional_context": {
        "ticker": "AAPL",
        "timeframe": "1d"
    }
}
```

Response:
```json
{
    "request_id": "uuid",
    "signal_explanation": {
        "signal_id": "tech_signal_001",
        "source": "technical",
        "signal_strength": 0.75,
        "confidence_score": 0.8,
        "key_drivers": [...],
        "risk_factors": [...],
        "alternative_scenarios": [...],
        "local_explanation": {...},
        "supporting_evidence": {...}
    },
    "global_explanation": {
        "feature_importance_ranking": [...],
        "feature_interactions": [...],
        "model_behavior_summary": "...",
        "performance_metrics": {...}
    },
    "confidence_metrics": {...},
    "visualization_data": {...},
    "recommendations": [...],
    "summary": "..."
}
``` 