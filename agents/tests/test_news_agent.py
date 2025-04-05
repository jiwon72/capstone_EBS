import pytest
from fastapi.testclient import TestClient
from datetime import datetime, timedelta
import uuid
from agents.news.api import app
from agents.news.models import NewsArticle, NewsSource, NewsCategory

client = TestClient(app)

def test_news_analysis_endpoint():
    """뉴스 분석 엔드포인트 테스트"""
    request_data = {
        "ticker": "AAPL",
        "start_date": (datetime.now() - timedelta(days=1)).isoformat(),
        "end_date": datetime.now().isoformat(),
        "sources": ["reuters", "bloomberg"],
        "limit": 10
    }

    response = client.post("/news", json=request_data)
    assert response.status_code == 200
    
    data = response.json()
    assert "request_timestamp" in data
    assert "ticker" in data
    assert "articles" in data
    assert "analysis" in data
    assert "aggregated_sentiment" in data
    assert "average_sentiment_score" in data
    assert "key_findings" in data

def test_news_sentiment_analysis():
    """뉴스 감성 분석 테스트"""
    article = NewsArticle(
        article_id=str(uuid.uuid4()),
        title="Company XYZ Reports Strong Q4 Earnings",
        content="The company reported earnings that exceeded analyst expectations...",
        source=NewsSource.REUTERS,
        url="https://example.com/news/1",
        published_at=datetime.now(),
        author="John Doe",
        categories=["earnings"],
        tickers=["XYZ"]
    )

    request_data = {
        "ticker": "XYZ",
        "start_date": (datetime.now() - timedelta(days=1)).isoformat(),
        "end_date": datetime.now().isoformat(),
        "limit": 1
    }

    response = client.post("/news", json=request_data)
    assert response.status_code == 200
    
    data = response.json()
    assert len(data["articles"]) > 0
    assert len(data["analysis"]) > 0
    assert "sentiment" in data["analysis"][0]
    assert "sentiment_score" in data["analysis"][0]

def test_invalid_request():
    """잘못된 요청 테스트"""
    invalid_data = {
        "ticker": "INVALID",
        "start_date": "invalid_date",
        "end_date": "invalid_date"
    }

    response = client.post("/news", json=invalid_data)
    assert response.status_code == 422  # Validation Error
    
    data = response.json()
    assert "detail" in data

def test_news_relevance_filtering():
    """뉴스 관련성 필터링 테스트"""
    request_data = {
        "ticker": "AAPL",
        "start_date": (datetime.now() - timedelta(days=1)).isoformat(),
        "end_date": datetime.now().isoformat(),
        "limit": 10
    }

    response = client.post("/news", json=request_data)
    assert response.status_code == 200
    
    data = response.json()
    assert len(data["articles"]) <= request_data["limit"]

def test_market_impact_analysis():
    """시장 영향 분석 테스트"""
    article = NewsArticle(
        article_id=str(uuid.uuid4()),
        title="Major Tech Company Announces Layoffs",
        content="The company announced significant workforce reduction...",
        source=NewsSource.BLOOMBERG,
        url="https://example.com/news/2",
        published_at=datetime.now(),
        author="Jane Smith",
        categories=["company_news"],
        tickers=["TECH"]
    )

    request_data = {
        "ticker": "TECH",
        "start_date": (datetime.now() - timedelta(days=1)).isoformat(),
        "end_date": datetime.now().isoformat(),
        "limit": 1
    }

    response = client.post("/news", json=request_data)
    assert response.status_code == 200
    
    data = response.json()
    assert len(data["analysis"]) > 0
    analysis = data["analysis"][0]
    assert "impact_analysis" in analysis
    assert "market_impact" in analysis["impact_analysis"]
    assert "confidence" in analysis["impact_analysis"]
    assert "factors" in analysis["impact_analysis"] 