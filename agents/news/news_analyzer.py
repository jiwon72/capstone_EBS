import uuid
import openai
import os
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import requests
import yfinance as yf
from newsapi import NewsApiClient
from transformers import pipeline
from .models import (
    NewsSource, NewsCategory, SentimentScore,
    NewsArticle, EntityMention, MarketImpact,
    NewsAnalysis, NewsAnalysisRequest, NewsAnalysisResponse
)

class NewsAnalyzer:
    def __init__(self, is_test: bool = False):
        # 테스트 환경에서는 더미 API 키 사용
        news_api_key = "test_key" if is_test else os.getenv("NEWS_API_KEY", "your-news-api-key")
        self.news_api = NewsApiClient(api_key=news_api_key)
        self.sentiment_analyzer = pipeline("sentiment-analysis")
        
        # 테스트 환경에서는 OpenAI 클라이언트 초기화하지 않음
        if not is_test:
            self.openai_client = openai.OpenAI()
        else:
            self.openai_client = None
        
    def _fetch_news_articles(
        self,
        tickers: List[str],
        time_range: str,
        sources: Optional[List[NewsSource]] = None,
        categories: Optional[List[NewsCategory]] = None
    ) -> List[NewsArticle]:
        """뉴스 기사 수집"""
        articles = []
        
        # 시간 범위 계산
        if time_range == "1d":
            from_date = datetime.now() - timedelta(days=1)
        elif time_range == "7d":
            from_date = datetime.now() - timedelta(days=7)
        else:  # 30d
            from_date = datetime.now() - timedelta(days=30)
            
        # NewsAPI를 통한 뉴스 수집
        for ticker in tickers:
            company = yf.Ticker(ticker)
            company_name = company.info.get('longName', ticker)
            
            news = self.news_api.get_everything(
                q=f"{company_name} OR {ticker}",
                from_param=from_date.strftime('%Y-%m-%d'),
                language='en',
                sort_by='relevancy'
            )
            
            for article in news['articles']:
                # 기사 관련성 및 감성 분석
                relevance_score = self._calculate_relevance_score(article['title'], article['description'], ticker)
                sentiment = self._analyze_sentiment(article['title'], article['description'])
                
                # 카테고리 분류
                category = self._classify_news_category(article['title'], article['description'])
                
                if sources and NewsSource(article['source']['name'].lower()) not in sources:
                    continue
                    
                if categories and category not in categories:
                    continue
                
                articles.append(NewsArticle(
                    article_id=str(uuid.uuid4()),
                    title=article['title'],
                    content=article['description'],
                    source=NewsSource(article['source']['name'].lower()),
                    url=article['url'],
                    published_at=datetime.strptime(article['publishedAt'], '%Y-%m-%dT%H:%M:%SZ'),
                    category=category,
                    tickers=[ticker],
                    sentiment=sentiment,
                    relevance_score=relevance_score
                ))
                
        return articles
    
    def _calculate_relevance_score(self, title: str, content: str, ticker: str) -> float:
        """기사 관련성 점수 계산"""
        # 간단한 관련성 점수 계산 로직
        text = f"{title} {content}".lower()
        ticker_count = text.count(ticker.lower())
        company = yf.Ticker(ticker)
        company_name = company.info.get('longName', '').lower()
        company_count = text.count(company_name)
        
        score = min((ticker_count + company_count) / 10, 1.0)
        return score
    
    def _analyze_sentiment(self, title: str, content: str) -> SentimentScore:
        """감성 분석"""
        text = f"{title} {content}"
        result = self.sentiment_analyzer(text)[0]
        
        # 감성 점수를 SentimentScore로 변환
        score = result['score']
        if result['label'] == 'POSITIVE':
            if score > 0.8:
                return SentimentScore.VERY_POSITIVE
            return SentimentScore.POSITIVE
        elif result['label'] == 'NEGATIVE':
            if score > 0.8:
                return SentimentScore.VERY_NEGATIVE
            return SentimentScore.NEGATIVE
        return SentimentScore.NEUTRAL
    
    def _classify_news_category(self, title: str, content: str) -> NewsCategory:
        """뉴스 카테고리 분류"""
        text = f"{title} {content}".lower()
        
        # 간단한 규칙 기반 분류
        if any(word in text for word in ['earnings', 'revenue', 'profit', 'loss']):
            return NewsCategory.EARNINGS
        elif any(word in text for word in ['merger', 'acquisition', 'takeover']):
            return NewsCategory.MERGER_ACQUISITION
        elif any(word in text for word in ['market', 'index', 'trading']):
            return NewsCategory.MARKET_MOVEMENT
        elif any(word in text for word in ['gdp', 'inflation', 'employment']):
            return NewsCategory.ECONOMIC_INDICATOR
        elif any(word in text for word in ['regulation', 'compliance', 'sec']):
            return NewsCategory.REGULATORY_NEWS
        elif any(word in text for word in ['industry', 'sector']):
            return NewsCategory.INDUSTRY_NEWS
        elif any(word in text for word in ['geopolitical', 'political', 'government']):
            return NewsCategory.GEOPOLITICAL
        else:
            return NewsCategory.COMPANY_NEWS
    
    def _extract_entities(self, article: NewsArticle) -> List[EntityMention]:
        """엔티티 추출 및 분석"""
        prompt = f"""
        Extract key entities from the following news article and analyze their sentiment:
        Title: {article.title}
        Content: {article.content}
        
        Return the entities with their types (COMPANY, PERSON, LOCATION) and sentiment.
        """
        
        response = self.openai_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        
        # GPT 응답 파싱 및 엔티티 생성
        entities = []
        # 실제 구현에서는 GPT 응답을 파싱하여 EntityMention 객체 생성
        
        return entities
    
    def _analyze_market_impact(
        self,
        article: NewsArticle,
        entities: List[EntityMention]
    ) -> MarketImpact:
        """시장 영향 분석"""
        prompt = f"""
        Analyze the potential market impact of the following news:
        Title: {article.title}
        Content: {article.content}
        Category: {article.category}
        Sentiment: {article.sentiment}
        
        Consider:
        1. Impact level (-1 to 1)
        2. Affected sectors
        3. Time horizon
        4. Key drivers
        """
        
        response = self.openai_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        
        # GPT 응답 파싱 및 MarketImpact 객체 생성
        # 실제 구현에서는 더 정교한 파싱 로직 필요
        impact = MarketImpact(
            impact_level=0.5,
            confidence_score=0.8,
            affected_sectors=["TECHNOLOGY"],
            affected_tickers=article.tickers,
            time_horizon="SHORT_TERM",
            key_drivers=["Market Sentiment", "Company Performance"]
        )
        
        return impact
    
    def analyze_news(self, request: NewsAnalysisRequest) -> NewsAnalysisResponse:
        """뉴스 분석 메인 로직"""
        # 1. 뉴스 기사 수집
        articles = self._fetch_news_articles(
            request.tickers,
            request.time_range,
            request.sources,
            request.categories
        )
        
        # 2. 각 기사 분석
        analyzed_articles = []
        for article in articles:
            if article.relevance_score < request.min_relevance_score:
                continue
                
            # 엔티티 추출
            entities = self._extract_entities(article)
            
            # 시장 영향 분석
            market_impact = self._analyze_market_impact(article, entities)
            
            # 분석 결과 생성
            analysis = NewsAnalysis(
                article=article,
                entities=entities,
                market_impact=market_impact,
                key_takeaways=self._generate_key_takeaways(article, market_impact),
                trading_signals=self._generate_trading_signals(article, market_impact)
            )
            analyzed_articles.append(analysis)
        
        # 3. 전체 분석 결과 생성
        return NewsAnalysisResponse(
            analyzed_articles=analyzed_articles,
            overall_sentiment=self._calculate_overall_sentiment(analyzed_articles),
            major_events=self._extract_major_events(analyzed_articles),
            trading_implications=self._generate_trading_implications(analyzed_articles),
            risk_factors=self._identify_risk_factors(analyzed_articles),
            summary=self._generate_summary(analyzed_articles)
        )
    
    def _generate_key_takeaways(
        self,
        article: NewsArticle,
        market_impact: MarketImpact
    ) -> List[str]:
        """주요 시사점 생성"""
        prompt = f"""
        Generate key takeaways from the following news:
        Title: {article.title}
        Content: {article.content}
        Impact: {market_impact.impact_level}
        """
        
        response = self.openai_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        
        # 실제 구현에서는 GPT 응답을 파싱하여 key takeaways 리스트 생성
        return ["Market sentiment is positive", "Strong growth potential"]
    
    def _generate_trading_signals(
        self,
        article: NewsArticle,
        market_impact: MarketImpact
    ) -> List[Dict]:
        """트레이딩 신호 생성"""
        signals = []
        
        if market_impact.impact_level > 0.5:
            signals.append({
                "ticker": article.tickers[0],
                "action": "BUY",
                "confidence": market_impact.confidence_score,
                "reason": "Strong positive news impact"
            })
        elif market_impact.impact_level < -0.5:
            signals.append({
                "ticker": article.tickers[0],
                "action": "SELL",
                "confidence": market_impact.confidence_score,
                "reason": "Strong negative news impact"
            })
            
        return signals
    
    def _calculate_overall_sentiment(self, analyses: List[NewsAnalysis]) -> SentimentScore:
        """전체 감성 수준 계산"""
        if not analyses:
            return SentimentScore.NEUTRAL
            
        sentiment_scores = {
            SentimentScore.VERY_NEGATIVE: -2,
            SentimentScore.NEGATIVE: -1,
            SentimentScore.NEUTRAL: 0,
            SentimentScore.POSITIVE: 1,
            SentimentScore.VERY_POSITIVE: 2
        }
        
        weighted_sum = sum(
            sentiment_scores[a.sentiment] * 0.8  # 가중치 대신 임의의 값 사용
            for a in analyses
        )
        total_weight = len(analyses) * 0.8  # 가중치의 합
        
        avg_score = weighted_sum / total_weight if total_weight > 0 else 0
        
        if avg_score > 1:
            return SentimentScore.VERY_POSITIVE
        elif avg_score > 0:
            return SentimentScore.POSITIVE
        elif avg_score < -1:
            return SentimentScore.VERY_NEGATIVE
        elif avg_score < 0:
            return SentimentScore.NEGATIVE
        else:
            return SentimentScore.NEUTRAL
    
    def _extract_major_events(self, analyses: List[NewsAnalysis]) -> List[Dict]:
        """주요 이벤트 추출"""
        events = []
        for analysis in analyses:
            if analysis.market_impact.impact_level > 0.7 or analysis.market_impact.impact_level < -0.7:
                events.append({
                    "title": analysis.article.title,
                    "category": analysis.article.category,
                    "impact_level": analysis.market_impact.impact_level,
                    "affected_tickers": analysis.market_impact.affected_tickers
                })
        return events
    
    def _generate_trading_implications(self, analyses: List[NewsAnalysis]) -> List[Dict]:
        """트레이딩 시사점 생성"""
        implications = []
        for analysis in analyses:
            implications.extend(analysis.trading_signals)
        return implications
    
    def _identify_risk_factors(self, analyses: List[NewsAnalysis]) -> List[Dict]:
        """리스크 요인 식별"""
        risk_factors = []
        for analysis in analyses:
            if analysis.market_impact.impact_level < -0.3:
                risk_factors.append({
                    "factor": analysis.article.title,
                    "severity": abs(analysis.market_impact.impact_level),
                    "affected_sectors": analysis.market_impact.affected_sectors,
                    "time_horizon": analysis.market_impact.time_horizon
                })
        return risk_factors
    
    def _generate_summary(self, analyses: List[NewsAnalysis]) -> str:
        """전체 분석 결과 요약"""
        if not analyses:
            return "No relevant news articles found."
            
        summary = f"Analyzed {len(analyses)} news articles.\n\n"
        
        # 주요 이벤트 요약
        major_events = self._extract_major_events(analyses)
        if major_events:
            summary += "Major Events:\n"
            for event in major_events[:3]:  # Top 3 events
                summary += f"- {event['title']}\n"
                
        # 전체 감성
        overall_sentiment = self._calculate_overall_sentiment(analyses)
        summary += f"\nOverall Market Sentiment: {overall_sentiment.value}\n"
        
        # 트레이딩 시사점
        implications = self._generate_trading_implications(analyses)
        if implications:
            summary += "\nKey Trading Implications:\n"
            for imp in implications[:3]:  # Top 3 implications
                summary += f"- {imp['action']} {imp['ticker']}: {imp['reason']}\n"
                
        return summary

    async def analyze(self, request: NewsAnalysisRequest) -> NewsAnalysisResponse:
        """뉴스 분석을 수행하고 결과를 반환합니다."""
        # TODO: 실제 뉴스 분석 로직 구현
        # 현재는 더미 데이터를 반환
        
        article = NewsArticle(
            article_id=str(uuid.uuid4()),
            title="Sample News Article",
            content="This is a sample news article content for testing purposes.",
            source=NewsSource.REUTERS,
            url="https://example.com/news/1",
            published_at=datetime.now() - timedelta(hours=1),
            author="John Doe",
            categories=["Finance", "Technology"],
            tickers=[request.ticker]
        )
        
        analysis = NewsAnalysis(
            article=article,
            sentiment=SentimentScore.POSITIVE,
            sentiment_score=0.75,
            key_topics=["Earnings", "Growth", "Innovation"],
            named_entities={
                "COMPANY": [request.ticker],
                "PERSON": ["John Doe"],
                "LOCATION": ["New York"]
            },
            summary="Positive news about company's performance",
            impact_analysis={
                "market_impact": "POSITIVE",
                "confidence": 0.8,
                "factors": ["Strong earnings", "Market expansion"]
            }
        )
        
        return NewsAnalysisResponse(
            ticker=request.ticker,
            articles=[article],
            analysis=[analysis],
            aggregated_sentiment=SentimentScore.POSITIVE,
            average_sentiment_score=0.75,
            key_findings="Overall positive sentiment based on recent news coverage."
        )