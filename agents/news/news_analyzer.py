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
from models import (
    NewsSource, NewsCategory, SentimentScore,
    NewsArticle, EntityMention, MarketImpact,
    NewsAnalysis, NewsAnalysisRequest, NewsAnalysisResponse
)

class NewsAnalyzer:
    def __init__(self, is_test: bool = False):
        """뉴스 분석기 초기화"""
        # News API 키 설정
        news_api_key = os.getenv('NEWS_API_KEY', "04942b7e848547bc9a08c9d50cb688ff")
        if is_test:
            news_api_key = "test_key"
            
        self.newsapi = NewsApiClient(api_key=news_api_key)
        
        # 주요 한국 기업 및 산업 키워드 설정
        self.market_keywords = {
            'general_market': ['코스피', 'KOSPI', '코스닥', 'KOSDAQ', '한국거래소', 'KRX'],
            'industries': ['반도체', '전기차', 'EV', '2차전지', '바이오', '제약', '금융', '자동차', 'IT'],
            'major_companies': ['삼성전자', 'SK하이닉스', 'LG에너지솔루션', '삼성바이오로직스', '삼성SDI', 
                              'LG화학', '현대차', '기아', '네이버', '카카오']
        }
        
        # OpenAI 클라이언트 초기화
        if not is_test:
            self.openai_client = openai.OpenAI()
        else:
            self.openai_client = None
        
        self.sentiment_analyzer = pipeline("sentiment-analysis", model="snunlp/KR-FinBert-SC")
        
    def _fetch_news_articles(self, time_range: str = "1d") -> List[NewsArticle]:
        """뉴스 기사 수집"""
        articles = []
        
        # 시간 범위 설정
        end_date = datetime.now()
        if time_range == "1d":
            start_date = end_date - timedelta(days=1)
        elif time_range == "7d":
            start_date = end_date - timedelta(days=7)
        else:  # 30d
            start_date = end_date - timedelta(days=30)
            
        try:
            # 1. 한국 비즈니스 뉴스 (top-headlines)
            kr_response = self.newsapi.get_top_headlines(
                country='kr',
                category='business'
            )
            
            if kr_response.get('status') == 'ok':
                print(f"Found {len(kr_response.get('articles', []))} Korean business headlines")
                self._process_articles(kr_response.get('articles', []), articles, start_date, 'MARKET')
            
            # 2. 산업별/기업별 뉴스 (everything)
            search_terms = (
                self.market_keywords['general_market'] +
                self.market_keywords['industries'] +
                self.market_keywords['major_companies']
            )
            
            for term in search_terms:
                try:
                    response = self.newsapi.get_everything(
                        q=term,
                        from_param=start_date.strftime("%Y-%m-%d"),
                        to=end_date.strftime("%Y-%m-%d"),
                        sort_by='relevancy'
                    )
                    
                    if response.get('status') == 'ok':
                        print(f"Found {len(response.get('articles', []))} articles for term: {term}")
                        self._process_articles(response.get('articles', []), articles, start_date, term)
                        
                except Exception as e:
                    print(f"Error fetching news for term {term}: {str(e)}")
                    continue
                    
        except Exception as e:
            print(f"Error fetching news: {str(e)}")
            
        print(f"Final filtered articles count: {len(articles)}")
        return articles
        
    def _process_articles(self, raw_articles: List[Dict], articles: List[NewsArticle], 
                         start_date: datetime, category: str) -> None:
        """기사 처리 및 필터링"""
        for article in raw_articles:
            try:
                # 필수 필드 확인
                if not all(field in article for field in ['title', 'url', 'publishedAt']):
                    print(f"Skipping article due to missing fields: {article.get('title', 'No title')}")
                    continue
                    
                # 발행일 확인
                published_at = datetime.strptime(article['publishedAt'], "%Y-%m-%dT%H:%M:%SZ")
                if published_at < start_date:
                    print(f"Skipping article due to old date: {article['title']}")
                    continue
                
                # 기사 소스 매핑
                source_name = article.get('source', {}).get('name', '').lower()
                source = self._map_source(source_name)
                
                # 내용이 없는 경우 description으로 대체
                content = article.get('content', '') or article.get('description', '')
                if not content:
                    print(f"Skipping article due to no content: {article['title']}")
                    continue
                
                # 관련성 점수 계산
                relevance_score = self._calculate_relevance_score(
                    article['title'],
                    content,
                    category
                )
                
                # 기사 추가 (관련성 점수 필터링은 analyze_news에서 수행)
                news_article = NewsArticle(
                    article_id=str(uuid.uuid4()),
                    title=article['title'],
                    url=article['url'],
                    source=source,
                    published_at=published_at,
                    content=content,
                    relevance_score=relevance_score,
                    tickers=[]  # 추후 티커 매핑 기능 추가
                )
                articles.append(news_article)
                print(f"Added article: {article['title']} (relevance: {relevance_score:.2f})")
                    
            except Exception as e:
                print(f"Error processing article: {str(e)}")
                continue
                
    def _map_source(self, source_name: str) -> NewsSource:
        """뉴스 소스 매핑"""
        source_mapping = {
            'reuters': NewsSource.REUTERS,
            'bloomberg': NewsSource.BLOOMBERG,
            'cnbc': NewsSource.CNBC,
            'financial times': NewsSource.FINANCIAL_TIMES,
            'ft.com': NewsSource.FINANCIAL_TIMES,
            'wall street journal': NewsSource.WALL_STREET_JOURNAL,
            'wsj': NewsSource.WALL_STREET_JOURNAL
        }
        
        for key, value in source_mapping.items():
            if key in source_name:
                return value
        return NewsSource.OTHER
        
    def _calculate_relevance_score(self, title: str, content: str, category: str) -> float:
        """기사 관련성 점수 계산"""
        text = f"{title} {content}".lower()
        
        # 카테고리별 키워드 매칭
        if category == 'MARKET':
            keywords = self.market_keywords['general_market']
        elif category in self.market_keywords['industries']:
            keywords = [category]
        elif category in self.market_keywords['major_companies']:
            keywords = [category]
        else:
            # KOSPI/KOSDAQ 관련 키워드
            if category in ['KOSPI', 'KOSDAQ', 'KRX']:
                keywords = self.market_keywords['general_market']
            # 산업 관련 영문 키워드
            elif category == 'EV':
                keywords = ['전기차', 'EV', '전기자동차']
            elif category == 'IT':
                keywords = ['IT', '기술', '소프트웨어', '인터넷']
            else:
                keywords = [category]
        
        # 키워드 매칭 점수 계산
        matches = 0
        for keyword in keywords:
            if keyword.lower() in text:
                # 제목에서 발견되면 더 높은 점수
                if keyword.lower() in title.lower():
                    matches += 1.5
                else:
                    matches += 1.0
        
        # 최종 점수 계산 (0.3 이상이면 관련 기사로 판단)
        score = matches / (len(keywords) * 1.5) if keywords else 0
        return min(max(score, 0.3), 1.0)  # 최소 0.3, 최대 1.0
    
    def _analyze_sentiment(self, title: str, content: str) -> SentimentScore:
        """감성 분석"""
        text = f"{title} {content}"
        result = self.sentiment_analyzer(text)[0]
        
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
        
        if any(word in text for word in ['earnings', 'revenue', 'profit']):
            return NewsCategory.EARNINGS
        elif any(word in text for word in ['merger', 'acquisition', 'M&A']):
            return NewsCategory.MERGER_ACQUISITION
        elif any(word in text for word in ['market', 'index', 'trading']):
            return NewsCategory.MARKET_MOVEMENT
        elif any(word in text for word in ['gdp', 'inflation', 'employment']):
            return NewsCategory.ECONOMIC_INDICATOR
        elif any(word in text for word in ['regulation', 'sec', 'law']):
            return NewsCategory.REGULATORY_NEWS
        elif any(word in text for word in ['industry', 'sector']):
            return NewsCategory.INDUSTRY_NEWS
        elif any(word in text for word in ['government', 'policy', 'political']):
            return NewsCategory.GEOPOLITICAL
        else:
            return NewsCategory.COMPANY_NEWS
    
    def _extract_entities(self, article: NewsArticle) -> List[EntityMention]:
        """엔티티 추출"""
        entities = []
        return entities
    
    def _analyze_market_impact(self, article: NewsArticle, entities: List[EntityMention]) -> MarketImpact:
        """시장 영향 분석"""
        # 기사 내용에서 영향받는 섹터 분석
        affected_sectors = []
        text = f"{article.title} {article.content}".lower()
        
        # 산업 키워드 매칭
        sector_keywords = {
            "TECHNOLOGY": ["반도체", "it", "소프트웨어", "인공지능", "ai", "클라우드"],
            "ENERGY": ["2차전지", "배터리", "전기차", "ev", "신재생에너지"],
            "FINANCE": ["금융", "은행", "보험", "증권", "투자"],
            "HEALTHCARE": ["바이오", "제약", "의료", "healthcare"],
            "AUTO": ["자동차", "모빌리티", "완성차"],
            "RETAIL": ["유통", "소매", "이커머스", "retail"]
        }
        
        for sector, keywords in sector_keywords.items():
            if any(keyword in text for keyword in keywords):
                affected_sectors.append(sector)
        
        if not affected_sectors:
            affected_sectors = ["GENERAL_MARKET"]
        
        impact = MarketImpact(
            impact_level=0.5,
            confidence_score=0.8,
            affected_sectors=affected_sectors,
            affected_tickers=[],  # 티커 정보는 비워둠
            time_horizon="SHORT_TERM",
            key_drivers=["Market Sentiment"]
        )
        return impact
    
    def _calculate_overall_sentiment(self, articles: List[NewsArticle]) -> str:
        """
        전체 뉴스 기사의 감성을 분석하여 종합적인 감성 점수를 반환합니다.
        """
        if not articles:
            return "neutral"
            
        sentiment_scores = {
            "very_positive": 2,
            "positive": 1,
            "neutral": 0,
            "negative": -1,
            "very_negative": -2
        }
        
        total_score = 0
        count = 0
        
        for article in articles:
            if article.sentiment in sentiment_scores:
                total_score += sentiment_scores[article.sentiment]
                count += 1
        
        if count == 0:
            return "neutral"
            
        avg_score = total_score / count
        
        if avg_score >= 1.5:
            return "very_positive"
        elif avg_score >= 0.5:
            return "positive"
        elif avg_score > -0.5:
            return "neutral"
        elif avg_score > -1.5:
            return "negative"
        else:
            return "very_negative"
    
    def _calculate_average_sentiment(self, articles: List[NewsArticle]) -> float:
        """
        전체 뉴스 기사의 평균 감성 점수를 계산합니다.
        """
        if not articles:
            return 0.0
            
        sentiment_scores = {
            "very_positive": 1.0,
            "positive": 0.5,
            "neutral": 0.0,
            "negative": -0.5,
            "very_negative": -1.0
        }
        
        total_score = 0.0
        valid_articles = 0
        
        for article in articles:
            if article.sentiment in sentiment_scores:
                total_score += sentiment_scores[article.sentiment]
                valid_articles += 1
        
        return total_score / valid_articles if valid_articles > 0 else 0.0
    
    def analyze_news(self, request: NewsAnalysisRequest) -> NewsAnalysisResponse:
        """뉴스 분석 메인 로직"""
        articles = []
        
        try:
            # 1. 한국 비즈니스 뉴스 (top-headlines)
            kr_response = self.newsapi.get_top_headlines(
                country='kr',
                category='business',
                page_size=100
            )
            
            if kr_response.get('status') == 'ok':
                print(f"Found {len(kr_response.get('articles', []))} Korean business headlines")
                self._process_articles(kr_response.get('articles', []), articles, 
                                    datetime.now() - timedelta(days=7), 'MARKET')
            
            # 2. 글로벌 비즈니스 뉴스
            global_response = self.newsapi.get_top_headlines(
                category='business',
                page_size=100
            )
            
            if global_response.get('status') == 'ok':
                print(f"Found {len(global_response.get('articles', []))} global business headlines")
                self._process_articles(global_response.get('articles', []), articles, 
                                    datetime.now() - timedelta(days=7), 'MARKET')
            
            # 3. 기업/산업별 뉴스 검색
            search_terms = (
                self.market_keywords['major_companies'] +
                [f"{term} korea" for term in self.market_keywords['major_companies']] +
                self.market_keywords['industries'] +
                [f"{term} korea" for term in self.market_keywords['industries']]
            )
            
            for term in search_terms:
                try:
                    response = self.newsapi.get_everything(
                        q=term,
                        from_param=(datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d"),
                        to=datetime.now().strftime("%Y-%m-%d"),
                        sort_by='relevancy',
                        page_size=100
                    )
                    
                    if response.get('status') == 'ok':
                        print(f"Found {len(response.get('articles', []))} articles for term: {term}")
                        self._process_articles(response.get('articles', []), articles, 
                                            datetime.now() - timedelta(days=7), term)
                        
                except Exception as e:
                    print(f"Error fetching news for term {term}: {str(e)}")
                    continue
                    
        except Exception as e:
            print(f"Error fetching news: {str(e)}")
            
        print(f"Total articles found before filtering: {len(articles)}")
        
        # 중복 제거 (URL 기준)
        unique_articles = {article.url: article for article in articles}.values()
        filtered_articles = list(unique_articles)
        
        print(f"Articles after removing duplicates: {len(filtered_articles)}")
        
        # 관련성 점수로 필터링
        filtered_articles = [
            article for article in filtered_articles 
            if article.relevance_score >= request.min_relevance_score
        ]
        
        print(f"Articles after relevance filtering: {len(filtered_articles)}")
        
        # 분석 수행
        analyzed_articles = []
        for article in filtered_articles[:50]:  # 상위 50개만 분석
            entities = self._extract_entities(article)
            market_impact = self._analyze_market_impact(article, entities)
            sentiment = self._analyze_sentiment(article.title, article.content)
            category = self._classify_news_category(article.title, article.content)
            
            analysis = NewsAnalysis(
                article=article,
                entities=entities,
                market_impact=market_impact,
                sentiment=sentiment,
                category=category,
                key_takeaways=[],
                trading_signals=[]
            )
            analyzed_articles.append(analysis)
        
        return NewsAnalysisResponse(
            request_timestamp=datetime.now(),
            ticker=request.tickers[0] if request.tickers else "",
            articles=filtered_articles,
            analyzed_articles=analyzed_articles,
            overall_sentiment=self._calculate_overall_sentiment(analyzed_articles),
            aggregated_sentiment=self._calculate_overall_sentiment(analyzed_articles),
            average_sentiment_score=self._calculate_average_sentiment(analyzed_articles),
            major_events=[],
            trading_implications=[],
            risk_factors=[],
            summary="News analysis completed",
            key_findings=f"Found {len(analyzed_articles)} relevant articles"
        )

    async def analyze(self, request: NewsAnalysisRequest) -> NewsAnalysisResponse:
        """뉴스 분석을 수행하고 결과를 반환합니다."""
        return self.analyze_news(request)