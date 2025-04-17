from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from enum import Enum
from datetime import datetime
import uuid

class NewsSource(str, Enum):
    REUTERS = "reuters"
    BLOOMBERG = "bloomberg"
    CNBC = "cnbc"
    FINANCIAL_TIMES = "financial_times"
    WALL_STREET_JOURNAL = "wall_street_journal"
    OTHER = "other"

class SentimentScore(str, Enum):
    VERY_NEGATIVE = "very_negative"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    POSITIVE = "positive"
    VERY_POSITIVE = "very_positive"

class NewsCategory(str, Enum):
    EARNINGS = "earnings"
    MERGER_ACQUISITION = "merger_acquisition"
    MARKET_MOVEMENT = "market_movement"
    ECONOMIC_INDICATOR = "economic_indicator"
    COMPANY_NEWS = "company_news"
    INDUSTRY_NEWS = "industry_news"
    REGULATORY_NEWS = "regulatory_news"
    GEOPOLITICAL = "geopolitical"

class NewsArticle(BaseModel):
    article_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    url: str
    source: NewsSource = NewsSource.OTHER
    published_at: datetime
    content: str
    relevance_score: float = 1.0
    tickers: List[str] = []
    category: Optional[NewsCategory] = None
    sentiment: Optional[SentimentScore] = None

class EntityMention(BaseModel):
    entity: str
    entity_type: str
    sentiment: SentimentScore
    mentions_count: int
    context_snippets: List[str]

class MarketImpact(BaseModel):
    impact_level: float
    confidence_score: float
    affected_sectors: List[str]
    affected_tickers: List[str]
    time_horizon: str
    key_drivers: List[str]

class NewsAnalysis(BaseModel):
    article: NewsArticle
    entities: List[EntityMention]
    market_impact: MarketImpact
    key_takeaways: List[str]
    trading_signals: List[Dict]

class NewsAnalysisRequest(BaseModel):
    tickers: List[str]
    time_range: str = "1d"
    min_relevance_score: float = 0.3

class NewsAnalysisResponse(BaseModel):
    request_timestamp: datetime
    ticker: str
    articles: List[NewsArticle]
    analyzed_articles: List[NewsAnalysis]
    overall_sentiment: SentimentScore
    aggregated_sentiment: SentimentScore
    average_sentiment_score: float
    major_events: List[Dict]
    trading_implications: List[Dict]
    risk_factors: List[Dict]
    summary: str
    key_findings: str 