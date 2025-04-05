from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from enum import Enum
from datetime import datetime

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
    article_id: str
    title: str
    content: str
    source: NewsSource
    url: str
    published_at: datetime
    author: Optional[str] = None
    categories: List[str] = []
    tickers: List[str] = []
    metadata: Optional[Dict[str, Any]] = None

class EntityMention(BaseModel):
    entity: str
    entity_type: str  # COMPANY, PERSON, LOCATION, etc.
    sentiment: SentimentScore
    mentions_count: int
    context_snippets: List[str]

class MarketImpact(BaseModel):
    impact_level: float = Field(..., ge=-1, le=1)  # -1 (very negative) to 1 (very positive)
    confidence_score: float = Field(..., ge=0, le=1)
    affected_sectors: List[str]
    affected_tickers: List[str]
    time_horizon: str  # SHORT_TERM, MEDIUM_TERM, LONG_TERM
    key_drivers: List[str]

class NewsAnalysis(BaseModel):
    article: NewsArticle
    sentiment: SentimentScore
    sentiment_score: float = Field(..., ge=-1.0, le=1.0)
    key_topics: List[str]
    named_entities: Dict[str, List[str]]
    summary: str
    impact_analysis: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.now)

class NewsAnalysisRequest(BaseModel):
    ticker: str
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    sources: Optional[List[NewsSource]] = None
    limit: int = Field(10, ge=1, le=100)

class NewsAnalysisResponse(BaseModel):
    request_timestamp: datetime = Field(default_factory=datetime.now)
    ticker: str
    articles: List[NewsArticle]
    analysis: List[NewsAnalysis]
    aggregated_sentiment: SentimentScore
    average_sentiment_score: float
    key_findings: str
    metadata: Optional[Dict[str, Any]] = None 