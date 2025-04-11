import { useState, useEffect } from 'react';
import axios from 'axios';
import { NewsAnalysis } from '../types';

const REFRESH_INTERVAL = 60000; // 1분 = 60000ms

interface BackendArticle {
  article: {
    title: string;
    tickers: string[];
    sentiment: string;
    published_at: string;
  };
  market_impact: {
    impact_level: number;
  };
}

interface BackendResponse {
  request_timestamp: string;
  analyzed_articles: BackendArticle[];
}

export const useNewsAnalysis = () => {
  const [newsItems, setNewsItems] = useState<NewsAnalysis[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastFetchTime, setLastFetchTime] = useState<Date | null>(null);

  const fetchNewsAnalysis = async () => {
    try {
      // 마지막 호출 시간 체크
      if (lastFetchTime && new Date().getTime() - lastFetchTime.getTime() < REFRESH_INTERVAL) {
        console.log('API 호출 제한: 1분 대기 중...');
        return;
      }

      setLoading(true);
      const response = await axios.get<BackendResponse>('http://localhost:8000/news');
      
      const formattedNews: NewsAnalysis[] = (response.data.analyzed_articles || []).map(item => ({
        ticker: item.article.tickers[0] || 'UNKNOWN',
        sentiment: item.article.sentiment || 'neutral',
        impact: item.market_impact?.impact_level || 0,
        summary: item.article.title,
        timestamp: item.article.published_at
      }));

      console.log('Formatted news items:', formattedNews); // 디버깅용
      setNewsItems(formattedNews);
      setLastFetchTime(new Date());
      setError(null);
    } catch (err) {
      if (axios.isAxiosError(err)) {
        console.log('API Response:', err.response?.data); // 디버깅용
        setError(
          err.response?.data?.detail || 
          '뉴스 분석 데이터를 불러오는데 실패했습니다.'
        );
      } else {
        setError('알 수 없는 오류가 발생했습니다.');
      }
      console.error('News analysis error:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchNewsAnalysis();
    
    // 1분마다 새로운 데이터 가져오기
    const interval = setInterval(fetchNewsAnalysis, REFRESH_INTERVAL);
    
    return () => clearInterval(interval);
  }, []);

  return { 
    newsItems, 
    loading, 
    error, 
    refetch: fetchNewsAnalysis,
    lastFetchTime 
  };
}; 