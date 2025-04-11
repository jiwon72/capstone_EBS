import React, { useEffect, useState } from 'react';
import axios from 'axios';
import {
  Card,
  CardContent,
  Typography,
  List,
  ListItem,
  ListItemText,
  Chip,
  Box,
  CircularProgress,
  Alert
} from '@mui/material';

interface NewsItem {
  ticker: string;
  sentiment: string;
  impact: number;
  summary: string;
  timestamp: string;
}

// Axios 인스턴스 생성
const api = axios.create({
  baseURL: 'http://localhost:8000',
  headers: {
    'Accept': 'application/json',
    'Content-Type': 'application/json'
  }
});

export const NewsAnalysis: React.FC = () => {
  const [newsItems, setNewsItems] = useState<NewsItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchNews = async () => {
      try {
        const { data } = await api.get('/news');
        const formattedNews = data.analyzed_articles.map((article: any) => ({
          ticker: article.article.tickers[0],
          sentiment: article.article.sentiment || 'neutral',
          impact: article.market_impact.impact_level,
          summary: article.article.title,
          timestamp: article.article.published_at
        }));
        
        setNewsItems(formattedNews);
        setError(null);
      } catch (err) {
        console.error('Error fetching news:', err);
        setError(
          axios.isAxiosError(err)
            ? err.response?.data?.detail || err.message
            : 'Failed to fetch news data'
        );
      } finally {
        setLoading(false);
      }
    };

    fetchNews();
    
    // 5초마다 새로운 뉴스 가져오기
    const interval = setInterval(fetchNews, 5000);
    return () => clearInterval(interval);
  }, []);

  const getSentimentColor = (sentiment: string) => {
    switch (sentiment.toLowerCase()) {
      case 'very_positive':
      case 'positive':
        return 'success';
      case 'very_negative':
      case 'negative':
        return 'error';
      default:
        return 'default';
    }
  };

  const getImpactColor = (impact: number) => {
    if (impact >= 0.7) return 'error';
    if (impact >= 0.4) return 'warning';
    return 'info';
  };

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="200px">
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Alert severity="error" sx={{ mb: 2 }}>
        {error}
      </Alert>
    );
  }

  return (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          뉴스 분석
        </Typography>
        
        <List>
          {newsItems.map((item, index) => (
            <ListItem key={index} divider={index < newsItems.length - 1}>
              <Box>
                <Box display="flex" alignItems="center" gap={1} mb={1}>
                  <Chip label={item.ticker} color="primary" size="small" />
                  <Chip
                    label={item.sentiment}
                    color={getSentimentColor(item.sentiment)}
                    size="small"
                  />
                  <Chip
                    label={`영향도: ${Math.round(item.impact * 100)}%`}
                    color={getImpactColor(item.impact)}
                    size="small"
                  />
                </Box>
                <Typography variant="body2" color="textPrimary" gutterBottom>
                  {item.summary}
                </Typography>
                <Typography variant="caption" color="textSecondary">
                  {new Date(item.timestamp).toLocaleString()}
                </Typography>
              </Box>
            </ListItem>
          ))}
        </List>
      </CardContent>
    </Card>
  );
}; 