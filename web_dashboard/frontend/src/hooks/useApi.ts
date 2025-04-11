import { useState, useEffect } from 'react';
import axios from 'axios';
import { DashboardData } from '../types';

const API_BASE_URL = 'http://localhost:8000/api';

export const useApi = () => {
  const [data, setData] = useState<DashboardData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchDashboardData = async () => {
    try {
      setLoading(true);
      const response = await axios.get<DashboardData>(`${API_BASE_URL}/dashboard`);
      setData(response.data);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDashboardData();
    const interval = setInterval(fetchDashboardData, 5000); // 5초마다 업데이트

    return () => clearInterval(interval);
  }, []);

  return { data, loading, error, refetch: fetchDashboardData };
}; 