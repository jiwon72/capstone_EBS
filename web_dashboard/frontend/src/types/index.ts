export interface Position {
  position_id: string;
  ticker: string;
  quantity: number;
  entry_price: number;
  current_price: number;
  unrealized_pnl: number;
  realized_pnl: number;
  side: 'BUY' | 'SELL';
  status: string;
  timestamp: string;
}

export interface RiskMetrics {
  var: number;
  drawdown: number;
  exposure: number;
  leverage: number;
}

export interface PortfolioStatus {
  portfolio_id: string;
  cash_balance: number;
  total_value: number;
  positions: Position[];
  risk_metrics: RiskMetrics;
  timestamp: string;
}

export interface NewsAnalysis {
  ticker: string;
  sentiment: string;
  impact: number;
  summary: string;
  timestamp: string;
}

export interface TechnicalSignal {
  ticker: string;
  signal_type: string;
  strength: number;
  direction: 'BUY' | 'SELL' | 'HOLD';
  indicators: Record<string, number>;
  timestamp: string;
}

export interface DashboardData {
  portfolio: PortfolioStatus;
  news: NewsAnalysis[];
  technical: TechnicalSignal[];
  timestamp: string;
} 