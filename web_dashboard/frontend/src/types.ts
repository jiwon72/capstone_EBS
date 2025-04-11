export interface NewsAnalysis {
  ticker: string;
  sentiment: string;
  impact: number;
  summary: string;
  timestamp: string;
}

export interface PortfolioStatus {
  total_value: number;
  cash_balance: number;
  positions: PortfolioPosition[];
  risk_metrics: PortfolioRiskMetrics;
}

export interface PortfolioPosition {
  position_id: string;
  ticker: string;
  quantity: number;
  entry_price: number;
  current_price: number;
  unrealized_pnl: number;
}

export interface PortfolioRiskMetrics {
  var: number;
  drawdown: number;
  exposure: number;
}

export interface Position {
  position_id: string;
  ticker: string;
  quantity: number;
  entry_price: number;
  current_price: number;
  unrealized_pnl: number;
}

export interface RiskMetrics {
  valueAtRisk: number;
  maxDrawdown: number;
  exposure: number;
}

export interface Portfolio {
  totalAssets: number;
  cashBalance: number;
  positions: Position[];
  riskMetrics: RiskMetrics;
}

export interface DashboardData {
  portfolio: Portfolio;
  newsAnalysis: NewsAnalysis[];
} 