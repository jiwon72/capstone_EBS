import React from 'react';
import { render, screen } from '@testing-library/react';
import '@testing-library/jest-dom';
import { Portfolio } from '../Portfolio';
import { PortfolioStatus } from '../../types';

describe('Portfolio 컴포넌트', () => {
  const mockPortfolio: PortfolioStatus = {
    total_value: 1000000,
    cash_balance: 300000,
    positions: [
      {
        position_id: '1',
        ticker: 'AAPL',
        quantity: 10,
        entry_price: 150.0,
        current_price: 170.0,
        unrealized_pnl: 200.0
      },
      {
        position_id: '2',
        ticker: 'GOOGL',
        quantity: 5,
        entry_price: 2800.0,
        current_price: 2900.0,
        unrealized_pnl: 500.0
      }
    ],
    risk_metrics: {
      var: 50000,
      drawdown: 0.1,
      exposure: 0.7
    }
  };

  it('포트폴리오 제목이 표시되어야 함', () => {
    render(<Portfolio data={mockPortfolio} />);
    expect(screen.getByText('포트폴리오 현황')).toBeInTheDocument();
  });

  it('총 자산과 현금 잔고가 표시되어야 함', () => {
    render(<Portfolio data={mockPortfolio} />);
    expect(screen.getByText(/총 자산: ₩1,000,000/)).toBeInTheDocument();
    expect(screen.getByText(/현금: ₩300,000/)).toBeInTheDocument();
  });

  it('모든 포지션이 표시되어야 함', () => {
    render(<Portfolio data={mockPortfolio} />);
    expect(screen.getByText('AAPL')).toBeInTheDocument();
    expect(screen.getByText('GOOGL')).toBeInTheDocument();
    expect(screen.getByText('10')).toBeInTheDocument();
    expect(screen.getByText('5')).toBeInTheDocument();
  });

  it('리스크 지표가 표시되어야 함', () => {
    render(<Portfolio data={mockPortfolio} />);
    expect(screen.getByText(/VaR \(95%\): ₩50,000/)).toBeInTheDocument();
    expect(screen.getByText(/최대 손실률: 10.00%/)).toBeInTheDocument();
    expect(screen.getByText(/투자 비중: 70.00%/)).toBeInTheDocument();
  });

  it('빈 포트폴리오 처리가 가능해야 함', () => {
    const emptyPortfolio: PortfolioStatus = {
      total_value: 0,
      cash_balance: 0,
      positions: [],
      risk_metrics: {
        var: 0,
        drawdown: 0,
        exposure: 0
      }
    };
    render(<Portfolio data={emptyPortfolio} />);
    expect(screen.getByText('포트폴리오 현황')).toBeInTheDocument();
    expect(screen.getByText(/총 자산: ₩0/)).toBeInTheDocument();
    expect(screen.queryByRole('row')).toBeInTheDocument(); // 헤더 행은 항상 표시됨
  });
}); 