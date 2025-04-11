/// <reference types="@testing-library/jest-dom" />
import React from 'react';
import { render, screen } from '@testing-library/react';
import '@testing-library/jest-dom';
import { NewsAnalysis } from '../NewsAnalysis';
import { NewsAnalysis as NewsAnalysisType } from '../../types';
import { describe, it, expect } from '@jest/globals';

describe('NewsAnalysis 컴포넌트', () => {
  const mockNewsItems: NewsAnalysisType[] = [
    {
      ticker: 'AAPL',
      sentiment: 'positive',
      impact: 0.8,
      summary: '애플 실적 예상치 상회',
      timestamp: '2024-03-20T10:00:00Z'
    },
    {
      ticker: 'GOOGL',
      sentiment: 'negative',
      impact: 0.6,
      summary: '구글 클라우드 성장세 둔화',
      timestamp: '2024-03-20T09:30:00Z'
    }
  ];

  it('뉴스 분석 제목이 표시되어야 함', () => {
    render(<NewsAnalysis newsItems={mockNewsItems} />);
    expect(screen.getByText('뉴스 분석')).toBeInTheDocument();
  });

  it('모든 뉴스 항목이 표시되어야 함', () => {
    render(<NewsAnalysis newsItems={mockNewsItems} />);
    expect(screen.getByText('AAPL')).toBeInTheDocument();
    expect(screen.getByText('GOOGL')).toBeInTheDocument();
    expect(screen.getByText('애플 실적 예상치 상회')).toBeInTheDocument();
    expect(screen.getByText('구글 클라우드 성장세 둔화')).toBeInTheDocument();
  });

  it('감성 분석 결과가 올바르게 표시되어야 함', () => {
    render(<NewsAnalysis newsItems={mockNewsItems} />);
    const positiveChip = screen.getByText('positive').closest('.MuiChip-root');
    const negativeChip = screen.getByText('negative').closest('.MuiChip-root');
    
    expect(positiveChip).toHaveClass('MuiChip-colorSuccess');
    expect(negativeChip).toHaveClass('MuiChip-colorError');
  });

  it('영향도가 올바르게 표시되어야 함', () => {
    render(<NewsAnalysis newsItems={mockNewsItems} />);
    expect(screen.getByText('영향도: 80%')).toBeInTheDocument();
    expect(screen.getByText('영향도: 60%')).toBeInTheDocument();
  });

  it('빈 뉴스 목록 처리가 가능해야 함', () => {
    render(<NewsAnalysis newsItems={[]} />);
    expect(screen.getByText('뉴스 분석')).toBeInTheDocument();
    expect(screen.queryByRole('listitem')).not.toBeInTheDocument();
  });
}); 