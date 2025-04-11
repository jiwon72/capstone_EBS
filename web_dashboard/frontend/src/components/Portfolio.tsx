import React from 'react';
import { 
  Card, 
  CardContent, 
  Typography, 
  Table, 
  TableBody, 
  TableCell, 
  TableHead, 
  TableRow,
  Box
} from '@mui/material';
import { PortfolioStatus, Position } from '../types';

interface PortfolioProps {
  data: PortfolioStatus;
}

export const Portfolio: React.FC<PortfolioProps> = ({ data }) => {
  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('ko-KR', {
      style: 'currency',
      currency: 'KRW'
    }).format(value);
  };

  return (
    <Card>
      <CardContent>
        <Typography variant="h5" gutterBottom>
          포트폴리오 현황
        </Typography>
        
        <Box mb={3}>
          <Typography variant="subtitle1">
            총 자산: {formatCurrency(data.total_value)}
          </Typography>
          <Typography variant="subtitle1">
            현금: {formatCurrency(data.cash_balance)}
          </Typography>
        </Box>

        <Typography variant="h6" gutterBottom>
          보유 포지션
        </Typography>
        
        <Table>
          <TableHead>
            <TableRow>
              <TableCell>종목</TableCell>
              <TableCell align="right">수량</TableCell>
              <TableCell align="right">평균단가</TableCell>
              <TableCell align="right">현재가</TableCell>
              <TableCell align="right">평가손익</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {data.positions.map((position: Position) => (
              <TableRow key={position.position_id}>
                <TableCell>{position.ticker}</TableCell>
                <TableCell align="right">{position.quantity}</TableCell>
                <TableCell align="right">
                  {formatCurrency(position.entry_price)}
                </TableCell>
                <TableCell align="right">
                  {formatCurrency(position.current_price)}
                </TableCell>
                <TableCell align="right" 
                  style={{ 
                    color: position.unrealized_pnl >= 0 ? 'green' : 'red' 
                  }}>
                  {formatCurrency(position.unrealized_pnl)}
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>

        <Box mt={3}>
          <Typography variant="h6" gutterBottom>
            리스크 지표
          </Typography>
          <Typography>
            VaR (95%): {formatCurrency(data.risk_metrics.var)}
          </Typography>
          <Typography>
            최대 손실률: {(data.risk_metrics.drawdown * 100).toFixed(2)}%
          </Typography>
          <Typography>
            투자 비중: {(data.risk_metrics.exposure * 100).toFixed(2)}%
          </Typography>
        </Box>
      </CardContent>
    </Card>
  );
}; 