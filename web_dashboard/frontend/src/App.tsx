import React from 'react';
import { BrowserRouter as Router, Route, Routes, Link } from 'react-router-dom';
import { ThemeProvider } from '@mui/material/styles';
import { AppBar, Toolbar, Typography, Container, Box } from '@mui/material';
import CssBaseline from '@mui/material/CssBaseline';
import theme from './theme';
import { NewsAnalysis } from './components/NewsAnalysis';
import { Portfolio } from './components/Portfolio';

// 임시 포트폴리오 데이터
const mockPortfolioData = {
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
    }
  ],
  risk_metrics: {
    var: 50000,
    drawdown: 0.1,
    exposure: 0.7
  }
};

const App: React.FC = () => {
  return (
    <Router>
      <ThemeProvider theme={theme}>
        <CssBaseline />
        <Box sx={{ display: 'flex', flexDirection: 'column', minHeight: '100vh' }}>
          <AppBar position="static">
            <Toolbar>
              <Typography variant="h6" sx={{ flexGrow: 1 }}>
                EBS Trading Dashboard
              </Typography>
              <Link to="/portfolio" style={{ color: 'white', textDecoration: 'none', marginRight: '20px' }}>
                포트폴리오
              </Link>
              <Link to="/news" style={{ color: 'white', textDecoration: 'none' }}>
                뉴스 분석
              </Link>
            </Toolbar>
          </AppBar>
          
          <Container component="main" sx={{ mt: 4, mb: 4, flex: 1 }}>
            <Routes>
              <Route path="/portfolio" element={<Portfolio data={mockPortfolioData} />} />
              <Route path="/news" element={<NewsAnalysis />} />
              <Route path="/" element={<Portfolio data={mockPortfolioData} />} />
            </Routes>
          </Container>

          <Box component="footer" sx={{ py: 3, bgcolor: 'background.paper' }}>
            <Container maxWidth="sm">
              <Typography variant="body2" color="text.secondary" align="center">
                © 2024 EBS Trading Dashboard. All rights reserved.
              </Typography>
            </Container>
          </Box>
        </Box>
      </ThemeProvider>
    </Router>
  );
};

export default App; 