import React from 'react';
import { Outlet, Link } from 'react-router-dom';
import {
  AppBar,
  Toolbar,
  Typography,
  Container,
  Box,
  Button,
  Paper
} from '@mui/material';

const Dashboard: React.FC = () => {
  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', minHeight: '100vh' }}>
      <AppBar position="static">
        <Toolbar>
          <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
            EBS Trading Dashboard
          </Typography>
          <Button color="inherit" component={Link} to="/">
            트레이딩 뷰
          </Button>
          <Button color="inherit" component={Link} to="/portfolio">
            포트폴리오
          </Button>
          <Button color="inherit" component={Link} to="/news">
            뉴스 분석
          </Button>
        </Toolbar>
      </AppBar>

      <Container component="main" sx={{ mt: 4, mb: 4, flex: 1 }}>
        <Paper elevation={3} sx={{ p: 3 }}>
          <Outlet />
        </Paper>
      </Container>

      <Box component="footer" sx={{ py: 3, bgcolor: 'background.paper' }}>
        <Container maxWidth="sm">
          <Typography variant="body2" color="text.secondary" align="center">
            © 2024 EBS Trading Dashboard. All rights reserved.
          </Typography>
        </Container>
      </Box>
    </Box>
  );
};

export default Dashboard; 