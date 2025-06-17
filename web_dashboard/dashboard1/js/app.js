// 전역 변수로 데이터 저장
let forecastData = null;
let dashboardData = null;
let stockList = []; // dashboard에서 가져온 종목 리스트

// 로딩 상태 표시 함수
function showLoading(message = '데이터를 불러오는 중...') {
  const summaryElement = document.querySelector('.summary-text');
  if (summaryElement) {
    summaryElement.textContent = message;
    summaryElement.style.color = '#666';
  }
  console.log('📋 ' + message);
}

// 에러 메시지 표시 함수
function showError(message) {
  const summaryElement = document.querySelector('.summary-text');
  if (summaryElement) {
    summaryElement.textContent = message;
    summaryElement.style.color = '#ff6b6b';
  }
  console.error('❌ ' + message);
}

// **추가된 함수**: 메인 데이터 로딩 함수
async function loadDashboardData() {
  try {
    showLoading('데이터를 불러오는 중...');
    
    // 병렬로 두 파일 로드
    const [forecastResponse, dashboardResponse] = await Promise.all([
      fetch('/system/forecast.json'),
      fetch('/system/dashboard.json')
    ]);

    if (!forecastResponse.ok) {
      throw new Error(`forecast.json 로드 실패: ${forecastResponse.status}`);
    }
    if (!dashboardResponse.ok) {
      throw new Error(`dashboard.json 로드 실패: ${dashboardResponse.status}`);
    }

    forecastData = await forecastResponse.json();
    dashboardData = await dashboardResponse.json();

    console.log('✅ 데이터 로딩 완료');
    console.log('📊 Forecast 종목 수:', Object.keys(forecastData).length);
    console.log('📊 Dashboard 종목 수:', dashboardData.strategy?.length || 0);

    // 데이터 검증 및 UI 업데이트
    validateDataConsistency();
    populateStockList();
    updateSummarySection();
    
    // **수정**: 첫 번째 종목으로 자동 초기화 및 선택 표시
    if (dashboardData.strategy && dashboardData.strategy.length > 0) {
      const firstStock = dashboardData.strategy[0].종목;
      
      // 첫 번째 종목을 선택된 상태로 표시
      setTimeout(() => {
        const firstStockElement = document.querySelector('.stock-list li:first-child');
        if (firstStockElement) {
          firstStockElement.classList.add('selected');
        }
      }, 100);
      
      // 첫 번째 종목의 데이터로 화면 업데이트
      await updateAllSections(firstStock);
    }

  } catch (error) {
    console.error('❌ 데이터 로딩 실패:', error);
    showError(`데이터 로딩 실패: ${error.message}`);
  }
}

// 데이터 일관성 검증 함수
function validateDataConsistency() {
  if (!dashboardData || !forecastData) return;
  
  console.log('🔍 데이터 일관성 검증 시작');
  
  // forecast의 종목명들
  const forecastStocks = Object.keys(forecastData);
  
  // dashboard의 종목명들
  const strategyStocks = dashboardData.strategy?.map(s => s.종목) || [];
  const riskStocks = dashboardData.risk?.map(r => r.종목) || [];
  const techStocks = dashboardData.tech?.map(t => t.종목) || [];
  const newsStocks = Object.keys(dashboardData.news || {});
  const portfolioStocks = dashboardData.portfolio?.map(p => p.종목명) || [];
  
  console.log('📊 데이터 분포:');
  console.log('  Forecast 종목:', forecastStocks.length, '개');
  console.log('  Dashboard Strategy:', strategyStocks.length, '개');
  console.log('  Dashboard에서 데이터가 있는 종목:', strategyStocks);
  
  // 매칭되는 종목과 매칭되지 않는 종목 찾기
  const matchingStocks = forecastStocks.filter(stock => strategyStocks.includes(stock));
  const missingInDashboard = forecastStocks.filter(stock => !strategyStocks.includes(stock));
  
  console.log('✅ 매칭되는 종목:', matchingStocks);
  console.log('⚠️ Dashboard에 없는 종목:', missingInDashboard);
}

// **수정된 함수**: CSS 스타일을 덮어쓰지 않고 원래 CSS 사용
function populateStockList() {
  const list = document.querySelector('.stock-list');
  if (!list) {
    console.error('❌ .stock-list 요소를 찾을 수 없습니다.');
    return;
  }
  
  list.innerHTML = '';

  // dashboard.json의 strategy에서 종목명 리스트 추출 (5개)
  const strategyStocks = dashboardData.strategy || [];

  strategyStocks.forEach(strategyItem => {
    const stockName = strategyItem.종목; // dashboard.json에서 종목명
    
    // forecast.json에서 해당 종목의 코드 찾기
    const stockInfo = forecastData[stockName];
    if (!stockInfo) {
      console.warn(`⚠️ ${stockName}의 forecast 데이터를 찾을 수 없습니다.`);
      return;
    }
    
    const stockCode = stockInfo.stock_code; // forecast.json에서 종목코드 가져오기
    const stockLabel = `${stockName}(${stockCode})`; // 종목명(종목코드) 형식

    const li = document.createElement('li');
    li.textContent = stockLabel;
    
    // CSS 스타일은 건드리지 않고 클래스만 추가
    li.addEventListener('click', async () => {
      // 선택된 종목 표시 (CSS 클래스 사용)
      document.querySelectorAll('.stock-list li').forEach(item => {
        item.classList.remove('selected');
      });
      li.classList.add('selected');
      
      await updateAllSections(stockName);  // 종목명으로 전달
    });

    list.appendChild(li);
  });

  console.log(`✅ 종목 리스트 UI 생성 완료: ${strategyStocks.length}개 종목 (dashboard.json의 5개 + forecast.json에서 코드 가져옴)`);
}

// **수정된 함수**: 데이터를 불러와서 화면에 반영하는 함수
async function updateAllSections(stockName) {
  try {
    console.log(`🔄 ${stockName} 데이터 업데이트 시작`);

    showLoading(`${stockName} 데이터를 불러오는 중...`);

    // forecast.json에서 해당 종목의 가격 데이터 가져오기
    const stockData = forecastData[stockName];

    if (!stockData) {
      throw new Error(`해당 종목(${stockName})의 예측 데이터를 찾을 수 없습니다.`);
    }

    // dashboard.json에서 해당 종목에 대한 정보 가져오기
    const strategy = dashboardData.strategy?.find(item => item.종목 === stockName);
    const news = dashboardData.news?.[stockName];
    const risk = dashboardData.risk?.find(item => item.종목 === stockName);
    const tech = dashboardData.tech?.find(item => item.종목 === stockName);
    const portfolio = dashboardData.portfolio?.find(item => item.종목명 === stockName);
    
    // HTML의 실제 클래스명에 정확히 맞춰 업데이트
    
    // 전략 분석 섹션 업데이트 (키워드 제거)
    updateElement('.stock-code', `${stockName}(${stockData.stock_code})`);
    updateElement('.strategy-name', strategy?.추천전략 || '-');
    updateElement('.strategy-text', strategy?.설명 || '해당 종목의 전략 분석 데이터가 없습니다.');
    
    // 뉴스/이슈 분석 섹션 업데이트 (키워드 추가)
    updateElement('.sentiment-score', news?.감성점수 ? news.감성점수.toFixed(3) : '-');
    updateElement('.market-impact', news?.시장영향도 ? news.시장영향도.toFixed(3) : '-');
    
    // 키워드를 뉴스/이슈 분석 카드에 추가
    if (news?.키워드) {
      const newsCard = document.querySelector('.card:nth-child(2) .card-text');
      if (newsCard) {
        newsCard.innerHTML = `
          감성점수: <strong class="sentiment-score">${news.감성점수.toFixed(3)}</strong> |
          시장 영향도: <strong class="market-impact">${news.시장영향도.toFixed(3)}</strong><br>
          키워드: <strong>${news.키워드}</strong>
        `;
      }
    }
    
    // 리스크 분석 섹션 업데이트 (HTML의 정확한 클래스명)
    updateElement('.risk-score', risk?.리스크점수 || '-');
    updateElement('.volatility', risk?.변동성 || '-');
    updateElement('.risk-grade', risk?.리스크레벨 || '-');
    
    // 기술적 분석 섹션 업데이트 (HTML의 정확한 클래스명)
    updateElement('.rsi', tech?.RSI || '-');
    updateElement('.macd', tech?.MACD || '-');
    updateElement('.bb-upper', tech?.볼린저밴드상단 ? tech.볼린저밴드상단.toLocaleString() : '-');
    updateElement('.bb-lower', tech?.볼린저밴드하단 ? tech.볼린저밴드하단.toLocaleString() : '-');
    updateElement('.close-price', tech?.최근종가 ? tech.최근종가.toLocaleString() : '-');
    
    // 포트폴리오 섹션 업데이트 (HTML의 정확한 클래스명)
    if (portfolio) {
      updateElement('.portfolio-quantity', portfolio.구매개수 ? portfolio.구매개수.toLocaleString() + '주' : '-');
      updateElement('.portfolio-investment', portfolio.투자금 ? portfolio.투자금.toLocaleString() + '원' : '-');
      updateElement('.portfolio-current-price', portfolio.현재가 ? portfolio.현재가.toLocaleString() + '원' : '-');
    } else {
      updateElement('.portfolio-quantity', '-');
      updateElement('.portfolio-investment', '-');
      updateElement('.portfolio-current-price', '-');
    }

    // 시계열 차트 업데이트
    drawForecastChart(stockName, stockData.dates, stockData.prices);

    // **수정**: 선택한 종목에 맞는 summary만 표시 (메시지가 아닌 실제 summary)
    updateSummaryForStock(stockName);

    console.log(`✅ ${stockName} 데이터 업데이트 완료`);

  } catch (error) {
    console.error(`❌ ${stockName} 데이터 업데이트 실패:`, error);
    showError(`${stockName} 데이터 업데이트 실패: ${error.message}`);
  }
}

// 요소 업데이트 헬퍼 함수
function updateElement(selector, value) {
  const element = document.querySelector(selector);
  if (element) {
    element.textContent = value || '데이터 없음';
  } else {
    console.warn(`⚠️ 요소를 찾을 수 없음: ${selector}`);
  }
}

// **수정된 함수**: 시계열 차트 그리기 (에러 처리 강화)
function drawForecastChart(stockName, dates, prices) {
  const canvas = document.getElementById('forecastChart');
  if (!canvas) {
    console.warn('⚠️ forecastChart 캔버스를 찾을 수 없습니다.');
    return;
  }

  // Chart.js가 로드되었는지 확인
  if (typeof Chart === 'undefined') {
    console.error('❌ Chart.js가 로드되지 않았습니다.');
    return;
  }

  try {
    // 기존 차트가 있으면 안전하게 제거
    if (window.forecastChart && typeof window.forecastChart.destroy === 'function') {
      window.forecastChart.destroy();
      window.forecastChart = null;
    }

    const ctx = canvas.getContext('2d');

    window.forecastChart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: dates.map(date => {
          const d = new Date(date);
          return `${d.getMonth() + 1}/${d.getDate()}`;
        }),
        datasets: [{
          label: `${stockName} 주가 예측`,
          data: prices,
          borderColor: '#007bff',
          backgroundColor: 'rgba(0, 123, 255, 0.1)',
          borderWidth: 2,
          pointBackgroundColor: '#007bff',
          pointBorderColor: '#ffffff',
          pointBorderWidth: 2,
          pointRadius: 4,
          fill: true,
          tension: 0.4
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: {
            title: {
              display: true,
              text: '날짜'
            },
            grid: {
              display: true,
              color: 'rgba(0,0,0,0.1)'
            }
          },
          y: {
            title: {
              display: true,
              text: '가격 (원)'
            },
            grid: {
              display: true,
              color: 'rgba(0,0,0,0.1)'
            },
            ticks: {
              callback: function(value) {
                return value.toLocaleString() + '원';
              }
            }
          }
        },
        plugins: {
          legend: {
            display: true,
            position: 'top'
          },
          tooltip: {
            callbacks: {
              label: function(context) {
                return `${context.dataset.label}: ${context.parsed.y.toLocaleString()}원`;
              }
            }
          }
        }
      }
    });

    console.log(`📈 ${stockName} 차트 업데이트 완료`);
  } catch (error) {
    console.error('❌ 차트 생성 실패:', error);
    // 차트 생성 실패 시 캔버스에 메시지 표시
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.font = '16px Arial';
    ctx.fillStyle = '#666';
    ctx.textAlign = 'center';
    ctx.fillText('차트를 불러올 수 없습니다', canvas.width / 2, canvas.height / 2);
  }
}

// **수정된 함수**: 선택한 종목에 맞는 summary만 표시 (종목명 제거)
function updateSummaryForStock(stockName) {
  if (!dashboardData?.summary) return;
  
  // 선택한 종목과 관련된 summary 찾기
  const stockSummary = dashboardData.summary.find(summary => 
    summary.includes(stockName)
  );
  
  const summaryElement = document.querySelector('.summary-content');
  if (summaryElement) {
    summaryElement.innerHTML = '';
    
    if (stockSummary) {
      const li = document.createElement('li');
      li.className = 'summary-text';
      
      // 종목명: 부분 제거하고 summary 내용만 표시
      const summaryText = stockSummary.split(': ')[1] || stockSummary;
      li.textContent = summaryText;
      
      summaryElement.appendChild(li);
    } else {
      // 해당 종목의 summary가 없는 경우
      const li = document.createElement('li');
      li.className = 'summary-text';
      li.textContent = `해당 종목의 종합 의견이 준비되지 않았습니다.`;
      summaryElement.appendChild(li);
    }
  }
}

// **수정된 함수**: 전체 요약이 아닌 종목별 처리
function updateSummarySection() {
  // 초기 로딩 시에는 첫 번째 종목의 summary 표시
  if (dashboardData?.strategy && dashboardData.strategy.length > 0) {
    const firstStock = dashboardData.strategy[0].종목;
    updateSummaryForStock(firstStock);
  }
}

// **추가된 함수**: 포트폴리오 전체 정보 업데이트
function updatePortfolioOverview() {
  if (!dashboardData?.portfolio) return;
  
  const totalInvestment = dashboardData.portfolio.reduce((sum, item) => sum + (item.투자금 || 0), 0);
  const totalCurrentValue = dashboardData.portfolio.reduce((sum, item) => {
    if (item.종목명 === '현금') return sum + item.투자금;
    return sum + ((item.구매개수 || 0) * (item.현재가 || 0));
  }, 0);
  
  const totalProfit = totalCurrentValue - totalInvestment;
  const totalProfitRate = totalInvestment > 0 ? (totalProfit / totalInvestment * 100).toFixed(2) : 0;
  
  updateElement('.total-investment', totalInvestment.toLocaleString() + '원');
  updateElement('.total-current-value', totalCurrentValue.toLocaleString() + '원');
  updateElement('.total-profit', `${totalProfit.toLocaleString()}원 (${totalProfitRate}%)`);
}

// 페이지가 모두 로딩되면 데이터 로드 함수 실행
document.addEventListener('DOMContentLoaded', function() {
  console.log('🚀 페이지 로딩 완료, 데이터 로딩 시작');
  loadDashboardData();
});

// 페이지 언로드 시 차트 정리
window.addEventListener('beforeunload', function() {
  if (window.forecastChart) {
    window.forecastChart.destroy();
  }
});