// js/debate.js

let forecastData = null;
let debateData = null;

window.addEventListener("DOMContentLoaded", async () => {
  try {
    // 필요한 데이터들 병렬로 로드
    const [forecastResponse, debateResponse] = await Promise.all([
      fetch("/system/forecast.json"),
      fetch("/system/debate_logs/debate_log_20250614.json")
    ]);

    forecastData = await forecastResponse.json();
    debateData = await debateResponse.json();
    
    console.log("[✅ 데이터 로딩 완료]");
    console.log("Forecast 종목 수:", Object.keys(forecastData).length);
    console.log("Debate 종목 수:", Object.keys(debateData).length);

    // 종목 리스트 생성 및 첫 번째 종목 자동 선택
    populateStockList();
    
  } catch (e) {
    console.error("데이터 불러오기 실패:", e);
    showError("데이터를 불러오는데 실패했습니다.");
  }
});

// **수정된 함수**: debate_log의 종목코드 기준으로 리스트 생성
function populateStockList() {
  const stockList = document.querySelector(".stock-list");
  if (!stockList) {
    console.error("❌ .stock-list 요소를 찾을 수 없습니다.");
    return;
  }

  stockList.innerHTML = "";

  // debate_log에 있는 종목코드들 가져오기
  const stockCodes = Object.keys(debateData);
  console.log("[📊 debate_log 종목코드들]", stockCodes);

  stockCodes.forEach((stockCode, index) => {
    // forecast.json에서 해당 종목코드의 종목명 찾기
    const stockName = findStockNameByCode(stockCode);
    
    if (!stockName) {
      console.warn(`⚠️ ${stockCode}의 종목명을 forecast.json에서 찾을 수 없습니다.`);
      return;
    }
    
    const stockLabel = `${stockName}(${stockCode})`; // 종목명(종목코드) 형식

    const li = document.createElement("li");
    li.textContent = stockLabel;
    li.style.cursor = "pointer";
    li.dataset.stockName = stockName; // 종목명 저장
    li.dataset.stockCode = stockCode; // 종목코드 저장
    
    // 클릭 이벤트
    li.addEventListener("click", () => {
      // 선택된 종목 하이라이트
      document.querySelectorAll('.stock-list li').forEach(item => {
        item.classList.remove('selected');
      });
      li.classList.add('selected');
      
      updateDebateSection(stockName, stockCode);
    });

    stockList.appendChild(li);
  });

  console.log(`✅ 종목 리스트 생성 완료: ${stockCodes.length}개 종목`);

  // 첫 번째 종목 자동 선택
  if (stockCodes.length > 0) {
    const firstStockCode = stockCodes[0];
    const firstStockName = findStockNameByCode(firstStockCode);
    
    if (firstStockName) {
      // 첫 번째 종목을 선택된 상태로 표시
      setTimeout(() => {
        const firstStockElement = document.querySelector('.stock-list li:first-child');
        if (firstStockElement) {
          firstStockElement.classList.add('selected');
        }
      }, 100);
      
      updateDebateSection(firstStockName, firstStockCode);
    }
  }
}

// **새로운 함수**: 종목코드로 종목명 찾기
function findStockNameByCode(targetCode) {
  for (const [stockName, stockInfo] of Object.entries(forecastData)) {
    if (stockInfo.stock_code === targetCode) {
      console.log(`[🔍 매핑 발견] ${targetCode} -> ${stockName}`);
      return stockName;
    }
  }
  console.warn(`[❌ 매핑 실패] ${targetCode}에 해당하는 종목명을 찾을 수 없음`);
  return null;
}

// **수정된 함수**: 종목명과 종목코드를 모두 받아서 처리
async function updateDebateSection(stockName, stockCode) {
  try {
    console.log(`[🔍 업데이트] ${stockName} (${stockCode})`);

    // 상단 제목 업데이트 - 종목명(종목코드) 형태
    const titleElement = document.getElementById("current-stock-name");
    if (titleElement) {
      titleElement.textContent = `${stockName}(${stockCode})`;
    }

    // debate_log에서 해당 종목코드의 데이터 찾기
    if (debateData[stockCode]) {
      const entries = debateData[stockCode];
      console.log(`[✅ 데이터 발견] ${stockCode}:`, entries.length, "개 항목");

      // 각 전문가별 의견 추출 함수
      const getExpertComment = (agentName) => {
        const match = entries.find(entry => entry.agent === agentName);
        return match?.전문가설명 || `${agentName} 전문가의 의견이 준비되지 않았습니다.`;
      };

      // 각 분야별로 전문가 의견 업데이트
      updateElement(".debate-news", getExpertComment("news_analyzer"));
      updateElement(".debate-strategy", getExpertComment("strategy_generator"));
      updateElement(".debate-technical", getExpertComment("technical_analyzer"));
      updateElement(".debate-risk", getExpertComment("risk_analyzer"));
      
    } else {
      console.warn(`⚠️ ${stockCode}에 대한 debate 데이터가 없습니다.`);
      
      // 데이터가 없는 경우 기본 메시지 표시
      const noDataMessage = `${stockName}에 대한 전문가 토론 데이터가 준비되지 않았습니다.`;
      updateElement(".debate-news", noDataMessage);
      updateElement(".debate-strategy", noDataMessage);
      updateElement(".debate-technical", noDataMessage);
      updateElement(".debate-risk", noDataMessage);
    }
  } catch (err) {
    console.error("❌ 데이터 업데이트 실패:", err);
    showError(`${stockName} 데이터 업데이트에 실패했습니다.`);
  }
}

// **유틸리티 함수**: 요소 업데이트
function updateElement(selector, text) {
  const element = document.querySelector(selector);
  if (element) {
    element.textContent = text;
  } else {
    console.warn(`⚠️ 요소를 찾을 수 없음: ${selector}`);
  }
}

// **유틸리티 함수**: 에러 메시지 표시
function showError(message) {
  console.error("❌", message);
  
  // 모든 debate 섹션에 에러 메시지 표시
  document.querySelectorAll(".debate-news, .debate-strategy, .debate-technical, .debate-risk")
    .forEach(el => {
      if (el) el.textContent = message;
    });
}