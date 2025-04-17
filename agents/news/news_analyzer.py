import urllib.request
import urllib.parse
import json
import ssl
import time
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from difflib import SequenceMatcher
from deep_translator import GoogleTranslator
from bs4 import BeautifulSoup
import pandas as pd
from typing import List, Dict
from datetime import datetime

client_id = "jWYb81zpxwjjBOvjTlc1"
client_secret = "CH69vSJ6hu"

class MarketImpactAnalyzer:
    def __init__(self):
        self.sector_keywords = {
            '반도체': ['반도체', '파운드리', 'DDR', 'DRAM', 'NAND', '웨이퍼'],
            '2차전지': ['2차전지', '배터리', 'LFP', 'NCM', '양극재', '음극재', '분리막'],
            '자동차': ['자동차', '전기차', 'EV', '내연기관', '하이브리드'],
            'IT': ['소프트웨어', '플랫폼', '클라우드', 'AI', '인공지능'],
            '바이오': ['제약', '바이오', '신약', '임상', '백신'],
            '금융': ['은행', '증권', '보험', '카드', '핀테크']
        }
        
        self.market_indicators = {
            '금리': ['기준금리', '국고채', '회사채', '금리인상', '금리인하'],
            '환율': ['원달러', '원화가치', '환율', '달러인덱스'],
            '원자재': ['유가', '구리', '리튬', '니켈', '코발트'],
            '경제지표': ['GDP', '물가', '고용', '수출', '무역수지']
        }

    def analyze_market_impact(self, news_text: str, stock_name: str) -> Dict:
        """시장 영향 분석"""
        impact_analysis = {
            'impact_level': 0.0,  # -1.0 ~ 1.0
            'confidence_score': 0.0,  # 0.0 ~ 1.0
            'affected_sectors': [],
            'market_indicators': [],
            'related_stocks': [],
            'time_horizon': 'short_term'  # short_term, mid_term, long_term
        }

        # 섹터 영향 분석
        for sector, keywords in self.sector_keywords.items():
            for keyword in keywords:
                if keyword in news_text:
                    if sector not in impact_analysis['affected_sectors']:
                        impact_analysis['affected_sectors'].append(sector)

        # 시장 지표 영향 분석
        for indicator, keywords in self.market_indicators.items():
            for keyword in keywords:
                if keyword in news_text:
                    if indicator not in impact_analysis['market_indicators']:
                        impact_analysis['market_indicators'].append(indicator)

        # 영향도 계산
        impact_analysis['impact_level'] = self._calculate_impact_level(news_text)
        impact_analysis['confidence_score'] = self._calculate_confidence_score(news_text)
        impact_analysis['time_horizon'] = self._determine_time_horizon(news_text)
        impact_analysis['related_stocks'] = self._find_related_stocks(news_text, stock_name)

        return impact_analysis

    def _calculate_impact_level(self, text: str) -> float:
        """뉴스의 시장 영향도 계산"""
        impact_keywords = {
            'positive': ['급등', '상승', '호실적', '수주', '흑자전환', '매출증가'],
            'negative': ['급락', '하락', '적자전환', '매출감소', '리스크', '우려']
        }
        
        score = 0.0
        for keyword in impact_keywords['positive']:
            if keyword in text:
                score += 0.2
        for keyword in impact_keywords['negative']:
            if keyword in text:
                score -= 0.2
                
        return max(min(score, 1.0), -1.0)

    def _calculate_confidence_score(self, text: str) -> float:
        """분석 신뢰도 점수 계산"""
        confidence_keywords = ['전망', '예상', '추정', '확인', '발표', '공시']
        score = 0.0
        
        for keyword in confidence_keywords:
            if keyword in text:
                score += 0.2
                
        return min(score, 1.0)

    def _determine_time_horizon(self, text: str) -> str:
        """영향 시간 범위 결정"""
        short_term = ['당일', '오늘', '내일', '이번주', '단기']
        mid_term = ['이번달', '다음달', '분기', '중기']
        long_term = ['내년', '장기', '중장기', '미래']
        
        for term in long_term:
            if term in text:
                return 'long_term'
        for term in mid_term:
            if term in text:
                return 'mid_term'
        return 'short_term'

    def _find_related_stocks(self, text: str, main_stock: str) -> List[str]:
        """연관 종목 찾기"""
        related = [main_stock]
        return related

class NewsAnalyzer:
    def __init__(self):
        self.market_impact_analyzer = MarketImpactAnalyzer()

    def get_top100_by_volume(self):
        url = "https://finance.naver.com/sise/sise_quant.naver"
        headers = {'User-Agent': 'Mozilla/5.0'}
        req = urllib.request.Request(url, headers=headers)
        context = ssl._create_unverified_context()
        res = urllib.request.urlopen(req, context=context)
        soup = BeautifulSoup(res, "html.parser")

        stock_names = []
        for item in soup.select("table.type_2 tr"):
            td = item.select("td")
            if len(td) > 1:
                name = td[1].get_text(strip=True)
                if name:
                    stock_names.append(name)
            if len(stock_names) >= 10:
                break
        return stock_names

    def is_similar(self, new_text, existing_texts, threshold=0.8):
        for text in existing_texts:
            similarity = SequenceMatcher(None, new_text, text).ratio()
            if similarity > threshold:
                return True
        return False

    def analyze_sentiment_for_stock(self, stock_name):
        encText = urllib.parse.quote(stock_name)
        displayNum = 20
        url = f"https://openapi.naver.com/v1/search/news?query={encText}&display={displayNum}&sort=sim"

        context = ssl._create_unverified_context()
        request = urllib.request.Request(url)
        request.add_header("X-Naver-Client-Id", client_id)
        request.add_header("X-Naver-Client-Secret", client_secret)

        try:
            response = urllib.request.urlopen(request, context=context)
        except:
            print(f"[요청 실패] {stock_name}")
            return None

        rescode = response.getcode()
        if rescode != 200:
            return None

        response_body = response.read().decode('utf-8')
        news_data = json.loads(response_body)

        analyzer = SentimentIntensityAnalyzer()
        sentiment_scores = []
        unique_descriptions = []
        latest_title = ""
        market_impacts = []

        for item in news_data['items']:
            title = item['title'].replace('<b>', '').replace('</b>', '')
            description = item['description'].replace('<b>', '').replace('</b>', '')

            if self.is_similar(description, unique_descriptions):
                continue
            unique_descriptions.append(description)
            latest_title = title

            text_kr = f"{title}. {description}"
            
            # 시장 영향 분석
            market_impact = self.market_impact_analyzer.analyze_market_impact(text_kr, stock_name)
            market_impacts.append(market_impact)

            try:
                translated = GoogleTranslator(source='ko', target='en').translate(text_kr)
            except Exception as e:
                print(f"[번역 실패] {e}")
                continue

            sentiment = analyzer.polarity_scores(translated)
            compound_score = sentiment['compound']
            sentiment_scores.append(compound_score)

        if sentiment_scores:
            avg_score = sum(sentiment_scores) / len(sentiment_scores)
            buy_probability = int(((avg_score + 1) / 2) * 100)
            
            # 시장 영향 종합
            avg_impact_level = sum(impact['impact_level'] for impact in market_impacts) / len(market_impacts)
            affected_sectors = list(set([sector for impact in market_impacts for sector in impact['affected_sectors']]))
            
            return {
                "종목명": stock_name,
                "감성점수": round(avg_score, 3),
                "매수확률": buy_probability,
                "뉴스갯수": len(sentiment_scores),
                "최신기사제목": latest_title,
                "추천": "매수 추천" if avg_score > 0 else "매도 추천" if avg_score < 0 else "중립",
                "시장영향도": round(avg_impact_level, 3),
                "영향섹터": affected_sectors
            }
        else:
            return {
                "종목명": stock_name,
                "감성점수": None,
                "매수확률": None,
                "뉴스갯수": 0,
                "최신기사제목": "",
                "추천": "분석불가",
                "시장영향도": 0.0,
                "영향섹터": []
            }

    def run_sentiment_analysis(self):
        top_stocks = self.get_top100_by_volume()
        print("📊 거래대금 상위 종목 불러오기 완료\n")

        results = []
        for stock in top_stocks:
            print(f"🔍 {stock} 분석 중...")
            result = self.analyze_sentiment_for_stock(stock)
            if result:
                results.append(result)
            time.sleep(1.5)

        df = pd.DataFrame(results)
        df.to_csv("sentiment_result.csv", index=False, encoding='utf-8-sig')
        
        with open("sentiment_result.json", "w", encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        return df, results

if __name__ == "__main__":
    analyzer = NewsAnalyzer()
    df_result, json_result = analyzer.run_sentiment_analysis()