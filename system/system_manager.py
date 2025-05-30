import logging
from datetime import datetime
from typing import List, Dict, Optional
import asyncio
import pandas as pd
from agents.decision.decision_maker import DecisionMaker
from agents.news.news_analyzer import NewsAnalyzer
from agents.strategy.strategy_generator import StrategyGenerator, TimeHorizon
from agents.risk.risk_analyzer import RiskAnalyzer
from agents.technical.technical_analyzer import TechnicalAnalyzer
from agents.decision.models import (
    DecisionRequest, DecisionResponse, MarketContext,
    RiskParameters, AgentAnalysis, DecisionType
)
from agents.risk.models import RiskAssessmentRequest
from agents.strategy.models import MarketCondition
import FinanceDataReader as fdr
import openai
import os

class SystemManager:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # 로그 포맷 설정
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # 파일 핸들러 추가
        fh = logging.FileHandler(f'logs/system_manager_{datetime.now().strftime("%Y%m%d")}.log')
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        
        # 콘솔 핸들러 추가
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
        
        # 에이전트 초기화
        self.news_analyzer = NewsAnalyzer()
        self.strategy_generator = StrategyGenerator()
        self.risk_analyzer = RiskAnalyzer()
        self.technical_analyzer = TechnicalAnalyzer()
        self.decision_maker = DecisionMaker()
        
        # 시스템 상태
        self.is_running = False
        self.current_symbol = None
        self.current_timeframe = None
        self.last_decision = None
        self.last_analysis_time = None
        # 종목명→코드 자동 매핑용 전체 종목 리스트 캐싱
        try:
            self.stock_df = fdr.StockListing('KRX')
            self.name_to_code = dict(zip(self.stock_df['Name'], self.stock_df['Code']))
        except Exception as e:
            self.logger.error(f"종목 리스트 로딩 실패: {str(e)}")
            self.stock_df = None
            self.name_to_code = {}

    async def start_system(self, symbol: str, timeframe: str):
        """
        시스템을 시작합니다.
        """
        try:
            self.logger.info(f"시스템 시작: {symbol} ({timeframe})")
            self.is_running = True
            self.current_symbol = symbol
            self.current_timeframe = timeframe
            
            # 에이전트 초기화
            await self._initialize_agents()
            
            # 주기적 분석 시작
            await self._start_periodic_analysis()
            
        except Exception as e:
            self.logger.error(f"시스템 시작 중 오류 발생: {str(e)}")
            raise

    async def stop_system(self):
        """
        시스템을 중지합니다.
        """
        try:
            self.logger.info("시스템 중지")
            self.is_running = False
        except Exception as e:
            self.logger.error(f"시스템 중지 중 오류 발생: {str(e)}")
            raise

    async def _initialize_agents(self):
        """
        모든 에이전트를 초기화합니다.
        """
        try:
            # 각 에이전트는 생성자에서 초기화를 수행하므로 별도의 initialize 호출이 필요하지 않음
            self.logger.info("모든 에이전트 초기화 완료")
            
        except Exception as e:
            self.logger.error(f"에이전트 초기화 중 오류 발생: {str(e)}")
            raise

    def _create_risk_parameters(self, risk_analysis: Dict) -> RiskParameters:
        """
        리스크 파라미터를 생성합니다.
        """
        try:
            if isinstance(risk_analysis, dict):
                return RiskParameters(
                    max_position_size=0.1,  # 기본값
                    stop_loss_percentage=risk_analysis.get('stop_loss', 0.05),
                    take_profit_percentage=risk_analysis.get('target_price', 0.1),
                    max_drawdown=risk_analysis.get('max_loss', 0.1),
                    risk_reward_ratio=2.0  # 기본값
                )
            else:
                return RiskParameters(
                    max_position_size=0.1,
                    stop_loss_percentage=0.05,
                    take_profit_percentage=0.1,
                    max_drawdown=0.1,
                    risk_reward_ratio=2.0
                )
        except Exception as e:
            self.logger.error(f"리스크 파라미터 생성 중 오류 발생: {str(e)}")
            return RiskParameters(
                max_position_size=0.1,
                stop_loss_percentage=0.05,
                take_profit_percentage=0.1,
                max_drawdown=0.1,
                risk_reward_ratio=2.0
            )

    async def _start_periodic_analysis(self):
        """
        주기적인 분석을 시작합니다. (상위 10개 종목 전체 반복 분석)
        """
        while self.is_running:
            try:
                # 1. 상위 10개 종목 선정 (거래량 기준)
                if self.stock_df is not None:
                    top10_df = self.stock_df.sort_values(by='Volume', ascending=False).head(10)
                    target_symbols = list(top10_df['Code'])
                else:
                    self.logger.error("종목 데이터가 없습니다. 기본 종목 리스트로 대체합니다.")
                    target_symbols = [self.current_symbol]

                news_analyses = []
                strategy_results = []
                technical_analyses = []
                risk_analyses = []

                for symbol in target_symbols:
                    self.logger.info(f"[{symbol}] 뉴스 분석 시작")
                    news = self.news_analyzer.analyze_sentiment_for_stock(symbol)
                    if not news or (isinstance(news, dict) and news.get('감성점수') is None):
                        self.logger.warning(f"[{symbol}] 뉴스 분석 실패 또는 결과 없음. 이후 분석 건너뜀.")
                        continue
                    self.logger.info(f"[{symbol}] 뉴스 분석 완료")
                    news_analyses.append(news)
                    # MarketCondition 객체 변환
                    market_trend = 'neutral'
                    if isinstance(news, dict):
                        impact = news.get('시장영향도', 0.0)
                        if impact > 0:
                            market_trend = 'bullish'
                        elif impact < 0:
                            market_trend = 'bearish'
                    market_condition_obj = MarketCondition(
                        market_trend=market_trend,
                        volatility_level='high' if news.get('뉴스갯수', 0) >= 10 else 'low',
                        trading_volume=0.0,
                        sector_performance={},
                        major_events=[],
                        timestamp=datetime.now()
                    )
                    self.logger.info(f"[{symbol}] 전략 분석 시작")
                    strategy = self.strategy_generator.generate_strategy(
                        user_input="",
                        market_conditions=market_condition_obj,
                        risk_tolerance=0.5,
                        time_horizon=TimeHorizon.MEDIUM_TERM
                    )
                    strategy_results.append(strategy)
                    self.logger.info(f"[{symbol}] 전략 분석 완료")
                    self.logger.info(f"[{symbol}] 기술적 분석 시작")
                    tech = self.technical_analyzer.analyze(symbol)
                    technical_analyses.append(tech)
                    self.logger.info(f"[{symbol}] 기술적 분석 완료")
                    self.logger.info(f"[{symbol}] 리스크 분석 시작")
                    if tech and isinstance(tech, dict):
                        market_data = tech.get('raw_data', pd.DataFrame())
                        technical_indicators = tech.get('indicators', {})
                    else:
                        market_data = pd.DataFrame()
                        technical_indicators = {}
                    risk = self.risk_analyzer.assess_risk(
                        request=RiskAssessmentRequest(
                            market_data=market_data,
                            technical_indicators=technical_indicators,
                            risk_tolerance=0.5,
                            portfolio_value=1000000.0,
                            positions=[],
                            market_context=None
                        )
                    )
                    risk_analyses.append(risk)
                    self.logger.info(f"[{symbol}] 리스크 분석 완료")

                # 2. 보고서 생성 (분석 결과 집계)
                await self._generate_investment_report(
                    target_symbols,
                    news_analyses,
                    strategy_results,
                    risk_analyses,
                    technical_analyses
                )

                # 3. 상태 업데이트 및 대기
                self.last_analysis_time = datetime.now()
                await self._wait_for_next_analysis()

            except Exception as e:
                self.logger.error(f"주기적 분석 중 오류 발생: {str(e)}")
                self.logger.error("시스템을 중단합니다.")
                self.is_running = False
                raise

    async def _generate_investment_report(self, symbols, news_analyses, strategies, risk_analyses, technical_analyses, debate_results=None):
        """
        상위 10개 종목 전체 분석 결과를 표/리스트로 구조화해 프롬프트에 포함, 보고서 파일로 저장
        """
        import openai
        import os
        from datetime import datetime
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            self.logger.error("OPENAI_API_KEY 환경변수가 설정되어 있지 않습니다.")
            return
        client = openai.OpenAI(api_key=api_key)

        def make_table(rows, columns):
            table = '| ' + ' | '.join(columns) + ' |\n'
            table += '| ' + ' | '.join(['---']*len(columns)) + ' |\n'
            for row in rows:
                table += '| ' + ' | '.join(str(row.get(col, '')) for col in columns) + ' |\n'
            return table

        # 각 분석 결과를 표/리스트로 요약
        news_rows = []
        for symbol, news in zip(symbols, news_analyses):
            if isinstance(news, dict):
                sentiment_score = news.get('sentiment_score', 'N/A')
                market_impact = news.get('시장영향도', 'N/A')
                keywords = ','.join(news.get('keywords', [])) if 'keywords' in news else ''
            else:
                sentiment_score = getattr(news, 'sentiment_score', 'N/A')
                market_impact = getattr(news, '시장영향도', 'N/A')
                keywords = ','.join(getattr(news, 'keywords', [])) if hasattr(news, 'keywords') else ''
            news_rows.append({
                '종목': symbol,
                '감성점수': sentiment_score,
                '시장영향도': market_impact,
                '키워드': keywords
            })
        news_table = make_table(news_rows, ['종목', '감성점수', '시장영향도', '키워드'])

        strategy_rows = []
        for symbol, strat in zip(symbols, strategies):
            if isinstance(strat, dict):
                recommendation = strat.get('recommendation', strat.get('strategy_type', 'N/A'))
                explanation = strat.get('explanation', '')
            else:
                recommendation = getattr(strat, 'strategy_type', 'N/A')
                explanation = getattr(strat, 'explanation', '')
            strategy_rows.append({
                '종목': symbol,
                '추천전략': recommendation,
                '설명': explanation
            })
        strategy_table = make_table(strategy_rows, ['종목', '추천전략', '설명'])

        risk_rows = []
        for symbol, risk in zip(symbols, risk_analyses):
            if isinstance(risk, dict):
                risk_score = risk.get('risk_score', 'N/A')
                volatility = risk.get('volatility', 'N/A')
                risk_level = risk.get('risk_level', 'N/A')
            else:
                risk_score = getattr(risk, 'risk_score', 'N/A')
                volatility = getattr(risk, 'volatility', 'N/A')
                risk_level = getattr(risk, 'risk_level', 'N/A')
            risk_rows.append({
                '종목': symbol,
                '리스크점수': risk_score,
                '변동성': volatility,
                '리스크레벨': risk_level
            })
        risk_table = make_table(risk_rows, ['종목', '리스크점수', '변동성', '리스크레벨'])

        tech_rows = []
        for symbol, tech in zip(symbols, technical_analyses):
            tech_row = {'종목': symbol}
            if isinstance(tech, dict):
                tech_row['RSI'] = tech.get('indicators', {}).get('rsi', 'N/A')
                tech_row['MACD'] = tech.get('indicators', {}).get('macd', 'N/A')
                tech_row['볼린저밴드상단'] = tech.get('indicators', {}).get('bb_upper', 'N/A')
                tech_row['볼린저밴드하단'] = tech.get('indicators', {}).get('bb_lower', 'N/A')
                tech_row['최근종가'] = tech.get('current_price', 'N/A')
            else:
                indicators = getattr(tech, 'indicators', {}) if hasattr(tech, 'indicators') else {}
                tech_row['RSI'] = indicators.get('rsi', 'N/A') if isinstance(indicators, dict) else 'N/A'
                tech_row['MACD'] = indicators.get('macd', 'N/A') if isinstance(indicators, dict) else 'N/A'
                tech_row['볼린저밴드상단'] = indicators.get('bb_upper', 'N/A') if isinstance(indicators, dict) else 'N/A'
                tech_row['볼린저밴드하단'] = indicators.get('bb_lower', 'N/A') if isinstance(indicators, dict) else 'N/A'
                tech_row['최근종가'] = getattr(tech, 'current_price', 'N/A')
            tech_rows.append(tech_row)
        tech_table = make_table(tech_rows, ['종목', 'RSI', 'MACD', '볼린저밴드상단', '볼린저밴드하단', '최근종가'])

        # debate 결과 기반 포트폴리오 비중 산출 함수
        def _calculate_portfolio_allocation(symbols, debate_results):
            # debate_results: List[List[Dict]] (각 종목별 에이전트 의견 리스트)
            allocations = {}
            total_score = 0.0
            for symbol, debates in zip(symbols, debate_results or [[]]*len(symbols)):
                # 각 debate: {'decision': 'BUY'/'HOLD'/'SELL', 'confidence': float, ...}
                buy_score = sum(1.0 * (d.get('decision') == 'BUY') * d.get('confidence', 0.5) for d in debates)
                hold_score = sum(0.5 * (d.get('decision') == 'HOLD') * d.get('confidence', 0.5) for d in debates)
                sell_score = sum(-1.0 * (d.get('decision') == 'SELL') * d.get('confidence', 0.5) for d in debates)
                score = max(buy_score + hold_score + sell_score, 0.0)
                allocations[symbol] = score
                total_score += score
            # 전체 신뢰도 낮으면 현금 비중↑
            avg_score = total_score / (len(symbols) or 1)
            cash_ratio = max(0.1, 1.0 - total_score) if total_score < 1.0 else 0.05
            # 비율 정규화
            if total_score > 0:
                for symbol in allocations:
                    allocations[symbol] = round((allocations[symbol] / total_score) * (1-cash_ratio), 3)
            else:
                for symbol in allocations:
                    allocations[symbol] = round(0.9 / len(symbols), 3)
            return allocations, round(cash_ratio, 3)

        # debate_results 인자가 없으면 빈 리스트로 처리
        allocations, cash_ratio = _calculate_portfolio_allocation(symbols, debate_results or [[] for _ in symbols])
        portfolio_rows = [
            {'자산': symbol, '비율': f"{int(allocations[symbol]*100)}%"} for symbol in symbols
        ]
        portfolio_rows.append({'자산': '현금', '비율': f"{int(cash_ratio*100)}%"})
        portfolio_table = make_table(portfolio_rows, ['자산', '비율'])

        # 프롬프트 템플릿(7번 삭제, 6번 뒤에 최종 포트폴리오 표 추가)
        prompt = f"""
        국내 주식 자동매매 시스템에서 최근 거래량이 가장 많은 상위 10개 종목을 대상으로, 아래 항목별로 포트폴리오 투자 분석 보고서를 작성하세요.

        - 각 항목은 **별도의 소제목**으로 구분하고, 실제 데이터(표, 수치, 차트 등)와 논리적 근거를 최대한 활용해 상세히 기술하세요.
        - 각 종목별 분석 결과는 표 또는 리스트로 정리하세요.

        1. **전략 분석**
        {strategy_table}
        2. **뉴스/이슈 분석**
        {news_table}
        3. **리스크 분석**
        {risk_table}
        4. **기술적 분석**
        {tech_table}
        5. **시계열(통계/머신러닝 기반) 분석**
        (각 종목별 시계열 예측 결과를 요약해 표로 정리)
        6. **포트폴리오 구성 이유 및 기대효과**
        (상기 분석 결과를 종합하여, 상위 10개 종목 포트폴리오를 구성한 최종 이유와 기대 효과를 제시)
        
        **최종 포트폴리오 구성**
        {portfolio_table}
        
        분석은 최근 데이터(최소 최근 1년 이내)를 기준으로 하며, 데이터가 부족한 부분은 추정 근거를 명확히 밝혀 주세요.
        """
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
            report_md = response.choices[0].message.content
            report_dir = "investment_reports"
            os.makedirs(report_dir, exist_ok=True)
            now = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(report_dir, f"investment_report_portfolio_{now}.md")
            with open(filename, "w", encoding="utf-8") as f:
                f.write(report_md)
            self.logger.info(f"포트폴리오 투자 분석 보고서가 {filename}에 저장되었습니다.")
        except Exception as e:
            self.logger.error(f"투자 분석 보고서 생성 중 오류 발생: {str(e)}")

    async def _wait_for_next_analysis(self):
        """
        다음 분석까지 대기합니다.
        """
        try:
            # 시간프레임에 따른 대기 시간 설정
            if self.current_timeframe == "1m":
                await asyncio.sleep(60)
            elif self.current_timeframe == "5m":
                await asyncio.sleep(300)
            elif self.current_timeframe == "15m":
                await asyncio.sleep(900)
            elif self.current_timeframe == "1h":
                await asyncio.sleep(3600)
            elif self.current_timeframe == "1d":
                await asyncio.sleep(86400)
            else:
                await asyncio.sleep(3600)  # 기본값 1시간
                
        except Exception as e:
            self.logger.error(f"대기 중 오류 발생: {str(e)}")
            raise 

    async def analyze_stock(self, symbol: str) -> Dict:
        """
        주식 분석을 수행합니다.
        """
        try:
            self.logger.info(f"{symbol} 분석 시작")
            
            # 1. 뉴스 분석
            self.logger.info(f"{symbol} 뉴스 분석 시작")
            news_analysis = await self.news_analyzer.analyze(symbol)
            self.logger.info(f"{symbol} 뉴스 분석 완료")
            
            # 2. 전략 수립
            self.logger.info(f"{symbol} 전략 수립 시작")
            strategy = await self.strategy_generator.generate_strategy(
                user_input="",  # 빈 문자열로 설정
                market_conditions=news_analysis,  # 뉴스 분석 결과에서 자동으로 생성
                risk_tolerance=0.5,  # 기본값
                time_horizon=TimeHorizon.MEDIUM_TERM  # 명시적으로 설정
            )
            self.logger.info(f"{symbol} 전략 수립 완료")
            
            # 3. 리스크 분석
            self.logger.info(f"{symbol} 리스크 분석 시작")
            risk_analysis = await self.risk_analyzer.analyze(symbol, strategy)
            self.logger.info(f"{symbol} 리스크 분석 완료")
            
            # 4. 기술적 분석 (전략에서 선정된 종목들에 대해)
            self.logger.info(f"{symbol} 기술적 분석 시작")
            technical_analyses = []
            if not strategy.target_assets:
                self.logger.warning("분석할 종목이 없습니다.")
            else:
                # 종목코드 기준으로만 분석, 중복 제거
                symbols_to_analyze = set()
                for symbol in strategy.target_assets:
                    symbol_code = self._get_stock_code(symbol)
                    if symbol_code:
                        symbols_to_analyze.add(symbol_code)
                    else:
                        self.logger.warning(f"{symbol}의 종목코드 변환에 실패했습니다.")
                for symbol_code in symbols_to_analyze:
                    # 뉴스 분석 결과에 포함된 종목만 분석
                    symbol_name = self._get_stock_name(symbol_code)
                    if symbol_name not in news_symbols:
                        self.logger.info(f"{symbol_code}({symbol_name})는 뉴스 분석 결과에 없으므로 기술적 분석을 건너뜁니다.")
                        continue
                    try:
                        symbol_yf = symbol_code + ".KS"
                        analysis = self.technical_analyzer.analyze(symbol_yf)
                        if analysis:
                            technical_analyses.append(analysis)
                        else:
                            self.logger.warning(f"{symbol_code}의 기술적 분석에 실패했습니다.")
                    except Exception as e:
                        self.logger.error(f"{symbol_code} 분석 중 오류 발생: {str(e)}")
                        continue
            if not technical_analyses:
                self.logger.warning("기술적 분석 결과가 없습니다.")
            else:
                self.logger.info("기술적 분석 완료")

            # 5. 리스크 분석 (전략 결과를 인풋으로)
            # 기술적 분석 결과에서 market_data, technical_indicators 추출
            if technical_analyses and isinstance(technical_analyses[0], dict):
                market_data = technical_analyses[0].get('raw_data', pd.DataFrame())
                technical_indicators = technical_analyses[0].get('indicators', {})
            else:
                market_data = pd.DataFrame()
                technical_indicators = {}
            risk_analysis = self.risk_analyzer.assess_risk(
                request=RiskAssessmentRequest(
                    market_data=market_data,
                    technical_indicators=technical_indicators,
                    risk_tolerance=0.5,
                    portfolio_value=1000000.0,
                    positions=[],
                    market_context=None
                )
            )
            if not risk_analysis:
                raise ValueError("리스크 분석에 실패했습니다.")
            self.logger.info("리스크 분석 완료")

            # 6. 토론 기반 최종 결정 의견 도출
            # 기술적 분석을 실제로 수행한 종목코드 리스트에서 첫 번째 종목코드를 사용
            analyzed_symbols = [symbol_code for symbol_code in symbols_to_analyze if self._get_stock_name(symbol_code) in news_symbols]
            if analyzed_symbols:
                debate_symbol = analyzed_symbols[0]
            else:
                debate_symbol = target_symbol if target_symbol and target_symbol.isdigit() else self._get_stock_code(target_symbol)

            debate_context = {
                'symbol': debate_symbol,
                'market_conditions': market_condition_obj  # news_json 전체가 아니라 변환된 객체
            }
            debate_opinions = self.agent_debate_round(debate_context)
            self.logger.info(f"토론 라운드 결과: {debate_opinions}")

            # 7. 최종 결정 (토론 결과와 각 분석 결과를 종합)
            decision = await self._make_final_decision(
                news_analysis,
                strategy,
                risk_analysis,
                technical_analyses
            )
            if not decision:
                raise ValueError("최종 결정 생성에 실패했습니다.")

            # 7-1. 투자 분석 보고서 자동 생성
            await self._generate_investment_report(symbol, news_analysis, strategy, risk_analysis, technical_analyses)

            # 8. 결정 실행
            await self._execute_decision(decision)

            # 9. 상태 업데이트
            self.last_decision = decision
            self.last_analysis_time = datetime.now()

            # 10. 대기
            await self._wait_for_next_analysis()

            # 11. 최종 분석 결과 종합
            self.logger.info(f"{symbol} 최종 분석 결과 종합 시작")
            analysis_result = {
                'symbol': symbol,
                'news_analysis': news_analysis,
                'strategy': strategy,
                'risk_analysis': risk_analysis,
                'technical_analysis': technical_analyses[0] if technical_analyses else None,
                'market_context': market_context,
                'timestamp': datetime.now().isoformat()
            }
            self.logger.info(f"{symbol} 최종 분석 결과 종합 완료")
            
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"{symbol} 분석 중 오류 발생: {str(e)}")
            return {}

    def _get_stock_code(self, symbol: str) -> str:
        """
        종목명을 종목코드로 변환합니다. (FinanceDataReader 기반 자동 매핑)
        """
        try:
            # 종목코드가 이미 숫자로만 이루어진 경우
            if symbol.isdigit():
                return symbol
            # 자동 매핑
            code = self.name_to_code.get(symbol)
            if code:
                return code
            # 부분 일치(공백 제거, 대소문자 무시)
            for name, code in self.name_to_code.items():
                if name.replace(' ', '').lower() == symbol.replace(' ', '').lower():
                    return code
            return None
        except Exception as e:
            self.logger.error(f"종목코드 변환 중 오류 발생: {str(e)}")
            return None

    def _get_stock_name(self, code: str) -> str:
        """
        종목코드를 종목명으로 변환합니다. (FinanceDataReader 기반)
        """
        try:
            if self.stock_df is not None:
                result = self.stock_df[self.stock_df['Code'] == code]
                if not result.empty:
                    return result.iloc[0]['Name']
            return code
        except Exception as e:
            self.logger.error(f'종목명 변환 중 오류 발생: {str(e)}')
            return code

    def agent_debate_round(self, context):
        """
        에이전트 토론 라운드: 각 에이전트가 의견을 제시하고, 타 에이전트 의견을 반영해 재평가합니다.
        """
        agents = [
            self.technical_analyzer,
            self.news_analyzer,
            self.strategy_generator,
            self.risk_analyzer
        ]
        self.logger.info("[디베이트] 1라운드: 각 에이전트의 최초 의견 수집 시작")
        opinions = []
        for agent in agents:
            if hasattr(agent, 'propose'):
                op = agent.propose(context)
                self.logger.info(f"[디베이트][1R] {op.get('agent', agent.__class__.__name__)}: {op.get('decision')} (신뢰도: {op.get('confidence')}, 이유: {op.get('reason')})")
                opinions.append(op)
            else:
                op = {'agent': agent.__class__.__name__, 'decision': 'HOLD', 'confidence': 0.5, 'reason': '기본값'}
                self.logger.info(f"[디베이트][1R] {op['agent']}: {op['decision']} (신뢰도: {op['confidence']}, 이유: {op['reason']})")
                opinions.append(op)

        self.logger.info("[디베이트] 2라운드: 타 에이전트 의견 반영 후 재평가 시작")
        final_opinions = []
        for idx, agent in enumerate(agents):
            others = [op for i, op in enumerate(opinions) if i != idx]
            if hasattr(agent, 'debate'):
                op = agent.debate(context, others)
                self.logger.info(f"[디베이트][2R] {op.get('agent', agent.__class__.__name__)}: {op.get('decision')} (신뢰도: {op.get('confidence')}, 이유: {op.get('reason')})")
                final_opinions.append(op)
            else:
                op = opinions[idx]
                self.logger.info(f"[디베이트][2R] {op.get('agent', agent.__class__.__name__)}: {op.get('decision')} (신뢰도: {op.get('confidence')}, 이유: {op.get('reason')})")
                final_opinions.append(op)

        self.logger.info(f"[디베이트] 최종 토론 결과: {final_opinions}")
        return final_opinions

    async def _make_final_decision(
        self,
        news_analysis: Dict,
        strategy: Dict,
        risk_analysis: Dict,
        technical_analyses: List[Dict]
    ) -> DecisionResponse:
        """
        최종 투자 결정을 생성합니다.
        """
        try:
            if not all([news_analysis, strategy, risk_analysis]):
                raise ValueError("필수 분석 데이터가 누락되었습니다.")
            
            # 리스크 파라미터 설정
            risk_parameters = self._create_risk_parameters(risk_analysis)
            
            # 모든 에이전트의 분석 결과를 종합
            agent_analyses = []
            
            # 뉴스 분석 결과 추가
            if isinstance(news_analysis, dict):
                agent_analyses.append(
                    AgentAnalysis(
                        agent_name="news_analyzer",
                        analysis_type="news",
                        confidence_score=news_analysis.get('confidence', 0.0),
                        recommendation=news_analysis,
                        timestamp=datetime.now()
                    )
                )
            
            # 전략 분석 결과 추가
            if isinstance(strategy, dict):
                agent_analyses.append(
                    AgentAnalysis(
                        agent_name="strategy_generator",
                        analysis_type="strategy",
                        confidence_score=strategy.get('confidence', 0.0),
                        recommendation=strategy,
                        timestamp=datetime.now()
                    )
                )
            
            # 리스크 분석 결과 추가
            if isinstance(risk_analysis, dict):
                agent_analyses.append(
                    AgentAnalysis(
                        agent_name="risk_analyzer",
                        analysis_type="risk",
                        confidence_score=risk_analysis.get('confidence', 0.0),
                        recommendation=risk_analysis,
                        timestamp=datetime.now()
                    )
                )
            
            # 기술적 분석 결과 추가
            if technical_analyses:
                for analysis in technical_analyses:
                    if isinstance(analysis, dict):
                        agent_analyses.append(
                            AgentAnalysis(
                                agent_name="technical_analyzer",
                                analysis_type="technical",
                                confidence_score=analysis.get('confidence', 0.0),
                                recommendation=analysis,
                                timestamp=datetime.now()
                            )
                        )
            
            # 시장 상황 분석
            market_context = await self._analyze_market_context(
                news_analysis,
                strategy,
                risk_analysis,
                technical_analyses
            )
            
            # 현재 가격 설정
            current_price = 0.0
            if technical_analyses and isinstance(technical_analyses[0], dict):
                current_price = technical_analyses[0].get('current_price', 0.0)
            
            # 최종 결정 요청
            request = DecisionRequest(
                symbol=self.current_symbol,
                timeframe=self.current_timeframe,
                current_price=current_price,
                market_context=market_context,
                risk_parameters=risk_parameters,
                agent_analyses=agent_analyses,
                technical_analysis=technical_analyses[0] if technical_analyses else None,
                news_analysis=news_analysis if isinstance(news_analysis, dict) else news_analysis.dict() if hasattr(news_analysis, 'dict') else {},
                strategy=strategy.dict() if hasattr(strategy, 'dict') else strategy,
                risk_assessment=risk_analysis.dict() if hasattr(risk_analysis, 'dict') else risk_analysis if isinstance(risk_analysis, dict) else {}
            )
            
            decision = self.decision_maker.make_decision(request)
            if not decision:
                raise ValueError("결정 생성에 실패했습니다.")
            
            return decision
            
        except Exception as e:
            self.logger.error(f"최종 결정 생성 중 오류 발생: {str(e)}")
            raise

    async def _analyze_market_context(
        self,
        news_analysis: Dict,
        strategy: Dict,
        risk_analysis: Dict,
        technical_analyses: List[Dict]
    ) -> MarketContext:
        """
        시장 상황을 분석합니다.
        """
        try:
            # 뉴스 감성과 시장 상황 통합
            market_sentiment = news_analysis.get('market_sentiment', 'NEUTRAL') if isinstance(news_analysis, dict) else 'NEUTRAL'
            
            # 기술적 분석에서 변동성과 추세 강도 추출
            volatility_level = 0.0
            trend_strength = 0.0
            if technical_analyses and isinstance(technical_analyses[0], dict):
                volatility_level = technical_analyses[0].get('volatility', 0.0)
                trend_strength = technical_analyses[0].get('trend_strength', 0.0)
            
            # 거래량 프로필 생성
            volume_profile = {}
            if technical_analyses and isinstance(technical_analyses[0], dict):
                volume_profile = technical_analyses[0].get('volume_profile', {})
            
            # 전략에서 시장 상황 추출
            market_condition = strategy.get('market_condition', 'UNKNOWN') if isinstance(strategy, dict) else 'UNKNOWN'
            
            return MarketContext(
                market_condition=market_condition,
                volatility_level=volatility_level,
                trend_strength=trend_strength,
                volume_profile=volume_profile,
                market_sentiment=market_sentiment
            )
            
        except Exception as e:
            self.logger.error(f"시장 상황 분석 중 오류 발생: {str(e)}")
            return MarketContext(
                market_condition="UNKNOWN",
                volatility_level=0.0,
                trend_strength=0.0,
                volume_profile={},
                market_sentiment="NEUTRAL"
            )

    async def _execute_decision(self, decision: DecisionResponse):
        """
        결정을 실행합니다.
        """
        try:
            if not decision:
                raise ValueError("실행할 결정이 없습니다.")
                
            if decision.error_message:
                self.logger.error(f"결정 실행 실패: {decision.error_message}")
                return
            
            self.logger.info("=== 결정 실행 시작 ===")
            self.logger.info(f"결정 타입: {decision.decision}")
            self.logger.info(f"신뢰도: {float(decision.confidence):.2f}")
            self.logger.info(f"포지션 크기: {float(decision.position_size):.1f}%")
            self.logger.info(f"진입가: {float(decision.entry_price):,.0f}원")
            self.logger.info(f"목표가: {float(decision.target_price):,.0f}원")
            self.logger.info(f"손절가: {float(decision.stop_loss):,.0f}원")
            self.logger.info(f"예상 수익률: {float(decision.expected_return):.1f}%")
            self.logger.info(f"최대 손실률: {float(decision.max_loss):.1f}%")
            
            self.logger.info("=== 결정 근거 ===")
            for reason in decision.reasons:
                self.logger.info(f"- {reason}")
            
            # 결정 실행 로직
            if decision.decision == DecisionType.BUY:
                self.logger.info("매수 신호 실행")
                # 매수 로직 구현
            elif decision.decision == DecisionType.SELL:
                self.logger.info("매도 신호 실행")
                # 매도 로직 구현
            elif decision.decision == DecisionType.HOLD:
                self.logger.info("홀드 신호 실행")
                # 홀드 로직 구현
            else:
                self.logger.info("대기 신호 실행")
                # 대기 로직 구현
            
            self.logger.info("=== 결정 실행 완료 ===")
            
        except Exception as e:
            self.logger.error(f"결정 실행 중 오류 발생: {str(e)}")
            raise

if __name__ == "__main__":
    import asyncio
    manager = SystemManager()
    # 예시: 삼성전자, 1시간봉
    asyncio.run(manager.start_system(symbol="005930", timeframe="1h"))
    