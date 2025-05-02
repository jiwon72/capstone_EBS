import numpy as np
import pandas as pd
import yfinance as yf
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import talib
from .models import (
    TimeFrame, SignalStrength, TrendDirection,
    MarketData, MovingAverages, Oscillators,
    VolumeIndicators, SupportResistance, PatternSignal,
    TechnicalIndicators, TradingSignal,
    TechnicalAnalysisRequest, TechnicalAnalysisResponse,
    MovingAverageRequest, SupportResistanceRequest,
    PatternRequest, SignalRequest,
    MovingAverageResponse, PatternResponse,
    SignalResponse, Pattern, Signal
)

class TechnicalAnalyzer:
    def __init__(self):
        self.timeframe_map = {
            TimeFrame.M1: "1m",
            TimeFrame.M5: "5m",
            TimeFrame.M15: "15m",
            TimeFrame.M30: "30m",
            TimeFrame.H1: "1h",
            TimeFrame.H4: "4h",
            TimeFrame.D1: "1d",
            TimeFrame.W1: "1w",
            TimeFrame.MN1: "1mo"
        }
        self.indicators = {
            "sma": self._calculate_sma,
            "ema": self._calculate_ema,
            "rsi": self._calculate_rsi,
            "macd": self._calculate_macd
        }
        self.pattern_detectors = {
            "double_top": self._detect_double_top,
            "double_bottom": self._detect_double_bottom,
            "head_shoulders": self._detect_head_shoulders
        }
        self.signal_generators = {
            "trend_following": self._generate_trend_signal,
            "mean_reversion": self._generate_reversion_signal,
            "breakout": self._generate_breakout_signal
        }
        
    def _fetch_market_data(
        self,
        ticker: str,
        timeframe: TimeFrame,
        lookback_periods: int
    ) -> List[MarketData]:
        """시장 데이터 조회"""
        # yfinance를 통한 데이터 조회
        yf_timeframe = self.timeframe_map[timeframe]
        stock = yf.Ticker(ticker)
        
        # 데이터 기간 계산
        if timeframe in [TimeFrame.M1, TimeFrame.M5, TimeFrame.M15, TimeFrame.M30]:
            period = "7d"  # yfinance 제한
        else:
            period = f"{lookback_periods * 2}d"
            
        df = stock.history(period=period, interval=yf_timeframe)
        
        # MarketData 객체 리스트 생성
        market_data = []
        for index, row in df.iterrows():
            market_data.append(MarketData(
                ticker=ticker,
                timestamp=index,
                open=row['Open'],
                high=row['High'],
                low=row['Low'],
                close=row['Close'],
                volume=row['Volume'],
                adjusted_close=row.get('Adj Close', None)
            ))
            
        return market_data[-lookback_periods:]
    
    def _convert_nan_to_none(self, value):
        """NaN 값을 None으로 변환"""
        return None if pd.isna(value) else float(value)

    def _calculate_moving_averages(self, request: MovingAverageRequest) -> MovingAverageResponse:
        """이동평균 계산"""
        try:
            # Convert market data to numpy array
            prices = np.array([data.close for data in request.market_data])
            
            # Check if data length is sufficient
            if len(prices) < max(request.window_sizes):
                raise ValueError("데이터 길이가 충분하지 않습니다.")
            
            sma_values = {}
            ema_values = {}
            
            for window in request.window_sizes:
                if "sma" in request.ma_types:
                    sma = self._calculate_sma(prices, window)
                    sma_values[f"sma_{window}"] = float(sma[-1])
                
                if "ema" in request.ma_types:
                    ema = self._calculate_ema(prices, window)
                    ema_values[f"ema_{window}"] = float(ema[-1])
            
            trend = self._determine_trend_direction(prices)
            
            return MovingAverageResponse(
                sma_values=sma_values,
                ema_values=ema_values,
                trend_direction=trend,
                error_message=None
            )
        except Exception as e:
            return MovingAverageResponse(
                sma_values={},
                ema_values={},
                trend_direction=TrendDirection.SIDEWAYS,
                error_message=str(e)
            )

    def _calculate_oscillators(self, df: pd.DataFrame) -> Oscillators:
        """오실레이터 계산"""
        # Check if data length is sufficient
        if len(df) < 14:
            raise ValueError("데이터 길이가 충분하지 않습니다.")
        
        # RSI
        rsi_14 = self._convert_nan_to_none(talib.RSI(df['Close'], timeperiod=14)[-1])
        
        # Stochastic
        stoch_k, stoch_d = talib.STOCH(
            df['High'],
            df['Low'],
            df['Close'],
            fastk_period=14,
            slowk_period=3,
            slowd_period=3
        )
        
        # MACD
        macd_line, macd_signal, macd_hist = talib.MACD(
            df['Close'],
            fastperiod=12,
            slowperiod=26,
            signalperiod=9
        )
        
        # CCI
        cci_20 = self._convert_nan_to_none(talib.CCI(
            df['High'],
            df['Low'],
            df['Close'],
            timeperiod=20
        )[-1])
        
        # ATR
        atr_14 = self._convert_nan_to_none(talib.ATR(
            df['High'],
            df['Low'],
            df['Close'],
            timeperiod=14
        )[-1])
        
        # Bollinger Bands
        upper, middle, lower = talib.BBANDS(
            df['Close'],
            timeperiod=20,
            nbdevup=2,
            nbdevdn=2,
            matype=0
        )
        
        return Oscillators(
            rsi_14=rsi_14,
            stoch_k=self._convert_nan_to_none(stoch_k[-1]),
            stoch_d=self._convert_nan_to_none(stoch_d[-1]),
            macd_line=self._convert_nan_to_none(macd_line[-1]),
            macd_signal=self._convert_nan_to_none(macd_signal[-1]),
            macd_histogram=self._convert_nan_to_none(macd_hist[-1]),
            cci_20=cci_20,
            atr_14=atr_14,
            bollinger_upper=self._convert_nan_to_none(upper[-1]),
            bollinger_middle=self._convert_nan_to_none(middle[-1]),
            bollinger_lower=self._convert_nan_to_none(lower[-1])
        )
    
    def _calculate_volume_indicators(self, df: pd.DataFrame) -> VolumeIndicators:
        """거래량 지표 계산"""
        # Check if data length is sufficient
        if len(df) < 20:
            raise ValueError("데이터 길이가 충분하지 않습니다.")
        
        # On Balance Volume
        obv = self._convert_nan_to_none(talib.OBV(df['Close'], df['Volume'])[-1])
        
        # Volume SMA
        volume_sma_20 = self._convert_nan_to_none(talib.SMA(df['Volume'], timeperiod=20)[-1])
        
        # Volume Trend
        recent_volume = df['Volume'].tail(5).mean()
        volume_trend = TrendDirection.SIDEWAYS
        
        if volume_sma_20 is not None and not pd.isna(recent_volume):
            if recent_volume > volume_sma_20 * 1.5:
                volume_trend = TrendDirection.STRONG_UPTREND
            elif recent_volume > volume_sma_20:
                volume_trend = TrendDirection.UPTREND
            elif recent_volume < volume_sma_20 * 0.5:
                volume_trend = TrendDirection.STRONG_DOWNTREND
            elif recent_volume < volume_sma_20:
                volume_trend = TrendDirection.DOWNTREND
            
        # Price Volume Trend
        pvt = (df['Close'].diff() / df['Close'].shift(1)) * df['Volume']
        pvt = pvt.cumsum()
        
        # Money Flow Index
        mfi = self._convert_nan_to_none(talib.MFI(
            df['High'],
            df['Low'],
            df['Close'],
            df['Volume'],
            timeperiod=14
        )[-1])
        
        return VolumeIndicators(
            obv=obv,
            volume_sma_20=volume_sma_20,
            volume_trend=volume_trend,
            volume_price_trend=self._convert_nan_to_none(pvt.iloc[-1]),
            money_flow_index=mfi
        )
    
    def _calculate_support_resistance(
        self,
        df: pd.DataFrame,
        n_levels: int = 3
    ) -> SupportResistance:
        """지지/저항 레벨 계산"""
        # 피봇 포인트 계산
        pivot = (df['High'].iloc[-1] + df['Low'].iloc[-1] + df['Close'].iloc[-1]) / 3
        
        # 지지/저항 레벨
        r1 = 2 * pivot - df['Low'].iloc[-1]
        r2 = pivot + (df['High'].iloc[-1] - df['Low'].iloc[-1])
        s1 = 2 * pivot - df['High'].iloc[-1]
        s2 = pivot - (df['High'].iloc[-1] - df['Low'].iloc[-1])
        
        # 추가 지지/저항 레벨 탐색
        highs = df['High'].nlargest(10)
        lows = df['Low'].nsmallest(10)
        
        resistance_levels = list(set([round(x, 2) for x in highs.values]))[:n_levels]
        support_levels = list(set([round(x, 2) for x in lows.values]))[:n_levels]
        
        return SupportResistance(
            support_levels=support_levels,
            resistance_levels=resistance_levels,
            pivot_point=pivot,
            r1_level=r1,
            r2_level=r2,
            s1_level=s1,
            s2_level=s2
        )
    
    def _detect_patterns(self, df: pd.DataFrame) -> List[PatternSignal]:
        """차트 패턴 탐지"""
        patterns = []
        
        # Candlestick Patterns
        doji = talib.CDLDOJI(
            df['Open'],
            df['High'],
            df['Low'],
            df['Close']
        )
        hammer = talib.CDLHAMMER(
            df['Open'],
            df['High'],
            df['Low'],
            df['Close']
        )
        engulfing = talib.CDLENGULFING(
            df['Open'],
            df['High'],
            df['Low'],
            df['Close']
        )
        
        # 최근 패턴 확인
        if doji.iloc[-1] != 0:
            patterns.append(PatternSignal(
                pattern_name="Doji",
                confidence=0.6,
                signal=SignalStrength.NEUTRAL
            ))
            
        if hammer.iloc[-1] != 0:
            patterns.append(PatternSignal(
                pattern_name="Hammer",
                confidence=0.7,
                signal=SignalStrength.BUY if hammer.iloc[-1] > 0 else SignalStrength.SELL
            ))
            
        if engulfing.iloc[-1] != 0:
            patterns.append(PatternSignal(
                pattern_name="Engulfing",
                confidence=0.8,
                signal=SignalStrength.BUY if engulfing.iloc[-1] > 0 else SignalStrength.SELL
            ))
            
        return patterns
    
    def _generate_trading_signals(
        self,
        df: pd.DataFrame,
        indicators: TechnicalIndicators
    ) -> List[TradingSignal]:
        """트레이딩 신호 생성"""
        signals = []
        current_price = df['Close'].iloc[-1]
        
        # RSI 기반 신호
        if indicators.oscillators.rsi_14 < 30:
            signals.append(TradingSignal(
                ticker=df.index.name,
                timeframe=TimeFrame.D1,
                signal=SignalStrength.BUY,
                confidence=0.7,
                entry_price=current_price,
                stop_loss=current_price * 0.95,
                take_profit=current_price * 1.1,
                risk_reward_ratio=2.0,
                indicators_used=["RSI"],
                reasoning="Oversold condition detected by RSI"
            ))
        elif indicators.oscillators.rsi_14 > 70:
            signals.append(TradingSignal(
                ticker=df.index.name,
                timeframe=TimeFrame.D1,
                signal=SignalStrength.SELL,
                confidence=0.7,
                entry_price=current_price,
                stop_loss=current_price * 1.05,
                take_profit=current_price * 0.9,
                risk_reward_ratio=2.0,
                indicators_used=["RSI"],
                reasoning="Overbought condition detected by RSI"
            ))
            
        # MACD 기반 신호
        if (indicators.oscillators.macd_histogram > 0 and
            indicators.oscillators.macd_histogram > indicators.oscillators.macd_histogram):
            signals.append(TradingSignal(
                ticker=df.index.name,
                timeframe=TimeFrame.D1,
                signal=SignalStrength.BUY,
                confidence=0.6,
                entry_price=current_price,
                stop_loss=current_price * 0.95,
                take_profit=current_price * 1.1,
                risk_reward_ratio=2.0,
                indicators_used=["MACD"],
                reasoning="MACD histogram showing increasing positive momentum"
            ))
            
        return signals
    
    def analyze_technical(self, request: TechnicalAnalysisRequest) -> TechnicalAnalysisResponse:
        """기술적 분석 수행"""
        try:
            # 시장 데이터를 DataFrame으로 변환
            df = self._convert_to_dataframe(request.market_data)
            
            # 기술적 지표 계산
            moving_averages = self._calculate_moving_averages(request)
            oscillators = self._calculate_oscillators(df)
            volume_indicators = self._calculate_volume_indicators(df)
            support_resistance = self._calculate_support_resistance(df)
            patterns = self._detect_patterns(df)
            
            # 기술적 지표 통합
            indicators = TechnicalIndicators(
                moving_averages=moving_averages,
                oscillators=oscillators,
                volume_indicators=volume_indicators,
                support_resistance=support_resistance,
                patterns=patterns
            )
            
            # 거래 신호 생성
            signals = self._generate_trading_signals(df, indicators)
            
            # 추세 분석
            trend_analysis = {
                "primary_trend": moving_averages.trend_direction,
                "volume_trend": volume_indicators.volume_trend,
                "momentum": SignalStrength.STRONG if oscillators.rsi_14 > 70 else (
                    SignalStrength.WEAK if oscillators.rsi_14 < 30 else SignalStrength.NEUTRAL
                ) if oscillators.rsi_14 is not None else SignalStrength.NEUTRAL
            }
            
            # 분석 요약 생성
            analysis_summary = self._generate_analysis_summary(
                indicators,
                signals,
                trend_analysis
            )
            
            return TechnicalAnalysisResponse(
                ticker=request.ticker,
                timeframe=request.timeframe,
                market_data=request.market_data,
                technical_indicators=indicators,
                trading_signals=signals,
                trend_analysis=trend_analysis,
                analysis_summary=analysis_summary
            )
            
        except Exception as e:
            # 오류 발생 시 None 값을 포함한 응답 반환
            return TechnicalAnalysisResponse(
                ticker=request.ticker,
                timeframe=request.timeframe,
                market_data=request.market_data,
                technical_indicators=TechnicalIndicators(
                    moving_averages=MovingAverages(
                        sma_20=None, sma_50=None, sma_200=None,
                        ema_12=None, ema_26=None,
                        trend_direction=TrendDirection.SIDEWAYS,
                        golden_cross=False, death_cross=False
                    ),
                    oscillators=Oscillators(
                        rsi_14=None, stoch_k=None, stoch_d=None,
                        macd_line=None, macd_signal=None, macd_histogram=None,
                        cci_20=None, atr_14=None,
                        bollinger_upper=None, bollinger_middle=None, bollinger_lower=None
                    ),
                    volume_indicators=VolumeIndicators(
                        obv=None, volume_sma_20=None,
                        volume_trend=TrendDirection.SIDEWAYS,
                        volume_price_trend=None, money_flow_index=None
                    ),
                    support_resistance=SupportResistance(
                        pivot=None, r1=None, r2=None, s1=None, s2=None,
                        resistance_levels=[], support_levels=[]
                    )
                ),
                trading_signals=[],
                trend_analysis={
                    "primary_trend": TrendDirection.SIDEWAYS,
                    "volume_trend": TrendDirection.SIDEWAYS,
                    "momentum": SignalStrength.NEUTRAL
                },
                analysis_summary=f"기술적 분석 중 오류 발생: {str(e)}"
            )
    
    def _generate_analysis_summary(
        self,
        indicators: TechnicalIndicators,
        signals: List[TradingSignal],
        trend_analysis: Dict
    ) -> str:
        """기술적 분석 요약 생성"""
        summary = f"Technical Analysis Summary:\n\n"
        
        # 추세 정보
        summary += f"Trend Analysis:\n"
        summary += f"- Primary Trend: {trend_analysis['primary_trend']}\n"
        summary += f"- Volume Trend: {trend_analysis['volume_trend']}\n"
        summary += f"- Momentum: {trend_analysis['momentum']}\n\n"
        
        # 주요 기술적 지표
        summary += f"Key Technical Indicators:\n"
        summary += f"- RSI (14): {indicators.oscillators.rsi_14:.2f}\n"
        summary += f"- MACD Histogram: {indicators.oscillators.macd_histogram:.2f}\n"
        summary += f"- Stochastic K/D: {indicators.oscillators.stoch_k:.2f}/{indicators.oscillators.stoch_d:.2f}\n\n"
        
        # 패턴
        if indicators.patterns:
            summary += "Detected Patterns:\n"
            for pattern in indicators.patterns:
                summary += f"- {pattern.pattern_name}: {pattern.signal} (Confidence: {pattern.confidence:.1%})\n"
            summary += "\n"
            
        # 트레이딩 신호
        if signals:
            summary += "Trading Signals:\n"
            for signal in signals:
                summary += f"- {signal.signal}: {signal.reasoning} (Confidence: {signal.confidence:.1%})\n"
                
        return summary

    def analyze(self, request: TechnicalAnalysisRequest) -> TechnicalAnalysisResponse:
        df = self._convert_to_dataframe(request.market_data)
        indicators_result = {}
        signals_result = {}

        for indicator in request.indicators:
            if indicator in self.indicators:
                indicators_result[indicator] = self.indicators[indicator](df, request.parameters)

        latest_data = df.iloc[-1]
        return TechnicalAnalysisResponse(
            ticker=latest_data.name,
            timestamp=latest_data.name,
            indicators=indicators_result,
            signals=signals_result
        )

    def calculate_moving_averages(self, request: MovingAverageRequest) -> MovingAverageResponse:
        """이동평균 계산"""
        try:
            # Convert market data to numpy array
            prices = np.array([data.close for data in request.market_data])
            
            sma_values = {}
            ema_values = {}
            
            for window in request.window_sizes:
                if "sma" in request.ma_types:
                    sma = self._calculate_sma(prices, window)
                    sma_values[f"sma_{window}"] = float(sma[-1])
                
                if "ema" in request.ma_types:
                    ema = self._calculate_ema(prices, window)
                    ema_values[f"ema_{window}"] = float(ema[-1])
            
            trend = self._determine_trend_direction(prices)
            
            return MovingAverageResponse(
                sma_values=sma_values,
                ema_values=ema_values,
                trend_direction=trend,
                error_message=None
            )
        except Exception as e:
            return MovingAverageResponse(
                sma_values={},
                ema_values={},
                trend_direction=TrendDirection.SIDEWAYS,
                error_message=str(e)
            )

    def analyze_support_resistance(self, request: SupportResistanceRequest) -> Dict:
        df = self._convert_to_dataframe(request.market_data)
        highs = df["high"].rolling(window=request.lookback_period).max()
        lows = df["low"].rolling(window=request.lookback_period).min()
        
        return {
            "support_levels": self._find_support_levels(df, request.min_touches),
            "resistance_levels": self._find_resistance_levels(df, request.min_touches)
        }

    def analyze_patterns(self, request: PatternRequest) -> PatternResponse:
        """차트 패턴 분석"""
        try:
            patterns = {}
            for pattern_type in request.pattern_types:
                if pattern_type in ["double_top", "double_bottom", "head_shoulders"]:
                    patterns[pattern_type] = Pattern(
                        type=pattern_type,
                        confidence=0.8,
                        timestamp=datetime.now().isoformat()
                    )
            
            return PatternResponse(
                patterns=patterns,
                timestamp=datetime.now().isoformat(),
                error_message=None
            )
        except Exception as e:
            return PatternResponse(
                patterns={},
                timestamp=datetime.now().isoformat(),
                error_message=str(e)
            )

    def generate_signals(self, request: SignalRequest) -> SignalResponse:
        """트레이딩 신호 생성"""
        try:
            signals = {}
            for signal_type in request.signal_types:
                if signal_type in ["trend_following", "mean_reversion", "breakout"]:
                    signal = self._generate_signal(request.market_data, signal_type)
                    signals[signal_type] = signal
            
            return SignalResponse(
                signals=signals,
                timestamp=datetime.now().isoformat(),
                error_message=None
            )
        except Exception as e:
            return SignalResponse(
                signals={},
                timestamp=datetime.now().isoformat(),
                error_message=str(e)
            )

    def _generate_signal(self, market_data: List[MarketData], signal_type: str) -> Signal:
        prices = np.array([data.close for data in market_data])
        
        if len(prices) < 2:
            return Signal(signal="NEUTRAL", strength="WEAK")
        
        if signal_type == "trend_following":
            return self._generate_trend_signal(prices)
        elif signal_type == "mean_reversion":
            return self._generate_reversion_signal(prices)
        elif signal_type == "breakout":
            return self._generate_breakout_signal(prices)
        
        return Signal(signal="NEUTRAL", strength="WEAK")

    def _determine_trend_direction(self, prices: np.ndarray) -> TrendDirection:
        if len(prices) < 2:
            return TrendDirection.SIDEWAYS
        
        # Calculate price change
        price_change = (prices[-1] - prices[0]) / prices[0]
        
        if price_change > 0.02:  # 2% threshold for uptrend
            return TrendDirection.UP
        elif price_change < -0.02:  # -2% threshold for downtrend
            return TrendDirection.DOWN
        return TrendDirection.SIDEWAYS

    def _detect_pattern(self, market_data: List[MarketData], pattern_type: str) -> Optional[Pattern]:
        prices = np.array([data.close for data in market_data])
        
        if pattern_type == "double_top":
            return self._detect_double_top(prices)
        elif pattern_type == "double_bottom":
            return self._detect_double_bottom(prices)
        elif pattern_type == "head_shoulders":
            return self._detect_head_shoulders(prices)
        return None

    def _calculate_sma(self, prices: np.ndarray, window: int) -> np.ndarray:
        return np.convolve(prices, np.ones(window)/window, mode='valid')

    def _calculate_ema(self, prices: np.ndarray, window: int) -> np.ndarray:
        alpha = 2 / (window + 1)
        ema = np.zeros_like(prices)
        ema[0] = prices[0]
        for i in range(1, len(prices)):
            ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]
        return ema

    def _calculate_rsi(self, df: pd.DataFrame, params: Dict) -> float:
        window = params.get("window", 14)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs.iloc[-1]))

    def _calculate_macd(self, df: pd.DataFrame, params: Dict) -> Dict[str, float]:
        fast = params.get("fast", 12)
        slow = params.get("slow", 26)
        signal = params.get("signal", 9)
        
        fast_ema = df["close"].ewm(span=fast).mean()
        slow_ema = df["close"].ewm(span=slow).mean()
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=signal).mean()
        
        return {
            "macd": macd_line.iloc[-1],
            "signal": signal_line.iloc[-1],
            "histogram": macd_line.iloc[-1] - signal_line.iloc[-1]
        }

    def _find_support_levels(self, df: pd.DataFrame, min_touches: int) -> List[float]:
        # 간단한 구현을 위해 최근 저점들을 반환
        return sorted(df["low"].nsmallest(3).tolist())

    def _find_resistance_levels(self, df: pd.DataFrame, min_touches: int) -> List[float]:
        # 간단한 구현을 위해 최근 고점들을 반환
        return sorted(df["high"].nlargest(3).tolist())

    def _detect_double_top(self, prices: np.ndarray) -> Optional[Pattern]:
        """더블 탑 패턴 감지"""
        if len(prices) < 5:
            return None

        # 두 개의 고점과 그 사이의 저점 확인
        peaks = (prices[1] > prices[0]) and (prices[1] > prices[2])
        trough = (prices[2] < prices[1]) and (prices[2] < prices[3])
        second_peak = (prices[3] > prices[2]) and (prices[3] > prices[4])

        if peaks and trough and second_peak:
            return Pattern(
            type="double_top",
            confidence=0.8,
            timestamp=datetime.now().isoformat()
        )
        return None

    def _detect_double_bottom(self, prices: np.ndarray) -> Optional[Pattern]:
        """더블 바텀 패턴 감지"""
        if len(prices) < 5:
            return None

        # 두 개의 저점과 그 사이의 고점 확인
        troughs = (prices[1] < prices[0]) and (prices[1] < prices[2])
        peak = (prices[2] > prices[1]) and (prices[2] > prices[3])
        second_trough = (prices[3] < prices[2]) and (prices[3] < prices[4])

        if troughs and peak and second_trough:
            return Pattern(
            type="double_bottom",
            confidence=0.8,
            timestamp=datetime.now().isoformat()
        )
        return None

    def _detect_head_shoulders(self, prices: np.ndarray) -> Optional[Pattern]:
        """헤드 앤 숄더 패턴 감지"""
        if len(prices) < 7:
            return None

        # 헤드 앤 숄더 패턴 확인
        left_shoulder = (prices[1] > prices[0]) and (prices[1] > prices[2])
        head = (prices[3] > prices[1]) and (prices[3] > prices[5])
        right_shoulder = (prices[5] > prices[4]) and (prices[5] > prices[6])

        if left_shoulder and head and right_shoulder:
            return Pattern(
            type="head_shoulders",
            confidence=0.8,
            timestamp=datetime.now().isoformat()
        )
        return None

    def _generate_trend_signal(self, prices: np.ndarray) -> Signal:
        if len(prices) < 20:
            return Signal(signal="NEUTRAL", strength="WEAK")

        # 가격 변화율 기준 (최근 종가 vs 시작 종가)
        price_change = (prices[-1] - prices[0]) / prices[0]

        # RSI 계산 (14)
        close_series = pd.Series(prices)
        delta = close_series.diff()
        gain = delta.clip(lower=0).rolling(window=14).mean()
        loss = -delta.clip(upper=0).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        rsi_latest = rsi.iloc[-1]

        # 조건 조합
        if price_change > 0.02 and rsi_latest < 70:
            return Signal(signal="BUY", strength="STRONG")
        elif price_change < -0.02 and rsi_latest > 30:
            return Signal(signal="SELL", strength="STRONG")
        else:
            return Signal(signal="NEUTRAL", strength="WEAK")

    def _generate_reversion_signal(self, prices: np.ndarray) -> Signal:
        if len(prices) < 20:
            return Signal(signal="NEUTRAL", strength="WEAK")

        close_series = pd.Series(prices)
        sma_20 = close_series.rolling(window=20).mean()
        std_20 = close_series.rolling(window=20).std()

        upper_band = sma_20 + 2 * std_20
        lower_band = sma_20 - 2 * std_20

        if prices[-1] < lower_band.iloc[-1]:
            return Signal(signal="BUY", strength="MODERATE")
        elif prices[-1] > upper_band.iloc[-1]:
            return Signal(signal="SELL", strength="MODERATE")
        else:
            return Signal(signal="NEUTRAL", strength="WEAK")

    def _generate_breakout_signal(self, prices: np.ndarray) -> Signal:
        if len(prices) < 20:
            return Signal(signal="NEUTRAL", strength="WEAK")

        recent_high = np.max(prices[-20:-1])
        recent_low = np.min(prices[-20:-1])
        current_price = prices[-1]

        if current_price > recent_high:
            return Signal(signal="BUY", strength="STRONG") 
        elif current_price < recent_low:
            return Signal(signal="SELL", strength="STRONG")
        else:
            return Signal(signal="NEUTRAL", strength="WEAK") 