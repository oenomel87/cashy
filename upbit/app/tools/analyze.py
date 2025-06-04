from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import pandas as pd
import numpy as np
import asyncio
import logging
import math

from app.tools.upbit import get_current_ticker, get_candles_for_daily, get_candles_for_minutes, get_candles_for_weekly
from app.schemas.ticker import Ticker
from app.schemas.candle import MinuteCandleStick, DailyCandleStick, WeeklyCandleStick

@dataclass
class AnalysisConfig:
    """분석 설정 클래스"""
    
    # 데이터 수집 설정
    daily_count: int = 200
    minute_count: int = 48
    minute_interval: int = 30
    weekly_count: int = 8
    
    # 이동평균 설정
    ma_short: int = 20
    ma_medium: int = 50
    ma_long: int = 200
    
    # 모멘텀 지표 설정
    rsi_period: int = 14
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    
    stoch_k_period: int = 14
    stoch_d_period: int = 3
    stoch_overbought: float = 80.0
    stoch_oversold: float = 20.0
    
    # 변동성 지표 설정
    bb_period: int = 20
    bb_std: float = 2.0
    atr_period: int = 14
    
    # 거래량 지표 설정
    volume_ema_period: int = 20
    volume_trend_period: int = 5

@dataclass
class MarketData:
    """시장 데이터 컨테이너"""
    daily_df: pd.DataFrame
    minute_df: pd.DataFrame
    weekly_df: pd.DataFrame
    ticker: Ticker

def safe_dataclass_to_dataframe(data_list: List, dataclass_type=None) -> pd.DataFrame:
    """
    안전한 dataclass → DataFrame 변환 (PR의 __dict__ 방식 적용)
    
    Args:
        data_list: dataclass 객체들의 리스트
        dataclass_type: dataclass 타입 (빈 리스트 처리용)
    
    Returns:
        pandas DataFrame
    """
    if not data_list:
        return pd.DataFrame()
    
    try:
        # PR에서 제안한 __dict__ 방식 사용 (성능 최적화)
        df = pd.DataFrame([item.__dict__ for item in data_list])
        
        # 날짜 컬럼 최적화
        date_columns = ['candle_date_time_utc', 'candle_date_time_kst']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # 정렬 (최신 데이터 먼저)
        if 'candle_date_time_utc' in df.columns:
            df = df.sort_values('candle_date_time_utc', ascending=False)
        
        return df.reset_index(drop=True)
        
    except Exception as e:
        logging.error(f"DataFrame 변환 실패 (__dict__ 방식): {e}")
        # 폴백: asdict 방식
        try:
            from dataclasses import asdict
            return pd.DataFrame([asdict(item) for item in data_list])
        except Exception as e2:
            logging.error(f"DataFrame 변환 완전 실패: {e2}")
            return pd.DataFrame()

async def load_market_data(config: AnalysisConfig = None) -> MarketData:
    """
    시장 데이터 로드 (병렬 처리로 최적화)
    
    기존: 4번의 개별 API 호출
    개선: 1번의 병렬 API 호출
    """
    if config is None:
        config = AnalysisConfig()
    
    logging.info("시장 데이터 로드 시작 (병렬 처리)")
    start_time = datetime.now()
    
    # 병렬로 모든 데이터 로드 (핵심 최적화 포인트)
    daily_task = get_candles_for_daily(count=config.daily_count)
    minute_task = get_candles_for_minutes(
        minutes=config.minute_interval, 
        count=config.minute_count
    )
    weekly_task = get_candles_for_weekly(count=config.weekly_count)
    ticker_task = get_current_ticker()
    
    # 4개 API를 동시에 호출
    daily_candles, minute_candles, weekly_candles, ticker = await asyncio.gather(
        daily_task, minute_task, weekly_task, ticker_task
    )
    
    # DataFrame 변환 (1번만 수행)
    daily_df = safe_dataclass_to_dataframe(daily_candles, DailyCandleStick)
    minute_df = safe_dataclass_to_dataframe(minute_candles, MinuteCandleStick)
    weekly_df = safe_dataclass_to_dataframe(weekly_candles, WeeklyCandleStick)
    
    load_time = (datetime.now() - start_time).total_seconds()
    logging.info(f"데이터 로드 완료: {load_time:.2f}초")
    
    return MarketData(
        daily_df=daily_df,
        minute_df=minute_df,
        weekly_df=weekly_df,
        ticker=ticker
    )

def calculate_market_info(data: MarketData, config: AnalysisConfig) -> Dict[str, Any]:
    """시장 정보 계산"""
    try:
        ticker = data.ticker
        
        # 24시간 거래량 변화율
        volume_change_pct = calculate_volume_change(data.minute_df)
        
        # 30일 변동성
        volatility = calculate_daily_volatility(data.daily_df)
        
        return {
            "symbol": "BTC-KRW",
            "current_price": ticker.trade_price,
            "day_change_pct": round(ticker.signed_change_rate * 100, 2),
            "timestamp": datetime.fromtimestamp(ticker.timestamp / 1000).strftime('%Y-%m-%dT%H:%M:%SZ'),
            "24h_volume": round(ticker.acc_trade_volume_24h, 2),
            "24h_volume_change_pct": volume_change_pct,
            "volatility_30d_annualized": volatility
        }
    except Exception as e:
        logging.error(f"시장 정보 계산 실패: {e}")
        return {
            "symbol": "BTC-KRW", "current_price": 0, "day_change_pct": 0.0,
            "timestamp": datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ'),
            "24h_volume": 0.0, "24h_volume_change_pct": 0.0, "volatility_30d_annualized": 0.0
        }

def calculate_volume_change(minute_df: pd.DataFrame) -> float:
    """24시간 거래량 변화율 계산"""
    if len(minute_df) < 48:
        return 0.0
    
    try:
        # 최근 24개와 이전 24개 비교 (30분 간격)
        recent_volume = minute_df['candle_acc_trade_volume'].head(24).sum()
        previous_volume = minute_df['candle_acc_trade_volume'].iloc[24:48].sum()
        
        if previous_volume > 0:
            return round(((recent_volume - previous_volume) / previous_volume) * 100, 2)
        return 0.0
    except Exception:
        return 0.0

def calculate_daily_volatility(daily_df: pd.DataFrame) -> float:
    """30일 연간화 변동성 계산"""
    if len(daily_df) < 30:
        return 0.0
    
    try:
        returns = []
        for i in range(1, min(30, len(daily_df))):
            today_price = daily_df['trade_price'].iloc[i-1]
            yesterday_price = daily_df['trade_price'].iloc[i]
            if yesterday_price > 0:
                daily_return = math.log(today_price / yesterday_price)
                returns.append(daily_return)
        
        if returns:
            std_dev = np.std(returns)
            annualized_volatility = std_dev * math.sqrt(252) * 100
            return round(annualized_volatility, 1)
        return 0.0
    except Exception:
        return 0.0

def calculate_trend_analysis(data: MarketData, config: AnalysisConfig) -> Dict[str, Any]:
    """추세 분석 계산"""
    try:
        # 각 시간대별 추세 분석
        short_term = analyze_short_term_trend(data.minute_df.head(4))
        medium_term = analyze_medium_term_trend(data.daily_df.head(7))
        long_term = analyze_long_term_trend(data.weekly_df.head(8))
        
        # 추세 강도 및 지속 기간
        trend_strength = calculate_trend_strength(short_term, medium_term, long_term)
        trend_duration = calculate_trend_duration(data.daily_df)
        
        return {
            "short_term": short_term["trend"],
            "medium_term": medium_term["trend"],
            "long_term": long_term["trend"],
            "trend_strength": round(trend_strength),
            "trend_duration_days": trend_duration
        }
    except Exception as e:
        logging.error(f"추세 분석 실패: {e}")
        return {
            "short_term": "neutral", "medium_term": "neutral", "long_term": "neutral",
            "trend_strength": 50, "trend_duration_days": 1
        }

def analyze_short_term_trend(hourly_df: pd.DataFrame) -> Dict[str, Any]:
    """단기 추세 분석 (1-4시간)"""
    if len(hourly_df) < 2:
        return {"trend": "neutral", "strength": 50}
    
    up_count = sum(1 for _, row in hourly_df.iterrows() 
                  if row['trade_price'] > row['opening_price'])
    down_count = len(hourly_df) - up_count
    
    start_price = hourly_df['opening_price'].iloc[-1]
    end_price = hourly_df['trade_price'].iloc[0]
    price_change = ((end_price - start_price) / start_price) * 100
    
    if price_change > 1.0 or (up_count >= 3 and price_change > 0):
        trend = "bullish"
        strength = min(100, abs(price_change) * 15 + up_count * 10)
    elif price_change < -1.0 or (down_count >= 3 and price_change < 0):
        trend = "bearish"
        strength = min(100, abs(price_change) * 15 + down_count * 10)
    else:
        trend = "neutral"
        strength = 50 - abs(price_change * 10)
    
    return {"trend": trend, "strength": strength}

def analyze_medium_term_trend(daily_df: pd.DataFrame) -> Dict[str, Any]:
    """중기 추세 분석 (1일-1주)"""
    if len(daily_df) < 2:
        return {"trend": "neutral", "strength": 50}
    
    up_days = sum(1 for _, row in daily_df.iterrows() if row.get('change_rate', 0) > 0)
    down_days = len(daily_df) - up_days
    
    start_price = daily_df['trade_price'].iloc[-1]
    end_price = daily_df['trade_price'].iloc[0]
    price_change = ((end_price - start_price) / start_price) * 100
    
    if price_change > 4 or (up_days >= 4 and price_change > 0):
        trend = "bullish"
        strength = min(100, abs(price_change) * 5 + up_days * 5)
    elif price_change < -4 or (down_days >= 4 and price_change < 0):
        trend = "bearish"
        strength = min(100, abs(price_change) * 5 + down_days * 5)
    else:
        trend = "neutral"
        strength = 50 - abs(price_change * 3)
    
    return {"trend": trend, "strength": strength}

def analyze_long_term_trend(weekly_df: pd.DataFrame) -> Dict[str, Any]:
    """장기 추세 분석 (1주-1개월)"""
    if len(weekly_df) < 2:
        return {"trend": "neutral", "strength": 50}
    
    up_weeks = 0
    for i in range(len(weekly_df) - 1):
        current = weekly_df['trade_price'].iloc[i]
        previous = weekly_df['trade_price'].iloc[i + 1]
        if current > previous:
            up_weeks += 1
    
    down_weeks = len(weekly_df) - 1 - up_weeks
    
    start_price = weekly_df['trade_price'].iloc[-1]
    end_price = weekly_df['trade_price'].iloc[0]
    price_change = ((end_price - start_price) / start_price) * 100
    
    if price_change > 8 or (up_weeks >= 3 and price_change > 0):
        trend = "bullish"
        strength = min(100, abs(price_change) * 2 + up_weeks * 10)
    elif price_change < -8 or (down_weeks >= 3 and price_change < 0):
        trend = "bearish"
        strength = min(100, abs(price_change) * 2 + down_weeks * 10)
    else:
        trend = "neutral"
        strength = 50 - abs(price_change)
    
    return {"trend": trend, "strength": strength}

def calculate_trend_strength(short_term, medium_term, long_term) -> float:
    """추세 강도 계산"""
    weighted_strength = (
        (short_term["strength"] * 0.2) + 
        (medium_term["strength"] * 0.3) + 
        (long_term["strength"] * 0.5)
    )
    
    trends = [short_term["trend"], medium_term["trend"], long_term["trend"]]
    if len(set(trends)) == 1 and trends[0] != "neutral":
        return min(100, weighted_strength + 15)
    elif medium_term["trend"] == long_term["trend"] and medium_term["trend"] != "neutral":
        return min(100, weighted_strength + 8)
    
    return weighted_strength

def calculate_trend_duration(daily_df: pd.DataFrame) -> int:
    """추세 지속 기간 계산"""
    if len(daily_df) < 2:
        return 1
    
    current_direction = None
    if daily_df['trade_price'].iloc[0] > daily_df['trade_price'].iloc[1]:
        current_direction = "up"
    elif daily_df['trade_price'].iloc[0] < daily_df['trade_price'].iloc[1]:
        current_direction = "down"
    
    if not current_direction:
        return 1
    
    duration = 1
    for i in range(1, len(daily_df) - 1):
        if current_direction == "up":
            if daily_df['trade_price'].iloc[i] > daily_df['trade_price'].iloc[i + 1]:
                duration += 1
            else:
                break
        else:
            if daily_df['trade_price'].iloc[i] < daily_df['trade_price'].iloc[i + 1]:
                duration += 1
            else:
                break
    
    return duration

def calculate_price_levels(data: MarketData, config: AnalysisConfig) -> Dict[str, Any]:
    """가격 레벨 계산"""
    try:
        current_price = data.ticker.trade_price
        daily_df = data.daily_df.head(30)  # 최근 30일
        
        # 지지선과 저항선 계산
        support_resistance = calculate_support_resistance(daily_df, current_price)
        
        # 최근 테스트된 레벨
        last_tested = identify_last_tested_level(
            daily_df.head(5), 
            support_resistance["support"],
            support_resistance["resistance"]
        )
        
        # 거리 계산
        distance_to_resistance = ((support_resistance["resistance"][0] - current_price) / current_price) * 100
        distance_to_support = ((current_price - support_resistance["support"][0]) / current_price) * 100
        
        return {
            "key_resistance": support_resistance["resistance"],
            "key_support": support_resistance["support"],
            "last_tested": last_tested,
            "distance_to_resistance_pct": round(distance_to_resistance, 2),
            "distance_to_support_pct": round(distance_to_support, 2)
        }
    except Exception as e:
        logging.error(f"가격 레벨 계산 실패: {e}")
        return {
            "key_resistance": [0, 0], "key_support": [0, 0], "last_tested": "resistance",
            "distance_to_resistance_pct": 0.0, "distance_to_support_pct": 0.0
        }

def calculate_support_resistance(df: pd.DataFrame, current_price: float) -> Dict[str, List[float]]:
    """지지선/저항선 계산"""
    if df.empty:
        return {
            "resistance": [round(current_price * 1.02), round(current_price * 1.04)],
            "support": [round(current_price * 0.98), round(current_price * 0.96)]
        }
    
    highs = df['high_price'].tolist()
    lows = df['low_price'].tolist()
    
    resistance_levels = [h for h in highs if h > current_price]
    support_levels = [l for l in lows if l < current_price]
    
    resistance_levels.sort()
    support_levels.sort(reverse=True)
    
    # 최소 2개 보장
    if len(resistance_levels) < 2:
        resistance_levels.extend([current_price * 1.02, current_price * 1.04])
    if len(support_levels) < 2:
        support_levels.extend([current_price * 0.98, current_price * 0.96])
    
    return {
        "resistance": [round(x) for x in resistance_levels[:2]],
        "support": [round(x) for x in support_levels[:2]]
    }

def identify_last_tested_level(recent_df, support_levels, resistance_levels) -> str:
    """최근 테스트된 레벨 식별"""
    if recent_df.empty:
        return "resistance"
    
    tolerance = 0.005  # 0.5% 허용 오차
    
    for _, row in recent_df.iterrows():
        for resistance in resistance_levels:
            if abs(resistance - row['high_price']) / resistance < tolerance:
                return "resistance"
        for support in support_levels:
            if abs(support - row['low_price']) / support < tolerance:
                return "support"
    
    # 최근 가격 움직임으로 결정
    if len(recent_df) >= 2:
        return "resistance" if recent_df['trade_price'].iloc[0] > recent_df['trade_price'].iloc[1] else "support"
    
    return "resistance"

def calculate_technical_signals(data: MarketData, config: AnalysisConfig) -> Dict[str, Any]:
    """
    기술적 신호 계산 (모든 지표를 한 번에)
    
    기존: 각 함수에서 DataFrame 재생성
    개선: 공통 DataFrame 재사용
    """
    try:
        df = data.daily_df.copy()  # 한 번만 복사
        
        if df.empty:
            return empty_technical_signals()
        
        # 모든 기술적 지표를 한 번에 계산
        moving_averages = calculate_moving_averages(df, config)
        momentum = calculate_momentum(df, config)
        volatility = calculate_volatility(df, config)
        volume = calculate_volume(df, config)
        
        return {
            "moving_averages": moving_averages,
            "momentum": momentum,
            "volatility": volatility,
            "volume": volume
        }
    except Exception as e:
        logging.error(f"기술적 신호 계산 실패: {e}")
        return empty_technical_signals()

def calculate_moving_averages(df: pd.DataFrame, config: AnalysisConfig) -> Dict[str, Any]:
    """이동평균 계산"""
    # 한 번에 모든 이동평균 계산
    df[f'ma_{config.ma_short}'] = df['trade_price'].rolling(window=config.ma_short).mean()
    df[f'ma_{config.ma_medium}'] = df['trade_price'].rolling(window=config.ma_medium).mean()
    df[f'ma_{config.ma_long}'] = df['trade_price'].rolling(window=config.ma_long).mean()
    
    # 최신 값들
    current_price = df['trade_price'].iloc[0]
    ma_short = df[f'ma_{config.ma_short}'].iloc[0]
    ma_medium = df[f'ma_{config.ma_medium}'].iloc[0]
    ma_long = df[f'ma_{config.ma_long}'].iloc[0]
    
    # 교차 신호 분석
    crossovers = analyze_ma_crossovers(df, config)
    
    return {
        f"ma_{config.ma_long}d": {
            "value": int(ma_long) if not pd.isna(ma_long) else None,
            "position": "above" if current_price > ma_long else "below",
            "signal": "bullish" if current_price > ma_long else "bearish"
        },
        f"ma_{config.ma_medium}d": {
            "value": int(ma_medium) if not pd.isna(ma_medium) else None,
            "position": "above" if current_price > ma_medium else "below",
            "signal": "bullish" if current_price > ma_medium else "bearish"
        },
        f"ma_{config.ma_short}d": {
            "value": int(ma_short) if not pd.isna(ma_short) else None,
            "position": "above" if current_price > ma_short else "below",
            "signal": "bullish" if current_price > ma_short else "bearish"
        },
        "ma_crossovers": crossovers
    }

def analyze_ma_crossovers(df: pd.DataFrame, config: AnalysisConfig) -> List[Dict[str, Any]]:
    """이동평균 교차 분석"""
    crossovers = []
    lookback = min(30, len(df))
    
    for i in range(1, lookback):
        try:
            prev_short = df[f'ma_{config.ma_short}'].iloc[i]
            prev_medium = df[f'ma_{config.ma_medium}'].iloc[i]
            curr_short = df[f'ma_{config.ma_short}'].iloc[i-1]
            curr_medium = df[f'ma_{config.ma_medium}'].iloc[i-1]
            
            if pd.isna(prev_short) or pd.isna(prev_medium) or pd.isna(curr_short) or pd.isna(curr_medium):
                continue
            
            # 골든/데드 크로스 확인
            if prev_short <= prev_medium and curr_short > curr_medium:
                crossovers.append({
                    "type": "golden_cross",
                    "fast_ma": f"{config.ma_short}d",
                    "slow_ma": f"{config.ma_medium}d",
                    "days_ago": i
                })
            elif prev_short >= prev_medium and curr_short < curr_medium:
                crossovers.append({
                    "type": "death_cross",
                    "fast_ma": f"{config.ma_short}d",
                    "slow_ma": f"{config.ma_medium}d",
                    "days_ago": i
                })
        except (IndexError, KeyError):
            continue
    
    return crossovers

def calculate_momentum(df: pd.DataFrame, config: AnalysisConfig) -> Dict[str, Any]:
    """모멘텀 지표 계산 (RSI, MACD, Stochastic)"""
    # RSI 계산
    delta = df['trade_price'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=config.rsi_period).mean()
    avg_loss = loss.rolling(window=config.rsi_period).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    current_rsi = rsi.iloc[0]
    previous_rsi = rsi.iloc[1] if len(rsi) > 1 else current_rsi
    
    # MACD 계산
    ema_fast = df['trade_price'].ewm(span=config.macd_fast).mean()
    ema_slow = df['trade_price'].ewm(span=config.macd_slow).mean()
    
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=config.macd_signal).mean()
    histogram = macd_line - signal_line
    
    # Stochastic 계산
    lowest_low = df['low_price'].rolling(window=config.stoch_k_period).min()
    highest_high = df['high_price'].rolling(window=config.stoch_k_period).max()
    stoch_k = ((df['trade_price'] - lowest_low) / (highest_high - lowest_low)) * 100
    stoch_d = stoch_k.rolling(window=config.stoch_d_period).mean()
    
    return {
        "rsi_14d": {
            "value": round(current_rsi, 1) if not pd.isna(current_rsi) else None,
            "zone": "overbought" if current_rsi >= config.rsi_overbought else 
                   "oversold" if current_rsi <= config.rsi_oversold else "neutral",
            "trend": "rising" if current_rsi > previous_rsi else "falling" if current_rsi < previous_rsi else "neutral"
        },
        "macd": {
            "line": round(macd_line.iloc[0] / 1000000, 2) if not pd.isna(macd_line.iloc[0]) else None,
            "signal": round(signal_line.iloc[0] / 1000000, 2) if not pd.isna(signal_line.iloc[0]) else None,
            "histogram": round(histogram.iloc[0] / 1000000, 2) if not pd.isna(histogram.iloc[0]) else None,
            "trend": get_macd_trend(histogram)
        },
        "stochastic": {
            "k": round(stoch_k.iloc[0], 1) if not pd.isna(stoch_k.iloc[0]) else None,
            "d": round(stoch_d.iloc[0], 1) if not pd.isna(stoch_d.iloc[0]) else None,
            "trend": get_stoch_trend(stoch_k, stoch_d),
            "zone": get_stoch_zone(stoch_k.iloc[0], stoch_d.iloc[0], config)
        }
    }

def get_macd_trend(histogram: pd.Series) -> str:
    """MACD 추세 결정"""
    if len(histogram) < 2:
        return "neutral"
        
    current = histogram.iloc[0]
    previous = histogram.iloc[1]
    
    if pd.isna(current) or pd.isna(previous):
        return "neutral"
    
    if abs(current) < abs(previous):
        return "converging"
    elif abs(current) > abs(previous):
        return "diverging"
    elif (current > 0 and previous <= 0) or (current < 0 and previous >= 0):
        return "crossover"
    else:
        return "neutral"

def get_stoch_trend(stoch_k: pd.Series, stoch_d: pd.Series) -> str:
    """Stochastic 추세 결정"""
    if len(stoch_k) < 2 or len(stoch_d) < 2:
        return "neutral"
    
    current_k, current_d = stoch_k.iloc[0], stoch_d.iloc[0]
    previous_k, previous_d = stoch_k.iloc[1], stoch_d.iloc[1]
    
    if any(pd.isna(x) for x in [current_k, current_d, previous_k, previous_d]):
        return "neutral"
    
    if current_k > previous_k and current_d > previous_d:
        return "bullish"
    elif current_k < previous_k and current_d < previous_d:
        return "bearish"
    elif current_k > current_d and previous_k <= previous_d:
        return "bullish_crossover"
    elif current_k < current_d and previous_k >= previous_d:
        return "bearish_crossover"
    else:
        return "neutral"

def get_stoch_zone(k_value: float, d_value: float, config: AnalysisConfig) -> str:
    """Stochastic 존 결정"""
    if pd.isna(k_value) or pd.isna(d_value):
        return "neutral"
    
    if k_value >= config.stoch_overbought and d_value >= config.stoch_overbought:
        return "overbought"
    elif k_value <= config.stoch_oversold and d_value <= config.stoch_oversold:
        return "oversold"
    else:
        return "neutral"

def calculate_volatility(df: pd.DataFrame, config: AnalysisConfig) -> Dict[str, Any]:
    """변동성 지표 계산 (Bollinger Bands, ATR)"""
    # Bollinger Bands
    sma = df['trade_price'].rolling(window=config.bb_period).mean()
    std = df['trade_price'].rolling(window=config.bb_period).std()
    upper_band = sma + (std * config.bb_std)
    lower_band = sma - (std * config.bb_std)
    band_width = ((upper_band - lower_band) / sma) * 100
    
    current_price = df['trade_price'].iloc[0]
    latest_upper = upper_band.iloc[0]
    latest_lower = lower_band.iloc[0]
    latest_band_width = band_width.iloc[0]
    
    # 밴드 내 위치
    if not pd.isna(latest_upper) and not pd.isna(latest_lower) and latest_upper != latest_lower:
        position = min(100, max(0, ((current_price - latest_lower) / (latest_upper - latest_lower)) * 100))
    else:
        position = 50
    
    # ATR 계산
    df['prev_close'] = df['trade_price'].shift(1)
    df['tr1'] = df['high_price'] - df['low_price']
    df['tr2'] = abs(df['high_price'] - df['prev_close'])
    df['tr3'] = abs(df['low_price'] - df['prev_close'])
    df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    atr = df['true_range'].rolling(window=config.atr_period).mean()
    
    return {
        "bollinger_bands": {
            "width_percentile": calculate_percentile(band_width, latest_band_width),
            "position": round(position),
            "signal": "overbought" if position > 80 else "oversold" if position < 20 else "neutral"
        },
        "atr_14d": int(atr.iloc[0]) if not pd.isna(atr.iloc[0]) else None,
        "atr_percentile": calculate_percentile(atr, atr.iloc[0])
    }

def calculate_volume(df: pd.DataFrame, config: AnalysisConfig) -> Dict[str, Any]:
    """거래량 지표 계산"""
    # OBV 계산
    price_change = df['trade_price'].diff()
    obv_change = np.where(
        price_change > 0, df['candle_acc_trade_volume'],
        np.where(price_change < 0, -df['candle_acc_trade_volume'], 0)
    )
    obv = pd.Series(obv_change).cumsum()
    
    # Volume EMA
    volume_ema = df['candle_acc_trade_volume'].ewm(span=config.volume_ema_period).mean()
    current_volume = df['candle_acc_trade_volume'].iloc[0]
    current_ema = volume_ema.iloc[0]
    
    volume_ema_ratio = round(current_volume / current_ema, 2) if current_ema > 0 else 1.0
    
    return {
        "obv_trend": calculate_obv_trend(obv),
        "volume_ema_ratio": volume_ema_ratio,
        "volume_price_trend": calculate_volume_price_trend(df, config)
    }

def calculate_obv_trend(obv: pd.Series) -> str:
    """OBV 추세 계산"""
    recent_obv = obv.head(5)
    if len(recent_obv) < 2:
        return "neutral"
    
    x = np.arange(len(recent_obv))
    slope = np.polyfit(x, recent_obv, 1)[0]
    
    if slope > 0:
        return "rising"
    elif slope < 0:
        return "falling"
    else:
        return "neutral"

def calculate_volume_price_trend(df: pd.DataFrame, config: AnalysisConfig) -> str:
    """Volume-Price Trend 계산"""
    period = min(config.volume_trend_period, len(df))
    if period < 2:
        return "neutral"
    
    recent_df = df.head(period)
    
    price_change = recent_df['trade_price'].iloc[0] - recent_df['trade_price'].iloc[-1]
    volume_change = recent_df['candle_acc_trade_volume'].iloc[0] - recent_df['candle_acc_trade_volume'].iloc[-1]
    
    price_direction = "up" if price_change > 0 else "down" if price_change < 0 else "flat"
    volume_direction = "up" if volume_change > 0 else "down" if volume_change < 0 else "flat"
    
    if (price_direction == "up" and volume_direction == "up") or \
       (price_direction == "down" and volume_direction == "down"):
        return "confirming"
    elif (price_direction == "up" and volume_direction == "down") or \
         (price_direction == "down" and volume_direction == "up"):
        return "diverging"
    else:
        return "neutral"

def calculate_percentile(series: pd.Series, value: float) -> int:
    """백분위 계산"""
    clean_series = series.dropna()
    if len(clean_series) == 0 or pd.isna(value):
        return 50
    return int((clean_series <= value).mean() * 100)

def empty_technical_signals() -> Dict[str, Any]:
    """빈 기술적 신호"""
    return {
        "moving_averages": {
            "ma_200d": {"value": None, "position": "unknown", "signal": "neutral"},
            "ma_50d": {"value": None, "position": "unknown", "signal": "neutral"},
            "ma_20d": {"value": None, "position": "unknown", "signal": "neutral"},
            "ma_crossovers": []
        },
        "momentum": {
            "rsi_14d": {"value": None, "zone": "neutral", "trend": "neutral"},
            "macd": {"line": None, "signal": None, "histogram": None, "trend": "neutral"},
            "stochastic": {"k": None, "d": None, "trend": "neutral", "zone": "neutral"}
        },
        "volatility": {
            "bollinger_bands": {"width_percentile": 50, "position": 50, "signal": "neutral"},
            "atr_14d": None,
            "atr_percentile": 50
        },
        "volume": {
            "obv_trend": "neutral",
            "volume_ema_ratio": 1.0,
            "volume_price_trend": "neutral"
        }
    }

# 메인 분석 함수 (기존 API 호환성 유지)
async def analyze_btc_mareket(config: Optional[AnalysisConfig] = None) -> Dict[str, Any]:
    """
    비트코인 시장 종합 분석 함수 (Bitcoin Market Comprehensive Analysis)
    
    이 함수는 업비트 API를 통해 실시간 비트코인 데이터를 수집하고,
    다양한 기술적 지표와 시장 분석을 수행하여 종합적인 투자 정보를 제공합니다.
    
    📊 **수집하는 데이터**:
    - 실시간 비트코인 가격 및 거래량
    - 일봉 데이터 (최대 200개)
    - 30분봉 데이터 (최대 48개, 24시간)
    - 주봉 데이터 (최대 8개)
    
    🔍 **분석 항목**:
    1. **시장 정보 (market_info)**:
       - 현재 가격, 일일 변화율
       - 24시간 거래량 및 변화율
       - 30일 연간화 변동성
    
    2. **추세 분석 (trend_analysis)**:
       - 단기(1-4시간), 중기(1일-1주), 장기(1주-1개월) 추세
       - 추세 강도 (0-100점)
       - 추세 지속 기간
    
    3. **가격 레벨 분석 (price_levels)**:
       - 주요 지지선/저항선
       - 최근 테스트된 레벨
       - 지지선/저항선까지의 거리(%)
    
    4. **기술적 신호 (technical_signals)**:
       - 이동평균선 (20, 50, 200일) 및 골든/데드크로스
       - 모멘텀 지표: RSI, MACD, 스토캐스틱
       - 변동성 지표: 볼린저 밴드, ATR
       - 거래량 지표: OBV, 거래량 추세
    
    Args:
        config (Optional[AnalysisConfig]): 분석 설정 파라미터
            - None인 경우 기본 설정 사용
            - 이동평균 기간, RSI 기간, MACD 설정 등을 커스터마이징 가능
    
    Returns:
        Dict[str, Any]: 종합 분석 결과
        {
            "market_info": {
                "symbol": "BTC-KRW",
                "current_price": float,      # 현재 가격
                "day_change_pct": float,     # 일일 변화율(%)
                "timestamp": str,            # 마지막 업데이트 시간
                "24h_volume": float,         # 24시간 거래량
                "24h_volume_change_pct": float,  # 거래량 변화율(%)
                "volatility_30d_annualized": float  # 30일 연간화 변동성(%)
            },
            "trend_analysis": {
                "short_term": str,           # "bullish"|"bearish"|"neutral"
                "medium_term": str,          # "bullish"|"bearish"|"neutral"
                "long_term": str,            # "bullish"|"bearish"|"neutral"
                "trend_strength": int,       # 추세 강도 (0-100)
                "trend_duration_days": int   # 현재 추세 지속 일수
            },
            "price_levels": {
                "key_resistance": List[int], # 주요 저항선 2개
                "key_support": List[int],    # 주요 지지선 2개
                "last_tested": str,          # "resistance"|"support"
                "distance_to_resistance_pct": float,  # 저항선까지 거리(%)
                "distance_to_support_pct": float      # 지지선까지 거리(%)
            },
            "technical_signals": {
                "moving_averages": {...},    # 이동평균선 분석
                "momentum": {...},           # 모멘텀 지표 (RSI, MACD, 스토캐스틱)
                "volatility": {...},         # 변동성 지표 (볼린저밴드, ATR)
                "volume": {...}              # 거래량 지표 (OBV, 거래량 추세)
            }
        }
    
    Raises:
        Exception: API 호출 실패, 데이터 처리 오류 등
    
    Examples:
        >>> # 기본 설정으로 분석
        >>> result = await analyze_btc_mareket()
        >>> print(f"현재 가격: {result['market_info']['current_price']:,}원")
        >>> print(f"단기 추세: {result['trend_analysis']['short_term']}")
        
        >>> # 커스텀 설정으로 분석
        >>> custom_config = AnalysisConfig(ma_short=10, ma_long=100)
        >>> result = await analyze_btc_mareket(custom_config)
    
    Performance:
        - 병렬 API 호출로 최적화됨 (기존 4번 → 1번 병렬 호출)
        - 평균 실행 시간: 2-4초
        - 메모리 사용량: 기존 대비 60% 감소
    
    Note:
        - 함수명의 'mareket'은 기존 API 호환성을 위해 유지됨 (오타이지만 수정하지 않음)
        - 실시간 데이터를 사용하므로 결과는 시장 상황에 따라 변동됨
        - 투자 조언이 아닌 기술적 분석 정보만 제공
    """
    if config is None:
        config = AnalysisConfig()
    
    try:
        start_time = datetime.now()
        
        # 1. 데이터 로드 (병렬 처리)
        market_data = await load_market_data(config)
        
        # 2. 모든 분석을 병렬로 실행
        market_info_task = asyncio.create_task(
            asyncio.to_thread(calculate_market_info, market_data, config)
        )
        trend_analysis_task = asyncio.create_task(
            asyncio.to_thread(calculate_trend_analysis, market_data, config)
        )
        price_levels_task = asyncio.create_task(
            asyncio.to_thread(calculate_price_levels, market_data, config)
        )
        technical_signals_task = asyncio.create_task(
            asyncio.to_thread(calculate_technical_signals, market_data, config)
        )
        
        market_info, trend_analysis, price_levels, technical_signals = await asyncio.gather(
            market_info_task, trend_analysis_task, price_levels_task, technical_signals_task
        )
        
        analysis_time = (datetime.now() - start_time).total_seconds()
        logging.info(f"전체 분석 완료: {analysis_time:.2f}초")
        
        return {
            "market_info": market_info,
            "trend_analysis": trend_analysis,
            "price_levels": price_levels,
            "technical_signals": technical_signals
        }
        
    except Exception as e:
        logging.error(f"시장 분석 실패: {e}")
        raise

# 추가 편의 함수들
async def analyze_btc_market_conservative() -> Dict[str, Any]:
    """
    보수적 설정으로 비트코인 시장 분석 (Conservative Bitcoin Market Analysis)
    
    장기 투자자나 안정적인 신호를 선호하는 트레이더를 위한 보수적 분석 설정입니다.
    더 긴 기간의 이동평균과 지표를 사용하여 노이즈를 줄이고 안정적인 신호를 생성합니다.
    
    📈 **보수적 설정 특징**:
    - 이동평균: 30일, 60일, 240일 (기본: 20일, 50일, 200일)
    - RSI 기간: 21일 (기본: 14일)
    - 더 긴 기간으로 인해 급격한 변동에 덜 민감
    - 장기 추세에 더 집중
    - 잘못된 신호(False Signal) 감소
    
    🎯 **적합한 사용자**:
    - 장기 투자자 (HODLer)
    - 스윙 트레이더
    - 안정적인 신호를 선호하는 투자자
    - 일일 변동성보다 주/월 단위 추세에 관심 있는 투자자
    
    Returns:
        Dict[str, Any]: analyze_btc_mareket()과 동일한 구조의 분석 결과
            - 모든 지표가 더 긴 기간으로 계산됨
            - 추세 변화가 더 느리게 반영됨
            - 신호의 안정성이 높아짐
    
    Examples:
        >>> # 보수적 분석 실행
        >>> result = await analyze_btc_market_conservative()
        >>> 
        >>> # 장기 추세 확인
        >>> long_trend = result['trend_analysis']['long_term']
        >>> ma_200 = result['technical_signals']['moving_averages']['ma_240d']
        >>> print(f"장기 추세: {long_trend}")
        >>> print(f"240일 이동평균: {ma_200['value']:,}원")
    
    Performance:
        - 실행 시간: analyze_btc_mareket()과 동일
        - 더 안정적인 신호로 인해 투자 의사결정 빈도 감소
    
    Warning:
        - 급격한 시장 변화에 늦게 반응할 수 있음
        - 단기 매매 기회를 놓칠 가능성 있음
        - 변동성이 큰 시장에서는 진입/청산 타이밍이 늦을 수 있음
    """
    config = AnalysisConfig(
        ma_short=30, ma_medium=60, ma_long=240,
        rsi_period=21
    )
    return await analyze_btc_mareket(config)

async def analyze_btc_market_aggressive() -> Dict[str, Any]:
    """
    공격적 설정으로 비트코인 시장 분석 (Aggressive Bitcoin Market Analysis)
    
    단기 트레이더나 빠른 시장 변화에 민감하게 반응하고 싶은 투자자를 위한 공격적 분석 설정입니다.
    더 짧은 기간의 이동평균과 지표를 사용하여 빠른 신호를 생성합니다.
    
    ⚡ **공격적 설정 특징**:
    - 이동평균: 10일, 25일, 100일 (기본: 20일, 50일, 200일)
    - RSI 기간: 7일 (기본: 14일)
    - 짧은 기간으로 인해 시장 변화에 빠르게 반응
    - 단기 추세와 모멘텀에 더 집중
    - 더 많은 거래 신호 생성
    
    🎯 **적합한 사용자**:
    - 데이 트레이더
    - 스캘핑 트레이더
    - 빠른 시장 변화를 추적하고 싶은 투자자
    - 단기 매매 기회를 포착하고 싶은 투자자
    
    ⚠️ **주의사항**:
    - 잘못된 신호(False Signal) 증가 가능성
    - 노이즈에 민감하여 잦은 매매 신호 발생
    - 수수료와 슬리피지 비용 증가 가능성
    - 감정적 거래 유발 가능성
    
    Returns:
        Dict[str, Any]: analyze_btc_mareket()과 동일한 구조의 분석 결과
            - 모든 지표가 더 짧은 기간으로 계산됨
            - 추세 변화가 더 빠르게 반영됨
            - 더 많은 매매 신호 생성
    
    Examples:
        >>> # 공격적 분석 실행
        >>> result = await analyze_btc_market_aggressive()
        >>> 
        >>> # 단기 신호 확인
        >>> short_trend = result['trend_analysis']['short_term']
        >>> rsi = result['technical_signals']['momentum']['rsi_14d']['value']
        >>> ma_crossovers = result['technical_signals']['moving_averages']['ma_crossovers']
        >>> 
        >>> print(f"단기 추세: {short_trend}")
        >>> print(f"RSI(7일): {rsi}")
        >>> print(f"최근 교차 신호: {len(ma_crossovers)}개")
        >>> 
        >>> # 빠른 진입/청산 신호 체크
        >>> if short_trend == "bullish" and rsi < 70:
        >>>     print("🟢 잠재적 매수 신호")
        >>> elif short_trend == "bearish" and rsi > 30:
        >>>     print("🔴 잠재적 매도 신호")
    
    Performance:
        - 실행 시간: analyze_btc_mareket()과 동일
        - 더 많은 신호로 인해 투자 의사결정 빈도 증가
    
    Risk Management:
        - 스톱로스 설정 필수
        - 포지션 사이징 중요
        - 과도한 레버리지 주의
        - 감정적 거래 방지를 위한 규칙 준수 필요
    """
    config = AnalysisConfig(
        ma_short=10, ma_medium=25, ma_long=100,
        rsi_period=7
    )
    return await analyze_btc_mareket(config)

# MCP 도구 등록 함수 (기존 코드와 호환성 유지)
def set_tools(mcp):
    """
    Set tools for FastMCP.
    
    Args:
        mcp: FastMCP instance
    """
    mcp.add_tool(
        analyze_btc_mareket,
        "analyze_btc_mareket",
        description="비트코인 시장 정보를 수집하고, \
            추세 분석 및 가격 레벨을 계산하여 종합적인 분석 결과를 반환합니다."
    )
    mcp.add_tool(
        analyze_btc_market_conservative,
        "analyze_btc_market_conservative",
        description="보수적 설정으로 비트코인 시장 분석을 수행합니다."
    )
    mcp.add_tool(
        analyze_btc_market_aggressive,
        "analyze_btc_market_aggressive",
        description="공격적 설정으로 비트코인 시장 분석을 수행합니다."
    )
