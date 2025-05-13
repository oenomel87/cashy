from typing import List
import pandas as pd
import numpy as np
import math
from datetime import datetime
from mcp.server.fastmcp import FastMCP

from app.tools.upbit import get_current_ticker, get_candles_for_daily, get_candles_for_minutes, get_candles_for_weekly
from app.schemas.ticker import Ticker
from app.schemas.candle import MinuteCandleStick, DailyCandleStick, WeeklyCandleStick

async def get_market_info() -> dict:
    """
    비트코인의 시장 정보(market_info)를 구하는 함수
    market_info는 비트코인 가격, 변동성, 거래량 변화율 등을 포함합니다.
    market_info 데이터
    - symbol: 비트코인 심볼 (KRW-BTC)
    - current_price: 현재 비트코인 가격
    - day_change_pct: 24시간 변동률
    - timestamp: 현재 시간 (UTC)
    - 24h_volume: 24시간 거래량
    - 24h_volume_change_pct: 24시간 거래량 변화율

    Returns:
        market_info: 비트코인 시장 정보
    """
    # 1. 현재 티커 정보 가져오기
    current_ticker: Ticker = await get_current_ticker()
    
    # 2. 일별 캔들스틱 데이터 가져오기 (최소 30일 - 변동성 계산용)
    daily_candles: List[DailyCandleStick] = await get_candles_for_daily(count=30)
    
    # 3. 분 단위 캔들스틱 데이터 가져오기 (48개 30분 캔들 = 24시간)
    minute_candles: List[MinuteCandleStick] = await get_candles_for_minutes(minutes=30, count=48)
    
    # 4. 24시간 거래량 변화율 계산
    volume_change_pct = calculate_volume_change(minute_candles)
    
    # 5. 30일 변동성 계산
    volatility = calculate_volatility(daily_candles)
    
    # 6. market_info 객체 구성
    market_info = {
        "symbol": "BTC-KRW",  # 변환된 심볼
        "current_price": current_ticker.trade_price,
        "day_change_pct": round(current_ticker.signed_change_rate * 100, 2),
        "timestamp": datetime.fromtimestamp(current_ticker.timestamp / 1000).strftime('%Y-%m-%dT%H:%M:%SZ'),
        "24h_volume": round(current_ticker.acc_trade_volume_24h, 2),
        "24h_volume_change_pct": volume_change_pct,
        "volatility_30d_annualized": volatility
    }
    
    return market_info

def calculate_volume_change(minute_candles: List[MinuteCandleStick]) -> float:
    """
    24시간 거래량 변화율 계산
    """
    # 최근 24개의 캔들(12시간)과 이전 24개의 캔들(12시간) 비교
    recent_volume = sum(candle.candle_acc_trade_volume for candle in minute_candles[:24])
    previous_volume = sum(candle.candle_acc_trade_volume for candle in minute_candles[24:])
    
    # 변화율 계산
    if previous_volume > 0:
        volume_change_pct = ((recent_volume - previous_volume) / previous_volume) * 100
        return round(volume_change_pct, 2)
    return 0.0

def calculate_volatility(daily_candles: List[DailyCandleStick]) -> float:
    """
    30일 연간화 변동성 계산
    """
    # 일별 수익률 계산
    returns = []
    for i in range(1, len(daily_candles)):
        today_price = daily_candles[i-1].trade_price
        yesterday_price = daily_candles[i].trade_price
        daily_return = math.log(today_price / yesterday_price)
        returns.append(daily_return)
    
    # 표준편차 계산
    if returns:
        std_dev = np.std(returns)
        # 연간화 (252 거래일 기준)
        annualized_volatility = std_dev * math.sqrt(252) * 100
        return round(annualized_volatility, 1)
    return 0.0

async def get_trend_analysis() -> dict:
    """
    비트코인의 추세 분석(trend_analysis) 정보를 구하는 함수
    trend_analysis는 단기, 중기, 장기 추세 분석 결과를 포함합니다.
    trend_analysis 데이터
    - short_term: 단기 추세 (1h-4h)
    - medium_term: 중기 추세 (1d-1w)
    - long_term: 장기 추세 (1w-1m)
    - trend_strength: 추세 강도 (0-100)
    - trend_duration_days: 현재 추세 지속 기간 (일수)
    """
    # 1. 시간별 캔들스틱 데이터 가져오기 (단기 추세 분석용)
    hourly_candles = await get_candles_for_minutes(minutes=60, count=60)
    
    # 2. 일별 캔들스틱 데이터 가져오기 (중기 추세 분석용)
    daily_candles = await get_candles_for_daily(count=14)
    
    # 3. 주별 캔들스틱 데이터 가져오기 (장기 추세 분석용)
    weekly_candles = await get_candles_for_weekly(count=8)
    
    # 4. 단기 추세 분석 (1-4시간)
    short_term_trend = analyze_short_term_trend(hourly_candles[:4])
    
    # 5. 중기 추세 분석 (1일-1주)
    medium_term_trend = analyze_medium_term_trend(daily_candles[:7])
    
    # 6. 장기 추세 분석 (1주-1개월)
    long_term_trend = analyze_long_term_trend(weekly_candles)
    
    # 7. 추세 강도 계산
    trend_strength = calculate_trend_strength(short_term_trend, medium_term_trend, long_term_trend)
    
    # 8. 추세 지속 기간 계산
    trend_duration = calculate_trend_duration(daily_candles)
    
    # 9. trend_analysis 객체 구성
    trend_analysis = {
        "short_term": short_term_trend["trend"],  # 단기 추세 (1h-4h)
        "medium_term": medium_term_trend["trend"],  # 중기 추세 (1d-1w)
        "long_term": long_term_trend["trend"],  # 장기 추세 (1w-1m)
        "trend_strength": round(trend_strength),  # 추세 강도 (0-100)
        "trend_duration_days": trend_duration  # 현재 추세 지속 기간
    }
    
    return trend_analysis

def analyze_short_term_trend(hourly_candles: List[MinuteCandleStick]) -> dict:
    """
    단기 추세 분석 (1-4시간)
    """
    # 상승/하락 캔들 수 계산
    up_count = 0
    down_count = 0
    
    for candle in hourly_candles:
        if candle.trade_price > candle.opening_price:
            up_count += 1
        elif candle.trade_price < candle.opening_price:
            down_count += 1
    
    # 시작 가격과 종료 가격의 차이 계산
    start_price = hourly_candles[-1].opening_price
    end_price = hourly_candles[0].trade_price
    price_change = ((end_price - start_price) / start_price) * 100
    
    # 추세 결정
    trend = "neutral"
    strength = 50 - abs(price_change * 10)
    
    if price_change > 1.0 or (up_count >= 3 and price_change > 0):
        trend = "bullish"
        strength = min(100, abs(price_change) * 15 + up_count * 10)
    elif price_change < -1.0 or (down_count >= 3 and price_change < 0):
        trend = "bearish"
        strength = min(100, abs(price_change) * 15 + down_count * 10)
    
    return {"trend": trend, "strength": strength, "price_change": price_change}

def analyze_medium_term_trend(daily_candles: List[DailyCandleStick]) -> dict:
    """
    중기 추세 분석 (1일-1주)
    """
    # 상승/하락일 계산
    up_days = 0
    down_days = 0
    
    for candle in daily_candles:
        if candle.change_rate > 0:
            up_days += 1
        elif candle.change_rate < 0:
            down_days += 1
    
    # 시작 가격과 종료 가격의 차이 계산
    start_price = daily_candles[-1].trade_price
    end_price = daily_candles[0].trade_price
    price_change = ((end_price - start_price) / start_price) * 100
    
    # 추세 결정
    trend = "neutral"
    strength = 50 - abs(price_change * 3)
    
    if price_change > 4 or (up_days >= 4 and price_change > 0):
        trend = "bullish"
        strength = min(100, abs(price_change) * 5 + up_days * 5)
    elif price_change < -4 or (down_days >= 4 and price_change < 0):
        trend = "bearish"
        strength = min(100, abs(price_change) * 5 + down_days * 5)
    
    return {"trend": trend, "strength": strength, "price_change": price_change}

def analyze_long_term_trend(weekly_candles: List[WeeklyCandleStick]) -> dict:
    """
    장기 추세 분석 (1주-1개월)
    """
    # 상승/하락 주 계산
    up_weeks = 0
    down_weeks = 0
    
    for i in range(len(weekly_candles) - 1):
        current_price = weekly_candles[i].trade_price
        prev_price = weekly_candles[i + 1].trade_price
        
        if current_price > prev_price:
            up_weeks += 1
        elif current_price < prev_price:
            down_weeks += 1
    
    # 시작 가격과 종료 가격의 차이 계산
    start_price = weekly_candles[-1].trade_price
    end_price = weekly_candles[0].trade_price
    price_change = ((end_price - start_price) / start_price) * 100
    
    # 추세 결정
    trend = "neutral"
    strength = 50 - abs(price_change)
    
    if price_change > 8 or (up_weeks >= 3 and price_change > 0):
        trend = "bullish"
        strength = min(100, abs(price_change) * 2 + up_weeks * 10)
    elif price_change < -8 or (down_weeks >= 3 and price_change < 0):
        trend = "bearish"
        strength = min(100, abs(price_change) * 2 + down_weeks * 10)
    
    return {"trend": trend, "strength": strength, "price_change": price_change}

def calculate_trend_strength(short_term, medium_term, long_term) -> float:
    """
    추세 강도 계산 - 단기, 중기, 장기 추세의 가중 평균
    """
    # 가중 평균 계산
    weighted_strength = (
        (short_term["strength"] * 0.2) + 
        (medium_term["strength"] * 0.3) + 
        (long_term["strength"] * 0.5)
    )
    
    # 추세 일치 보너스
    if (short_term["trend"] == medium_term["trend"] and 
        medium_term["trend"] == long_term["trend"]):
        return min(100, weighted_strength + 15)
    
    # 중장기 추세 일치 보너스
    if medium_term["trend"] == long_term["trend"]:
        return min(100, weighted_strength + 8)
    
    return weighted_strength

def calculate_trend_duration(daily_candles: List[DailyCandleStick]) -> int:
    """
    현재 추세가 지속된 일수 계산
    """
    # 최근 추세 방향 결정
    current_direction = None
    
    # 최근 2일 종가 비교로 추세 방향 결정
    if daily_candles[0].trade_price > daily_candles[1].trade_price:
        current_direction = "up"
    elif daily_candles[0].trade_price < daily_candles[1].trade_price:
        current_direction = "down"
    else:
        # 같은 가격일 경우 이전 추세 확인
        for i in range(2, len(daily_candles)):
            if daily_candles[1].trade_price != daily_candles[i].trade_price:
                current_direction = "up" if daily_candles[1].trade_price > daily_candles[i].trade_price else "down"
                break
    
    # 추세 지속 일수 계산
    duration_days = 1  # 최소 하루부터 시작
    
    if current_direction:
        for i in range(1, len(daily_candles) - 1):
            comparison = False
            if current_direction == "up":
                comparison = daily_candles[i].trade_price > daily_candles[i + 1].trade_price
            else:
                comparison = daily_candles[i].trade_price < daily_candles[i + 1].trade_price
            
            if comparison:
                duration_days += 1
            else:
                break
    
    return duration_days

async def get_price_levels() -> dict:
    """
    비트코인의 가격 레벨(price_levels) 정보를 구하는 함수
    price_levels는 주요 지지선과 저항선, 최근 테스트된 레벨, 저항선과 지지선까지의 거리 등을 포함합니다.
    price_levels 데이터
    - key_resistance: 주요 저항선 (상위 2개)
    - key_support: 주요 지지선 (상위 2개)
    - last_tested: 최근 테스트된 레벨 (지지선/저항선)
    - distance_to_resistance_pct: 저항선까지의 거리 (퍼센트)
    - distance_to_support_pct: 지지선까지의 거리 (퍼센트)
    """
    # 1. 현재 티커 정보 가져오기
    current_ticker = await get_current_ticker()
    current_price = current_ticker.trade_price
    
    # 2. 일별 캔들스틱 데이터 가져오기 (분석용)
    daily_candles = await get_candles_for_daily(count=30)
    
    # 3. 가격 클러스터 식별
    price_clusters = identify_price_clusters(daily_candles)
    
    # 4. 주요 지지선과 저항선 결정
    support_resistance = determine_support_and_resistance(price_clusters, current_price)
    
    # 5. 최근 테스트된 레벨 확인
    last_tested = identify_last_tested_level(daily_candles, 
                                             support_resistance["support"], 
                                             support_resistance["resistance"])
    
    # 6. 저항선과 지지선까지의 거리 계산
    distance_to_resistance = calculate_distance_to_level(
        current_price, support_resistance["resistance"][0])
    distance_to_support = calculate_distance_to_level(
        support_resistance["support"][0], current_price)
    
    # 7. price_levels 객체 구성
    price_levels = {
        "key_resistance": support_resistance["resistance"],
        "key_support": support_resistance["support"],
        "last_tested": last_tested,
        "distance_to_resistance_pct": round(distance_to_resistance, 2),
        "distance_to_support_pct": round(distance_to_support, 2)
    }
    
    return price_levels

def identify_price_clusters(candles: List[DailyCandleStick]) -> List[dict]:
    """
    가격 클러스터 식별 - 중요한 가격 구간 찾기
    """
    # 고점과 저점 데이터 수집
    high_prices = [candle.high_price for candle in candles]
    low_prices = [candle.low_price for candle in candles]
    all_prices = high_prices + low_prices
    
    # 가격 범위 정의 (최대 0.5% 편차)
    price_tolerance = 0.005
    clusters = []
    
    # 처리된 가격을 추적하기 위한 집합
    processed_prices = set()
    
    # 각 가격에 대해 클러스터링
    for i, price in enumerate(all_prices):
        # 이미 처리된 가격이면 건너뜀
        if price in processed_prices:
            continue
        
        # 이 가격을 중심으로 하는 클러스터 찾기
        cluster = {
            "central_price": price,
            "prices": [price],
            "touch_count": 1,
            "type": "resistance" if i < len(high_prices) else "support"
        }
        
        # 비슷한 가격 범위에 있는 다른 가격 찾기
        for j in range(i + 1, len(all_prices)):
            other_price = all_prices[j]
            deviation = abs(price - other_price) / price
            
            if deviation <= price_tolerance:
                cluster["prices"].append(other_price)
                cluster["touch_count"] += 1
                processed_prices.add(other_price)
        
        # 클러스터 중심 가격을 평균으로 업데이트
        if len(cluster["prices"]) > 1:
            cluster["central_price"] = sum(cluster["prices"]) / len(cluster["prices"])
        
        # 클러스터 유형 결정 (고점이 더 많으면 저항선, 저점이 더 많으면 지지선)
        high_count = sum(1 for p in cluster["prices"] if p in high_prices)
        low_count = sum(1 for p in cluster["prices"] if p in low_prices)
        
        if high_count > low_count:
            cluster["type"] = "resistance"
        else:
            cluster["type"] = "support"
        
        # 클러스터 강도 계산
        cluster["strength"] = calculate_cluster_strength(cluster, candles)
        
        # 클러스터 추가
        clusters.append(cluster)
        
        # 이 가격도 처리된 것으로 표시
        processed_prices.add(price)
    
    # 강도 기준 내림차순 정렬
    return sorted(clusters, key=lambda x: x["strength"], reverse=True)

def calculate_cluster_strength(cluster: dict, candles: List[DailyCandleStick]) -> float:
    """
    클러스터 강도 계산
    """
    strength = cluster["touch_count"]
    
    # 최근성에 기반한 추가 강도 (최근 캔들일수록 더 중요)
    for i, candle in enumerate(candles):
        recency_factor = max(1, 5 - i * 0.5)  # 최근 캔들은 더 높은 가중치
        
        # 이 캔들이 클러스터에 터치했는지 확인
        if cluster["type"] == "resistance":
            deviation = abs(cluster["central_price"] - candle.high_price) / cluster["central_price"]
            if deviation <= 0.005:
                strength += recency_factor
        else:
            deviation = abs(cluster["central_price"] - candle.low_price) / cluster["central_price"]
            if deviation <= 0.005:
                strength += recency_factor
    
    # 최근 추세와 일치하면 추가 강도 부여
    recent_trend = "up" if candles[0].trade_price > candles[4].trade_price else "down"
    if (recent_trend == "up" and cluster["type"] == "resistance") or \
       (recent_trend == "down" and cluster["type"] == "support"):
        strength += 2
    
    return strength

def determine_support_and_resistance(clusters, current_price: float) -> dict:
    """
    주요 지지선과 저항선 결정
    """
    # 저항선 (현재 가격보다 높은 클러스터)
    resistance_clusters = sorted(
        [c for c in clusters if c["central_price"] > current_price and c["type"] == "resistance"],
        key=lambda x: x["central_price"]
    )
    
    # 지지선 (현재 가격보다 낮은 클러스터)
    support_clusters = sorted(
        [c for c in clusters if c["central_price"] < current_price and c["type"] == "support"],
        key=lambda x: x["central_price"],
        reverse=True
    )
    
    # 상위 2개의 저항선과 지지선 선택
    resistance_levels = [round(c["central_price"]) for c in resistance_clusters[:2]]
    support_levels = [round(c["central_price"]) for c in support_clusters[:2]]
    
    # 저항선이 충분하지 않은 경우, 기술적 레벨 추가
    if len(resistance_levels) < 2:
        if resistance_levels:
            # 상위 저항선이 있으면 그 위에 1% 추가
            resistance_levels.append(round(resistance_levels[0] * 1.01))
        else:
            # 저항선이 하나도 없으면 현재 가격에서 상승
            resistance_levels.append(round(current_price * 1.01))
            resistance_levels.append(round(current_price * 1.02))
    
    # 지지선이 충분하지 않은 경우, 기술적 레벨 추가
    if len(support_levels) < 2:
        if support_levels:
            # 상위 지지선이 있으면 그 아래에 1% 추가
            support_levels.append(round(support_levels[0] * 0.99))
        else:
            # 지지선이 하나도 없으면 현재 가격에서 하락
            support_levels.append(round(current_price * 0.99))
            support_levels.append(round(current_price * 0.98))
    
    return {
        "resistance": resistance_levels,
        "support": support_levels
    }

def identify_last_tested_level(candles: List[DailyCandleStick], support_levels, resistance_levels) -> str:
    """
    최근 테스트된 레벨 식별
    """
    # 최근 5개 캔들 확인
    recent_candles = candles[:5]
    price_tolerance = 0.005  # 0.5% 오차 허용
    
    last_tested_resistance = -1  # 가장 최근에 저항선이 테스트된 캔들 인덱스
    last_tested_support = -1  # 가장 최근에 지지선이 테스트된 캔들 인덱스
    
    # 각 캔들에 대해 레벨 테스트 확인
    for i, candle in enumerate(recent_candles):
        # 저항선 테스트 확인
        for resistance in resistance_levels:
            high_diff = abs(resistance - candle.high_price) / resistance
            if high_diff < price_tolerance:
                last_tested_resistance = i
                break
        
        # 지지선 테스트 확인
        for support in support_levels:
            low_diff = abs(support - candle.low_price) / support
            if low_diff < price_tolerance:
                last_tested_support = i
                break
        
        # 둘 다 테스트된 경우, 가장 최근 것을 반환
        if last_tested_resistance != -1 and last_tested_support != -1:
            return "resistance" if last_tested_resistance <= last_tested_support else "support"
    
    # 하나만 테스트된 경우
    if last_tested_resistance != -1:
        return "resistance"
    if last_tested_support != -1:
        return "support"
    
    # 둘 다 테스트되지 않은 경우, 최근 가격 움직임으로 결정
    if candles[0].trade_price > candles[1].trade_price:
        return "resistance"  # 상승 중이므로 저항선 방향으로 이동 중
    else:
        return "support"  # 하락 중이므로 지지선 방향으로 이동 중

def calculate_distance_to_level(from_price, to_price) -> float:
    """
    두 가격 레벨 간의 거리를 백분율로 계산
    """
    return ((to_price - from_price) / from_price) * 100

async def analyze_btc_mareket() -> dict:
    """
    비트코인 시장 분석 시스템 실행
    이 함수는 비트코인 시장 정보를 수집하고, 추세 분석 및 가격 레벨을 계산하여 종합적인 분석 결과를 반환합니다.

    분석 결과는 다음과 같은 항목을 포함합니다:
    1. market_info: 비트코인 시장 정보 (현재 가격, 변동성, 거래량 변화율 등)
    - symbol: 비트코인 심볼 (KRW-BTC)
    - current_price: 현재 비트코인 가격
    - day_change_pct: 24시간 변동률
    - timestamp: 현재 시간 (UTC)
    - 24h_volume: 24시간 거래량
    - 24h_volume_change_pct: 24시간 거래량 변화율
    2 trend_analysis: 비트코인 추세 분석 (단기, 중기, 장기)
    - short_term: 단기 추세 (1h-4h)
    - medium_term: 중기 추세 (1d-1w)
    - long_term: 장기 추세 (1w-1m)
    - trend_strength: 추세 강도 (0-100)
    - trend_duration_days: 현재 추세 지속 기간 (일수)
    3. price_levels: 비트코인 가격 레벨 (주요 지지선, 저항선 등)
    - key_resistance: 주요 저항선 (상위 2개)
    - key_support: 주요 지지선 (상위 2개)
    - last_tested: 최근 테스트된 레벨 (지지선/저항선)
    - distance_to_resistance_pct: 저항선까지의 거리 (퍼센트)
    - distance_to_support_pct: 지지선까지의 거리 (퍼센트)
    """
    # 1. market_info 항목 구하기
    market_info = await get_market_info()
    print("Market Info:", market_info)
    
    # 2. trend_analysis 항목 구하기
    trend_analysis = await get_trend_analysis()
    print("Trend Analysis:", trend_analysis)
    
    # 3. price_levels 항목 구하기
    price_levels = await get_price_levels()
    print("Price Levels:", price_levels)
    
    # 4. 전체 분석 결과 통합
    bitcoin_analysis = {
        "market_info": market_info,
        "trend_analysis": trend_analysis,
        "price_levels": price_levels
    }
    
    return bitcoin_analysis

def set_tools(mcp: FastMCP):
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
