import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from app.schemas.candle import MinuteCandleStick
from app.schemas.ticker import Ticker

def test_minute_candlestick_from_dict():
    data = {
        "market": "KRW-BTC",
        "candle_date_time_utc": "2024-01-01T00:00:00",
        "candle_date_time_kst": "2024-01-01T09:00:00",
        "opening_price": "100.0",
        "high_price": "110.0",
        "low_price": "90.0",
        "trade_price": "105.0",
        "timestamp": "1609459200000",
        "candle_acc_trade_price": "1000.0",
        "candle_acc_trade_volume": "10.0",
        "unit": "1",
    }

    candle = MinuteCandleStick.from_dict(data)

    assert candle.market == data["market"]
    assert candle.candle_date_time_utc == data["candle_date_time_utc"]
    assert candle.opening_price == 100.0
    assert candle.high_price == 110.0
    assert candle.low_price == 90.0
    assert candle.trade_price == 105.0
    assert candle.timestamp == 1609459200000
    assert candle.candle_acc_trade_price == 1000.0
    assert candle.candle_acc_trade_volume == 10.0
    assert candle.unit == 1


def test_ticker_from_dict():
    data = {
        "market": "KRW-BTC",
        "trade_date": "20240101",
        "trade_time": "123000",
        "trade_date_kst": "20240101",
        "trade_time_kst": "213000",
        "trade_timestamp": "1609459200000",
        "opening_price": "100.0",
        "high_price": "110.0",
        "low_price": "90.0",
        "trade_price": "105.0",
        "prev_closing_price": "95.0",
        "change": "RISE",
        "change_price": "10.0",
        "change_rate": "0.1",
        "signed_change_price": "10.0",
        "signed_change_rate": "0.1",
        "trade_volume": "5.0",
        "acc_trade_price": "500.0",
        "acc_trade_price_24h": "1000.0",
        "acc_trade_volume": "50.0",
        "acc_trade_volume_24h": "100.0",
        "highest_52_week_price": "120.0",
        "highest_52_week_date": "2024-01-01",
        "lowest_52_week_price": "80.0",
        "lowest_52_week_date": "2023-06-01",
        "timestamp": "1609459300000",
    }

    ticker = Ticker.from_dict(data)

    assert ticker.market == data["market"]
    assert ticker.trade_timestamp == 1609459200000
    assert ticker.opening_price == 100.0
    assert ticker.high_price == 110.0
    assert ticker.low_price == 90.0
    assert ticker.trade_price == 105.0
    assert ticker.prev_closing_price == 95.0
    assert ticker.change == "RISE"
    assert ticker.change_price == 10.0
    assert ticker.change_rate == 0.1
    assert ticker.signed_change_price == 10.0
    assert ticker.signed_change_rate == 0.1
    assert ticker.trade_volume == 5.0
    assert ticker.acc_trade_price == 500.0
    assert ticker.acc_trade_price_24h == 1000.0
    assert ticker.acc_trade_volume == 50.0
    assert ticker.acc_trade_volume_24h == 100.0
    assert ticker.highest_52_week_price == 120.0
    assert ticker.highest_52_week_date == "2024-01-01"
    assert ticker.lowest_52_week_price == 80.0
    assert ticker.lowest_52_week_date == "2023-06-01"
    assert ticker.timestamp == 1609459300000
