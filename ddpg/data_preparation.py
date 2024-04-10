# 바이낸스 API
from binance.client import Client
from binance.exceptions import BinanceAPIException
from binance.enums import *

# Time 동기화
import time
import win32api

# 보조지표 계산/출력 라이브러리
import talib
import math
import matplotlib.pyplot as plt

# Numpy / pandas
import numpy as np
import pandas as pd
import pytz

# CSV파일
import os
import csv

# Dict 깔끔한 출력
import pprint

# minmax scaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scale_cols = ['open', 'high', 'low', 'close',
              'volume', 'sma7', 'sma20', 'sma60',
              'sma120', 'rsi', 'vol_sma', 'upperband',
              'lowerband', 'atr', 'macd', 'macdsignal', 'macdhist',
              'cci', 'adx']


kline_interval_mapping = {
    "1w": Client.KLINE_INTERVAL_1WEEK,
    "1d": Client.KLINE_INTERVAL_1DAY,
    "4h": Client.KLINE_INTERVAL_4HOUR,
    "1h": Client.KLINE_INTERVAL_1HOUR,
    "15m": Client.KLINE_INTERVAL_15MINUTE,
    "5m": Client.KLINE_INTERVAL_5MINUTE,
    "1m": Client.KLINE_INTERVAL_1MINUTE,
}
ticker = ["BTCUSDT","ETHUSDT", "BNBUSDT","SOLUSDT","XRPUSDT","DOGEUSDT", "ADAUSDT", "AVAXUSDT",
           "SHIBUSDT","DOTUSDT", "LINKUSDT", "TRXUSDT", "MATICUSDT",
           "BCHUSDT", "ICPUSDT", "NEARUSDT", "UNIUSDT", "APTUSDT", "LTCUSDT", "STXUSDT"]
tickers = ["APTUSDT"]

# API 파일 경로
api_key_file_path = "../api.txt"

# 디렉토리 생성
data_dir = 'candle_data'

# 클라이언트 변수
_client = None

### Initiation
# row 생략 없이 출력
pd.set_option('display.max_rows', 20)
# col 생략 없이 출력
pd.set_option('display.max_columns', None)
# 가져올 분봉 데이터의 개수 (최대 500개까지 가능)
limit = 500
# 캔들 데이터 가져오기
symbol = "BTCUSDT"

#시스템 시간 동기화
def set_system_time(serv_time):
    gmtime = time.gmtime(int((serv_time["serverTime"])/1000))
    win32api.SetSystemTime(gmtime[0],
                           gmtime[1],
                           0,
                           gmtime[2],
                           gmtime[3],
                           gmtime[4],
                           gmtime[5],
                           0)

# API 키를 읽어오는 함수
def read_api_keys(file_path):
    with open(file_path, "r") as file:
        api_key = file.readline().strip()
        api_secret = file.readline().strip()
    return api_key, api_secret

def create_client():
    global _client
    ### 계좌 연결
    binance_access_key, binance_secret_key = read_api_keys(api_key_file_path)
    try:
        _client = Client(binance_access_key, binance_secret_key)
        server_time = _client.get_server_time()
        set_system_time(server_time)
    except BinanceAPIException as e:
        print(e)
        exit()
    return

# USDT 잔고 출력
def get_usdt_balance(client, isprint):
    usdt_balance = None
    futures_account = client.futures_account_balance()
    for asset in futures_account:
        if asset['asset'] == "USDT":
            usdt_balance = float(asset['balance'])
            break
    if usdt_balance is not None:
        if isprint:
            print(f"USDT 잔고: {usdt_balance}")
    else:
        print("USDT 잔고를 찾을 수 없습니다.")
    return usdt_balance

def get_klines(client, symbol, limit, interval):
    # klines 데이터 형태
    # 0=Open time(ms), 1=Open, 2=High, 3=Low, 4=Close, 5=Voume,
    # 6=Close time, 7=Quote asset vloume, 8=Number of trades
    # 9=Taker buy base asset volume 10=Take buy quote asset volume [2차원 list]
    klines_1m = client.get_klines(symbol=symbol, interval=interval, limit=limit)
    col_name = ['time', 'open', 'high', 'low', 'close', 'volume', 'close time', 'quote', 'trade_num', 'taker_buy_base',
                'taker_buy_quote', 'ignored']
    return pd.DataFrame(klines_1m, columns=col_name)

def get_klines_by_date(client, symbol, limit, interval, start_time, end_time):
    start_timestamp = int(start_time.timestamp() * 1000)  # 밀리초 단위로 변환
    end_timestamp = int(end_time.timestamp() * 1000)  # 밀리초 단위로 변환

    candles = client.get_klines(symbol=symbol, interval=interval, limit=limit,
                                startTime=start_timestamp, endTime=end_timestamp)
    col_name = ['time', 'open', 'high', 'low', 'close', 'volume', 'close time', 'quote', 'trade_num', 'taker_buy_base',
                'taker_buy_quote', 'ignored']
    return pd.DataFrame(candles, columns=col_name)

def get_candle_data_to_csv(ticker, scale, start_time, end_time): # "1 Jan, 2021", "30 Jun, 2023"
    # csv 파일 생성
    filename = f"{ticker}_{scale}.csv"
    filepath = os.path.join(data_dir, filename)

    with open(filepath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['time', 'open', 'high', 'low', 'close', 'volume'])

        print("Open Ok")

        klines = _client.get_historical_klines(ticker, kline_interval_mapping.get(scale), start_time, end_time)
        print("Get Candles OK")

        for k in klines:
            timestamp = k[0]
            open_price = k[1]
            high_price = k[2]
            low_price = k[3]
            close_price = k[4]
            volume = k[5]
            writer.writerow([timestamp, open_price, high_price, low_price, close_price, volume])

    print("Data fetching and saving completed.")

def get_candle_data_to_csv_1m(ticker, scale, start_time, end_time): #너무 커서 잘라서 쓰자
    # csv 파일 생성
    i=0
    y=2021
    st1 = "1 Jan"
    st2 = "1 Jul"
    en1 = "30 Jun"
    en2 = "31 DEC"

    for y in range (2021, 2024):
        for i in range (0, 2):
            filename = f"{ticker}_{scale}_{y}_{i}.csv"
            print(filename)
            filepath = os.path.join(data_dir, filename)
            with open(filepath, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['time', 'open', 'high', 'low', 'close', 'volume'])

                print("Open Ok")

                if i==0:
                    klines = _client.get_historical_klines(ticker, kline_interval_mapping.get(scale),f"{st1}, {y}" , f"{en1}, {y}")
                else:
                    klines = _client.get_historical_klines(ticker, kline_interval_mapping.get(scale), f"{st2}, {y}",
                                                           f"{en2}, {y}")

                print("Get Candles OK")

                for k in klines:
                    timestamp = k[0]
                    open_price = k[1]
                    high_price = k[2]
                    low_price = k[3]
                    close_price = k[4]
                    volume = k[5]
                    writer.writerow([timestamp, open_price, high_price, low_price, close_price, volume])

            print("Data fetching and saving completed.")

def get_candle_datas_to_csv(ticker, start_time, end_time):
    for key in kline_interval_mapping.keys():
        if key == '1m' or key == '5m': # 1m은 너무커서 좀 그렇다
            continue
        print(f"get_candle_data {ticker}_{key}.csv")
        get_candle_data_to_csv(ticker, key, start_time, end_time)

def get_candle_subdatas(candles):

    if candles.empty :
        return

    ### 데이터 분석
    # 문자열 -> 숫자 변환 && Pd Series
    close = candles['close'].apply(pd.to_numeric).to_numpy() # 종가 값 활용
    high = candles['high'].apply(pd.to_numeric).to_numpy()
    low = candles['low'].apply(pd.to_numeric).to_numpy()
    volume = candles['volume'].apply(pd.to_numeric).to_numpy()

    # Numpy밖에 못 쓴다 -> .to_numpy()
    sma7 = pd.Series(talib.SMA(close, timeperiod=7), name="sma7")
    sma20 = pd.Series(talib.SMA(close, timeperiod=20), name="sma20")
    sma60 = pd.Series(talib.SMA(close, timeperiod=60), name="sma60")
    sma120 = pd.Series(talib.SMA(close, timeperiod=120), name="sma120")

    rsi = pd.Series(talib.RSI(close, timeperiod=14), name="rsi")
    volume_sma = pd.Series(talib.SMA(volume, timeperiod=20), name="vol_sma")
    ### 한국 시간으로 맞춰주기 + DateTime으로 변환
    # korea_tz = pytz.timezone('Asia/Seoul')
    # datetime = pd.to_datetime(candles['time'], unit='ms')
    # candles['time'] = datetime.dt.tz_localize(pytz.utc).dt.tz_convert(korea_tz)
    # 볼린저 밴드
    upperband, middleband, lowerband = talib.BBANDS(candles['close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    upperband.name = "upperband"
    lowerband.name = "lowerband"

    # atr
    atr = pd.Series(talib.ATR(high, low, close, timeperiod=14), name="atr")
    # macd
    macd, macdsignal, macdhist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    macd = pd.Series(macd, name="macd")
    macdsignal = pd.Series(macdsignal, name="macdsignal")
    macdhist = pd.Series(macdhist, name="macdhist")
    # cci
    real = pd.Series(talib.CCI(high, low, close, timeperiod=14), name="cci")
    # adx
    adx = pd.Series(talib.ADX(high, low, close, timeperiod=14),name="adx")
    # 트렌드
    # inclination = calculate_trends(candles, 0)
    # 연결
    data = pd.concat([candles, sma7, sma20, sma60, sma120, rsi, volume_sma, upperband, lowerband, atr, macd, macdsignal, macdhist, real, adx],
                     axis=1)

    data.fillna(-1, inplace=True)
    return data

def get_subdatas():

    for ticker in tickers:
        for key in kline_interval_mapping.keys():
            if key == '1m':
                for y in range (2021, 2024):
                    for i in range(0, 2):
                        print(f"making {ticker}_{key}_{y}_{i}..")
                        df = pd.read_csv(f"candle_data/{ticker}_{key}_{y}_{i}.csv")
                        df_sub = get_candle_subdatas(df)
                        if df_sub is not None :
                            df_sub.to_csv(f"candle_data/{ticker}_{key}_{y}_{i}_sub.csv")
            else:
                print(f"making {ticker}_{key}..")
                df = pd.read_csv(f"candle_data/{ticker}_{key}.csv")
                df_sub = get_candle_subdatas(df)
                if df_sub is not None:
                    df_sub.to_csv(f"candle_data/{ticker}_{key}_sub.csv")

    return


if __name__ == '__main__':
    create_client()
    get_usdt_balance(_client, True)

    # get_candle_data_to_csv("BTCUSDT", "5m", "1 Jan, 2021", "30 Jun, 2023")
    get_subdatas()



