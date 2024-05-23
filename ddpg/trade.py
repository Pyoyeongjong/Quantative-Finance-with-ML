# 바이낸스 API
from binance.client import Client
from binance.exceptions import BinanceAPIException
from binance.enums import *

# A2C
from a2c import A2Cagent
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import json

# 경고 무시
import warnings
from sklearn.exceptions import DataConversionWarning

# 특정 경고 무시
warnings.filterwarnings(action='ignore', category=UserWarning)

# 텔레그램
import telegram
import asyncio

# 반내림, 루트
from decimal import Decimal, ROUND_DOWN
import math

### Telegram
bot = telegram.Bot(token="6332731064:AAEOlgnRBgM8RZxW9CnkUPJHvEo54SZoEH8")
chat_id = 1735838793
import talib

# 시간 동기화
import win32api
import time
from datetime import datetime
from datetime import timedelta

# 보조지표 계산/출력 라이브러리
import math
import matplotlib.pyplot as plt

# Numpy / pandas
import numpy as np
import pandas as pd
import pytz

# CSV파일
import os
import csv

# 클라이언트 변수
client = None
home = "D:\\vscode\Quantative-Finance-with-ML\ddpg\save_weights\\"

# 차트 데이터 다운로드
def get_klines(client, symbol, limit, interval):
    # klines 데이터 형태
    # 0=Open time(ms), 1=Open, 2=High, 3=Low, 4=Close, 5=Voume,
    # 6=Close time, 7=Quote asset vloume, 8=Number of trades
    # 9=Taker buy base asset volume 10=Take buy quote asset volume [2차원 list]
    klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
    col_name = ['time', 'open', 'high', 'low', 'close', 'volume', 'close time', 'quote', 'trade_num', 'taker_buy_base',
                'taker_buy_quote', 'ignored']
    data = pd.DataFrame(klines, columns=col_name)
    data = data[['time', 'open', 'high', 'low', 'close', 'volume']].apply(pd.to_numeric)
    return data

def get_candle_subdatas(candles):

    if candles.empty :
        return

    ### 데이터 분석
    # 문자열 -> 숫자 변환 && Pd Series
    close = candles['close'].apply(pd.to_numeric).to_numpy() # 종가 값 활용
    high = candles['high'].apply(pd.to_numeric).to_numpy()
    low = candles['low'].apply(pd.to_numeric).to_numpy()
    volume = candles['volume'].apply(pd.to_numeric).to_numpy()

    front_close = candles['close'].shift(1).apply(pd.to_numeric).to_numpy()

    highr = ( candles['high'] - front_close ) / front_close * 100
    highr.name = 'highp'
    lowr = ( candles['low'] - front_close ) / front_close * 100
    lowr.name = 'lowp'
    closer = ( candles['close'] - front_close ) / front_close * 100
    closer.name = 'closep'
    openr = ( candles['open'] - front_close ) / front_close * 100
    openr.name = 'openp'

    # Numpy밖에 못 쓴다 -> .to_numpy()
    sma5 = (pd.Series(talib.SMA(close, timeperiod=5), name="sma5") - candles['close']) / candles['close'] * 100
    sma5.name = 'sma5p'
    sma10 = (pd.Series(talib.SMA(close, timeperiod=10), name="sma10") - candles['close']) / candles['close'] * 100
    sma10.name = 'sma10p'
    sma20 = (pd.Series(talib.SMA(close, timeperiod=20), name="sma20") - candles['close']) / candles['close'] * 100
    sma20.name = 'sma20p'
    sma40 = (pd.Series(talib.SMA(close, timeperiod=40), name="sma40") - candles['close']) / candles['close'] * 100
    sma40.name = 'sma40p'
    sma60 = (pd.Series(talib.SMA(close, timeperiod=60), name="sma60") - candles['close']) / candles['close'] * 100
    sma60.name = 'sma60p'
    sma90 = (pd.Series(talib.SMA(close, timeperiod=90), name="sma90") - candles['close']) / candles['close'] * 100
    sma90.name = 'sma90p'
    sma120 = (pd.Series(talib.SMA(close, timeperiod=120), name="sma120") - candles['close']) / candles['close'] * 100
    sma120.name = 'sma120p'

    ema5 = (pd.Series(talib.EMA(close, timeperiod=5), name="ema5") - candles['close']) / candles['close'] * 100
    ema5.name = 'ema5p'
    ema20 = (pd.Series(talib.EMA(close, timeperiod=20), name="ema20") - candles['close']) / candles['close'] * 100
    ema20.name = 'ema20p'
    ema60 = (pd.Series(talib.EMA(close, timeperiod=60), name="ema60") - candles['close']) / candles['close'] * 100
    ema60.name = 'ema60p'
    ema120 = (pd.Series(talib.EMA(close, timeperiod=120), name="ema120") - candles['close']) / candles['close'] * 100
    ema120.name = 'ema120p'

    rsi = pd.Series(talib.RSI(close, timeperiod=14), name="rsi")
    volume_sma = pd.Series(talib.SMA(volume, timeperiod=20), name="vol_sma") / candles['volume']
    volume_sma.name = "volp"

    ### 한국 시간으로 맞춰주기 + DateTime으로 변환
    # korea_tz = pytz.timezone('Asia/Seoul')
    # datetime = pd.to_datetime(candles['time'], unit='ms')
    # candles['time'] = datetime.dt.tz_localize(pytz.utc).dt.tz_convert(korea_tz)
    # 볼린저 밴드
    upperband, middleband, lowerband = talib.BBANDS(candles['close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    upperband = ( upperband - candles['close'] ) / candles['close'] * 100
    upperband.name = "upperbandp"
    lowerband = ( lowerband - candles['close'] ) / candles['close'] * 100
    lowerband.name = "lowerbandp"
    # atr
    atr = pd.Series(talib.ATR(high, low, close, timeperiod=14), name="atr")
    # macd
    # macd, macdsignal, macdhist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    # macd = pd.Series(macd, name="macd")
    # macdsignal = pd.Series(macdsignal, name="macdsignal")
    # macdhist = pd.Series(macdhist, name="macdhist")
    # cci
    real = pd.Series(talib.CCI(high, low, close, timeperiod=14), name="cci")
    # adx
    adx = pd.Series(talib.ADX(high, low, close, timeperiod=14),name="adx")
    # 트렌드
    # inclination = calculate_trends(candles, 0)
    # 연결
    data = pd.concat([candles, openr, highr, lowr, closer, sma5, sma10, sma20, sma40, sma60, sma90, sma120, ema5, ema20, ema60, ema120, rsi, volume_sma, upperband, lowerband, atr, real, adx],
                     axis=1)

    data.fillna(0, inplace=True)
    return data

# API 파일 경로
api_key_file_path = "api.txt"
# api_key_file_path = "/home/ubuntu/Bitcoin/Binance/api.txt"

# API 키를 읽어오는 함수
def read_api_keys(file_path):
    with open(file_path, "r") as file:
        api_key = file.readline().strip()
        api_secret = file.readline().strip()
    return api_key, api_secret


# 시스템 시간 동기화
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

# 클라이언트 로그인
def create_client():
    global client
    ### 계좌 연결
    binance_access_key, binance_secret_key = read_api_keys(api_key_file_path)
    try:
        client = Client(binance_access_key, binance_secret_key)
        server_time = client.get_server_time()
        set_system_time(server_time)
    except BinanceAPIException as e:
        print(e)
        exit()
    print("Log in OK")
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

# ticker 정보찾기
def find_long_precision(symbol):

    exchange_info = client.get_exchange_info()
    symbols = exchange_info['symbols']

    for s in symbols:
        if s['symbol'] == symbol:
            filters = {f['filterType']: f for f in s['filters']}
            step_size = Decimal(filters['LOT_SIZE']['stepSize']).normalize()
            tick_size = Decimal(filters['PRICE_FILTER']['tickSize']).normalize()
            min_notional = Decimal(filters['NOTIONAL']['minNotional']).normalize()
            quantity_precision = -step_size.as_tuple().exponent
            price_precision = -tick_size.as_tuple().exponent
            return price_precision, quantity_precision, min_notional
        
def find_short_precision(symbol):
    info = client.futures_exchange_info()
    symbols = info['symbols']

    for s in symbols:
        if s['symbol'] == symbol:
            price_precision = s['pricePrecision']
            quantity_precision = s['quantityPrecision']
            for f in s['filters']:
                if f['filterType'] == 'MIN_NOTIONAL':
                    min_notional = f['notional']
            return price_precision, quantity_precision, float(min_notional)



# 롱 전용 오더
def create_order_market(ticker, quantity):
    order = client.order_market_buy(
        symbol=ticker,
        quantity=quantity,
    )
    return order

def close_order_market(ticker, quantity):
    order = client.order_market_sell(
        symbol=ticker,
        quantity=quantity,
    )
    return order

def create_stop_loss(ticker, quantity, price, lossp, price_precision):
    decimal_format = f'0.{"0" * (int(price_precision)-1)}1'

    OVER = 1.005 # 트리거 값

    order = client.create_order(
        symbol=ticker,
        side=SIDE_SELL,
        type=ORDER_TYPE_STOP_LOSS_LIMIT, # 왜 LIMIT밖에 안돼
        timeInForce=TIME_IN_FORCE_GTC, # 주문의 유효시간
        quantity=quantity,
        stopPrice = Decimal(price * lossp * OVER).quantize(Decimal(decimal_format), rounding=ROUND_DOWN),
        price = Decimal(price * lossp).quantize(Decimal(decimal_format), rounding=ROUND_DOWN)
    )
    return order

def cancel_order(ticker, order):
    if order == None:
        return
    result = client.cancel_order(
        symbol=ticker,
        orderId=order['orderId']
    )
    return result

def cancel_all_open_order(ticker):
    orders = client.get_open_orders(symbol=ticker)
    for order in orders:
        result = client.cancel_order(
            symbol=ticker,
            orderId = order['orderId']
        )
    return

# 숏 전용 오더
def future_create_order_market(symbol, quantity):
    f_order = client.futures_create_order(
        symbol=symbol,
        side=SIDE_SELL,
        type=FUTURE_ORDER_TYPE_MARKET,
        quantity=quantity
    )
    return f_order

def future_create_stop_loss(symbol, quantity, price, lossp, price_precision):
    decimal_format = f'0.{"0" * (int(price_precision)-1)}1'
    f_order = client.futures_create_order(
        symbol=symbol,
        side=SIDE_BUY,
        type=FUTURE_ORDER_TYPE_STOP_MARKET,
        quantity=quantity,
        stopPrice= Decimal(price*lossp).quantize(Decimal(decimal_format), rounding=ROUND_DOWN)
    )
    return f_order

def future_close_order_market(symbol, quantity):
    f_order = client.futures_create_order(
        symbol=symbol,
        side=SIDE_BUY,
        type=FUTURE_ORDER_TYPE_MARKET,
        quantity=quantity
    )
    return f_order

def future_cancel_order(symbol, order):
    if order == None:
        return
    f_order = client.futures_cancel_order(
        symbol=symbol,
        orderID = order['orderId']
    )
    return f_order

def future_cancel_all_open_order(symbol):
    f_order = client.futures_cancel_all_open_orders(
        symbol=symbol
    )
    return f_order

# 수량 확인
def get_balance(ticker):
    client.get_asset_balance(asset=ticker)


LONG_TYPE = 0
SHORT_TYPE = 1
ALL_TYPE = 2

def adjust_precision(precision, number):

    decimal_format = f'0.{"0" * (int(precision)-1)}1'
    decimal_value = Decimal(number).quantize(Decimal(decimal_format), rounding=ROUND_DOWN)
    return float(decimal_value)

# 각 트레이드 센터에서 Ticker에 대한 모든 것들을 관리한다.
class TradeCenter:
    def __init__(self, ticker, agent, amount, lossp):
        self.tick = ticker
        self.ticker = f"{ticker}USDT"
        self.amount = amount
        self.lossp = lossp

        self.stop_order = None
        #self.set_myqt()

        # ticker 정보 모으기 
        # price_precision: 가격 정확도  quantity_precision: 수량 정확도  min_notional: 최소주문금액(USDT)
        self.l_price_precision, self.l_quantity_precision, self.l_min_notional = find_long_precision(self.ticker)
        self.s_price_precision, self.s_quantitiy_precision, self.s_min_notional = find_short_precision(self.ticker)

        self.data_1w = None
        self.data_1d = None
        self.data_4h = None
        self.data_1h = None

        self.data_1w_obs = None
        self.data_1d_obs = None
        self.data_4h_obs = None
        self.data_1h_obs = None
        self.obs_len = None

        self.data_1w_mv = None
        self.data_1d_mv = None
        self.data_4h_mv = None
        self.data_1h_mv = None

        # Z-score Standardization
        self.scaler_standard = StandardScaler()
        self.scaler_minmax = MinMaxScaler()
        self.scaler_minmax.data_max_ = 100
        self.scaler_minmax.data_min_ = 0

        self.intervals = {
            '1w': {'interval': Client.KLINE_INTERVAL_1WEEK, 'data': 'data_1w'},
            '1d': {'interval': Client.KLINE_INTERVAL_1DAY, 'data': 'data_1d'},
            '4h': {'interval': Client.KLINE_INTERVAL_4HOUR, 'data': 'data_4h'},
            '1h': {'interval': Client.KLINE_INTERVAL_1HOUR, 'data': 'data_1h'}
        }

        # Ticker의 mean, variance를 불러온다.
        self.download_mv()

        # 데이터 다운로드 일련 속도

        self.download_datas()
        self.make_obs_datas()
        self.normalize_obs()
        self.get_obs_row()

        self.TradeAgent = A2Cagent(time_steps=0, state_dim=self.obs_len, action_dim=3)
        self.LongAgent= A2Cagent(time_steps=0, state_dim=self.obs_len, action_dim=2)
        self.ShortAgent = A2Cagent(time_steps=0, state_dim=self.obs_len, action_dim=2)

        self.load_weights(agent)

        # 1시간마다 반복해야 할 것 테스트
        # start = time.time()
        # self.download_datas()
        # self.make_obs_datas()
        # self.normalize_obs()
        # self.get_obs_row()
        # print(time.time()-start)

    def normalize_obs(self):
        z_cols = ['openp', 'highp', 'lowp', 'closep',
          'sma5p', 'sma20p',  'sma60p', 'sma120p', 'volp', 
          'upperbandp', 'lowerbandp', 'atr', 'cci', 'adx']
        min_max_cols = ['rsi']

        start = time.time()

        for key, info in self.intervals.items():
            data = getattr(self, info['data']+"_obs")
            data_mv = getattr(self, info['data']+"_mv")
            for col in z_cols:
                # print(data_mv[col][0], data_mv[col][1])
                self.scaler_standard.mean_ = np.array([data_mv[col][0]])
                self.scaler_standard.scale_ = np.array([math.sqrt(data_mv[col][1])])
                data[col] = self.scaler_standard.transform(data[[col]])

            for col in min_max_cols:
                data[col] = self.scaler_minmax.fit_transform(data[[col]])

        # print(time.time()-start)
                

    def download_mv(self):
        start = time.time()
        self.data_1w_mv = pd.read_csv(f"./ddpg/mv_table/{self.ticker}_data_1w_mv_table.csv").drop(columns='Unnamed: 0').dropna().apply(pd.to_numeric)
        self.data_1d_mv = pd.read_csv(f"./ddpg/mv_table/{self.ticker}_data_1d_mv_table.csv").drop(columns='Unnamed: 0').dropna().apply(pd.to_numeric)
        self.data_4h_mv = pd.read_csv(f"./ddpg/mv_table/{self.ticker}_data_4h_mv_table.csv").drop(columns='Unnamed: 0').dropna().apply(pd.to_numeric)
        self.data_1h_mv = pd.read_csv(f"./ddpg/mv_table/{self.ticker}_data_1h_mv_table.csv").drop(columns='Unnamed: 0').dropna().apply(pd.to_numeric)
        # print(time.time() - start)

    def load_weights(self, path):
        path = home+path
        self.TradeAgent.actor.model.load_weights(f"{path}/Trade_actor.weights.h5")
        self.TradeAgent.critic.model.load_weights(f"{path}/Trade_critic.weights.h5")
        self.LongAgent.actor.model.load_weights(f"{path}/Long_actor.weights.h5")
        self.LongAgent.critic.model.load_weights(f"{path}/Long_critic.weights.h5")
        self.ShortAgent.actor.model.load_weights(f"{path}/Short_actor.weights.h5")
        self.ShortAgent.critic.model.load_weights(f"{path}/Short_critic.weights.h5")
        print("Weights Load Okay")
        return
        
    def download_datas(self):
        start = time.time()
        self.data_1w = get_klines(client=client, symbol=self.ticker, limit=150, interval=Client.KLINE_INTERVAL_1WEEK)
        self.data_1d = get_klines(client=client, symbol=self.ticker, limit=150, interval=Client.KLINE_INTERVAL_1DAY)
        self.data_4h = get_klines(client=client, symbol=self.ticker, limit=150, interval=Client.KLINE_INTERVAL_4HOUR)
        self.data_1h = get_klines(client=client, symbol=self.ticker, limit=150, interval=Client.KLINE_INTERVAL_1HOUR)
        print("Downloaded data time = ", datetime.fromtimestamp(int(self.data_1h['time'].iloc[-1])/1000))
        # print(time.time() - start)

    def update_datas(self): # download_datas와 걸리는 시간이 얼마 차이 안난다.
        start = time.time()

        for key, info in self.intervals.items():
            # 현재 인터벌에 대한 새 캔들 데이터를 가져옴
            line = get_klines(client=client, symbol=self.ticker, limit=1, interval=info['interval'])
            data = getattr(self, info['data'])
            # 새 행이 감지되면 기존 데이터 업데이트
            if line['time'].iloc[0] != data['time'].iloc[-1]:
                info['data'] = pd.concat([data.drop(0), line], ignore_index=True)
                # 인스턴스 변수 업데이트
                setattr(self, f"data_{key}", f"{info['data']}")
        
        self.make_obs_datas()        
        # print(time.time() - start)

    def make_obs_datas(self):
        start = time.time()
        drop_list = ['open', 'high', 'low', 'close', 'volume', 'sma10p', 'sma40p', 'sma90p', 'ema5p', 'ema20p','ema60p','ema120p']
        self.data_1w_obs = get_candle_subdatas(self.data_1w)
        self.data_1w_obs = self.data_1w_obs.drop(columns=drop_list)
        self.data_1d_obs = get_candle_subdatas(self.data_1d)
        self.data_1d_obs = self.data_1d_obs.drop(columns=drop_list)
        self.data_4h_obs = get_candle_subdatas(self.data_4h)
        self.data_4h_obs = self.data_4h_obs.drop(columns=drop_list)
        self.data_1h_obs = get_candle_subdatas(self.data_1h)
        self.data_1h_obs = self.data_1h_obs.drop(columns=drop_list)
        # print(time.time() - start)

    def get_obs_row(self):

        start = time.time()
        row_list = []

        for key, info in self.intervals.items():
            data_obs = getattr(self, info['data']+'_obs')
            # print(datetime.fromtimestamp(int(row['time'])/1000))
            row = data_obs.iloc[-2].drop('time')
            row_list.append(row)

        obs = np.concatenate(row_list)
        self.obs_len = len(obs)
        # print(time.time() - start)
        return obs
    
    def check_long_usdt(self):
        balance = client.get_asset_balance(asset='USDT')
        amt = balance['free']
        return float(amt)
    
    def check_short_usdt(self):
        futures_account = client.futures_account_balance()
        for asset in futures_account:
            if asset['asset'] == "USDT":
                amt = asset['availableBalance']
                # amt = asset['Balance']
        return float(amt)

    def check_long_amount(self):
        balance = client.get_asset_balance(asset=self.tick)
        amt = float(balance['free']) + float(balance['locked'])
        return float(amt)
    
    def check_short_amount(self):
        positions = client.futures_position_information()
        amt = None
        for position in positions:
            if position['symbol'] == self.ticker:
                amt = float(position['positionAmt'])
        # return adjust_precision(self.smqyt, float(amt))
        return float(amt)


    def trade(self, now):

        self.download_datas()
        self.make_obs_datas()
        self.normalize_obs()
        obs = self.get_obs_row()

        if adjust_precision(self.l_quantity_precision, self.check_long_amount()) > 0: # if Long
            act = self.LongAgent.get_action(obs)
            act = np.argmax(act)
            if act == 0: # Hold
                # Long 유지 알림
                print(f"{now} Long Holding")
                bot.send_message(chat_id=chat_id, text=f"###[딥러닝 자동매매 봇]-{self.ticker}###\n{now} : 롱 포지션 유지")
                return
            else: # Sell
                self.close_long(now)
                return

        elif adjust_precision(self.s_quantitiy_precision, self.check_short_amount()) < 0: # if Short
            act = self.ShortAgent.get_action(obs)
            act = np.argmax(act)
            if act == 0: # Hold
                # Short 유지 알림
                print(f"{now} Short Holding")
                bot.send_message(chat_id=chat_id, text=f"###[딥러닝 자동매매 봇]-{self.ticker}###\n{now} : 숏 포지션 유지")
                return
            else: # Sell
                self.close_short(now)
                return
            
        else: # No position
            act = self.TradeAgent.get_action(obs)
            raw_act = act
            print(f"{now} Raw_Act = ",raw_act)
            bot.send_message(chat_id=chat_id, text=f"###[딥러닝 자동매매 봇]-{self.ticker}###\n{now} : Raw_Act = {raw_act}")
            act = np.argmax(act)
            if raw_act[act] >= 0.9:
                if act == 0: # Long
                    self.open_long(now)
                    return
                elif act == 1:
                    self.open_short(now)
                    return
                    
            # 미진입 알림
            print(f"{now} STAY")
            bot.send_message(chat_id=chat_id, text=f"###[딥러닝 자동매매 봇]-{self.ticker}###\n{now} : 관망")
            return
        
    def open_long(self, now):
        # 가용 usdt 확인
        available_usdt = self.check_long_usdt() * self.amount
        # 현재 비트코인 시세 확인
        tick_price = self.data_1h['close'].iloc[-1]
        # 포지션 들어가기
        amt = available_usdt / tick_price
        # quantity precision 교정
        amt = adjust_precision(self.l_quantity_precision, amt)
        if float(available_usdt) >= self.l_min_notional and float(amt) > 0 :
            # 롱 진입
            create_order_market(self.ticker, amt)
            # 손절가 설정
            self.stop_order = create_stop_loss(self.ticker, amt, tick_price, (1-self.lossp), self.l_price_precision)
            # Long 진입 알림
            print(f"{now} Long Entered {tick_price} {amt}")
            bot.send_message(chat_id=chat_id, text=f"###[딥러닝 자동매매 봇]-{self.ticker}###\n{now} : 롱 포지션 진입\n진입가 : {tick_price}\n진입량 : {amt}")
            return
        else:
            print(f"{now} Long Enter Failed - No Balance")
            bot.send_message(chat_id=chat_id, text=f"###[딥러닝 자동매매 봇]-{self.ticker}###\n{now} : 롱 진입 자금이 부족합니다.")
            return
        
    def open_short(self, now):
        # 가용 usdt 확인
        available_usdt = self.check_short_usdt() * self.amount
        # 현재 비트코인 시세 확인
        tick_price = self.data_1h['close'].iloc[-1]
        # 포지션 들어가기
        client.futures_change_leverage(symbol=self.ticker, leverage=1)
        amt = available_usdt / tick_price
        # quantitiy precision 교정
        amt = adjust_precision(self.s_quantitiy_precision, amt)
        if float(available_usdt) >= self.s_min_notional and float(amt) > 0: # smqyt 는 ticker마다 다른데, 일일이 찾아봐야 하는 것 같다.     
            # 숏 진입       
            future_create_order_market(self.ticker, amt)
            # 손절가 설정
            self.stop_order = future_create_stop_loss(self.ticker, amt, tick_price, (1+self.lossp), self.s_price_precision)
            # Short 진입 알림
            print(f"{now} Short Entered {tick_price} {amt}")
            bot.send_message(chat_id=chat_id, text=f"###[딥러닝 자동매매 봇]-{self.ticker}###\n{now} : 숏 포지션 진입\n진입가 : {tick_price}\n진입량 : {amt}")
            return
        else:
            print(f"{now} Short Enter Failed - No Balance")
            bot.send_message(chat_id=chat_id, text=f"###[딥러닝 자동매매 봇]-{self.ticker}###\n{now} : 숏 진입 자금이 부족합니다.")
            return
        
    def close_long(self, now):
        cancel_all_open_order(self.ticker)
        amt = self.check_long_amount()
        amt = adjust_precision(self.l_quantity_precision, amt)
        if amt <= 0 :
            return
        
        close_order_market(self.ticker, float(amt))
        # Long 거래종료 알림
        print(f"{now} Long closed {amt}")
        bot.send_message(chat_id=chat_id, text=f"###[딥러닝 자동매매 봇]-{self.ticker}###\n{now} : 롱 포지션 종료\n종료량 : {amt}")
        return

    def close_short(self, now):
        future_cancel_all_open_order(self.ticker)
        amt = self.check_short_amount()
        amt = adjust_precision(self.s_quantitiy_precision, amt)
        if amt >= 0 :
            return
        
        future_close_order_market(self.ticker, float(abs(amt)))
        # Short 거래종료 알림
        print(f"{now} Short closed {amt}")
        bot.send_message(chat_id=chat_id, text=f"###[딥러닝 자동매매 봇]-{self.ticker}###\n{now} : 숏 포지션 종료\n종료량 : {amt}")
        return
        



def main():

    bot.send_message(chat_id=chat_id, text="###[딥러닝 자동매매 봇]###\nCOPYRIGHT by 표영종")
    centers = []
    ETH_Center = TradeCenter("ETH", "a2c_17", 0.3, 0.1)
    ADA_Center = TradeCenter("ADA", "a2c_17", 0.3, 0.1)
    XRP_Center = TradeCenter("XRP", "a2c_17", 0.3, 0.1)
    centers.append(ETH_Center)
    centers.append(ADA_Center)
    centers.append(XRP_Center)
    bot.send_message(chat_id=chat_id, text="###[딥러닝 자동매매 봇]###\n데이터 로드 OK, 매매 시작")

    now = datetime.now()
    formatted_now = now.strftime('%Y-%m-%d %H:%M:%S') # 출력용

    while(1):
        now = datetime.now()
        if now.minute == 0 and now.second < 1: # 충분한 딜레이가 있어야 시간봉을 가져올 수 있다.
            continue
        else:
            if now.minute == 0 and now.second < 5:
                formatted_now = now.strftime('%Y-%m-%d %H:%M:%S') # 출력용
                for center in centers:
                    center.trade(formatted_now)
                time_to_sleep = 60 - now.second  
            else:
                time_to_sleep = 60 - now.second  
            time.sleep(time_to_sleep)

if __name__=='__main__':

    create_client() # 0.5초 걸림
    get_usdt_balance(client, True)
    main()