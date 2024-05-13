# 바이낸스 API
from binance.client import Client
from binance.exceptions import BinanceAPIException
from binance.enums import *


# 텔레그램
import telegram
import asyncio

### Telegram
bot = telegram.Bot(token="6332731064:AAEOlgnRBgM8RZxW9CnkUPJHvEo54SZoEH8")
chat_id = 1735838793

# 시간 동기화
# import win32api
import time
from datetime import datetime
from datetime import timedelta

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

# 클라이언트 변수
client = None

# API 키를 읽어오는 함수
def read_api_keys(file_path):
    with open(file_path, "r") as file:
        api_key = file.readline().strip()
        api_secret = file.readline().strip()
    return api_key, api_secret

