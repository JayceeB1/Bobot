import ccxt
import time
import config
import numpy as np
import talib
import ta
import pandas as pd
import mysql.connector
from mysql.connector import Error
import plotly.graph_objects as go
import plotly.io as pio
from datetime import datetime
import pytz
import binance
from binance import *
import math

##################################################################################################################################################
# Configuration de l'API Binance
##################################################################################################################################################


client = Client(config.API_KEY, config.API_SECRET)

binance = ccxt.binance({
    "apiKey": config.API_KEY,
    "secret": config.API_SECRET,
    "enableRateLimit": True
})


# Fonction pour arrondir une valeur en fonction d'une taille de pas spécifiée
def round_step(value, step_size):
    # Divise la valeur par la taille de pas, arrondit à la baisse le résultat, puis multiplie par la taille de pas pour obtenir la valeur ajustée
    return math.floor(value / step_size) * step_size

# Fonction pour formater un nombre flottant en fonction de la précision donnée
def format_float(value, precision):
    return float(f"{value:.{precision}f}")

# Fonction pour ajuster les données de précision sur l'historique
def adjust_historical_data(data, market_info):    
    adjusted_data = []
    for entry in data:
        timestamp, open_price, high, low, close_price, volume = entry        
        # Récupère les informations de précision du prix et du volume
        price_precision = int(market_info['quoteAssetPrecision'])
        volume_precision = int(market_info['baseAssetPrecision'])
        # Ajuste le prix et le volume
        adjusted_open = round(float(open_price), price_precision)
        adjusted_high = round(float(high), price_precision)
        adjusted_low = round(float(low), price_precision)
        adjusted_close = round(float(close_price), price_precision)
        adjusted_volume = round(float(volume), volume_precision)
        adjusted_data.append([timestamp, adjusted_open, adjusted_high, adjusted_low, adjusted_close, adjusted_volume])
    return adjusted_data

##################################################################################################################################################
# Stratégies 
##################################################################################################################################################


def ichimoku_signals(data):
    df = pd.DataFrame(data, columns=['timestamp', 'open_price', 'high_price', 'low_price', 'close_price', 'volume'])
    ichimoku = ta.trend.IchimokuIndicator(high=df['high_price'], low=df['low_price'], window1=9, window2=26, window3=52)
    df['ichimoku_a'] = ichimoku.ichimoku_a()  # Leading Span A
    df['ichimoku_b'] = ichimoku.ichimoku_b()  # Leading Span B
    df['ichimoku_base_line'] = ichimoku.ichimoku_base_line()  # Base Line (Kijun-Sen)
    df['ichimoku_conversion_line'] = ichimoku.ichimoku_conversion_line()  # Conversion Line (Tenkan-Sen)

    df['ichimoku_a'] = df['ichimoku_a'].shift(-26)
    df['ichimoku_b'] = df['ichimoku_b'].shift(-26)

    return df

def calculate_rsi(data, period):
    close_prices = np.array([entry[4] for entry in data])
    rsi = talib.RSI(close_prices, timeperiod=period)
    data_with_rsi = data.copy()
    data_with_rsi.append(rsi[-1])
    return data_with_rsi

def detect_buy_sell_signals(data, per=50, mult=3.0):
    close_prices = np.array([entry[4] for entry in data])
    abs_diff = np.abs(np.diff(close_prices))
    sma = pd.DataFrame(abs_diff).rolling(window=per).mean().values.flatten()
    smooth_range = talib.EMA(sma, timeperiod=per) * mult
    range_filter = [close_prices[0]]
    for i in range(1, len(close_prices) - 1):
        if close_prices[i] > range_filter[-1]:
            range_filter.append(max(close_prices[i] - smooth_range[i], range_filter[-1]))
        else:
            range_filter.append(min(close_prices[i] + smooth_range[i], range_filter[-1]))

    range_filter.append(close_prices[-1] + smooth_range[-1])
    range_filter = np.array(range_filter)

    upward = np.where(range_filter[1:] > range_filter[:-1], 1, 0)
    downward = np.where(range_filter[1:] < range_filter[:-1], 1, 0)

    data_with_signals = data.copy()
    data_with_signals.append((upward, downward))

    return data_with_signals

##################################################################################################################################################
# autres fonctions
##################################################################################################################################################

# Fonction pour récupérer les données historiques
def fetch_historical_data(symbol, timeframe, since):
    data = []
    limit = 1000
    while True:
        batch = binance.fetch_ohlcv(symbol, timeframe, since, limit)
        time.sleep(binance.rateLimit / 1000) # Respect de la limite de taux de requête
        if len(batch) == 0:
            break
        data += batch
        since = batch[-1][0] + 1
    return data

def convert_to_heiken_ashi(data):
    heiken_ashi_data = []
    previous_ha_candle = None

    for index, row in enumerate(data):
        if not previous_ha_candle:
            ha_open = (row[1] + row[4]) / 2
        else:
            ha_open = (previous_ha_candle[1] + previous_ha_candle[4]) / 2

        ha_close = (row[1] + row[2] + row[3] + row[4]) / 4
        ha_high = max(row[2], ha_open, ha_close)
        ha_low = min(row[3], ha_open, ha_close)

        heiken_ashi_data.append([row[0], ha_open, ha_high, ha_low, ha_close, row[5]])
        previous_ha_candle = heiken_ashi_data[-1]

    return heiken_ashi_data

# Fonction pour créer une connexion à la base de données
def create_db_connection(host, user, database, password):
    connection = None
    try:
        connection = mysql.connector.connect(
            host=host,
            user=user,
            password=password,
            database=database
        )
    except Error as e:
        print(f"Connection: The error '{e}' occurred")

    return connection

# Fonction pour exécuter une requête SQL
def execute_query(connection, query):
    cursor = connection.cursor()
    try:
        cursor.execute(query)
        connection.commit()
    except Error as e:
        print(f"Requête: The error '{e}' occurred")

# Fonction vider la table historical_data
def truncate_historical_data():
    query = f"""
        TRUNCATE TABLE historical_data_ml
    """
    execute_query(connection, query)

# Fonction pour insérer les données historiques dans la table historical_data
def insert_historical_data(connection, symbol, interval, data):
    query = f"""
        INSERT INTO historical_data_ml (symbol, `interval`, timestamp, open, high, low, close, volume, 
                                         ichimoku_a, ichimoku_b, ichimoku_base_line, ichimoku_conversion_line,
                                         rsi, upward_signal, downward_signal)
        VALUES ('{symbol}', '{interval}', {data[0]}, {data[1]}, {data[2]}, {data[3]}, {data[4]}, {data[5]}, 
                {data[6]}, {data[7]}, {data[8]}, {data[9]}, {data[10]}, {data[11]}, {data[12]})
    """
    execute_query(connection, query)



def is_positive(num):
    return num > 0

##################################################################################################################################################
# On rentre dans le dur
##################################################################################################################################################

# choix de la paire, timeframe et date de début
symbol = "RNDR/BUSD"
timeframe = "1h"
since = binance.parse8601("2023-01-01T00:00:00Z")

# Paramètres détection
per=50
mult=3

# récupération des infos de marché
binsymbol=symbol.replace("/", "")
market_info = client.get_symbol_info(binsymbol)
print(market_info)  # Obtient les informations du marché
price_precision = int(market_info['quoteAssetPrecision']) # Obtient la précision
volume_precision = int(market_info['baseAssetPrecision'])
print("price_precision ", price_precision)
# Recherche dans les informations du marché
price_filter = None
lot_size_filter = None
min_notional_filter = None
for filter in market_info["filters"]:
    if filter["filterType"] == "PRICE_FILTER":
        price_filter = filter
    elif filter["filterType"] == "LOT_SIZE":
        lot_size_filter = filter
    elif filter["filterType"] == "NOTIONAL":
        min_notional_filter = filter

# Récupération des valeurs 
min_qty = float(lot_size_filter["minQty"])  # Quantité minimale pour un ordre (en token)
max_qty = float(lot_size_filter["maxQty"])  # Quantité maximale pour un ordre (en token)
step_size = float(lot_size_filter["stepSize"])  # Taille de pas pour ajuster les quantités d'ordre (en token)

min_price = float(price_filter["minPrice"])  # Prix minimum du token
max_price = float(price_filter["maxPrice"])  # Prix maximum du token

min_cost = float(min_notional_filter["minNotional"])  # Coût minimum pour un ordre (BUSD etc)
max_cost = float(min_notional_filter["maxNotional"])  # Coût maximum pour un ordre (BUSD etc)

# on récupère l'historique
historical_data = fetch_historical_data(symbol, timeframe, since)

historical_data = adjust_historical_data(historical_data, market_info)


# on converti pour les bougies Heiken Ashi
heiken_ashi_data = convert_to_heiken_ashi(historical_data)




host = config.host
user = config.user
database = config.database
password = config.password
connection = create_db_connection(host, user, database, password)

for entry in historical_data:
    insert_historical_data(connection, symbol, timeframe, entry)

# Fonction pour convertir les timestamp au bon format
def convert_timestamp_to_paris(timestamp):
    timestamp_seconds = timestamp / 1000
    utc_time = datetime.utcfromtimestamp(timestamp_seconds)
    paris_tz = pytz.timezone('Europe/Paris')
    paris_time = utc_time.replace(tzinfo=pytz.utc).astimezone(paris_tz)
    return paris_time.strftime('%Y-%m-%d %H:%M:%S')


connection.close()