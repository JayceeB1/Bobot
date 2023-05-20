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
    # Convertir les données en DataFrame
    df = pd.DataFrame(data, columns=['timestamp', 'open_price', 'high_price', 'low_price', 'close_price', 'volume'])

    # Calculer les composants de l'indicateur Ichimoku
    ichimoku = ta.trend.IchimokuIndicator(high=df['high_price'], low=df['low_price'], window1=9, window2=26, window3=52)
    df['ichimoku_a'] = ichimoku.ichimoku_a()  # Leading Span A
    df['ichimoku_b'] = ichimoku.ichimoku_b()  # Leading Span B
    df['ichimoku_base_line'] = ichimoku.ichimoku_base_line()  # Base Line (Kijun-Sen)
    df['ichimoku_conversion_line'] = ichimoku.ichimoku_conversion_line()  # Conversion Line (Tenkan-Sen)

    # Décaler les lignes Span A et Span B de 26 périodes en avant
    df['ichimoku_a'] = df['ichimoku_a'].shift(-26)
    df['ichimoku_b'] = df['ichimoku_b'].shift(-26)

    # Créer des signaux d'achat et de vente basés sur l'indicateur Ichimoku
    # Signal d'achat : le prix de clôture traverse le nuage (Span A et Span B) à la hausse
    # Signal de vente : le prix de clôture traverse le nuage à la baisse
    ichimoku_buy_signals = (df['close_price'].shift(-1) < df[['ichimoku_a', 'ichimoku_b']].min(axis=1)) & (df['close_price'] > df[['ichimoku_a', 'ichimoku_b']].max(axis=1))
    ichimoku_sell_signals = (df['close_price'].shift(-1) > df[['ichimoku_a', 'ichimoku_b']].max(axis=1)) & (df['close_price'] < df[['ichimoku_a', 'ichimoku_b']].min(axis=1))

    return ichimoku_buy_signals, ichimoku_sell_signals

def calculate_rsi(data, period):
    close_prices = np.array([entry[4] for entry in data])
    rsi = talib.RSI(close_prices, timeperiod=period)
    return rsi[-1]  # Retourne seulement la dernière valeur du RSI

# Fonction pour détecter les signaux d'achat/vente
def detect_buy_sell_signals(data, per=50, mult=3.0):
    close_prices = np.array([entry[4] for entry in data])
    # Calculer le smooth range
    abs_diff = np.abs(np.diff(close_prices))
    # abs_diff_series = pd.Series(abs_diff)
    sma = pd.DataFrame(abs_diff).rolling(window=per).mean().values.flatten()
    smooth_range = talib.EMA(sma, timeperiod=per) * mult

    # Appliquer le range filter
    range_filter = [close_prices[0]]
    for i in range(1, len(close_prices) - 1):
        if close_prices[i] > range_filter[-1]:
            range_filter.append(max(close_prices[i] - smooth_range[i], range_filter[-1]))
        else:
            range_filter.append(min(close_prices[i] + smooth_range[i], range_filter[-1]))

    # Ajouter la dernière valeur de smooth_range
    range_filter.append(close_prices[-1] + smooth_range[-1])

    range_filter = np.array(range_filter)

    # Calculer les upward et downward
    upward = np.where(range_filter[1:] > range_filter[:-1], 1, 0)
    downward = np.where(range_filter[1:] < range_filter[:-1], 1, 0)

    # Trouver les conditions d'achat et de vente
    long_condition = (close_prices[:-1] > range_filter[:-1]) & (upward == 1)
    short_condition = (close_prices[:-1] < range_filter[:-1]) & (downward == 1)

    return long_condition, short_condition

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
        TRUNCATE TABLE historical_data
    """
    execute_query(connection, query)

# Fonction pour insérer les données historiques dans la table historical_data
def insert_historical_data(connection, symbol, interval, data):
    query = f"""
        INSERT INTO historical_data (symbol, `interval`, timestamp, open, high, low, close, volume)
        VALUES ('{symbol}', '{interval}', {data[0]}, {data[1]}, {data[2]}, {data[3]}, {data[4]}, {data[5]})
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


# on lance la détection
buy_signals, sell_signals = detect_buy_sell_signals(heiken_ashi_data, per, mult)
rsi = calculate_rsi(historical_data, 7)
ichimoku_buy_signal,ichimoku_sell_signal = ichimoku_signals(historical_data)

host = config.host
user = config.user
database = config.database
password = config.password
connection = create_db_connection(host, user, database, password)

# purge de l'historique déjà dans la BDD
truncate_historical_data()

for entry in historical_data:
    insert_historical_data(connection, symbol, timeframe, entry)

# Fonction pour insérer les performances dans la table performance
def insert_performance(connection, symbol, timeframe, strategy, start_timestamp, end_timestamp, initial_balance, final_balance, pnl, success_rate, buy_count, sell_count, max_drawdown, avg_trades_per_day):
    query = f"""
        INSERT INTO performance (strategy, symbol, timeframe, start_timestamp, end_timestamp, initial_balance, final_balance, pnl, success_rate, buy_count, sell_count, max_drawdown, avg_trades_per_day)
        VALUES ('{symbol}', '{timeframe}', '{strategy}', {start_timestamp}, {end_timestamp}, {initial_balance}, {final_balance}, {pnl}, {success_rate}, {buy_count}, {sell_count}, {max_drawdown}, {avg_trades_per_day})
    """
    execute_query(connection, query)

# Fonction pour convertir les timestamp au bon format
def convert_timestamp_to_paris(timestamp):
    timestamp_seconds = timestamp / 1000
    utc_time = datetime.utcfromtimestamp(timestamp_seconds)
    paris_tz = pytz.timezone('Europe/Paris')
    paris_time = utc_time.replace(tzinfo=pytz.utc).astimezone(paris_tz)
    return paris_time.strftime('%Y-%m-%d %H:%M:%S')

# SIMULATION TRADING

initial_balance = 100
risk_fraction = 0.9  # Ajout d'une variable pour déterminer le pourcentage du capital à investir dans chaque trade
trading_fee = 0.001  # Frais de trading de 0.1%
# Paramètres de pyramiding
pyramid_levels = 1  # Nombre d'achats possibles (pyramid levels)
pyramid_ratio = 0.6  # Pourcentage de la balance restante à investir dans chaque niveau pyramidé
pyramid_balance_min = 20  # Balance minimum restante pour effectuer un achat supplémentaire

savings = 0 # on va essayer de faire l'écureuil

position = None
buy_count = 0
sell_count = 0

balance = initial_balance
max_balance = initial_balance
temp_drawdown = 0
max_drawdown = 0

successful_trades = 0
total_trades = 0

evolution_balance = []

buy_timestamps, buy_prices = [], []
sell_timestamps, sell_prices = [], []

# Variables de pyramiding
pyramid_position = None
pyramid_count = 0
pyramid_balance = balance

# Variable pour enregistrement des ventes
sold_before_end = False

assert len(heiken_ashi_data) == len(historical_data)

# Boucle de simulation
for i, (entry, hist_entry) in enumerate(zip(heiken_ashi_data[:-1], historical_data[:-1])):
    timestamp, open_price, high, low, close_price, volume = entry
    hist_timestamp, hist_open_price, hist_high, hist_low, hist_close_price, hist_volume = hist_entry

    is_buy = buy_signals[i]
    is_sell = sell_signals[i]

    is_ichimoku_buy = ichimoku_buy_signal[i]
    is_ichimoku_sell = ichimoku_sell_signal[i]
   

    # Calculer le RSI pour les données historiques jusqu'à l'index actuel
    rsi_values_buy = calculate_rsi(historical_data[:i+1], 7)
    rsi_values_sell = calculate_rsi(historical_data[:i+1], 22)
    # Vérifier les signaux de surachat et de survente
    is_buy_rsi = rsi_values_buy < 9.5
    is_sell_rsi = rsi_values_sell > 65

    activate_range_buy = True
    activate_range_sell = True
    activate_rsi_buy = False
    activate_rsi_sell = False
    activate_ichimoku_buy = False
    activate_ichimoku_sell = False

    # Si on détecte un bottom, on achète
    if (is_buy and activate_range_buy) or (is_buy_rsi and activate_rsi_buy) or (is_ichimoku_buy and activate_ichimoku_buy):
        # Si c'est le premier achat
        if position is None:
            amount_to_invest = format_float(balance * risk_fraction, price_precision)
            if amount_to_invest > max_cost:
                amount_to_invest = max_cost
            if amount_to_invest > min_cost:
                position = format_float(amount_to_invest / format_float(close_price,price_precision), price_precision)
                position = format_float(round_step(position, step_size), price_precision)
                fee = format_float(amount_to_invest * trading_fee, price_precision)
                balance -= format_float(amount_to_invest + fee, price_precision)
                balance_achat = format_float(amount_to_invest - fee, price_precision)
                buy_count += 1
                total_trades += 1
                buy_timestamps.append(timestamp)
                buy_prices.append(close_price)
                sold_before_end = False
                trade_time = convert_timestamp_to_paris(timestamp)
                print(f"Achat! Quand: {trade_time},  Prix : {format_float(close_price,price_precision)}, Montant investi : {amount_to_invest}, Balance : {balance}, Max Balance: {max_balance}, Frais: {fee}")

                # Si on doit pyramider
                if pyramid_count < pyramid_levels:
                    pyramid_position = position
                    pyramid_balance = balance
                    pyramid_count += 1

        # Si on est en train de pyramider et qu'il reste suffisamment de balance
        elif pyramid_count > 0 and pyramid_balance - pyramid_position >= pyramid_balance_min and pyramid_count < pyramid_levels:
            amount_to_invest = format_float(pyramid_balance * pyramid_ratio, price_precision)
            if amount_to_invest > max_cost:
                amount_to_invest = max_cost
            if amount_to_invest > min_cost:
                position = position + format_float(amount_to_invest / format_float(close_price,price_precision), price_precision)
                position = format_float(round_step(position, step_size), price_precision)
                fee = format_float(amount_to_invest * trading_fee, price_precision)
                balance -= format_float(amount_to_invest + fee, price_precision)
                balance_achat -= format_float(amount_to_invest + fee, price_precision)
                pyramid_balance -= format_float(amount_to_invest + fee, price_precision)
                buy_count += 1
                total_trades += 1
                buy_timestamps.append(timestamp)
                buy_prices.append(close_price)
                pyramid_count += 1
                trade_time = convert_timestamp_to_paris(timestamp)
                print(f"Achat Pyramidé! Quand: {trade_time}, Prix : {close_price}, Montant investi : {amount_to_invest}, Balance : {balance}, Max Balance: {max_balance}, Frais: {fee}")

    # Si on détecte un top, on vend
    elif (is_sell and position is not None and activate_range_sell) or (is_sell_rsi and position is not None and activate_rsi_sell) or (is_ichimoku_sell and position is not None and activate_ichimoku_sell):
        position_to_sell = format_float(round_step(position, step_size), price_precision)
        if position_to_sell > max_qty:
            position_to_sell = max_qty
        if position_to_sell > min_qty:
            amount_to_sell = format_float(position_to_sell * close_price, price_precision)
            fee = format_float(amount_to_sell * trading_fee, price_precision)
            balance += format_float(amount_to_sell - fee, price_precision)
            evolution_balance.append(balance)
            balance_vente = format_float(amount_to_sell - fee, price_precision)
            test_gain = balance_vente - balance_achat
            profit_ou_pas = is_positive(test_gain)
            if profit_ou_pas :
                successful_trades += 1
            position = None
            sell_count += 1
            total_trades += 1            
            sell_timestamps.append(timestamp)
            sell_prices.append(close_price)
            sold_before_end = True
            pyramid_position = None
            pyramid_count = 0
            pyramid_balance = balance
            trade_time = convert_timestamp_to_paris(timestamp)
            print(f"Vente! Quand: {trade_time}, Prix : {format_float(close_price,price_precision)}, Montant vendu : {amount_to_sell}, Balance : {balance}, Max Balance: {max_balance}, Max Drawdown: {max_drawdown}, Frais: {fee}")   

    if position is None:
        old_max_balance = max_balance
        max_balance = max(max_balance, balance)
        if old_max_balance < max_balance:
                savings += math.floor(balance * 0.001)
                balance -= savings
        temp_drawdown = max_balance - balance
        max_drawdown = max(max_drawdown, temp_drawdown)

if not sold_before_end:
    if position is not None:
        buy_count -= pyramid_count
        total_trades -= pyramid_count
        balance += position * close_price
        position = None
        buy_timestamps = buy_timestamps[:-pyramid_count]
        buy_prices = buy_prices[:-pyramid_count]


# récupération des valeurs pour la BDD
if total_trades != 0:
    final_balance = balance
    pnl = final_balance - initial_balance
    success_rate = (successful_trades / total_trades) * 100
    start_timestamp = historical_data[0][0]
    end_timestamp = historical_data[-1][0]

    trading_days = (end_timestamp - start_timestamp) / (1000 * 60 * 60 * 24) # Convertir la durée en jours
    avg_trades_per_day = total_trades / trading_days

    strategy = f"H/L/HA: {symbol} {timeframe} PER {per} MULT {mult}"


    insert_performance(connection, strategy, symbol, timeframe, historical_data[0][0], historical_data[-1][0], initial_balance, final_balance, pnl, success_rate, buy_count, sell_count, max_drawdown, avg_trades_per_day)
    
if total_trades == 0: 
    print("Pas de trades!")


print(f"Bravo t'as mis de côté {savings} BUSD, t'es un winner mec.")


# Fonction pour afficher le graphique des données de balance
def plot_evolution_balance(evolution_balance, sell_timestamps):
    # Convertir les timestamps en dates
    sell_dates = [datetime.fromtimestamp(ts/1000.0) for ts in sell_timestamps]

    # Créer un objet graphique
    fig = go.Figure()

    # Ajouter une trace pour l'évolution de la balance
    fig.add_trace(go.Scatter(
        x=sell_dates,
        y=evolution_balance,
        mode='lines',
        name='Balance',
        line=dict(color='lightgreen')
    ))

    # Mettre à jour le layout du graphique
    fig.update_layout(
        title='Evolution de la balance',
        xaxis_title='Temps',
        yaxis_title='Balance',
        plot_bgcolor='black',
        paper_bgcolor='black',
        font=dict(
            color="white"
        ),
        template='plotly_dark'
    )

    # Afficher le graphique
    pio.show(fig, port=8080)


# Fonction pour afficher le graphique des données de trade
def plot_signals(data, long_signals, short_signals, per, mult, buy_timestamps, sell_timestamps):
    # Convertir les données en DataFrame
    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')  # Convertir le timestamp en datetime
    df.set_index('timestamp', inplace=True)  # Utiliser le timestamp comme index

    # Créer le graphique avec Plotly
    fig = go.Figure()

    # Ajouter les bougies
    fig.add_trace(go.Candlestick(x=df.index,
                                  open=df['open'],
                                  high=df['high'],
                                  low=df['low'],
                                  close=df['close'],
                                  name='Bougies'))

    # Ajouter les points d'achat
    buy_timestamps = [pd.to_datetime(ts, unit='ms') for ts in buy_timestamps]
    fig.add_trace(go.Scatter(x=buy_timestamps, y=[df.loc[ts, 'close'] for ts in buy_timestamps], mode='markers', marker=dict(size=10, color='lime', symbol='triangle-up'), name='Achats'))

    # Ajouter les points de vente
    sell_timestamps = [pd.to_datetime(ts, unit='ms') for ts in sell_timestamps]
    fig.add_trace(go.Scatter(x=sell_timestamps, y=[df.loc[ts, 'close'] for ts in sell_timestamps], mode='markers', marker=dict(size=10, color='orange', symbol='triangle-down'), name='Ventes'))

    # Configurer les titres et les légendes
    fig.update_layout(title='Prix de clôture avec signaux d\'achat et de vente',
                      xaxis_title='Temps',
                      yaxis_title='Prix',
                      # Définir le fond noir pour le graphique
                       plot_bgcolor='#1f1f21',
                      paper_bgcolor='#18181a',
                      font=dict(color='white'))

    # Afficher le graphique avec un port différent
    pio.show(fig, port=8080)


plot_evolution_balance(evolution_balance, sell_timestamps)

plot_signals(historical_data, buy_signals, sell_signals, per, mult, buy_timestamps, sell_timestamps)


connection.close()