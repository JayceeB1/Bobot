import numpy as np
import pandas as pd
import os
from stable_baselines3 import PPO
from stable_baselines3 import A2C
import torch as th
import gym
from gym import spaces
from sqlalchemy import create_engine
from sklearn.preprocessing import MinMaxScaler
import config
from binance import *

# Connect to your MySQL database
db_connection_str = f'mysql+mysqlconnector://{config.user}:{config.password}@{config.host}/{config.database}'
connection = create_engine(db_connection_str)

# Config binance client
client = Client(config.API_KEY, config.API_SECRET)

# Récupérer le symbol
query = "SELECT symbol FROM historical_data_ml" 
symbol = pd.read_sql(query, con=connection)["symbol"].iloc[0]

symbol=symbol.replace("/", "")
print(symbol)

market_info = client.get_symbol_info(symbol)
print(market_info)

print(market_info)  # Obtient les informations du marché
price_precision = int(market_info['quoteAssetPrecision']) # Obtient la précision

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
min_cost = float(min_notional_filter["minNotional"])  # Coût minimum pour un ordre (BUSD etc)
max_cost = float(min_notional_filter["maxNotional"])  # Coût maximum pour un ordre (BUSD etc)

INITIAL_BALANCE = 100
trading_fees = 0.001

class TradingEnv(gym.Env):
    def __init__(self, data):
        super(TradingEnv, self).__init__()

        self.balance = INITIAL_BALANCE  # Solde initial du portefeuille
        self.position = 0  # Quantité actuelle d'actifs détenus
        self.last_trade_price = 0  # Prix de la dernière transaction

        self.data = data
        self.observation_space = spaces.Box(low=0, high=1, shape=(len(data.columns),))
        self.action_space = spaces.Discrete(3)

        self.current_step = 0
        self.max_steps = len(data)
        self.trade_count = 0  # Variable pour compter le nombre de trades

        self.trading_fees = trading_fees

    def reset(self):
        self.current_step = 0
        observation = self.data.iloc[self.current_step].values
        self.balance = INITIAL_BALANCE
        self.position = 0
        self.last_trade_price = 0
        self.trade_count = 0  # Réinitialiser le compteur de trades
        return observation

    def step(self, action):

        

        if self.current_step >= self.max_steps - 1:
            done = True
        else:
            done = False

        self.current_step += 1
        if self.current_step >= self.max_steps:
            return self.data.iloc[self.current_step-1].values, 0, done, {}

        observation = self.data.iloc[self.current_step].values

        current_price = observation[3]  # Ajoutez cette ligne pour obtenir le prix actuel de l'observation

        prev_portfolio_value = self.balance + self.position * self.last_trade_price

        if action == 0:  # Action "Ne rien faire"
            pass

        elif action == 1:  # Action "Acheter"
            if self.balance > 0:
                # Calculez la quantité à acheter en fonction de votre stratégie
                quantity_to_buy = self.balance / current_price
                # Vérifiez si la quantité à acheter est supérieure à la quantité minimale autorisée
                if quantity_to_buy >= min_qty:
                    # Vérifiez si la quantité à acheter est inférieure à la quantité maximale autorisée
                    if quantity_to_buy <= max_qty:
                        # Ajustez la quantité à acheter en fonction de la taille de pas
                        quantity_to_buy = round(quantity_to_buy / step_size) * step_size
                        # Calculez le coût total de l'achat
                        total_cost = quantity_to_buy * current_price
                        # Vérifiez si le coût total de l'achat est supérieur ou égal au coût minimum autorisé
                        if total_cost >= min_cost:
                            # Mettez à jour le portefeuille
                            self.position += quantity_to_buy
                            self.fees = quantity_to_buy * current_price * self.trading_fees 
                            self.balance -= quantity_to_buy * current_price
                            self.balance -= self.fees
                            self.last_trade_price = current_price
                            self.trade_count += 1  # Incrémenter le compteur de trades
                        else:
                            # Le coût total de l'achat est inférieur au coût minimum autorisé, ne rien faire
                            pass
                    else:
                        # La quantité à acheter dépasse la quantité maximale autorisée, ne rien faire
                        pass
                else:
                    # La quantité à acheter est inférieure à la quantité minimale autorisée, ne rien faire
                    pass

        elif action == 2:  # Action "Vendre"
            if self.position > 0:
                # Vérifiez si la quantité à vendre est supérieure à la quantité minimale autorisée
                if self.position >= min_qty:
                    # Vérifiez si la quantité à vendre est inférieure à la quantité maximale autorisée
                    if self.position <= max_qty:
                        # Ajustez la quantité à vendre en fonction de la taille de pas
                        quantity_to_sell = round(self.position / step_size) * step_size
                        # Mettez à jour le portefeuille
                        self.fees = quantity_to_sell * current_price * self.trading_fees 
                        self.balance += quantity_to_sell * current_price
                        self.balance -= self.fees
                        self.position -= quantity_to_sell
                        self.last_trade_price = current_price
                        self.trade_count += 1  # Incrémenter le compteur de trades
                    else:
                        # La quantité à vendre dépasse la quantité maximale autorisée, ne rien faire
                        pass
                else:
                    # La quantité à vendre est inférieure à la quantité minimale autorisée, ne rien faire
                    pass

        curr_portfolio_value = self.balance + self.position * current_price

        # Calcul de la récompense en utilisant calculate_reward() en fonction de votre stratégie
        reward = calculate_reward(action, prev_portfolio_value, curr_portfolio_value)

        return observation, reward, done, {}


def calculate_reward(action, prev_portfolio_value, curr_portfolio_value):
        
    if prev_portfolio_value == 0:
        return_rate = 0  # Valeur par défaut du taux de rendement si prev_portfolio_value est zéro
    else:
        return_rate = (curr_portfolio_value - prev_portfolio_value) / prev_portfolio_value

    # Définir les récompenses en fonction de l'action et du gain par trade
    if action == 0:  # Action "Ne rien faire"
        reward = 0
    elif action == 1:  # Action "Acheter"
        reward = return_rate - (trading_fees * 2)  # Rendement positif moins les frais d'achat et de vente
    elif action == 2:  # Action "Vendre"
        if return_rate < 0:
            reward = -return_rate - (trading_fees * 2)  # Récompense négative pour une vente avec perte moins les frais d'achat et de vente
        else:
            reward = return_rate * (curr_portfolio_value - prev_portfolio_value) / prev_portfolio_value - (trading_fees * 2)
            # Récompense basée sur le gain par trade moins les frais d'achat et de vente

    return reward


# Query your table to retrieve the data
query = "SELECT * FROM historical_data_ml" 
df = pd.read_sql(query, con=connection)

# Drop the 'symbol' column from the DataFrame
df = df.drop(['id', 'symbol', 'interval'], axis=1)
# Supprimer les 200 premiers enregistrements
df = df[200:]

# Prepare your data (e.g., normalize, scale, etc.)
numerical_columns = [
    'open', 'high', 'low', 'close', 'volume',
    'ichimoku_a', 'ichimoku_b', 'ichimoku_base_line', 'ichimoku_conversion_line',
    'rsi', 'ema200', 'stochrsi', 'macd', 'sma5', 'sma10', 'sma21', 'adx', 'wma5', 'k', 'd', 'j',
    'boll_upper_band', 'sma', 'boll_lower_band'
]
scaler = MinMaxScaler()
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

# Create the trading environment
env = TradingEnv(df)

# chemin vers le modèle
save_file = "C:/wamp64/www/tradingbot/modele/maxibot.zip"

learning_rate = 0.0003
n_steps = 2048
ent_coef = 0.01
clip_range = 0.2
gae_lambda = 0.95

timesteps = 20000

# # Créer un dictionnaire pour spécifier l'architecture du réseau de neurones
# policy_kwargs = dict(
#     activation_fn=th.nn.ReLU,
#     net_arch=dict(pi=[128, 64, 32])
# )

# Vérifier si le modèle existe
if os.path.exists(save_file):
    # Charger le modèle existant
    model = PPO.load(save_file, print_system_info=True)
    print("Modèle chargé!")
    model.set_env(env)
else:
    # Créer un nouveau modèle  policy_kwargs=policy_kwargs
    model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=learning_rate,
    n_steps=n_steps,
    ent_coef=ent_coef,
    clip_range=clip_range,
    gae_lambda=gae_lambda
    )
    print("Modèle créé!")

model.learn(total_timesteps=timesteps)

# Use the trained agent for trading decisions
obs = env.reset()
for _ in range(len(df)):
    action, _ = model.predict(obs)
    obs, reward, done, _ = env.step(action)
    if done:
        break

# Utilisez le modèle formé pour prendre des décisions de trading
obs = env.reset()

portfolio_values = []
asset_values = []
returns = []
balance = INITIAL_BALANCE

for _ in range(len(df)):
    action, _ = model.predict(obs)
    obs, reward, done, _ = env.step(action)

    # Mettez à jour les variables du portefeuille
    current_price = obs[3]
    balance = balance + env.position * (current_price - env.last_trade_price)
    env.last_trade_price = current_price

    # Enregistrez les métriques pertinentes
    portfolio_value = env.balance + env.position * env.last_trade_price
    asset_value = env.position * env.last_trade_price

    portfolio_values.append(portfolio_value)
    asset_values.append(asset_value)

    if len(returns) == 0:
        cumulative_return = 0
    else:
        cumulative_return = returns[-1] + reward

    returns.append(cumulative_return)

    if done:
        break

model.save(save_file)

# Retrieve test data and evaluate model performance
test_query = "SELECT * FROM historical_data_ml" 
test_df = pd.read_sql(test_query, con=connection)
test_df = test_df.drop(['id', 'symbol', 'interval'], axis=1)
# Supprimer les 200 premiers enregistrements
test_df = test_df[200:]

test_df[numerical_columns] = scaler.transform(test_df[numerical_columns])

test_portfolio_values = []
test_asset_values = []
test_returns = []
test_max_trade_gain = 0.0
test_metrics = {}
test_balance = INITIAL_BALANCE

# Reset the environment
test_obs = env.reset()

for _ in range(len(test_df)):
    test_action, _ = model.predict(test_obs)
    test_obs, test_reward, test_done, _ = env.step(test_action)

    # Mettez à jour les variables du portefeuille pour les données de test
    test_current_price = test_obs[3]
    test_balance = test_balance + env.position * (test_current_price - env.last_trade_price)
    env.last_trade_price = test_current_price

    # Enregistrez les métriques pertinentes pour les données de test
    test_portfolio_value = env.balance + env.position * env.last_trade_price
    test_asset_value = env.position * env.last_trade_price

    test_portfolio_values.append(test_portfolio_value)
    test_asset_values.append(test_asset_value)

    if len(test_returns) == 0:
        test_cumulative_return = 0
    else:
        test_cumulative_return = test_returns[-1] + test_reward

    test_returns.append(test_cumulative_return)

    if test_reward > 0 and test_reward > test_max_trade_gain:
        test_max_trade_gain = test_reward

    if test_done:
        break

# Enregistrez les métriques finales
metrics = {}
metrics['Final Balance'] = balance
metrics['Number of Trades'] = env.trade_count
metrics['Success Rate (%)'] = (env.trade_count / len(df)) * 100
metrics['Maximum Drawdown'] = np.max(returns) - np.min(returns)
metrics['Max Trade Gain'] = test_max_trade_gain

test_metrics = {}
test_metrics['Final Balance'] = test_balance
test_metrics['Number of Trades'] = env.trade_count
test_metrics['Success Rate (%)'] = (env.trade_count / len(test_df)) * 100
test_metrics['Maximum Drawdown'] = np.max(test_returns) - np.min(test_returns)
test_metrics['Max Trade Gain'] = test_max_trade_gain

print("Metrics (Train):", metrics)
print("Metrics (Test):", test_metrics)

# Close the database connection
connection.dispose()
