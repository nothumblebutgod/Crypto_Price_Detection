import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import datetime as dt

url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
params = {"vs_currency": "usd", "days": "90"}
resp = requests.get(url, params=params)
resp.raise_for_status()
data = resp.json()

prices = data['prices']
market_caps = data['market_caps']
volumes = data['total_volumes']

df_prices = pd.DataFrame(prices, columns=['timestamp', 'price'])
df_mcap = pd.DataFrame(market_caps, columns=['timestamp', 'market_cap'])
df_vol = pd.DataFrame(volumes, columns=['timestamp', 'volume'])

for d in (df_prices, df_mcap, df_vol):
    d['timestamp'] = pd.to_datetime(d['timestamp'], unit='ms')
    d.set_index('timestamp', inplace=True)

df = df_prices.join(df_mcap, how='outer').join(df_vol, how='outer')
df = df.sort_index().ffill().bfill()

print("Data berhasil diambil (tail):")
print(df[['price', 'market_cap', 'volume']].tail())

url2 = "https://api.coingecko.com/api/v3/coins/markets"
params2 = {"vs_currency": "usd", "ids": "bitcoin"}
resp2 = requests.get(url2, params=params2)
resp2.raise_for_status()
summary = resp2.json()[0]

current_price = summary.get('current_price')
market_cap_now = summary.get('market_cap')
market_cap_change_24h = summary.get('market_cap_change_percentage_24h')
volume_24h = summary.get('total_volume')
fdv = summary.get('fully_diluted_valuation')
circulating = summary.get('circulating_supply')
max_supply = summary.get('max_supply')

print("\nRingkasan:")
print(f"Current Price: ${current_price:,.2f}")
print(f"Market Cap: ${market_cap_now:,.0f}")
print(f"24h Volume: ${volume_24h:,.0f}")
print(f"24h Change: {market_cap_change_24h:.2f}%")
print(f"Circulating: {circulating:,.0f} BTC, Max: {max_supply}")

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df[['price']])

X_train, y_train = [], []
for i in range(60, len(scaled_data)):
    X_train.append(scaled_data[i-60:i, 0])
    y_train.append(scaled_data[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

model = Sequential([
    LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    Dropout(0.2),
    LSTM(units=50, return_sequences=False),
    Dropout(0.2),
    Dense(units=25),
    Dense(units=1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
print("\nTraining model...")
model.fit(X_train, y_train, epochs=25, batch_size=32, verbose=1)

last_60_days = scaled_data[-60:].copy()
predictions = []

for _ in range(7):
    x_input = np.reshape(last_60_days, (1, 60, 1))
    pred_price = model.predict(x_input, verbose=0)
    predictions.append(pred_price[0, 0])
    last_60_days = np.append(last_60_days[1:], pred_price)
    last_60_days = np.reshape(last_60_days, (60, 1))

predicted_prices = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
last_date = df.index[-1]
future_dates = [last_date + dt.timedelta(days=i+1) for i in range(7)]

plt.figure(figsize=(16, 8))

plt.plot(df.index, df['price'], label='Data Aktual (90 hari)', color='blue', linewidth=2)

plt.plot(future_dates, predicted_prices, 'o--', color='orange', 
         label='Prediksi (7 hari)', linewidth=2, markersize=8)

for date, price in zip(future_dates, predicted_prices.flatten()):
    plt.text(date, price, f"${price:,.0f}", 
             fontsize=9, ha='center', va='bottom', 
             rotation=45, color='darkorange', fontweight='bold')

info_text = (
    f"💰 Market Cap: ${market_cap_now/1e12:.2f}T\n"
    f"📊 Volume (24h): ${volume_24h/1e9:.2f}B\n"
    f"🔁 Change (24h): {market_cap_change_24h:+.2f}%\n"
    f"🔢 Circulating: {circulating:,.0f} BTC\n"
    f"🔢 Max Supply: {max_supply if max_supply else 'N/A'}"
)

plt.gcf().text(0.73, 0.72, info_text, fontsize=10, 
               bbox=dict(facecolor='lightyellow', alpha=0.9, 
                        boxstyle='round,pad=1', edgecolor='gray'))

# Styling
plt.title('Prediksi Harga Bitcoin Menggunakan LSTM', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Tanggal', fontsize=12)
plt.ylabel('Harga (USD)', fontsize=12)
plt.legend(loc='upper left', fontsize=11)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

# Format y-axis untuk mata uang
ax = plt.gca()
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

plt.show()

print("\n✅ Prediksi selesai!")
print(f"Harga terakhir: ${df['price'].iloc[-1]:,.2f}")
print(f"Prediksi 7 hari ke depan:")
for i, (date, price) in enumerate(zip(future_dates, predicted_prices.flatten()), 1):
    print(f"  Hari {i} ({date.strftime('%Y-%m-%d')}): ${price:,.2f}")