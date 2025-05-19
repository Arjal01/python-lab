import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
df = pd.read_csv('stocks.csv')
df['Date'] = pd.to_datetime(df['Date'])

# 1. Data Loading and Basic Info
print("1. Dataset Info:")
print(df.info())
print("\nFirst 5 rows:")
print(df.head())

# 2. Highest and Lowest Closing Prices
highest_close = df.loc[df['Close'].idxmax()]
lowest_close = df.loc[df['Close'].idxmin()]
print("\n2. Highest Closing Price:")
print(highest_close[['Ticker', 'Date', 'Close']])
print("\nLowest Closing Price:")
print(lowest_close[['Ticker', 'Date', 'Close']])

# 3. Top 5 Performing Stocks (Percentage Gain)
df = df.sort_values('Date')
percentage_gains = {}
for ticker in df['Ticker'].unique():
    ticker_data = df[df['Ticker'] == ticker]
    first_close = ticker_data['Close'].iloc[0]
    last_close = ticker_data['Close'].iloc[-1]
    gain = ((last_close - first_close) / first_close) * 100
    percentage_gains[ticker] = gain
gains_df = pd.DataFrame.from_dict(percentage_gains, orient='index', columns=['Percentage Gain'])
gains_df = gains_df.sort_values(by='Percentage Gain', ascending=False)
print("\n3. Top Performing Stocks (Percentage Gain):")
print(gains_df.head())

# 4. Most Volatile Stocks (Standard Deviation of Close Price)
volatility = df.groupby('Ticker')['Close'].std().sort_values(ascending=False)
print("\n4. Most Volatile Stocks (Standard Deviation of Close Price):")
print(volatility)

# 5. Daily Gainers and Losers
df['Daily Change %'] = ((df['Close'] - df['Open']) / df['Open']) * 100
daily_gainers = df.loc[df.groupby('Date')['Daily Change %'].idxmax()][['Date', 'Ticker', 'Daily Change %']]
daily_losers = df.loc[df.groupby('Date')['Daily Change %'].idxmin()][['Date', 'Ticker', 'Daily Change %']]
print("\n5. Daily Gainers:")
print(daily_gainers)
print("\nDaily Losers:")
print(daily_losers)

# 6. Average Opening and Closing Price per Company
avg_prices = df.groupby('Ticker')[['Open', 'Close']].mean()
print("\n6. Average Opening and Closing Prices per Company:")
print(avg_prices)

# 7. Companies with Highest Total Trading Volume
total_volume = df.groupby('Ticker')['Volume'].sum().sort_values(ascending=False)
print("\n7. Companies with Highest Total Trading Volume:")
print(total_volume)

# 8. Volume Spikes Detection (Outliers)
spikes = []
for ticker in df['Ticker'].unique():
    ticker_data = df[df['Ticker'] == ticker]
    Q1 = ticker_data['Volume'].quantile(0.25)
    Q3 = ticker_data['Volume'].quantile(0.75)
    IQR = Q3 - Q1
    threshold = Q3 + 1.5 * IQR
    spike_data = ticker_data[ticker_data['Volume'] > threshold][['Ticker', 'Date', 'Volume']]
    spikes.append(spike_data)
spikes_df = pd.concat(spikes)
print("\n8. Volume Spikes (Outliers):")
print(spikes_df)

# 9. Monthly Stock Price Trends (Line Plot)
plt.figure(figsize=(10, 6))
for ticker in df['Ticker'].unique():
    ticker_data = df[df['Ticker'] == ticker]
    plt.plot(ticker_data['Date'], ticker_data['Close'], label=ticker)
plt.title('Monthly Closing Price Trends')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.legend()
plt.grid(True)
plt.show()

# 10. Moving Average (SMA) Trend Analysis
plt.figure(figsize=(10, 6))
for ticker in df['Ticker'].unique():
    ticker_data = df[df['Ticker'] == ticker].sort_values('Date')
    ticker_data['10-day SMA'] = ticker_data['Close'].rolling(window=10).mean()
    plt.plot(ticker_data['Date'], ticker_data['10-day SMA'], label=f'{ticker} 10-day SMA')
plt.title('10-day Simple Moving Average (SMA) Trend')
plt.xlabel('Date')
plt.ylabel('SMA')
plt.legend()
plt.grid(True)
plt.show()

# 11. Price Change Before and After Key Date
key_date = pd.Timestamp('2023-03-01')
before = df[df['Date'] < key_date].groupby('Ticker')['Close'].mean()
after = df[df['Date'] >= key_date].groupby('Ticker')['Close'].mean()
change = ((after - before) / before * 100).sort_values(ascending=False)
print("\n11. Price Change Before and After 2023-03-01 (%):")
print(change)

# 12. Correlation Between Stocks (Close Prices)
pivot_df = df.pivot(index='Date', columns='Ticker', values='Close')
correlation = pivot_df.corr()
print("\n12. Correlation Between Stocks (Close Prices):")
print(correlation)

# 13. Stock Return Distribution (Histogram)
df = df.sort_values(['Ticker', 'Date'])
df['Daily Return'] = df.groupby('Ticker')['Close'].pct_change() * 100
plt.figure(figsize=(10, 6))
for ticker in df['Ticker'].unique():
    ticker_data = df[df['Ticker'] == ticker]
    plt.hist(ticker_data['Daily Return'].dropna(), bins=20, alpha=0.5, label=ticker)
plt.title('Distribution of Daily Returns')
plt.xlabel('Daily Return (%)')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
plt.show()

# 14. Heatmap of Company Performance (Total Gain %)
gains_matrix = gains_df.values  # Get percentage gains as array
tickers = gains_df.index
plt.figure(figsize=(8, 4))
cax = plt.imshow(gains_matrix, cmap='coolwarm', aspect='auto')
plt.colorbar(cax, label='Percentage Gain')
plt.xticks(np.arange(len(tickers)), tickers)
plt.yticks([])  # Single row, no y-axis labels
plt.title('Total Stock Gain % Heatmap')
for i in range(len(tickers)):
    plt.text(i, 0, f'{gains_matrix[i, 0]:.2f}', ha='center', va='center', color='black')
plt.show()