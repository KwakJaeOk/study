import pandas as pd
import matplotlib.pyplot as plt
import requests
from io import StringIO

# 데이터 로드 및 전처리
def load_and_preprocess_data(filepath):
    price_data = pd.read_csv(filepath)
    price_data['날짜'] = pd.to_datetime(price_data['날짜'])
    price_data.set_index('날짜', inplace=True)
    price_data.fillna(method='ffill', inplace=True)
    return price_data[['SPXT Index', 'KBPMGO10 Index']]

# 불황 기간 데이터 가져오기
def get_recession_dates():
    fred_url = 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=USREC'
    response = requests.get(fred_url)
    data = response.content.decode('utf-8')
    recession_data = pd.read_csv(StringIO(data))
    recession_data['DATE'] = pd.to_datetime(recession_data['DATE'])
    recession_data.set_index('DATE', inplace=True)
    recession_periods = recession_data[recession_data['USREC'] == 1].index.to_series().groupby(
        (recession_data['USREC'] != recession_data['USREC'].shift()).cumsum()).agg(['first', 'last'])
    return recession_periods

# 변동성 계산 함수
def calculate_volatility(df, window):
    return df.rolling(window=window).std()

# 상관관계 계산 함수
def calculate_correlation(df, window):
    return df['SPXT Index'].rolling(window=window).corr(df['KBPMGO10 Index'])

# 포트폴리오 변동성 계산
def calculate_portfolio_volatility(returns_data, weights, window):
    portfolio_returns = returns_data.dot(weights)
    portfolio_volatility = calculate_volatility(portfolio_returns.to_frame('Portfolio'), window)
    return portfolio_volatility

# 포트폴리오 및 주가 그래프 그리기
def plot_portfolio_and_stock_prices(filtered_data, investment_normalized, portfolio_values_rebalanced, rebalance_dates,
                                    spx_weights, combined_spx_weights, combined_portfolio_30_70, recession_periods,
                                    rebalance_threshold):
    fig, axes = plt.subplots(3, 1, figsize=(14, 15))

    ax1 = axes[0]
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Portfolio Value')
    ax1.plot(combined_portfolio_30_70.index, combined_portfolio_30_70,
             label='Combined Portfolio (30% SPX, 70% KBPMGO10)', color='tab:blue')
    ax1.plot(filtered_data.index, portfolio_values_rebalanced,
             label='Rebalanced Portfolio (30% SPX, 70% KBPMGO10, ±5% Threshold)', color='tab:orange')
    ax1.plot(filtered_data.index, investment_normalized['SPXT Index'] * 1000, label='SPXT Investment',
             color='tab:green', linestyle='dashed')
    ax1.plot(filtered_data.index, investment_normalized['KBPMGO10 Index'] * 1000,
             label='KBPMGO10 Investment', color='tab:red', linestyle='dashed')

    for rebalance_date in rebalance_dates:
        ax1.axvline(x=rebalance_date, color='red', linestyle='--', linewidth=0.8)

    for start, end in recession_periods.values:
        ax1.axvspan(start, end, color='grey', alpha=0.5)

    ax1.legend(loc='upper left')
    ax1.grid(False)

    # Adding the bar chart below the first chart
    ax2 = axes[1]
    ax2.plot(filtered_data.index, spx_weights, label='SPX Weight (Rebalanced)', color='tab:green')
    ax2.axhline(0.3 + rebalance_threshold, color='tab:blue', linestyle='--', linewidth=0.8,
                label='SPX Target Weight + 5%')
    ax2.axhline(0.3 - rebalance_threshold, color='tab:orange', linestyle='--', linewidth=0.8,
                label='SPX Target Weight - 5%')

    ax3 = axes[2]
    ax3.plot(filtered_data.index, combined_spx_weights, label='SPX Weight (Combined)', color='tab:green')
    ax3.axhline(0.3 + rebalance_threshold, color='tab:blue', linestyle='--', linewidth=0.8,
                label='SPX Target Weight + 5%')
    ax3.axhline(0.3 - rebalance_threshold, color='tab:orange', linestyle='--', linewidth=0.8,
                label='SPX Target Weight - 5%')

    fig.tight_layout()
    plt.show()

# 변동성 및 상관관계 차트 그리기
def plot_volatility_and_correlation(vol_90, combined_vol_90, rebalanced_vol_90, corr_90):
    fig, axes = plt.subplots(2, 1, figsize=(14, 14))

    ax_vol = axes[0]
    ax_vol.plot(vol_90.index, vol_90['SPXT Index'], label='90-Day Volatility SPX', color='tab:green')
    ax_vol.axhline(vol_90['SPXT Index'].mean(), color='tab:blue', linestyle='--', linewidth=1, label='Mean 90-Day Volatility SPX')
    ax_vol.plot(vol_90.index, vol_90['KBPMGO10 Index'], label='90-Day Volatility KBPMGO10', color='tab:red')
    ax_vol.axhline(vol_90['KBPMGO10 Index'].mean(), color='tab:blue', linestyle='--', linewidth=1, label='Mean 90-Day Volatility KBPMGO10')
    ax_vol.plot(combined_vol_90.index, combined_vol_90['Combined Portfolio'], label='90-Day Volatility Combined Portfolio (30% SPX, 70% KBPMGO10)', color='tab:orange')
    ax_vol.axhline(combined_vol_90['Combined Portfolio'].mean(), color='tab:purple', linestyle='--', linewidth=1, label='Mean 90-Day Volatility Combined Portfolio')
    ax_vol.plot(rebalanced_vol_90.index, rebalanced_vol_90['Rebalanced Portfolio'], label='90-Day Volatility Rebalanced Portfolio (30% SPX, 70% KBPMGO10)', color='tab:pink')
    ax_vol.axhline(rebalanced_vol_90['Rebalanced Portfolio'].mean(), color='tab:brown', linestyle='--', linewidth=1, label='Mean 90-Day Volatility Rebalanced Portfolio')
    ax_vol.legend()
    ax_vol.set_title('Volatility')

    ax_corr = axes[1]
    ax_corr.plot(corr_90.index, corr_90, label='90-Day Correlation', color='tab:orange')
    ax_corr.axhline(corr_90.mean(), color='tab:blue', linestyle='--', linewidth=1, label='Mean 90-day corr')
    ax_corr.legend()
    ax_corr.set_title('Correlation')

    plt.tight_layout()
    plt.show()

# 메인 함수
def main():
    filepath = "PRICE_DATA_20240808_py.csv"
    price_data = load_and_preprocess_data(filepath)

    start_date = '2010-01-05'
    end_date = '2024-08-01'
    rebalance_threshold = 0.05

    filtered_data = price_data.copy()
    if start_date is not None:
        filtered_data = filtered_data.loc[start_date:]
    if end_date is not None:
        filtered_data = filtered_data.loc[:end_date]

    investment_normalized = filtered_data / filtered_data.iloc[0]
    return_data = filtered_data.pct_change().dropna()

    portfolio_value = 1000
    spx_shares = (portfolio_value * 0.3) / filtered_data.iloc[0]['SPXT Index']
    kbpmgo10_shares = (portfolio_value * 0.7) / filtered_data.iloc[0]['KBPMGO10 Index']

    portfolio_values_rebalanced = []
    rebalance_dates = []
    spx_weights = []
    for date, prices in filtered_data.iterrows():
        spx_value = spx_shares * prices['SPXT Index']
        kbpmgo10_value = kbpmgo10_shares * prices['KBPMGO10 Index']
        portfolio_value = spx_value + kbpmgo10_value

        spx_weight = spx_value / portfolio_value
        spx_weights.append(spx_weight)
        if abs(0.3 - spx_weight) > rebalance_threshold:
            rebalance_dates.append(date)
            spx_shares = (portfolio_value * 0.3) / prices['SPXT Index']
            kbpmgo10_shares = (portfolio_value * 0.7) / prices['KBPMGO10 Index']

        portfolio_values_rebalanced.append(portfolio_value)

    combined_portfolio_30_70 = investment_normalized * [300, 700]
    combined_portfolio_30_70 = combined_portfolio_30_70.sum(axis=1)

    combined_spx_weights = (investment_normalized['SPXT Index'] * 300) / combined_portfolio_30_70

    recession_periods = get_recession_dates()

    vol_90 = calculate_volatility(return_data, 90)
    corr_90 = calculate_correlation(return_data, 90)

    combined_portfolio_returns = combined_portfolio_30_70.pct_change().dropna()
    rebalanced_portfolio_returns = pd.Series(portfolio_values_rebalanced, index=filtered_data.index).pct_change().dropna()

    combined_vol_90 = calculate_volatility(combined_portfolio_returns.to_frame('Combined Portfolio'), 90)
    rebalanced_vol_90 = calculate_volatility(rebalanced_portfolio_returns.to_frame('Rebalanced Portfolio'), 90)

    plot_portfolio_and_stock_prices(filtered_data, investment_normalized, portfolio_values_rebalanced, rebalance_dates,
                                    spx_weights, combined_spx_weights, combined_portfolio_30_70, recession_periods,
                                    rebalance_threshold)
    plot_volatility_and_correlation(vol_90, combined_vol_90, rebalanced_vol_90, corr_90)

if __name__ == "__main__":
    main()