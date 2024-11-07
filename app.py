import streamlit as st
import numpy as np
import pandas as pd
import vectorbt as vbt
import pandas_ta as ta

st.title("DelorianB Backtest")
st.sidebar.header("Backtest Settings")

if "run_main_backtest" not in st.session_state:
    st.session_state.run_main_backtest = False

if "run_manual_backtest" not in st.session_state:
    st.session_state.run_manual_backtest = False

assets = st.sidebar.text_input("Enter up to 50 comma-separated symbols:", "BTC-USD,ETH-USD,LTC-USD,SOL-USD,BNB-USD").split(',')
assets = [asset.strip() for asset in assets[:50]]  
interval = "15m"  
period = "1mo"  

tp_ratios = np.arange(0.01, 0.13, 0.01)  
sl_ratios = np.arange(0.01, 0.06, 0.01) 

def calculate_wavetrend(hlc3, channel_length, avg_length, ma_length):
    esa = ta.ema(hlc3, channel_length)
    de = ta.ema(abs(hlc3 - esa), channel_length)
    ci = (hlc3 - esa) / (0.015 * de)
    wt1 = ta.ema(ci, avg_length)
    wt2 = ta.sma(wt1, ma_length)
    return wt1, wt2

def detect_divergences(price, indicator, overbought_level, oversold_level):
    bullish_divergence = (price < price.shift(1)) & (indicator > indicator.shift(1)) & (indicator <= oversold_level)
    bearish_divergence = (price > price.shift(1)) & (indicator < indicator.shift(1)) & (indicator >= overbought_level)
    return bullish_divergence, bearish_divergence

def generate_signals(asset_close, asset_high, asset_low):
    hlc3 = (asset_high + asset_low + asset_close) / 3
    wt1, wt2 = calculate_wavetrend(hlc3, 9, 12, 3)
    rsi = ta.rsi(asset_close, 14)
    wt_bullish_div, wt_bearish_div = detect_divergences(asset_close, wt2, 53, -53)
    rsi_bullish_div, rsi_bearish_div = detect_divergences(asset_close, rsi, 60, 30)
    long_entries = (wt_bullish_div | rsi_bullish_div).vbt.signals.fshift()
    short_entries = (wt_bearish_div | rsi_bearish_div).vbt.signals.fshift()
    long_exits = short_entries
    short_exits = long_entries
    return long_entries, long_exits, short_entries, short_exits

if st.button("Run Backtest for Top 10 Coins"):
    st.session_state.run_main_backtest = True

if st.session_state.run_main_backtest:
    st.write("Running Backtest for Top 10 Coins, Sorting by Total Return...")
    
    performance_results = []
    data = vbt.YFData.download(assets, interval=interval, period=period, tz_localize='UTC').concat()

    for asset in assets:
        asset_close = data['Close'][asset]
        asset_high = data['High'][asset]
        asset_low = data['Low'][asset]
        
        long_entries, long_exits, short_entries, short_exits = generate_signals(asset_close, asset_high, asset_low)
        
        for tp_ratio in tp_ratios:
            for sl_ratio in sl_ratios:
                pf = vbt.Portfolio.from_signals(
                    close=asset_close,
                    entries=long_entries,
                    exits=long_exits,
                    short_entries=short_entries,
                    short_exits=short_exits,
                    sl_stop=sl_ratio,
                    tp_stop=tp_ratio,
                    init_cash=1000,
                    size=0.1,
                    freq=interval
                )
                performance_results.append({
                    "asset": asset,
                    "tp_ratio": tp_ratio,
                    "sl_ratio": sl_ratio,
                    "total_return": pf.total_return(),
                    "sharpe_ratio": pf.sharpe_ratio(),
                    "max_drawdown": pf.max_drawdown()
                })

    performance_df = pd.DataFrame(performance_results)
    top_10_df = performance_df.sort_values(by="total_return", ascending=False).head(10)
    st.write("Top 10 Coins by Total Return with Optimal TP and SL:")
    st.dataframe(top_10_df[['asset', 'tp_ratio', 'sl_ratio', 'total_return', 'sharpe_ratio', 'max_drawdown']])

st.sidebar.header("Manual Backtest")
manual_symbol = st.sidebar.text_input("Symbol", "BTC-USD")
manual_tp_ratio = st.sidebar.number_input("TP Ratio (e.g., 0.02 for 2%)", step=0.01, value=0.03)
manual_sl_ratio = st.sidebar.number_input("SL Ratio (e.g., 0.02 for 2%)", step=0.01, value=0.01)

if st.sidebar.button("Run Manual Backtest"):
    st.session_state.run_manual_backtest = True

if st.session_state.run_manual_backtest:
    manual_data = vbt.YFData.download(symbols=manual_symbol, interval=interval, period=period, tz_localize='UTC').concat()
    asset_close = manual_data['Close'][manual_symbol]
    asset_high = manual_data['High'][manual_symbol]
    asset_low = manual_data['Low'][manual_symbol]
    
    long_entries, long_exits, short_entries, short_exits = generate_signals(asset_close, asset_high, asset_low)
    
    pf = vbt.Portfolio.from_signals(
        close=asset_close,
        entries=long_entries,
        exits=long_exits,
        short_entries=short_entries,
        short_exits=short_exits,
        sl_stop=manual_sl_ratio,
        tp_stop=manual_tp_ratio,
        init_cash=1000,
        size=0.1,
        freq=interval
    )
    
    st.write(f"Manual Backtest Results for {manual_symbol} (TP: {manual_tp_ratio}, SL: {manual_sl_ratio}):")
    st.subheader('Performance metrics')
    st.write(pf.stats())
    fig = pf.plot()
    st.plotly_chart(fig)
    st.subheader('Trades Stats')
    st.write(pf.trades.stats())
    st.subheader('Positions')
    st.write(pf.positions.records_readable)
    st.subheader('Drawdowns')
    drawdowns = pf.drawdowns.records_readable
    st.write(drawdowns)
    st.subheader('Drawdown Plot')
    drawdown_fig = pf.drawdowns.plot()
    st.plotly_chart(drawdown_fig)
