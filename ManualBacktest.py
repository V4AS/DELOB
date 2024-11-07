import streamlit as st
import vectorbt as vbt
import pandas_ta as ta
import pandas as pd 

st.title("DelorianB Manual Backtest")

st.sidebar.header("Manual Backtest Settings")

interval = "15m"  
period = "1mo"    

manual_symbol = st.sidebar.text_input("Symbol", "BTC-USD")
manual_tp_ratio = st.sidebar.number_input("TP Ratio (e.g., 0.02 for 2%)", step=0.01, value=0.03)
manual_sl_ratio = st.sidebar.number_input("SL Ratio (e.g., 0.02 for 2%)", step=0.01, value=0.01)

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

if st.sidebar.button("Run Manual Backtest"):
    st.write(f"Running Manual Backtest for {manual_symbol} (TP: {manual_tp_ratio}, SL: {manual_sl_ratio})...")
    
    # Download data for the manual symbol
    manual_data = vbt.YFData.download([manual_symbol], interval=interval, period=period, tz_localize='UTC').concat()

    # Check data type and access accordingly
    if isinstance(manual_data['Close'], pd.Series):
        asset_close = manual_data['Close']
        asset_high = manual_data['High']
        asset_low = manual_data['Low']
    else:
        asset_close = manual_data['Close'][manual_symbol]
        asset_high = manual_data['High'][manual_symbol]
        asset_low = manual_data['Low'][manual_symbol]
    
    # Generate signals
    long_entries, long_exits, short_entries, short_exits = generate_signals(asset_close, asset_high, asset_low)
    
    # Run the backtest
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
    
    # Display results
    st.subheader("Manual Backtest Results")
    st.write(f"**Symbol:** {manual_symbol} | **TP Ratio:** {manual_tp_ratio} | **SL Ratio:** {manual_sl_ratio}")
    st.subheader('Performance Metrics')
    st.write(pf.stats())
    
    st.subheader('Portfolio Value Plot')
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
