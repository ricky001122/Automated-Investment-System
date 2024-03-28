# Importing necessary libraries
import streamlit as st
from datetime import date
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import plotly.subplots as sp
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import plotly.io as pio
from IPython.display import display
from plotly.offline import init_notebook_mode
init_notebook_mode(connected=True)

# Hiding Warnings
import warnings
warnings.filterwarnings('ignore')

def perform_portfolio_analysis(df, tickers_weights):


    # Starting DataFrame and Series 
    individual_cumsum = pd.DataFrame()
    individual_vol = pd.Series(dtype=float)
    individual_sharpe = pd.Series(dtype=float)


    # Iterating through tickers and weights in the tickers_weights dictionary
    for ticker, weight in tickers_weights.items():
        if ticker in df.columns: # Confirming that the tickers are available
            individual_returns = df[ticker].pct_change() # Computing individual daily returns for each ticker
            individual_cumsum[ticker] = ((1 + individual_returns).cumprod() - 1) * 100 # Computing cumulative returns over the period for each ticker 
            vol = (individual_returns.std() * np.sqrt(252)) * 100 # Computing annualized volatility
            individual_vol[ticker] = vol # Adding annualized volatility for each ticker
            individual_excess_returns = individual_returns - 0.01 / 252 # Computing the excess returns
            sharpe = (individual_excess_returns.mean() / individual_returns.std() * np.sqrt(252)).round(2) # Computing Sharpe Ratio
            individual_sharpe[ticker] = sharpe # Adding Sharpe Ratio for each ticker

            # Creating subplots for comparison across securities
            fig1 = make_subplots(rows = 1, cols = 2, horizontal_spacing=0.2,
                            column_titles=['Historical Performance Assets', 'Risk-Reward'],
                            column_widths=[.55, .45],
                            shared_xaxes=False, shared_yaxes=False)
        
    # Adding the historical returns for each ticker on the first subplot    
    for ticker in individual_cumsum.columns:
        fig1.add_trace(go.Scatter(x=individual_cumsum.index,
                                  y=individual_cumsum[ticker],
                                  mode = 'lines',
                                  name = ticker,
                                  hovertemplate = '%{y:.2f}%',
                                  showlegend=False),
                            row=1, col=1)

    # Defining colors for markers on the second subplot
    sharpe_colors = [individual_sharpe[ticker] for ticker in individual_cumsum.columns]

    # Adding markers for each ticker on the second subplot
    fig1.add_trace(go.Scatter(x=individual_vol.tolist(),
                              y=individual_cumsum.iloc[-1].tolist(),
                              mode='markers+text',
                              marker=dict(size=75, color = sharpe_colors, 
                                          colorscale = 'Bluered_r',
                                          colorbar=dict(title='Sharpe Ratio'),
                                          showscale=True),
                              name = 'Returns',
                              text = individual_cumsum.columns.tolist(),
                              textfont=dict(color='white'),
                              showlegend=False,
                              hovertemplate = '%{y:.2f}%<br>Annualized Volatility: %{x:.2f}%<br>Sharpe Ratio: %{marker.color:.2f}',
                              textposition='middle center'),
                        row=1, col=2)
            
    # Updating layout
    fig1.update_layout(title={
        'text': f'<b>Portfolio Analysis</b>',
        'font': {'size': 24}
    },
                       template = 'plotly_white',
                       height = 650, width = 1250,
                       hovermode = 'x unified')
        
    fig1.update_yaxes(title_text='Returns (%)', col=1)
    fig1.update_yaxes(title_text='Returns (%)', col = 2)
    fig1.update_xaxes(title_text = 'Date', col = 1)
    fig1.update_xaxes(title_text = 'Annualized Volatility (%)', col =2)
            
    return fig1 # Returning figure

def portfolio_vs_benchmark(port_returns, benchmark_returns):


    # Computing the cumulative returns for the portfolio and the benchmark
    portfolio_cumsum = (((1 + port_returns).cumprod() - 1) * 100).round(2)
    benchmark_cumsum = (((1 + benchmark_returns).cumprod() - 1) * 100).round(2)

    # Computing the annualized volatility for the portfolio and the benchmark
    port_vol = ((port_returns.std() * np.sqrt(252)) * 100).round(2)
    benchmark_vol = ((benchmark_returns.std() * np.sqrt(252)) * 100).round(2)

    # Computing Sharpe Ratio for the portfolio and the benchmark
    excess_port_returns = port_returns - 0.01 / 252
    port_sharpe = (excess_port_returns.mean() / port_returns.std() * np.sqrt(252)).round(2)
    exces_benchmark_returns = benchmark_returns - 0.01 / 252
    benchmark_sharpe = (exces_benchmark_returns.mean() / benchmark_returns.std() * np.sqrt(252)).round(2)

    # Creating a subplot to compare portfolio performance with the benchmark
    fig2 = make_subplots(rows = 1, cols = 2, horizontal_spacing=0.2,
                        column_titles=['Cumulative Returns', 'Portfolio Risk-Reward'],
                        column_widths=[.55, .45],
                        shared_xaxes=False, shared_yaxes=False)

    # Adding the cumulative returns for the portfolio
    fig2.add_trace(go.Scatter(x=portfolio_cumsum.index, 
                             y = portfolio_cumsum,
                             mode = 'lines', name = 'Portfolio', showlegend=False,
                             hovertemplate = '%{y:.2f}%'),
                             row=1,col=1)
    
    # Adding the cumulative returns for the benchmark
    fig2.add_trace(go.Scatter(x=benchmark_cumsum.index, 
                             y = benchmark_cumsum,
                             mode = 'lines', name = 'Benchmark', showlegend=False,
                             hovertemplate = '%{y:.2f}%'),
                             row=1,col=1)
    

    # Creating risk-reward plot for the benchmark and the portfolio
    fig2.add_trace(go.Scatter(x = [port_vol, benchmark_vol], y = [portfolio_cumsum.iloc[-1], benchmark_cumsum.iloc[-1]],
                             mode = 'markers+text', 
                             marker=dict(size = 75, 
                                         color = [port_sharpe, benchmark_sharpe],
                                         colorscale='Bluered_r',
                                         colorbar=dict(title='Sharpe Ratio'),
                                         showscale=True),
                             name = 'Returns', 
                             text=['Portfolio', 'Benchmark'], textposition='middle center',
                             textfont=dict(color='white'),
                             hovertemplate = '%{y:.2f}%<br>Annualized Volatility: %{x:.2f}%<br>Sharpe Ratio: %{marker.color:.2f}',
                             showlegend=False),
                             row = 1, col = 2)
    
    
    # Configuring layout
    fig2.update_layout(title={
        'text': f'<b>Portfolio vs Benchmark</b>',
        'font': {'size': 24}
    },
                      template = 'plotly_white',
                      height = 650, width = 1250,
                      hovermode = 'x unified')
    
    fig2.update_yaxes(title_text='Cumulative Returns (%)', col=1)
    fig2.update_yaxes(title_text='Cumulative Returns (%)', col = 2)
    fig2.update_xaxes(title_text = 'Date', col = 1)
    fig2.update_xaxes(title_text = 'Annualized Volatility (%)', col =2)

    return fig2 # Returning subplots


def portfolio_returns(tickers_and_values, start_date, end_date, benchmark):

   

    # Obtaining tickers data with yfinance
    df = yf.download(tickers=list(tickers_and_values.keys()),
                     start=start_date, end=end_date)

    # Checking if there is data available in the given date range
    if isinstance(df.columns, pd.MultiIndex):
        missing_data_tickers = []
        for ticker in tickers_and_values.keys():
            first_valid_index = df['Adj Close'][ticker].first_valid_index()
            if first_valid_index is None or first_valid_index.strftime('%Y-%m-%d') > start_date:
                missing_data_tickers.append(ticker)

        if missing_data_tickers:
            error_message = f"No data available for the following tickers starting from {start_date}: {', '.join(missing_data_tickers)}"
            return "error", error_message
    else:
        # For a single ticker, simply check the first valid index
        first_valid_index = df['Adj Close'].first_valid_index()
        if first_valid_index is None or first_valid_index.strftime('%Y-%m-%d') > start_date:
            error_message = f"No data available for the ticker starting from {start_date}"
            return "error", error_message
    
    # Calculating portfolio value
    total_portfolio_value = sum(tickers_and_values.values())

    # Calculating the weights for each security in the portfolio
    tickers_weights = {ticker: value / total_portfolio_value for ticker, value in tickers_and_values.items()}

    # Checking if dataframe has MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df = df['Adj Close'].fillna(df['Close']) # If 'Adjusted Close' is not available, use 'Close'

    # Checking if there are more than just one security in the portfolio
    if len(tickers_weights) > 1:
        weights = list(tickers_weights.values()) # Obtaining weights
        weighted_returns = df.pct_change().mul(weights, axis = 1) # Computed weighted returns
        port_returns = weighted_returns.sum(axis=1) # Sum weighted returns to build portfolio returns
    # If there is only one security in the portfolio...
    else:
        df = df['Adj Close'].fillna(df['Close'])  # Obtaining 'Adjusted Close'. If not available, use 'Close'
        port_returns = df.pct_change() # Computing returns without weights

    # Obtaining benchmark data with yfinance
    benchmark_df = yf.download(benchmark, 
                               start=start_date, end=end_date) 
    # Obtaining 'Adjusted Close'. If not available, use 'Close'.
    benchmark_df = benchmark_df['Adj Close'].fillna(benchmark_df['Close'])

    # Computing benchmark returns
    benchmark_returns = benchmark_df.pct_change()


    # Plotting a pie plot
    fig = go.Figure(data=[go.Pie(
        labels=list(tickers_weights.keys()), # Obtaining tickers 
        values=list(tickers_weights.values()), # Obtaining weights
        hoverinfo='label+percent', 
        textinfo='label+percent',
        hole=.65,
        marker=dict(colors=px.colors.qualitative.G10)
    )])

    # Defining layout
    fig.update_layout(title={
        'text': '<b>Portfolio Allocation</b>',
        'font': {'size': 24}
    }, height=550, width=1250)

    # Running function to compare portfolio and benchmark
    fig2 = portfolio_vs_benchmark(port_returns, benchmark_returns)    

    #fig.show() # Displaying Portfolio Allocation plot

    # If we have more than one security in the portfolio, 
    # we run function to evaluate each security individually
    fig1 = None
    if len(tickers_weights) > 1:
        fig1 = perform_portfolio_analysis(df, tickers_weights)
        #fig1.show()
    # Displaying Portfolio vs Benchmark plot    
    #fig2.show()
    return "success", (fig, fig1, fig2)

# Defining page settings
st.set_page_config(
    page_title="Investment Portfolio Management",
    page_icon=":heavy_dollar_sign:",
    layout='wide',
    initial_sidebar_state='expanded'
)


if 'num_pairs' not in st.session_state:
    st.session_state['num_pairs'] = 1

def add_input_pair():
    st.session_state['num_pairs'] += 1

# Function to display error message
def display_error_message(message):
    st.error(message)

# Function to validate input data
def validate_input(tickers_and_values, benchmark):
    if not benchmark:
        display_error_message("Please enter a benchmark ticker before running the analysis.")
        return False
    if not tickers_and_values:
        display_error_message("Please add at least one ticker with a non-zero investment value before running the analysis.")
        return False
    return True

# Title and introduction
title = '<h1 style="font-family:Didot; font-size: 64px; text-align:left">InvestPro</h1>'
st.markdown(title, unsafe_allow_html=True)

intro_text = """
<div style="background-color: #f0f0f0; padding: 20px;">
    <img src="https://static.vecteezy.com/system/resources/previews/001/176/913/non_2x/the-investment-background-vector.jpg" alt="Banner Image" style="width: 100%; max-width: 800px; display: block; margin: 0 auto 20px;">
    <h1 style="font-size: 24px; color: #333; text-align: center; margin-bottom: 20px;">Welcome to InvestPro</h1>
    <p style="font-size: 18px; color: #555; text-align: left; line-height: 1.5;">
        InvestPro is an intuitive app that streamlines your investment portfolio management. Effortlessly monitor your assets, benchmark against market standards, and discover valuable insights with just a few clicks.
        <br><br>
        Here's what Our project can do:
        <ul>
            <li>Enter the ticker symbols exactly as they appear on Yahoo Finance and the total amount invested for each security in your portfolio.</li>
            <li>Set a benchmark to compare your portfolio's performance against market indices or other chosen standards.</li>
            <li>Select the start and end dates for the period you wish to analyze and gain historical insights. <br><em>Note: The app cannot analyze dates before a company's IPO or use non-business days as your start or end dates.</em></li>
            <li>Click "Run Analysis" to visualize historical returns, obtain volatility metrics, and unveil the allocation percentages of your portfolio.</li>
        </ul>
        Empower your investment strategy with cutting-edge financial APIs and visualization tools.
        <br><br>
        Start making informed decisions to elevate your financial future today.
    </p>
</div>
<br/><br/>
"""
st.markdown(intro_text, unsafe_allow_html=True)


# Input section
if 'num_pairs' not in st.session_state:
    st.session_state['num_pairs'] = 1

tickers_and_values = {}
for n in range(st.session_state['num_pairs']):
    col1, col2 = st.columns(2)
    with col1:
        ticker = st.text_input(f"Ticker {n+1}", key=f"ticker_{n+1}", placeholder="Enter the symbol for a security.")
    with col2:
        value = st.number_input(f"Value Invested in Ticker {n+1} ($)", min_value=0.0, format="%.2f", key=f"value_{n+1}")
    tickers_and_values[ticker] = value


add_button_col, run_button_col = st.columns([1, 3])
with add_button_col:
    st.button("Add Another Ticker", on_click=add_input_pair)
with run_button_col:
    start_date = st.date_input("Start Date", value=date.today().replace(year=date.today().year-1), min_value=date(1900, 1, 1))
    end_date = st.date_input("End Date", value=date.today(), min_value=date(1900, 1, 1))
    benchmark = st.text_input("Benchmark", placeholder="Enter the symbol for a benchmark.")
    if st.button("Run Analysis"):
        tickers_and_values = {k: v for k, v in tickers_and_values.items() if k and v > 0}
        if validate_input(tickers_and_values, benchmark):
            start_date_str = start_date.strftime('%Y-%m-%d')
            end_date_str = end_date.strftime('%Y-%m-%d')
            status, result = portfolio_returns(tickers_and_values, start_date_str, end_date_str, benchmark)
            if status == "error":
                display_error_message(result)
            else:
                fig, fig1, fig2 = result
                if fig is not None:
                    st.plotly_chart(fig)
                if fig1 is not None:
                    st.plotly_chart(fig1)
                if fig2 is not None:
                    st.plotly_chart(fig2)
