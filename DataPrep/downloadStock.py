from datetime import datetime  # Import the datetime class from the datetime module
from yahoofinancials import YahooFinancials
import os
import pandas as pd


def get_ticker_data(ticker: str, param_start_date, param_end_date) -> dict:
    raw_data = YahooFinancials(ticker)
    return raw_data.get_historical_price_data(param_start_date, param_end_date, "daily").copy()

def fetch_ticker_data(ticker: str, start_date, end_date) -> pd.DataFrame:
    date_range = pd.bdate_range(start=start_date, end=end_date)
    values = pd.DataFrame({'Date': date_range})
    values['Date'] = pd.to_datetime(values['Date'])
    raw_data = get_ticker_data(ticker, start_date, end_date)
    return pd.DataFrame(raw_data[ticker]["prices"])[['date', 'open', 'high', 'low', 'adjclose', 'volume']]


# Assuming fetch_ticker_data function is already defined

def download_and_save_stock_data(symbol, start_date, end_date, base_directory="c:\\stock"):
    """
    Fetches stock data for a given symbol and date range, and saves it to a CSV file.

    :param symbol: The stock symbol to fetch data for.
    :param start_date: The start date of the data.
    :param end_date: The end date of the data.
    :param base_directory: The base directory to save the CSV files. Defaults to "c:\\stock".
    """
    # Fetch the stock data
    stock = fetch_ticker_data(symbol, start_date, end_date)
    
    # Set the directory and filename
    directory = base_directory
    # filename = f"{symbol}_{start_date}_to_{end_date}.csv"
    # filepath = os.path.join(directory, filename)
    filename = f"{symbol}.csv"
    filepath = os.path.join(directory, filename)

    # Create the directory if it does not exist
    os.makedirs(directory, exist_ok=True)

    # Save the dataframe to CSV
    try:
        stock.to_csv(filepath, index=False)
        print(f"File successfully saved to {filepath}")
    except Exception as e:
        print(f"An error occurred while saving {symbol}: {e}")

# Choose a date range
start_date = '2019-01-01'
end_date = datetime.now().strftime('%Y-%m-%d')

# List of symbols to fetch
symbols_to_fetch = [ 'AAPL', 'GOOGL', "INDI", "NDAQ", "SPX", "ES=F"]  # Add your symbols here
symbols_to_fetch = [ 'SX7E.Z']  # Add your symbols here
symbols_to_fetch = [ 'YM=F']  # Add your symbols here

for symbol in symbols_to_fetch:
    download_and_save_stock_data(symbol, start_date, end_date)