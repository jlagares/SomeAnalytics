import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
from scipy.stats import linregress
import imageio
import itertools
import os

NUM_DAYS = 120
WINDOW_DAYS = 360
PREDICT_DAYS = 10

def calculate_moving_average(values, N):
    """
    Calculates the moving average of the last N values in an array and returns the last 30 values of this moving average.

    Parameters:
    - values: An array of 120 values.
    - N: The number of values to calculate the moving average over.

    Returns:
    - A NumPy array containing the last 30 values of the moving average.
    """
    # Ensure the input is a NumPy array for efficient computation
    values = np.array(values)
    
    # Check if the input array has the expected length
    if len(values) != WINDOW_DAYS:
        raise ValueError(f"The input array must contain exactly {WINDOW_DAYS} values.")
    
    # Calculate the moving average using NumPy's convolve function
    # The 'valid' mode ensures the output length is len(values) - N + 1
    moving_avg = np.convolve(values, np.ones(N)/N, mode='valid')
    
    # Select and return the last 30 values of the moving average
    return moving_avg[-NUM_DAYS:]

def normalize_series(data):
    """
    Normalize an array of prices to be between 0 and 1, tolerating NaN values.
    
    Parameters:
    - data: A NumPy array of values.
    
    Returns:
    - A NumPy array of normalized values between 0 and 1.
    """
    # Calculate min and max, ignoring NaN values
    min_price = np.nanmin(data)
    max_price = np.nanmax(data)
    
    # Normalize, ignoring NaN values in the division
    normalized_data = (data - min_price) / (max_price - min_price)
    
    # Replace NaN values with the mean of the normalized data
    # Calculate the mean, ignoring NaN values
    mean_normalized = np.nanmean(normalized_data)
    # Replace NaN values in normalized_data with the mean
    normalized_data = np.where(np.isnan(normalized_data), mean_normalized, normalized_data)
    
    # print(f"Normalized {min_price}-{max_price}")
    return normalized_data

import numpy as np

def normalize_two_series(data1, data2):
    """
    Normalize two arrays of values to be between 0 and 1, based on the combined min and max of both series.
    NaN values are replaced with the mean of the non-NaN values before normalization.
    
    Parameters:
    - data1: A NumPy array of values.
    - data2: A NumPy array of values.
    
    Returns:
    - Two NumPy arrays of normalized values between 0 and 1.
    """
    # Combine data1 and data2, ignoring NaN values for min and max calculation
    combined_data = np.concatenate([data1, data2])
    
    # Calculate combined min and max, ignoring NaN values
    min_price = np.nanmin(combined_data)
    max_price = np.nanmax(combined_data)
    
    # Replace NaN values with the mean of the combined non-NaN values
    mean_combined = np.nanmean(combined_data)
    data1_no_nan = np.where(np.isnan(data1), mean_combined, data1)
    data2_no_nan = np.where(np.isnan(data2), mean_combined, data2)
    
    # Normalize data1 and data2
    normalized_data1 = (data1_no_nan - min_price) / (max_price - min_price)
    normalized_data2 = (data2_no_nan - min_price) / (max_price - min_price)
    
    # print(f"Normalized using combined min {min_price} and max {max_price}")
    return normalized_data1, normalized_data2

def calculate_rsi(prices, period=14):
    deltas = np.diff(prices)
    gain = np.where(deltas > 0, deltas, 0)
    loss = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.convolve(gain, np.ones(period)/period, mode='valid')
    avg_loss = np.convolve(loss, np.ones(period)/period, mode='valid')
    # AVOID ERROR: Division by zero in the calculation of relative strength
    # Create a copy of avg_loss for shifting
    avg_loss_shifted = np.roll(avg_loss, 1)
    # Replace zeros with the next value
    avg_loss = np.where(avg_loss==0, avg_loss_shifted, avg_loss)
    # If the first element is zero, replace it with the next non-zero value
    if avg_loss[0] == 0:
        avg_loss[0] = avg_loss[avg_loss != 0][0]
    rs = avg_gain[-NUM_DAYS:] / avg_loss[-NUM_DAYS:]
    rsi = normalize_series(100 - (100 / (1 + rs)))
    return np.concatenate([np.full(period-1, 0.5), rsi])  # Pad the start with NaNs

def calculate_mfi(high, low, close, volume, period=14):
    typical_price = (high + low + close) / 3
    raw_money_flow = typical_price * volume
    positive_flow = np.where(typical_price > np.roll(typical_price, 1), raw_money_flow, 0)
    negative_flow = np.where(typical_price < np.roll(typical_price, 1), raw_money_flow, 0)
    positive_flow[0] = negative_flow[0] = 0  # Correct the first element
    pos_flow_sum = np.convolve(positive_flow, np.ones(period), 'valid')
    neg_flow_sum = np.convolve(negative_flow, np.ones(period), 'valid')
    mfi = normalize_series(100 - (100 / (1 + pos_flow_sum / neg_flow_sum)))
    return np.concatenate([np.full(period-1, 0.5), mfi])  # Pad the start with NaNs

def calculate_stochastic_oscillator(close, high, low, k_period=14, d_period=3):
    lowest_low = np.convolve(low, np.ones(k_period), 'valid') / k_period
    highest_high = np.convolve(high, np.ones(k_period), 'valid') / k_period
    k_values = 100 * (close[k_period-1:] - lowest_low) / (highest_high - lowest_low)
    d_values = np.convolve(k_values, np.ones(d_period), 'valid') / d_period
    return normalize_series(k_values), normalize_series(d_values)

def calculate_atr(high, low, close, period=14):
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))
    true_ranges = np.maximum(high_low, high_close, low_close)
    true_ranges[0] = 0  # Correct the first element since np.roll introduces a shift
    atr = normalize_series(np.convolve(true_ranges, np.ones(period)/period, mode='valid'))
    return np.concatenate([np.full(period-1, 0.5), atr])  # Pad the start with NaNs

def calculate_ema(prices, period):
    weights = np.exp(np.linspace(-1., 0., period))
    weights /= weights.sum()
    ema = np.convolve(prices, weights, mode='valid')
    return np.concatenate([np.full(period-1, np.nan), ema])

def calculate_macd(prices, slow_period=26, fast_period=12, signal_period=9):
    slow_ema = calculate_ema(prices, slow_period)
    fast_ema = calculate_ema(prices, fast_period)
    macd_line = fast_ema - slow_ema
    signal_line = calculate_ema(macd_line[~np.isnan(macd_line)], signal_period)
    macd_histogram = macd_line[-len(signal_line):] - signal_line
    macd_line2, signal_line2 = normalize_two_series(macd_line, signal_line)
    macd_histogram = normalize_series(macd_histogram)
    return macd_line2, signal_line2, macd_histogram



def indicator_ma(selected_data_column,window_size, normalize = True):
   
    moving_avg = calculate_moving_average(selected_data_column, window_size)[-NUM_DAYS*window_size:]  # Last NUM_DAYS*AVG_N values
    if(normalize):
        return(normalize_series(moving_avg))
    else:
        return(moving_avg)

def repeat_series(data):
    # Reshape final_data to (1, 128) for stacking
    # Convert final_data to a NumPy array
    data_np = np.array(data)
    data_reshaped =data_np.reshape(1, len(data_np))
    # Stack final_data 3 times to create the middle rows
    rep_data = np.repeat(data_reshaped, 1, axis=0)
    return (rep_data)

def is_good_trend(predict_data, percentage_diff_threshold):
    # Calculate the average of the last 3 values
    avg_last_3 = predict_data['adjclose'].tail(3).mean()
    # Get the first value
    first_value = predict_data['adjclose'].iloc[0]
    # Calculate the percentage difference
    percentage_diff = ((avg_last_3 - first_value) / first_value) * 100
    # Check if the first 3 elements are monotonically increasing
    first_3_increasing = predict_data['adjclose'].head(3).is_monotonic_increasing
    print(f"The percentage difference between the first value and the average of the last 3 values is: {percentage_diff}%")
    return percentage_diff > percentage_diff_threshold and first_3_increasing

# Execute with the 5 mo
# Assuming the modified indicator functions are defined here (calculate_rsi, calculate_mfi, calculate_stochastic_oscillator)
def create_stock_snapshot_N(csv_file, start_index, window_size):
    # Load CSV file
    df = pd.read_csv(csv_file)
    # start_index = random.randint(WINDOW_DAYS, max_start_index-PREDICT_DAYS)
    #print(f" PREDICTING FROM VALUE {start_index}")
    selected_data = df.iloc[start_index-WINDOW_DAYS:start_index]
    # predict_data = df.iloc[start_index:start_index+PREDICT_DAYS]
    
    # Window size
    N = window_size
    # Calculate the rolling window average
    windowed_avg = selected_data.rolling(window=N).mean()

    # Since the first N-1 elements don't have enough data to calculate the windowed average,
    # they will be NaN. You might want to drop these NaN values.
    selected_data = windowed_avg.dropna()

    # Calculate indicators
    rsi = calculate_rsi(selected_data['adjclose'].values)
    mfi = calculate_mfi(selected_data['high'].values, selected_data['low'].values, selected_data['adjclose'].values, selected_data['volume'].values)
    k_values, d_values = calculate_stochastic_oscillator(selected_data['adjclose'].values, selected_data['high'].values, selected_data['low'].values)
    atr = calculate_atr(selected_data['high'].values,selected_data['adjclose'].values, selected_data['low'].values)
    macd_line, signal_line, macd_histogram = calculate_macd(selected_data['adjclose'].values)
    # prepare output data
    final_data = normalize_series(selected_data['adjclose'].values[-NUM_DAYS:])
    final_rsi = rsi[-NUM_DAYS:]
    final_mfi = mfi[-NUM_DAYS:]
    final_k_values = k_values[-NUM_DAYS:]
    final_d_values = d_values[-NUM_DAYS:]
    final_atr = atr[-NUM_DAYS:]
    final_macd_line = macd_line[-NUM_DAYS:]
    final_signal_line = signal_line[-NUM_DAYS:]
    final_macd_histogram = macd_histogram[-NUM_DAYS:]

    # List of all final arrays for iteration
    final_arrays = [final_data, final_rsi, final_mfi, final_k_values, final_d_values, final_atr, final_macd_line, final_signal_line, final_macd_histogram]
    zeros_array = np.zeros((1, len(final_data)))

    # Initialize the final structured array
    structured_array = zeros_array #zeros_array  # Assuming the data is 1D, adjust the reshape accordingly if it's not

    for arr in final_arrays:
        # For each array, we first add 2 rows of zeros
        final_data_reshaped = repeat_series(arr)
        structured_array = np.vstack([structured_array, final_data_reshaped])
        # structured_array = np.vstack([structured_array, zeros_array])
    return(structured_array)    

def create_stock_snapshot_levels(csv_file,start_index, level_array):
    # Use a list comprehension to generate the arrays
    snapshots = [create_stock_snapshot_N(csv_file=csv_file, start_index = start_index, window_size=i) for i in level_array]
    # Stack the arrays vertically
    return(np.vstack(snapshots))

def is_good_trend_csv(df, start_index, percentage_diff_threshold, debug=False, GoodTrendOnly=True, chartsPath = "charts", trainPath = "train",resultPath = "Good", suffix = ""):
    # Randomly select WINDOW_DAYS consecutive days
    predict_data = df.iloc[start_index:start_index+PREDICT_DAYS]
    if(is_good_trend(predict_data, percentage_diff_threshold) == GoodTrendOnly):
        # print("Good trend")
        # Calculate regression to do the classification
        plt.figure(figsize=(4, 4))
        regression_data = predict_data['adjclose']
        x = regression_data.index.values
        y = regression_data.values  # Corrected line
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        print(f"The slope of the regression line is: {slope}")
        # Plot the predict_data
        plt.scatter(x, y, color='blue', label='Data')
        # Calculate the values of the regression line
        regression_line = [slope*xi + intercept for xi in x]
        # Plot the regression line
        plt.plot(x, regression_line, color='red', label='Regression Line')
        figName = f"{chartsPath}\\{resultPath}\\predchart{start_index}-{suffix}.png"
        print(f"Saving to {figName}")
        plt.savefig(figName)
        if debug:
            plt.show()
        return GoodTrendOnly   
    
def create_full_snapshot_levels(file_list, debug=False, GoodTrendOnly = True, chartsPath = "charts", trainPath = "train", resultPath = "Good", suffix = ""):

    # Check if the trend is good
    df = pd.read_csv(file_list[0])
    max_start_index = len(df) - WINDOW_DAYS
    start_index = random.randint(WINDOW_DAYS, max_start_index-PREDICT_DAYS)
    print(f" PREDICTING FROM VALUE {start_index} {suffix}")
    while (is_good_trend_csv(df, start_index, 5, debug=debug, GoodTrendOnly=GoodTrendOnly, chartsPath=chartsPath, trainPath=trainPath, resultPath=resultPath, suffix=suffix) != GoodTrendOnly):
        start_index = random.randint(WINDOW_DAYS, max_start_index-PREDICT_DAYS)
        print("Bad trend")

    # print(f"Good trend at index {start_index}")
    stock_view = [create_stock_snapshot_levels(filename, start_index, level_array=[1,5,10]) for filename in file_list]
    # Stack the arrays vertically
    fullPic = np.vstack(stock_view)    
    # swap rows to get all same indicators together.
    if(debug):    
        plt.imshow(fullPic, cmap='gray', aspect='auto')
        plt.colorbar()  # Optionally add a colorbar
        plt.show()     
        print(fullPic.shape)  
    # Number of rows in each group
    group_size = 30 # 10 indicators
    # Number of complete groups
    num_groups = fullPic.shape[0] // group_size
    # Initialize an empty list to hold the reordered rows
    reordered_rows = []
    # Loop over each row within the groups
    for i in range(group_size):
        # Extract the i-th row from each group and concatenate them
        rows = [fullPic[j * group_size + i] for j in range(num_groups) if j * group_size + i < fullPic.shape[0]]
        if rows:  # If there are rows extracted, extend the list
            reordered_rows.extend(rows)
    # Convert the list of reordered rows back to a numpy array
    reordered_arr = np.array(reordered_rows)
    
    # Save the array as a PNG image
    reordered_arr_normalized = ((reordered_arr - reordered_arr.min()) * (1/(reordered_arr.max() - reordered_arr.min()) * 255)).astype('uint8')
    figName = f"{trainPath}\\{resultPath}\\indicator{start_index}-{suffix}.png"
    imageio.imsave(figName, reordered_arr_normalized)
    if(debug):
        print(reordered_arr.shape)
        plt.imshow(reordered_arr, cmap='gray', aspect='auto')
        # plt.colorbar()  # Optionally add a colorbar
        plt.show()   
    
# Visualize in deep-------------------------------------------------------------
def select_and_plot_indicators(csv_file):
    # Load CSV file
    # random.seed(42)
    df = pd.read_csv(csv_file)

    # Randomly select WINDOW_DAYS consecutive days
    max_start_index = len(df) - WINDOW_DAYS
    while True:
        start_index = random.randint(WINDOW_DAYS, max_start_index-PREDICT_DAYS)
        # print(f" PREDICTING FROM VALUE {start_index}")
        selected_data = df.iloc[start_index-WINDOW_DAYS:start_index]
        predict_data = df.iloc[start_index:start_index+PREDICT_DAYS]

        # Calculate regression to do the classification
        regression_data = predict_data['adjclose']
        x = regression_data.index.values
        y = regression_data.values  # Corrected line

        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        # print(f"The slope of the regression line is: {slope}")
        if is_good_trend(predict_data, 5):
            break
    # Plot the predict_data
    plt.scatter(x, y, color='blue', label='Data')

    # Calculate the values of the regression line
    regression_line = [slope*xi + intercept for xi in x]

    # Plot the regression line
    plt.plot(x, regression_line, color='red', label='Regression Line')

    plt.legend()
    plt.show()
    # Window size
    N = 5
    # Calculate the rolling window average
    windowed_avg = selected_data.rolling(window=N).mean()

    # Since the first N-1 elements don't have enough data to calculate the windowed average,
    # they will be NaN. You might want to drop these NaN values.
    selected_data = windowed_avg.dropna()

    # Calculate indicators
    rsi = calculate_rsi(selected_data['adjclose'].values)
    mfi = calculate_mfi(selected_data['high'].values, selected_data['low'].values, selected_data['adjclose'].values, selected_data['volume'].values)
    k_values, d_values = calculate_stochastic_oscillator(selected_data['adjclose'].values, selected_data['high'].values, selected_data['low'].values)
    atr = calculate_atr(selected_data['high'].values,selected_data['adjclose'].values, selected_data['low'].values)
    macd_line, signal_line, macd_histogram = calculate_macd(selected_data['adjclose'].values)
    # prepare output data
    final_data = normalize_series(selected_data['adjclose'].values[-NUM_DAYS:])
    final_rsi = rsi[-NUM_DAYS:]
    final_mfi = mfi[-NUM_DAYS:]
    final_k_values = k_values[-NUM_DAYS:]
    final_d_values = d_values[-NUM_DAYS:]
    final_atr = atr[-NUM_DAYS:]
    final_macd_line = macd_line[-NUM_DAYS:]
    final_signal_line = signal_line[-NUM_DAYS:]
    final_macd_histogram = macd_histogram[-NUM_DAYS:]

    # List of all final arrays for iteration
    final_arrays = [final_data, final_rsi, final_mfi, final_k_values, final_d_values, final_atr, final_macd_line, final_signal_line, final_macd_histogram]
    zeros_array = np.zeros((2, len(final_data)))

    # Initialize the final structured array
    structured_array = zeros_array  # Assuming the data is 1D, adjust the reshape accordingly if it's not

    for arr in final_arrays:
        # For each array, we first add 2 rows of zeros
        final_data_reshaped = repeat_series(arr)
        structured_array = np.vstack([structured_array, final_data_reshaped])
        structured_array = np.vstack([structured_array, zeros_array])
                
    # print(structured_array.shape)
    # print(final_data)
    plt.imshow((structured_array * 255).astype(np.uint8), cmap='gray',aspect='auto')
    plt.colorbar()  # Optionally add a colorbar to show the mapping from data values to colors
    plt.show()
    # Focus on the last NUM_DAYS days for plotting
    LAST_WINDOW_DATES = selected_data['date'][-NUM_DAYS:]
    
    # Normalize output series
    close_prices_normalized = normalize_series(selected_data['adjclose'][-NUM_DAYS:])
    rsi_normalized = rsi
    # Plotting
    plt.figure(figsize=(14, 10))
    N_PLOTS = 10

    # Adjusted stock adjusted close prices for the last NUM_DAYS days
    plt.subplot(N_PLOTS, 1, 1)
    plt.plot( close_prices_normalized, label='Adj Close Prices')
    plt.title(f'Stock Prices (Last {NUM_DAYS} Days)')
    plt.xticks(rotation=45)
    plt.legend()

    
    # Adjusted RSI Plotting for the last 30 days
    plt.subplot(N_PLOTS, 1, 2)
    if len(rsi) >= NUM_DAYS:
        plt.plot(LAST_WINDOW_DATES, rsi[-NUM_DAYS:], label='RSI', color='purple')
    else:
        # If the RSI array has fewer than 30 elements, plot as is
        plt.plot(selected_data['date'][-len(rsi):], rsi, label='RSI>', color='purple')
    plt.title(f'Relative Strength Index (Last {NUM_DAYS} Days)')
    plt.xticks(rotation=45)
    plt.legend()

    
    # Adjusted MFI Plotting for the last 30 days
    plt.subplot(N_PLOTS, 1, 3)
    if len(mfi) >= NUM_DAYS:
        plt.plot( mfi[-NUM_DAYS:], label='MFI', color='orange')
    else:
        plt.plot(mfi, label='MFI', color='orange')
    plt.title(f'Money Flow Index (Last {NUM_DAYS} Days)')
    plt.xticks(rotation=45)
    plt.legend()
    
     # Adjusted Stochastic Oscillator Plotting for the last 30 days
    plt.subplot(N_PLOTS, 1, 4)
    if len(k_values) >= NUM_DAYS:
        plt.plot(LAST_WINDOW_DATES, k_values[-NUM_DAYS:], label='%K', color='green')
        plt.plot(LAST_WINDOW_DATES, d_values[-NUM_DAYS:], label='%D', color='red')
    else:
        adjusted_dates_k = selected_data['date'][-len(k_values):]
        plt.plot(adjusted_dates_k, k_values, label='%K', color='green')
        plt.plot(adjusted_dates_k, d_values, label='%D', color='red')
    plt.title(f'Stochastic Oscillator (Last {NUM_DAYS} Days)')
    plt.xticks(rotation=45)
    plt.legend()
    # Assuming atr is calculated and available
    plt.subplot(N_PLOTS, 1, 5)  # Adjust the index based on your subplot arrangement
    if len(atr) >= NUM_DAYS:
        plt.plot(LAST_WINDOW_DATES, atr[-NUM_DAYS:], label='ATR', color='blue')
    else:
        # If the ATR array has fewer than NUM_DAYS elements, plot as is
        plt.plot(selected_data['date'][-len(atr):], atr, label='ATR', color='blue')
    plt.title(f'Average True Range (Last {NUM_DAYS} Days)')
    plt.xticks(rotation=45)
    plt.legend()

    # Plot stock adjusted close prices for the selected 120 days and predict data
    plt.subplot(N_PLOTS, 1, 6)
    plt.plot(selected_data['date'], (selected_data['adjclose']), label='Selected Data', color='blue')
    plt.plot(predict_data['date'], (predict_data['adjclose']), label='Predict Data', color='red', linestyle='--')
    plt.title('Stock Prices- to predict')
    plt.xticks(rotation=45)
    plt.legend()
    
    # Plotting Moving Average for the last 30 days
    # plt.subplot(N_PLOTS, 1, 7)  # Ensure this is the correct subplot index
    # plt.plot(LAST_WINDOW_DATES, moving_avg1, label=f'{AVG_N1}-Day Moving Average', color='magenta')
    # plt.plot(LAST_WINDOW_DATES, moving_avg2, label=f'{AVG_N2}-Day Moving Average', color='red')
    # plt.plot(LAST_WINDOW_DATES, moving_avg3, label=f'{AVG_N2}-Day Moving Average', color='red')
    # plt.title(f'Moving Average (2,4) (Last {NUM_DAYS} Days)')
    # plt.xticks(rotation=45)
    # plt.legend()  
    
    # # Plotting MACD
    plt.subplot(N_PLOTS, 1, 8)  # Adjust the index as needed based on your plotting arrangement
    plt.plot(selected_data['date'][-len(macd_line):], macd_line, label='MACD Line', color='darkgreen')
    plt.plot(selected_data['date'][-len(signal_line):], signal_line, label='Signal Line', color='magenta')
    plt.plot(selected_data['date'][-len(macd_histogram):], macd_histogram, label='MACD Histogram', color='blue')
    plt.title('MACD (Moving Average Convergence Divergence)')

 
    plt.show()

    # AVG_N1 = 2  # Period for the moving average
    # AVG_N2 = 5  # Period for the moving average
    # AVG_N3 = 10  # Period for the moving average
    # # moving_avg1 = calculate_moving_average(selected_data['adjclose'], AVG_N1)[-NUM_DAYS*AVG_N1:]  # Last NUM_DAYS*AVG_N values
    # # LAST_WINDOW_DATES_MA2 = selected_data['date'][-NUM_DAYS*AVG_N1:]
    # # # Focus on the last 30 days for plotting
    # # LAST_WINDOW_DATES = selected_data['date'][-NUM_DAYS:]
    # moving_avg1 = indicator_ma(selected_data_column=selected_data['adjclose'],window_size=AVG_N1)
    # moving_avg2 = indicator_ma(selected_data_column=selected_data['adjclose'],window_size=AVG_N2)
    # moving_avg3 = indicator_ma(selected_data_column=selected_data['adjclose'],window_size=AVG_N3)
    
        
# select_and_plot_indicators('c:\stock\AAPL.csv')
# create_full_snapshot_levels([r'c:\stock\AAPL.csv',r'c:\stock\GOOGL.csv',r'c:\stock\NDAQ.csv'])
# Keep the first element
stockArray = [ 'AAPL', 'ES=F', "YM=F", "NQ=F", 'YM=F',"^FTSE","GC=F", "SI=F"]
# Add 'c:\\stock\\Clean\\' to the start and '.csv' to the end of each symbol
stockArray = [r'c:\stock\Clean\{}.csv'.format(symbol) for symbol in stockArray]

first_element = [stockArray[0]]
# Permute all but the first element
permutations = list(itertools.permutations(stockArray[1:]))
# Combine the first element with each permutation
permuted_lists = [first_element + list(permutation) for permutation in permutations]
# print(permuted_lists)
# for i in range(1000):
#     print(f"Iteration {i} ")
#     print(permuted_lists[i%len(permuted_lists)])
# Get the basenames and take the first two characters
initials = [os.path.basename(path)[:4] for path in stockArray]
for i in range(5000):
    debug = False
    stockList = permuted_lists[i%len(permuted_lists)]
    initials = [os.path.basename(path)[:4] for path in stockList]
    initials_string = '-'.join(initials)
    print(f"Initials {initials_string}")
    create_full_snapshot_levels(stockArray,debug=debug, GoodTrendOnly = True, chartsPath = "outputData\\charts", trainPath = "outputData\\train", resultPath = "Good", suffix = initials_string)
    create_full_snapshot_levels(stockArray,debug=debug, GoodTrendOnly = False, chartsPath = "outputData\\charts", trainPath = "outputData\\train", resultPath = "Bad", suffix = initials_string)


# select_and_plot_indicators(r'c:\stock\ES=F.csv')
