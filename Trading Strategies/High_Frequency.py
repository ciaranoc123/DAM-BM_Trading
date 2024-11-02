import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt


# Simulate a charging or discharging operation based on price indexes.
# The bottleneck-controlled strategy is applied to maximize profit while considering battery constraints.
# Adjust charging or discharging dynamically based on the order of minimum and maximum price periods.
def process_prices_DAM(charge_level, capacity, ramp_rate, min_charge_level, eff_1, eff_2, prices, current_df, min_price_index, max_price_index):
    # Initialize profit to zero in case no valid trade occurs
    profit = 0

    if min_price_index < max_price_index:
        # Charging: only proceed if there is capacity available
        charge_amount = min(capacity - charge_level, ramp_rate)
        
        if charge_amount > 0:
            charge_level += charge_amount

            # Discharge only if there is charge above min_charge_level
            discharge_amount = min(charge_level - min_charge_level, ramp_rate)
            if discharge_amount > 0:
                charge_level -= discharge_amount
                profit = (current_df.loc[max_price_index, 'Price'] * discharge_amount * eff_1) - \
                         (current_df.loc[min_price_index, 'Price'] * charge_amount / eff_2)
            
    elif min_price_index > max_price_index:
        # Discharging: only proceed if there is enough charge
        discharge_amount = min(charge_level - min_charge_level, ramp_rate)
        
        if discharge_amount > 0:
            charge_level -= discharge_amount

            # Charge only if there is capacity available
            charge_amount = min(capacity - charge_level, ramp_rate)
            if charge_amount > 0:
                charge_level += charge_amount
                profit = (current_df.loc[max_price_index, 'Price'] * discharge_amount * eff_1) - \
                         (current_df.loc[min_price_index, 'Price'] * charge_amount / eff_2)

    # Append trade details if a valid trade occurred
    if profit != 0:
        prices.append((min_price_index, current_df.loc[min_price_index, 'Price'], max_price_index, 
                       current_df.loc[max_price_index, 'Price'], profit, charge_level))
#         print(f"Trade executed - minPriceIndex: {min_price_index}, maxPriceIndex: {max_price_index}, profit: {profit:.2f}, chargeLevel after trade: {charge_level}")
    
    return charge_level



# Recursive function to explore possible trade pairs within identified price subsets.
# The function iteratively identifies trade pairs within price data and tracks the state of the charge level.
# It considers price data before, in-between, and after each trade pair, maximizing trading opportunities.
def recursive_process_prices_DAM(charge_level, capacity, ramp_rate, min_charge_level, eff_1, eff_2, prices, current_df, remaining_prices_A, remaining_prices_B, current_Q_A, current_Q_B, day, trades_today, max_trades_per_day, profit_threshold):
    if len(remaining_prices_A) <= 1 or trades_today >= max_trades_per_day:
        return charge_level, trades_today

    max_price_index = remaining_prices_A['Price'].idxmax()
    min_price_index = remaining_prices_B['Price'].idxmin()

    # Expected profit check for recursive trades
    expected_profit = (current_Q_A.loc[max_price_index, 'Price'] * eff_1) - (current_Q_B.loc[min_price_index, 'Price'] / eff_2)
    if expected_profit >= profit_threshold and trades_today < max_trades_per_day:
        charge_level = process_prices_DAM(charge_level, capacity, ramp_rate, min_charge_level, eff_1, eff_2, prices, current_df, min_price_index, max_price_index)
        trades_today += 1

    # Recurse into smaller subsets if there are more trades allowed
    smaller_index = min(min_price_index, max_price_index)
    larger_index = max(min_price_index, max_price_index)
    remaining_prices_A = current_Q_A[(current_Q_A['level_0'] == day) & (current_Q_A.index > smaller_index) & (current_Q_A.index < larger_index)]
    remaining_prices_B = current_Q_B[(current_Q_B['level_0'] == day) & (current_Q_B.index > smaller_index) & (current_Q_B.index < larger_index)]

    return recursive_process_prices_DAM(charge_level, capacity, ramp_rate, min_charge_level, eff_1, eff_2, prices, current_df, remaining_prices_A, remaining_prices_B, current_Q_A, current_Q_B, day, trades_today, max_trades_per_day, profit_threshold)
    
    

# Execute the bottleneck-controlled trading strategy for TS3.
# This strategy aims to maximize profit by considering battery constraints and flexible timestamps.
# It identifies trade pairs within various price subsets and iterates through the available trading opportunities.
def electricity_strategy_DAM_HF(df, Q_A_Preds, Q_B_Preds, eff_1, eff_2, capacity, charge_level, ramp_rate, min_charge_level, profit_threshold=0, max_trades_per_day=3):
    prices = []
    day_index = df['level_0'].unique()
    
    for day in day_index:
        trades_today = 0
        current_df = df[df['level_0'] == day]
        current_Q_A = Q_A_Preds[Q_A_Preds['level_0'] == day]
        current_Q_B = Q_B_Preds[Q_B_Preds['level_0'] == day]

        max_price_index = current_Q_A['Price'].idxmax()
        min_price_index = current_Q_B['Price'].idxmin()        

        # First Trade
        expected_profit_T1 = (current_Q_A.loc[max_price_index, 'Price'] * eff_1) - (current_Q_B.loc[min_price_index, 'Price'] / eff_2)
        if expected_profit_T1 >= profit_threshold and trades_today < max_trades_per_day:
            charge_level = process_prices_DAM(charge_level, capacity, ramp_rate, min_charge_level, eff_1, eff_2, prices, current_df, min_price_index, max_price_index)
            trades_today += 1

        # Trades within, before, and after T1
        smaller_index = min(min_price_index, max_price_index)
        larger_index = max(min_price_index, max_price_index)
        
        prices_inbetween_T1_A = Q_A_Preds[(Q_A_Preds['level_0'] == day) & (Q_A_Preds.index > smaller_index) & (Q_A_Preds.index < larger_index)]
        prices_inbetween_T1_B = Q_B_Preds[(Q_B_Preds['level_0'] == day) & (Q_B_Preds.index > smaller_index) & (Q_B_Preds.index < larger_index)]
        charge_level, trades_today = recursive_process_prices_DAM(charge_level, capacity, ramp_rate, min_charge_level, eff_1, eff_2, prices, current_df, prices_inbetween_T1_A, prices_inbetween_T1_B, current_Q_A, current_Q_B, day, trades_today, max_trades_per_day, profit_threshold)

        if trades_today >= max_trades_per_day:
            continue

        # Additional pre- and post-trades using the same logic
        prices_before_T1_A = Q_A_Preds[(Q_A_Preds['level_0'] == day) & (Q_A_Preds.index < smaller_index)]
        prices_before_T1_B = Q_B_Preds[(Q_B_Preds['level_0'] == day) & (Q_B_Preds.index < smaller_index)]
        charge_level, trades_today = recursive_process_prices_DAM(charge_level, capacity, ramp_rate, min_charge_level, eff_1, eff_2, prices, current_df, prices_before_T1_A, prices_before_T1_B, current_Q_A, current_Q_B, day, trades_today, max_trades_per_day, profit_threshold)

        if trades_today >= max_trades_per_day:
            continue

        prices_after_T1_A = Q_A_Preds[(Q_A_Preds['level_0'] == day) & (Q_A_Preds.index > larger_index)]
        prices_after_T1_B = Q_B_Preds[(Q_B_Preds['level_0'] == day) & (Q_B_Preds.index > larger_index)]
        charge_level, trades_today = recursive_process_prices_DAM(charge_level, capacity, ramp_rate, min_charge_level, eff_1, eff_2, prices, current_df, prices_after_T1_A, prices_after_T1_B, current_Q_A, current_Q_B, day, trades_today, max_trades_per_day, profit_threshold)

    return pd.DataFrame(prices, columns=['minPriceIndex', 'minPrice', 'maxPriceIndex', 'maxPrice', 'profit', 'chargeLevel'])
    
    




def calculate_trading_results_DAM_HF(Y_r, Q_10, Q_30, Q_50, Q_70, Q_90):
    eff_1 = 0.8
    eff_2 = 0.98
    
    r_dam_50_50 = electricity_strategy_DAM_HF(df=Y_r, Q_A_Preds=Q_50, Q_B_Preds=Q_50, eff_1=0.8, eff_2=0.98, capacity=1,charge_level=0, ramp_rate=1, min_charge_level=0, profit_threshold=0, max_trades_per_day=3)
    r_dam_10_30 = electricity_strategy_DAM_HF(df=Y_r, Q_A_Preds=Q_10, Q_B_Preds=Q_30, eff_1=0.8, eff_2=0.98, capacity=1,charge_level=0, ramp_rate=1, min_charge_level=0, profit_threshold=0, max_trades_per_day=3)
    r_dam_30_50 = electricity_strategy_DAM_HF(df=Y_r, Q_A_Preds=Q_30, Q_B_Preds=Q_50, eff_1=0.8, eff_2=0.98, capacity=1,charge_level=0, ramp_rate=1, min_charge_level=0, profit_threshold=0, max_trades_per_day=3)
    r_dam_50_70 = electricity_strategy_DAM_HF(df=Y_r, Q_A_Preds=Q_50, Q_B_Preds=Q_70, eff_1=0.8, eff_2=0.98, capacity=1,charge_level=0, ramp_rate=1, min_charge_level=0, profit_threshold=0, max_trades_per_day=3)
    r_dam_70_90 = electricity_strategy_DAM_HF(df=Y_r, Q_A_Preds=Q_70, Q_B_Preds=Q_90, eff_1=0.8, eff_2=0.98, capacity=1,charge_level=0, ramp_rate=1, min_charge_level=0, profit_threshold=0, max_trades_per_day=3)
    r_dam_30_70 = electricity_strategy_DAM_HF(df=Y_r, Q_A_Preds=Q_30, Q_B_Preds=Q_70, eff_1=0.8, eff_2=0.98, capacity=1,charge_level=0, ramp_rate=1, min_charge_level=0, profit_threshold=0, max_trades_per_day=3)
    r_dam_10_90 = electricity_strategy_DAM_HF(df=Y_r, Q_A_Preds=Q_10, Q_B_Preds=Q_90, eff_1=0.8, eff_2=0.98, capacity=1,charge_level=0, ramp_rate=1, min_charge_level=0, profit_threshold=0, max_trades_per_day=3)    
    PF_DAM = electricity_strategy_DAM_HF(df=Y_r, Q_A_Preds=Y_r, Q_B_Preds=Y_r, eff_1=0.8, eff_2=0.98, capacity=1,charge_level=0, ramp_rate=1, min_charge_level=0, profit_threshold=0, max_trades_per_day=3)
    
    results = {
        'r_dam_50_50': np.round(sum(r_dam_50_50.iloc[:, 4:5].values), 2),
        'r_dam_10_30': np.round(sum(r_dam_10_30.iloc[:, 4:5].values), 2),
        'r_dam_30_50': np.round(sum(r_dam_30_50.iloc[:, 4:5].values), 2),
        'r_dam_50_70': np.round(sum(r_dam_50_70.iloc[:, 4:5].values), 2),
        'r_dam_70_90': np.round(sum(r_dam_70_90.iloc[:, 4:5].values), 2),
        'r_dam_30_70': np.round(sum(r_dam_30_70.iloc[:, 4:5].values), 2),
        'r_dam_10_90': np.round(sum(r_dam_10_90.iloc[:, 4:5].values), 2),
        'PF_DAM': np.round(sum(PF_DAM.iloc[:, 4:5].values), 2)
    }
    
    return results






















# Simulate a charging or discharging operation based on price indexes.
# The bottleneck-controlled strategy is applied to maximize profit while considering battery constraints.
# Adjust charging or discharging dynamically based on the order of minimum and maximum price periods.
def process_prices_BM(charge_level, capacity, ramp_rate, min_charge_level, eff_1, eff_2, prices, current_df, min_price_index, max_price_index):
    # Initialize profit to zero in case no valid trade occurs
    profit = 0

    if min_price_index < max_price_index:
        # Charging: only proceed if there is capacity available
        charge_amount = min(capacity - charge_level, ramp_rate)
        
        if charge_amount > 0:
            charge_level += charge_amount

            # Discharge only if there is charge above min_charge_level
            discharge_amount = min(charge_level - min_charge_level, ramp_rate)
            if discharge_amount > 0:
                charge_level -= discharge_amount
                profit = (current_df.loc[max_price_index, 'Price'] * discharge_amount * eff_1) - \
                         (current_df.loc[min_price_index, 'Price'] * charge_amount / eff_2)
            
    elif min_price_index > max_price_index:
        # Discharging: only proceed if there is enough charge
        discharge_amount = min(charge_level - min_charge_level, ramp_rate)
        
        if discharge_amount > 0:
            charge_level -= discharge_amount

            # Charge only if there is capacity available
            charge_amount = min(capacity - charge_level, ramp_rate)
            if charge_amount > 0:
                charge_level += charge_amount
                profit = (current_df.loc[max_price_index, 'Price'] * discharge_amount * eff_1) - \
                         (current_df.loc[min_price_index, 'Price'] * charge_amount / eff_2)

    # Append trade details if a valid trade occurred
    if profit != 0:
        prices.append((min_price_index, current_df.loc[min_price_index, 'Price'], max_price_index, 
                       current_df.loc[max_price_index, 'Price'], profit, charge_level))
#         print(f"Trade executed - minPriceIndex: {min_price_index}, maxPriceIndex: {max_price_index}, profit: {profit:.2f}, chargeLevel after trade: {charge_level}")
    
    return charge_level


# Recursive function to explore possible trade pairs within identified price subsets.
# The function iteratively identifies trade pairs within price data and tracks the state of the charge level.
# It considers price data before, in-between, and after each trade pair, maximizing trading opportunities.
def recursive_process_prices_BM(charge_level, capacity, ramp_rate, min_charge_level, eff_1, eff_2, prices, current_df, remaining_prices_A, remaining_prices_B, current_Q_A, current_Q_B, day, trades_today, max_trades_per_day, profit_threshold):
    if len(remaining_prices_A) <= 1 or trades_today >= max_trades_per_day:
        return charge_level, trades_today

    max_price_index = remaining_prices_A['Price'].idxmax()
    min_price_index = remaining_prices_B['Price'].idxmin()

    # Expected profit check for recursive trades
    expected_profit = (current_Q_A.loc[max_price_index, 'Price'] * eff_1) - (current_Q_B.loc[min_price_index, 'Price'] / eff_2)
    if expected_profit >= profit_threshold and trades_today < max_trades_per_day:
        charge_level = process_prices_BM(charge_level, capacity, ramp_rate, min_charge_level, eff_1, eff_2, prices, current_df, min_price_index, max_price_index)
        trades_today += 1

    # Recurse into smaller subsets if there are more trades allowed
    smaller_index = min(min_price_index, max_price_index)
    larger_index = max(min_price_index, max_price_index)
    remaining_prices_A = current_Q_A[(current_Q_A['level_0'] == day) & (current_Q_A.index > smaller_index) & (current_Q_A.index < larger_index)]
    remaining_prices_B = current_Q_B[(current_Q_B['level_0'] == day) & (current_Q_B.index > smaller_index) & (current_Q_B.index < larger_index)]

    return recursive_process_prices_BM(charge_level, capacity, ramp_rate, min_charge_level, eff_1, eff_2, prices, current_df, remaining_prices_A, remaining_prices_B, current_Q_A, current_Q_B, day, trades_today, max_trades_per_day, profit_threshold)
    
    
    

# Execute the bottleneck-controlled trading strategy for TS3.
# This strategy aims to maximize profit by considering battery constraints and flexible timestamps.
# It identifies trade pairs within various price subsets and iterates through the available trading opportunities.
def electricity_strategy_BM_HF(df, Q_A_Preds, Q_B_Preds,  eff_1, eff_2, capacity,charge_level, ramp_rate, min_charge_level, profit_threshold=0, max_trades_per_day=3):
    # Initialize an empty list to store trade details and set the initial charge level.
    prices = []
    day_index = df['level_0'].unique()
    
    for day in day_index:
        trades_today = 0
        current_df = df[df['level_0'] == day]
        current_Q_A = Q_A_Preds[Q_A_Preds['level_0'] == day]
        current_Q_B = Q_B_Preds[Q_B_Preds['level_0'] == day]

        max_price_index = current_Q_A['Price'].idxmax()
        min_price_index = current_Q_B['Price'].idxmin()        

        # First Trade
        expected_profit_T1 = (current_Q_A.loc[max_price_index, 'Price'] * eff_1) - (current_Q_B.loc[min_price_index, 'Price'] / eff_2)
        if expected_profit_T1 >= profit_threshold and trades_today < max_trades_per_day:
            charge_level = process_prices_BM(charge_level, capacity, ramp_rate, min_charge_level, eff_1, eff_2, prices, current_df, min_price_index, max_price_index)
            trades_today += 1

        # Trades within, before, and after T1
        smaller_index = min(min_price_index, max_price_index)
        larger_index = max(min_price_index, max_price_index)
        
        prices_inbetween_T1_A = Q_A_Preds[(Q_A_Preds['level_0'] == day) & (Q_A_Preds.index > smaller_index) & (Q_A_Preds.index < larger_index)]
        prices_inbetween_T1_B = Q_B_Preds[(Q_B_Preds['level_0'] == day) & (Q_B_Preds.index > smaller_index) & (Q_B_Preds.index < larger_index)]
        charge_level, trades_today = recursive_process_prices_BM(charge_level, capacity, ramp_rate, min_charge_level, eff_1, eff_2, prices, current_df, prices_inbetween_T1_A, prices_inbetween_T1_B, current_Q_A, current_Q_B, day, trades_today, max_trades_per_day, profit_threshold)

        if trades_today >= max_trades_per_day:
            continue

        # Additional pre- and post-trades using the same logic
        prices_before_T1_A = Q_A_Preds[(Q_A_Preds['level_0'] == day) & (Q_A_Preds.index < smaller_index)]
        prices_before_T1_B = Q_B_Preds[(Q_B_Preds['level_0'] == day) & (Q_B_Preds.index < smaller_index)]
        charge_level, trades_today = recursive_process_prices_BM(charge_level, capacity, ramp_rate, min_charge_level, eff_1, eff_2, prices, current_df, prices_before_T1_A, prices_before_T1_B, current_Q_A, current_Q_B, day, trades_today, max_trades_per_day, profit_threshold)

        if trades_today >= max_trades_per_day:
            continue

        prices_after_T1_A = Q_A_Preds[(Q_A_Preds['level_0'] == day) & (Q_A_Preds.index > larger_index)]
        prices_after_T1_B = Q_B_Preds[(Q_B_Preds['level_0'] == day) & (Q_B_Preds.index > larger_index)]
        charge_level, trades_today = recursive_process_prices_BM(charge_level, capacity, ramp_rate, min_charge_level, eff_1, eff_2, prices, current_df, prices_after_T1_A, prices_after_T1_B, current_Q_A, current_Q_B, day, trades_today, max_trades_per_day, profit_threshold)

    return pd.DataFrame(prices, columns=['minPriceIndex', 'minPrice', 'maxPriceIndex', 'maxPrice', 'profit', 'chargeLevel'])
    



def calculate_bm_trading_results_HF(Y_r, Q_10, Q_30, Q_50, Q_70, Q_90):
    eff_1 = 0.8
    eff_2 = 0.98
    
    r_bm_50_50 = electricity_strategy_BM_HF(df=Y_r, Q_A_Preds=Q_50, Q_B_Preds=Q_50, eff_1=0.8, eff_2=0.98, capacity=1,charge_level=0, ramp_rate=1, min_charge_level=0, profit_threshold=0, max_trades_per_day=3)
    r_bm_10_30 = electricity_strategy_BM_HF(df=Y_r, Q_A_Preds=Q_10, Q_B_Preds=Q_30, eff_1=0.8, eff_2=0.98, capacity=1,charge_level=0, ramp_rate=1, min_charge_level=0, profit_threshold=0, max_trades_per_day=3)
    r_bm_30_50 = electricity_strategy_BM_HF(df=Y_r, Q_A_Preds=Q_30, Q_B_Preds=Q_50, eff_1=0.8, eff_2=0.98, capacity=1,charge_level=0, ramp_rate=1, min_charge_level=0, profit_threshold=0, max_trades_per_day=3)
    r_bm_50_70 = electricity_strategy_BM_HF(df=Y_r, Q_A_Preds=Q_50, Q_B_Preds=Q_70, eff_1=0.8, eff_2=0.98, capacity=1,charge_level=0, ramp_rate=1, min_charge_level=0, profit_threshold=0, max_trades_per_day=3)
    r_bm_70_90 = electricity_strategy_BM_HF(df=Y_r, Q_A_Preds=Q_70, Q_B_Preds=Q_90, eff_1=0.8, eff_2=0.98, capacity=1,charge_level=0, ramp_rate=1, min_charge_level=0, profit_threshold=0, max_trades_per_day=3)
    r_bm_30_70 = electricity_strategy_BM_HF(df=Y_r, Q_A_Preds=Q_30, Q_B_Preds=Q_70, eff_1=0.8, eff_2=0.98, capacity=1,charge_level=0, ramp_rate=1, min_charge_level=0, profit_threshold=0, max_trades_per_day=3)
    r_bm_10_90 = electricity_strategy_BM_HF(df=Y_r, Q_A_Preds=Q_10, Q_B_Preds=Q_90, eff_1=0.8, eff_2=0.98, capacity=1,charge_level=0, ramp_rate=1, min_charge_level=0, profit_threshold=0, max_trades_per_day=3)    
    PF_BM = electricity_strategy_BM_HF(df=Y_r, Q_A_Preds=Y_r, Q_B_Preds=Y_r, eff_1=0.8, eff_2=0.98, capacity=1,charge_level=0, ramp_rate=1, min_charge_level=0, profit_threshold=0, max_trades_per_day=3)
    
    results = {
        'r_bm_50_50': np.round(sum(r_bm_50_50.iloc[:, 4:5].values), 2),
        'r_bm_10_30': np.round(sum(r_bm_10_30.iloc[:, 4:5].values), 2),
        'r_bm_30_50': np.round(sum(r_bm_30_50.iloc[:, 4:5].values), 2),
        'r_bm_50_70': np.round(sum(r_bm_50_70.iloc[:, 4:5].values), 2),
        'r_bm_70_90': np.round(sum(r_bm_70_90.iloc[:, 4:5].values), 2),
        'r_bm_30_70': np.round(sum(r_bm_30_70.iloc[:, 4:5].values), 2),
        'r_bm_10_90': np.round(sum(r_bm_10_90.iloc[:, 4:5].values), 2),
        'PF_BM': np.round(sum(PF_BM.iloc[:, 4:5].values), 2)
    }
    
    return results



# Define the function to plot profit for a single trade in the BM strategy
def plot_profit_High_Frequency_Strategy_BM(Y_r_bm, Q_A_Preds, Q_B_Preds, eff_1, eff_2, capacity,charge_level, ramp_rate, min_charge_level, profit_threshold=0, max_trades_per_day=3):
    # Run electricity strategy for BM
    HF_trade_bm = electricity_strategy_BM_HF(df=Y_r_bm, Q_A_Preds=Q_A_Preds, Q_B_Preds=Q_B_Preds, eff_1=eff_1, eff_2=eff_2, capacity=1,charge_level=0, ramp_rate=1, min_charge_level=0, profit_threshold=0, max_trades_per_day=3)
    # Plot the profit obtained
    plt.figure(figsize=(15, 6))
    plt.plot(HF_trade_bm.iloc[:, 4:5].values, marker='o', linestyle='-', color='r', label='Profit')
    plt.title('Profit Obtained from High Frequency Strategy (TS3-BM)')
    plt.xlabel('Index')
    plt.ylabel('Profit')
    plt.grid(True)
    plt.legend()
    plt.show()
    

# Define the function to plot profit for a single trade in the BM strategy
def plot_profit_High_Frequency_Strategy_DAM(Y_r, Q_A_Preds, Q_B_Preds, eff_1, eff_2, capacity,charge_level, ramp_rate, min_charge_level, profit_threshold=0, max_trades_per_day=3):
    # Run electricity strategy for BM
    HF_trade_dam = electricity_strategy_DAM_HF(df=Y_r, Q_A_Preds=Q_A_Preds, Q_B_Preds=Q_B_Preds, eff_1=eff_1, eff_2=eff_2, capacity=1,charge_level=0, ramp_rate=1, min_charge_level=0, profit_threshold=0, max_trades_per_day=3)
    # Plot the profit obtained
    plt.figure(figsize=(15, 6))
    plt.plot(HF_trade_dam.iloc[:, 4:5].values, marker='o', linestyle='-', color='b', label='Profit')
    plt.title('Profit Obtained from High Frequency Strategy (TS3-DAM)')
    plt.xlabel('Index')
    plt.ylabel('Profit')
    plt.grid(True)
    plt.legend()
    plt.show()
    
    
    
    


