import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt




def electricity_strategy_Multi_Trade_DAM(df, Q_A_Preds, Q_B_Preds, eff_1, eff_2, profit_threshold=0, max_trades_per_day=3):
    prices = []
    day_index = df['level_0'].unique()

    for day in day_index:
        current_df = df[df['level_0'] == day]
        current_Q_A = Q_A_Preds[Q_A_Preds['level_0'] == day]
        current_Q_B = Q_B_Preds[Q_B_Preds['level_0'] == day]
        
        trades_today = 0
        
        # Find the maximum price for that day          
        max_price_index = current_Q_A['Price'].idxmax()
        prices_before_max = Q_B_Preds[(Q_B_Preds['level_0'] == day) & (Q_B_Preds.index < max_price_index)]
        min_price_index = current_Q_B['Price'].idxmin()
        prices_after_min = Q_A_Preds[(Q_A_Preds['level_0'] == day) & (Q_A_Preds.index > min_price_index)]

        min_price_index1 = None
        max_price_index1 = None
        
        if len(prices_before_max) > 0:
            min_price_index1 = prices_before_max['Price'].idxmin()
        if len(prices_after_min) > 0:
            max_price_index1 = prices_after_min['Price'].idxmax()

        if max_price_index is not None and min_price_index1 is not None and max_price_index1 is not None:
            if (current_Q_A.loc[max_price_index, 'Price'] - current_Q_B.loc[min_price_index1, 'Price']) > (current_Q_A.loc[max_price_index1, 'Price'] - current_Q_B.loc[min_price_index, 'Price']):
                T1_max_price_index = max_price_index
                T1_min_price_index = min_price_index1
            else:
                T1_max_price_index = max_price_index1
                T1_min_price_index = min_price_index
        elif max_price_index is not None and min_price_index1 is not None:
            T1_max_price_index = max_price_index
            T1_min_price_index = min_price_index1
        elif max_price_index1 is not None and min_price_index is not None:
            T1_max_price_index = max_price_index1
            T1_min_price_index = min_price_index

        # Calculate expected profit using predictions for T1
        if T1_max_price_index in current_Q_A.index and T1_min_price_index in current_Q_B.index and trades_today<max_trades_per_day:
            Exp_profit_T1 = ((current_Q_A.loc[T1_max_price_index, 'Price']) * eff_1) - ((current_Q_B.loc[T1_min_price_index, 'Price']) / eff_2)
            if Exp_profit_T1 >= (profit_threshold-10):
                profit = ((current_df.loc[T1_max_price_index, 'Price']) * eff_1) - ((current_df.loc[T1_min_price_index, 'Price']) / eff_2)
                prices.append((T1_min_price_index, current_df.loc[T1_min_price_index, 'Price'], T1_max_price_index, current_df.loc[T1_max_price_index, 'Price'], profit))
                trades_today += 1
        
        # Repeat similar logic for T2
        current_df_before_min = current_df[current_df.index < T1_min_price_index]
        current_Q_A_before_min = Q_A_Preds[(Q_A_Preds['level_0'] == day) & (Q_A_Preds.index < T1_min_price_index)]
        max_price_index_before_min = None
        if not current_Q_A_before_min.empty:
            max_price_index_before_min = current_Q_A_before_min['Price'].idxmax()
        prices_before_max = Q_B_Preds[(Q_B_Preds['level_0'] == day) & (Q_B_Preds.index < max_price_index_before_min)]
        min_price_index2 = None
        max_price_index2 = None
        if len(prices_before_max) > 0:
            min_price_index2 = prices_before_max['Price'].idxmin()

        T2_max_price_index = None
        T2_min_price_index = None
        if max_price_index_before_min is not None and min_price_index2 is not None and trades_today<max_trades_per_day:
            T2_max_price_index = max_price_index_before_min
            T2_min_price_index = min_price_index2
            Exp_profit_T2 = ((current_Q_A.loc[T2_max_price_index, 'Price']) * eff_1) - ((current_Q_B.loc[T2_min_price_index, 'Price']) / eff_2)
            if Exp_profit_T2 >= profit_threshold:
                profit = ((current_df.loc[T2_max_price_index, 'Price']) * eff_1) - ((current_df.loc[T2_min_price_index, 'Price']) / eff_2)
                prices.append((T2_min_price_index, current_df.loc[T2_min_price_index, 'Price'], T2_max_price_index, current_df.loc[T2_max_price_index, 'Price'], profit))
                trades_today += 1
        
        # Repeat similar logic for T3
        current_df_after_T1max = current_df[current_df.index > T1_max_price_index]
        current_Q_A_after_T1max = Q_A_Preds[(Q_A_Preds['level_0'] == day) & (Q_A_Preds.index > T1_max_price_index)]
        max_price_index_after_T1max = None
        if not current_Q_A_after_T1max.empty:
            max_price_index_after_T1max = current_Q_A_after_T1max['Price'].idxmax()
        T3_prices_before_max = Q_B_Preds[(Q_B_Preds['level_0'] == day) & (Q_B_Preds.index < max_price_index_after_T1max) & (Q_B_Preds.index > T1_max_price_index)]
        min_price_index3 = None
        max_price_index3 = None
        if len(T3_prices_before_max) > 0:
            min_price_index3 = T3_prices_before_max['Price'].idxmin()
        
        T3_max_price_index = None
        T3_min_price_index = None
        if max_price_index_after_T1max is not None and min_price_index3 is not None and trades_today<max_trades_per_day:
            T3_max_price_index = max_price_index_after_T1max
            T3_min_price_index = min_price_index3
            Exp_profit_T3 = ((current_Q_A.loc[T3_max_price_index, 'Price']) * eff_1) - ((current_Q_B.loc[T3_min_price_index, 'Price']) / eff_2)
            if Exp_profit_T3 >= profit_threshold:
                profit = ((current_df.loc[T3_max_price_index, 'Price']) * eff_1) - ((current_df.loc[T3_min_price_index, 'Price']) / eff_2)
                prices.append((T3_min_price_index, current_df.loc[T3_min_price_index, 'Price'], T3_max_price_index, current_df.loc[T3_max_price_index, 'Price'], profit))
                trades_today += 1

    return pd.DataFrame(prices, columns=['minPriceIndex', 'minPrice', 'maxPriceIndex', 'maxPrice', 'profit'])
    
    
    
    
    
    
def electricity_strategy_Multi_Trade_BM(df, Q_A_Preds, Q_B_Preds, eff_1, eff_2, profit_threshold=0, max_trades_per_day=3, ramp_rate=0.5):
    prices = []
    day_index = df['level_0'].unique()

    for day in day_index:
        current_df = df[df['level_0'] == day]
        current_Q_A = Q_A_Preds[Q_A_Preds['level_0'] == day]
        current_Q_B = Q_B_Preds[Q_B_Preds['level_0'] == day]
        
        trades_today = 0
        
        # Find the maximum price for that day          
        max_price_index = current_Q_A['Price'].idxmax()
        prices_before_max = Q_B_Preds[(Q_B_Preds['level_0'] == day) & (Q_B_Preds.index < max_price_index)]
        min_price_index = current_Q_B['Price'].idxmin()
        prices_after_min = Q_A_Preds[(Q_A_Preds['level_0'] == day) & (Q_A_Preds.index > min_price_index)]

        min_price_index1 = None
        max_price_index1 = None
        
        if len(prices_before_max) > 0:
            min_price_index1 = prices_before_max['Price'].idxmin()
        if len(prices_after_min) > 0:
            max_price_index1 = prices_after_min['Price'].idxmax()

        if max_price_index is not None and min_price_index1 is not None and max_price_index1 is not None:
            if (current_Q_A.loc[max_price_index, 'Price'] - current_Q_B.loc[min_price_index1, 'Price']) > (current_Q_A.loc[max_price_index1, 'Price'] - current_Q_B.loc[min_price_index, 'Price']):
                T1_max_price_index = max_price_index
                T1_min_price_index = min_price_index1
            else:
                T1_max_price_index = max_price_index1
                T1_min_price_index = min_price_index
        elif max_price_index is not None and min_price_index1 is not None:
            T1_max_price_index = max_price_index
            T1_min_price_index = min_price_index1
        elif max_price_index1 is not None and min_price_index is not None:
            T1_max_price_index = max_price_index1
            T1_min_price_index = min_price_index

        # Calculate expected profit using predictions for T1
        if T1_max_price_index in current_Q_A.index and T1_min_price_index in current_Q_B.index and trades_today<max_trades_per_day:
            Exp_profit_T1 = ((current_Q_A.loc[T1_max_price_index, 'Price']) * eff_1) - ((current_Q_B.loc[T1_min_price_index, 'Price']) / eff_2)  
            if Exp_profit_T1 >= (profit_threshold):
                profit = ((current_df.loc[T1_max_price_index, 'Price']) * ramp_rate  * eff_1) - (((current_df.loc[T1_min_price_index, 'Price']) / eff_2) * ramp_rate )
                prices.append((T1_min_price_index, current_df.loc[T1_min_price_index, 'Price'], T1_max_price_index, current_df.loc[T1_max_price_index, 'Price'], profit))
                trades_today += 1
        
        # Repeat similar logic for T2
        current_df_before_min = current_df[current_df.index < T1_min_price_index]
        current_Q_A_before_min = Q_A_Preds[(Q_A_Preds['level_0'] == day) & (Q_A_Preds.index < T1_min_price_index)]
        max_price_index_before_min = None
        if not current_Q_A_before_min.empty:
            max_price_index_before_min = current_Q_A_before_min['Price'].idxmax()
        prices_before_max = Q_B_Preds[(Q_B_Preds['level_0'] == day) & (Q_B_Preds.index < max_price_index_before_min)]
        min_price_index2 = None
        max_price_index2 = None
        if len(prices_before_max) > 0:
            min_price_index2 = prices_before_max['Price'].idxmin()

        T2_max_price_index = None
        T2_min_price_index = None
        if max_price_index_before_min is not None and min_price_index2 is not None and trades_today<max_trades_per_day:
            T2_max_price_index = max_price_index_before_min
            T2_min_price_index = min_price_index2
            Exp_profit_T2 = ((current_Q_A.loc[T2_max_price_index, 'Price']) * ramp_rate  * eff_1) - ((current_Q_B.loc[T2_min_price_index, 'Price']) / eff_2) * ramp_rate 
            if Exp_profit_T2 >= profit_threshold:
                profit = ((current_df.loc[T2_max_price_index, 'Price']) * ramp_rate  * eff_1) - (((current_df.loc[T2_min_price_index, 'Price']) / eff_2) * ramp_rate )
                prices.append((T2_min_price_index, current_df.loc[T2_min_price_index, 'Price'], T2_max_price_index, current_df.loc[T2_max_price_index, 'Price'], profit))
                trades_today += 1
        
        # Repeat similar logic for T3
        current_df_after_T1max = current_df[current_df.index > T1_max_price_index]
        current_Q_A_after_T1max = Q_A_Preds[(Q_A_Preds['level_0'] == day) & (Q_A_Preds.index > T1_max_price_index)]
        max_price_index_after_T1max = None
        if not current_Q_A_after_T1max.empty:
            max_price_index_after_T1max = current_Q_A_after_T1max['Price'].idxmax()
        T3_prices_before_max = Q_B_Preds[(Q_B_Preds['level_0'] == day) & (Q_B_Preds.index < max_price_index_after_T1max) & (Q_B_Preds.index > T1_max_price_index)]
        min_price_index3 = None
        max_price_index3 = None
        if len(T3_prices_before_max) > 0:
            min_price_index3 = T3_prices_before_max['Price'].idxmin()
        
        T3_max_price_index = None
        T3_min_price_index = None
        if max_price_index_after_T1max is not None and min_price_index3 is not None and trades_today<max_trades_per_day:
            T3_max_price_index = max_price_index_after_T1max
            T3_min_price_index = min_price_index3
            Exp_profit_T3 = ((current_Q_A.loc[T3_max_price_index, 'Price']) * eff_1) - ((current_Q_B.loc[T3_min_price_index, 'Price']) / eff_2)
            if Exp_profit_T3 >= profit_threshold:
                profit = ((current_df.loc[T3_max_price_index, 'Price']) * ramp_rate  * eff_1) - (((current_df.loc[T3_min_price_index, 'Price']) / eff_2) * ramp_rate )
                prices.append((T3_min_price_index, current_df.loc[T3_min_price_index, 'Price'], T3_max_price_index, current_df.loc[T3_max_price_index, 'Price'], profit))
                trades_today += 1

    return pd.DataFrame(prices, columns=['minPriceIndex', 'minPrice', 'maxPriceIndex', 'maxPrice', 'profit'])

         
         
         
         
            
def calculate_trading_results_DAM_MT(Y_r, Q_10, Q_30, Q_50, Q_70, Q_90):
    eff_1 = 0.8
    eff_2 = 0.98
    
    r_dam_50_50 = electricity_strategy_Multi_Trade_DAM(df=Y_r, Q_A_Preds=Q_50, Q_B_Preds=Q_50, eff_1=eff_1, eff_2=eff_2, profit_threshold=0, max_trades_per_day=3)
    r_dam_10_30 = electricity_strategy_Multi_Trade_DAM(df=Y_r, Q_A_Preds=Q_10, Q_B_Preds=Q_30, eff_1=eff_1, eff_2=eff_2, profit_threshold=0, max_trades_per_day=3)
    r_dam_30_50 = electricity_strategy_Multi_Trade_DAM(df=Y_r, Q_A_Preds=Q_30, Q_B_Preds=Q_50, eff_1=eff_1, eff_2=eff_2, profit_threshold=0, max_trades_per_day=3)
    r_dam_50_70 = electricity_strategy_Multi_Trade_DAM(df=Y_r, Q_A_Preds=Q_50, Q_B_Preds=Q_70, eff_1=eff_1, eff_2=eff_2, profit_threshold=0, max_trades_per_day=3)
    r_dam_70_90 = electricity_strategy_Multi_Trade_DAM(df=Y_r, Q_A_Preds=Q_70, Q_B_Preds=Q_90, eff_1=eff_1, eff_2=eff_2, profit_threshold=0, max_trades_per_day=3)
    r_dam_30_70 = electricity_strategy_Multi_Trade_DAM(df=Y_r, Q_A_Preds=Q_30, Q_B_Preds=Q_70, eff_1=eff_1, eff_2=eff_2, profit_threshold=0, max_trades_per_day=3)
    r_dam_10_90 = electricity_strategy_Multi_Trade_DAM(df=Y_r, Q_A_Preds=Q_10, Q_B_Preds=Q_90, eff_1=eff_1, eff_2=eff_2, profit_threshold=0, max_trades_per_day=3)
    
    PF_DAM = electricity_strategy_Multi_Trade_DAM(df=Y_r, Q_A_Preds=Y_r, Q_B_Preds=Y_r, eff_1=eff_1, eff_2=eff_2, profit_threshold=0, max_trades_per_day=3)
    
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

def print_results_DAM(results):
    print("Trading results for different quantile pairs in the DAM:")
    for key, value in results.items():
        if key.startswith('r_dam'):
            quantiles = key.split('_')[2:]
            label = f"{quantiles[0]}-{quantiles[1]}"
            print(f"Total sum for trading quantile {label} pair in the DAM is: {value}")
        elif key == 'PF_DAM':
            print(f"Total sum for the Perfect Forecast pair in the DAM is: {value}")
            
            





def calculate_bm_trading_results_MT(Y_r, Q_10, Q_30, Q_50, Q_70, Q_90):
    eff_1 = 0.8
    eff_2 = 0.98
    
    r_bm_50_50 = electricity_strategy_Multi_Trade_BM(df=Y_r, Q_A_Preds=Q_50, Q_B_Preds=Q_50, eff_1=eff_1, eff_2=eff_2, profit_threshold=0, max_trades_per_day=2, ramp_rate=0.5)
    r_bm_10_30 = electricity_strategy_Multi_Trade_BM(df=Y_r, Q_A_Preds=Q_10, Q_B_Preds=Q_30, eff_1=eff_1, eff_2=eff_2, profit_threshold=0, max_trades_per_day=2, ramp_rate=0.5)
    r_bm_30_50 = electricity_strategy_Multi_Trade_BM(df=Y_r, Q_A_Preds=Q_30, Q_B_Preds=Q_50, eff_1=eff_1, eff_2=eff_2, profit_threshold=0, max_trades_per_day=2, ramp_rate=0.5)
    r_bm_50_70 = electricity_strategy_Multi_Trade_BM(df=Y_r, Q_A_Preds=Q_50, Q_B_Preds=Q_70, eff_1=eff_1, eff_2=eff_2, profit_threshold=0, max_trades_per_day=2, ramp_rate=0.5)
    r_bm_70_90 = electricity_strategy_Multi_Trade_BM(df=Y_r, Q_A_Preds=Q_70, Q_B_Preds=Q_90, eff_1=eff_1, eff_2=eff_2, profit_threshold=0, max_trades_per_day=2, ramp_rate=0.5)
    r_bm_30_70 = electricity_strategy_Multi_Trade_BM(df=Y_r, Q_A_Preds=Q_30, Q_B_Preds=Q_70, eff_1=eff_1, eff_2=eff_2, profit_threshold=0, max_trades_per_day=2, ramp_rate=0.5)
    r_bm_10_90 = electricity_strategy_Multi_Trade_BM(df=Y_r, Q_A_Preds=Q_10, Q_B_Preds=Q_90, eff_1=eff_1, eff_2=eff_2, profit_threshold=0, max_trades_per_day=2, ramp_rate=0.5)
    
    PF_BM = electricity_strategy_Multi_Trade_BM(df=Y_r, Q_A_Preds=Y_r, Q_B_Preds=Y_r, eff_1=eff_1, eff_2=eff_2, profit_threshold=0, max_trades_per_day=2, ramp_rate=0.5)
    
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

def print_results_BM(results):
    print("Trading results for different quantile pairs in the BM:")
    for key, value in results.items():
        if key.startswith('r_bm'):
            quantiles = key.split('_')[2:]
            label = f"{quantiles[0]}-{quantiles[1]}"
            print(f"Total sum for trading quantile {label} pair in the BM is: {value}")
        elif key == 'PF_BM':
            print(f"Total sum for the Perfect Forecast pair in the BM is: {value}")
            
            
            
# Define the function to plot profit for a single trade in the BM strategy
def plot_profit_Multi_trade_BM(Y_r_bm, Q_A_Preds, Q_B_Preds, eff_1, eff_2, profit_threshold=0, max_trades_per_day=2, ramp_rate=0.5):
    # Run electricity strategy for BM
    Multi_trade_bm = electricity_strategy_Multi_Trade_BM(df=Y_r_bm, Q_A_Preds=Q_A_Preds, Q_B_Preds=Q_B_Preds, eff_1=eff_1, eff_2=eff_2, profit_threshold=0, max_trades_per_day=2, ramp_rate=0.5)
    # Plot the profit obtained
    plt.figure(figsize=(15, 6))
    plt.plot(Multi_trade_bm.iloc[:, 4:5].values, marker='o', linestyle='-', color='r', label='Profit')
    plt.title('Profit Obtained from Multi-Trade Strategy (TS2-BM)')
    plt.xlabel('Index')
    plt.ylabel('Profit')
    plt.grid(True)
    plt.legend()
    plt.show()

# Define the function to plot profit for a single trade in the BM strategy
def plot_profit_Multi_trade_DAM(Y_r, Q_A_Preds, Q_B_Preds, eff_1, eff_2, profit_threshold=0, max_trades_per_day=2):
    # Run electricity strategy for BM
    Multi_trade_dam = electricity_strategy_Multi_Trade_DAM(df=Y_r, Q_A_Preds=Q_A_Preds, Q_B_Preds=Q_B_Preds, eff_1=eff_1, eff_2=eff_2, profit_threshold=0, max_trades_per_day=2)
    # Plot the profit obtained
    plt.figure(figsize=(15, 6))
    plt.plot(Multi_trade_dam.iloc[:, 4:5].values, marker='o', linestyle='-', color='b', label='Profit')
    plt.title('Profit Obtained from Multi-Trade Strategy (TS2-DAM)')
    plt.xlabel('Index')
    plt.ylabel('Profit')
    plt.grid(True)
    plt.legend()
    plt.show()

            
            
