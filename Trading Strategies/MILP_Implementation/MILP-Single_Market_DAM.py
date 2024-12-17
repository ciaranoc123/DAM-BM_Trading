import pandas as pd
import numpy as np
import datetime as dt
from pulp import LpMaximize, LpProblem, LpVariable, lpSum, LpStatus
from matplotlib.pyplot import figure

# Load DAM data
date_format_dam = "%d/%m/%Y %H:%M"
date_parse_dam = lambda date: dt.datetime.strptime(date, date_format_dam)
dat1 = pd.read_csv("/home/ciaran/QRA&Q-Ave(QR&CP)/DAM_QRA/QR/rf_Q_DAM_1-12.csv")
dat1 = pd.DataFrame(dat1)

quantiles = [10, 30, 50, 70, 90]
Q_DAM = {}
for q in quantiles:
    column_names = [f"EURPrices+{i}_Forecast_{q}" for i in range(24)]
    Q_DAM[q] = dat1[column_names].dropna().stack().reset_index()
    Q_DAM[q]["Price"] = Q_DAM[q].iloc[:, 2]
    Q_DAM[q] = Q_DAM[q].reindex(Q_DAM[q].index.repeat(2)).reset_index(drop=True)

column_names_dam = [f"EURPrices+{i}" for i in range(24)]
Y_r_DAM = dat1[column_names_dam].dropna().stack().reset_index()
Y_r_DAM = Y_r_DAM.reindex(Y_r_DAM.index.repeat(2)).reset_index(drop=True)
Y_r_DAM["Price"] = Y_r_DAM.iloc[:, 2]

# Set battery parameters
battery_efficiency_charge = 0.80
battery_efficiency_discharge = 0.98
Total_Daily_Volume = 6
max_battery_capacity = 1
min_battery_capacity = 0  
ramp_rate_charge = 1
ramp_rate_discharge = 1

# Define time periods (48-hour windows)
time_periods = range(0, len(Y_r_DAM), 48)

# Define quantile pairs
quantile_pairs = [(0.5, 0.5), (0.3, 0.7), (0.1, 0.9)]

# Initialize dictionaries to store total profits for each quantile pair
total_profits = {pair: 0 for pair in quantile_pairs}

# Initialize battery charge at the start of the first day
initial_charge = min_battery_capacity  

# Loop through each quantile pair
for alpha, complement_alpha in quantile_pairs:
    results = []
    current_charge = initial_charge

    for start_idx in time_periods:
        end_idx = start_idx + 48

        if end_idx > len(Y_r_DAM):
            break

        print(f"Initial charge: {current_charge}")

        alpha_key = int(alpha * 100)
        complement_alpha_key = int(complement_alpha * 100)

        if alpha_key not in Q_DAM or complement_alpha_key not in Q_DAM:
            continue

        p_max = Q_DAM[alpha_key]["Price"].iloc[start_idx:end_idx].values
        p_min = Q_DAM[complement_alpha_key]["Price"].iloc[start_idx:end_idx].values

        prob = LpProblem("Multiple_Trades_Quantile_Strategy", LpMaximize)
        buy_action = LpVariable.dicts("Buy_Action", range(48), cat='Binary')
        sell_action = LpVariable.dicts("Sell_Action", range(48), cat='Binary')
        buy = LpVariable.dicts("Buy", range(48), lowBound=0, cat='Continuous')
        sell = LpVariable.dicts("Sell", range(48), lowBound=0, cat='Continuous')
        charge = LpVariable.dicts("Charge", range(49), lowBound=min_battery_capacity, upBound=max_battery_capacity, cat='Continuous')

        # Initial battery charge constraint
        prob += charge[0] == current_charge, "Initial_Charge"

        # Define the total daily volume constraint
        prob += lpSum([buy[t] for t in range(48)]) + lpSum([sell[t] for t in range(48)]) == Total_Daily_Volume, "Total_Daily_Volume"

        # Ensure that each hour has at most one buy or sell action
        for t in range(48):
            prob += buy_action[t] + sell_action[t] <= 1, f"One_Action_Per_Hour_{t}"
            
        # Ensure t < T (48). no Sequential buy/sell
        for t in range(1,48):  
            prob += buy_action[t-1] + sell_action[t-1] + buy_action[t] + sell_action[t] <= 1, f"Sequential_Action_Constraint_{t}"


        # Link action variables to buy and sell variables
        for t in range(48):
            prob += buy[t] <= ramp_rate_charge * buy_action[t], f"Buy_Action_Link_{t}"
            prob += sell[t] <= ramp_rate_discharge * sell_action[t], f"Sell_Action_Link_{t}"

        # Battery charge constraints
        for t in range(48):
            prob += charge[t + 1] == charge[t] + buy[t] - sell[t], f"Charge_Update_{t}"
            prob += charge[t] <= max_battery_capacity, f"Max_Capacity_{t}"
            prob += charge[t] >= min_battery_capacity, f"Min_Capacity_{t}"

            # Charging constraints
            prob += buy[t] <= max_battery_capacity - charge[t], f"Charging_Constraint_{t}"
            prob += buy[t] <= ramp_rate_charge, f"Charging_Ramp_Rate_Constraint_{t}"

            # Discharging constraints
            prob += sell[t] <= charge[t] - min_battery_capacity, f"Discharging_Constraint_{t}"
            prob += sell[t] <= ramp_rate_discharge, f"Discharging_Ramp_Rate_Constraint_{t}"

        # Define the objective function (maximize profit)
        prob += lpSum([(battery_efficiency_discharge * p_max[t] * sell[t]) - ((p_min[t] / battery_efficiency_charge) * buy[t]) for t in range(48)])

        # Solve the problem
        prob.solve()

        # Check if the solution is optimal
        if LpStatus[prob.status] == 'Optimal':
            # Extract buy and sell times and amounts
            buy_times = [t for t in range(48) if buy[t].varValue > 0]
            sell_times = [t for t in range(48) if sell[t].varValue > 0]

            # Ensure unique buy-sell pairs
            used_hours = set()
            valid_trades = []
            for bt in buy_times:
                if bt in used_hours:
                    continue
                for st in sell_times:
                    if st in used_hours or bt >= st:
                        continue

                    # Calculate expected profit
                    buy_price = p_min[bt]
                    sell_price = p_max[st]
                    expected_profit = (battery_efficiency_discharge * sell_price * sell[st].varValue) - ((buy_price * buy[bt].varValue) / battery_efficiency_charge)

                    # Only consider the trade if expected profit is greater than 0
                    if expected_profit > 0:
                        valid_trades.append((bt, st))
                        used_hours.add(bt)
                        used_hours.add(st)
                        break  # Only one valid sell per buy to ensure no repeated actions

            for bt, st in valid_trades:
                # Calculate profit using real prices
                buy_price = Y_r_DAM.iloc[start_idx + bt]["Price"]
                sell_price = Y_r_DAM.iloc[start_idx + st]["Price"]
                buy_amount = buy[bt].varValue
                sell_amount = sell[st].varValue
                profit = (battery_efficiency_discharge * sell_price * sell_amount) - ((buy_price * buy_amount) / battery_efficiency_charge)
                results.append(profit)
                print(f"Day {start_idx // 48 + 1}, Alpha {alpha}-{complement_alpha}: Buy at {bt} ({buy_amount} MW), Sell at {st} ({sell_amount} MW), Profit: {profit}")

            # Update the current charge for the next day
            current_charge = charge[48].varValue

    # Calculate the sum of profits for the current quantile pair
    total_profit = sum(results)
    total_profits[(alpha, complement_alpha)] = total_profit
    print(f"Total Profit for Alpha {alpha}-{complement_alpha}: {total_profit}\n")

# Print the total profits for all quantile pairs
for pair, profit in total_profits.items():
    print(f"Total Profit for Alpha {pair[0]}-{pair[1]}: {profit}")
    
    
    
    
    
