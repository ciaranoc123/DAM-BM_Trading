import pandas as pd
import numpy as np
from pulp import LpMaximize, LpProblem, LpVariable, lpSum, LpStatus

# Load DAM data
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

# Load BM data
dat_BM = pd.read_csv("/home/ciaran/QRA&Q-Ave(QR&CP)/BM_QRA/QR/rf_Q_1-12.csv")
dat_BM = pd.DataFrame(dat_BM)

# Create quantile dataframes for BM
Q_BM = {}
for q in quantiles:
    column_names = [f"lag_{i}y_Forecast_{q}" for i in range(2, 18)]
    Q_BM[q] = dat_BM[column_names].dropna().stack().reset_index()
    Q_BM[q]["Price"] = Q_BM[q].iloc[:, 2]

# Create a dataframe 'Y_r_BM' with real price data
column_names = [f"lag_{i}y" for i in range(2, 18)]
Y_r_BM = dat_BM[column_names].dropna().stack().reset_index()
Y_r_BM["Price"] = Y_r_BM.iloc[:, 2]

battery_efficiency_charge = 0.80
battery_efficiency_discharge = 0.98
Total_Daily_Volume = 6
max_battery_capacity = 1
min_battery_capacity = 0
ramp_rate_charge = 1
ramp_rate_discharge = 1

time_periods = range(0, len(Y_r_DAM), 48)
quantile_pairs = [(0.5, 0.5)]
total_profits = {pair: 0 for pair in quantile_pairs}
initial_charge = min_battery_capacity

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

        p_max_DAM = Q_DAM[alpha_key]["Price"].iloc[start_idx:end_idx].values
        p_min_DAM = Q_DAM[complement_alpha_key]["Price"].iloc[start_idx:end_idx].values

        # Optimize for DAM
        prob_dam = LpProblem("DAM_Quantile_Strategy", LpMaximize)
        buy_action_dam = LpVariable.dicts("Buy_Action_DAM", range(48), cat='Binary')
        sell_action_dam = LpVariable.dicts("Sell_Action_DAM", range(48), cat='Binary')
        buy_dam = LpVariable.dicts("Buy_DAM", range(48), lowBound=0, cat='Continuous')
        sell_dam = LpVariable.dicts("Sell_DAM", range(48), lowBound=0, cat='Continuous')
        charge_dam = LpVariable.dicts("Charge_DAM", range(49), lowBound=min_battery_capacity, upBound=max_battery_capacity, cat='Continuous')

        prob_dam += charge_dam[0] == current_charge, "Initial_Charge_DAM"
        prob_dam += lpSum([buy_dam[t] for t in range(48)]) + lpSum([sell_dam[t] for t in range(48)]) == Total_Daily_Volume, "Total_Daily_Volume_DAM"

        for t in range(48):
            prob_dam += buy_action_dam[t] + sell_action_dam[t] <= 1, f"One_Action_Per_Hour_DAM_{t}"
            prob_dam += buy_dam[t] <= ramp_rate_charge * buy_action_dam[t], f"Buy_Action_Link_DAM_{t}"
            prob_dam += sell_dam[t] <= ramp_rate_discharge * sell_action_dam[t], f"Sell_Action_Link_DAM_{t}"
            prob_dam += charge_dam[t + 1] == charge_dam[t] + buy_dam[t] - sell_dam[t], f"Charge_Update_DAM_{t}"
            prob_dam += charge_dam[t] <= max_battery_capacity, f"Max_Capacity_DAM_{t}"
            prob_dam += charge_dam[t] >= min_battery_capacity, f"Min_Capacity_DAM_{t}"
            prob_dam += buy_dam[t] <= max_battery_capacity - charge_dam[t], f"Charging_Constraint_DAM_{t}"
            prob_dam += buy_dam[t] <= ramp_rate_charge, f"Charging_Ramp_Rate_Constraint_DAM_{t}"
            prob_dam += sell_dam[t] <= charge_dam[t] - min_battery_capacity, f"Discharging_Constraint_DAM_{t}"
            prob_dam += sell_dam[t] <= ramp_rate_discharge, f"Discharging_Ramp_Rate_Constraint_DAM_{t}"

        for t in range(1,48):  
            prob_dam += buy_action_dam[t-1] + sell_action_dam[t-1] + buy_action_dam[t] + sell_action_dam[t] <= 1, f"Sequential_Action_Constraint_{t}"

        prob_dam += lpSum([(battery_efficiency_discharge * p_max_DAM[t] * sell_dam[t]) - ((p_min_DAM[t] / battery_efficiency_charge) * buy_dam[t]) for t in range(48)])
        prob_dam.solve()

        if LpStatus[prob_dam.status] == 'Optimal':
            buy_times_dam = [t for t in range(48) if buy_dam[t].varValue > 0]
            sell_times_dam = [t for t in range(48) if sell_dam[t].varValue > 0]

            used_hours_dam = set()
            valid_trades_dam = []
            for bt in buy_times_dam:
                if bt in used_hours_dam:
                    continue
                for st in sell_times_dam:
                    if st in used_hours_dam or bt >= st:
                        continue

                    buy_price_dam = p_min_DAM[bt]
                    sell_price_dam = p_max_DAM[st]
                    expected_profit_dam = (battery_efficiency_discharge * sell_price_dam * sell_dam[st].varValue) - ((buy_price_dam * buy_dam[bt].varValue) / battery_efficiency_charge)

                    if expected_profit_dam > 0:
                        valid_trades_dam.append((bt, st))
                        used_hours_dam.add(bt)
                        used_hours_dam.add(st)
                        break

            for bt, st in valid_trades_dam:
                buy_price_dam = Y_r_DAM.iloc[start_idx + bt]["Price"]
                sell_price_dam = Y_r_DAM.iloc[start_idx + st]["Price"]
                buy_amount_dam = buy_dam[bt].varValue
                sell_amount_dam = sell_dam[st].varValue
                profit_dam = (battery_efficiency_discharge * sell_price_dam * sell_amount_dam) - ((buy_price_dam * buy_amount_dam) / battery_efficiency_charge)
                results.append(profit_dam)
                print(f"Day {start_idx // 48 + 1}, Alpha {alpha}-{complement_alpha}: Buy at {bt} ({buy_amount_dam} MW), Sell at {st} ({sell_amount_dam} MW), Profit: {profit_dam}")

        # Set the DAM positions as constraints for BM
        dam_buy_hours = {bt: buy_dam[bt].varValue for bt in buy_times_dam}
        dam_sell_hours = {st: sell_dam[st].varValue for st in sell_times_dam}

        p_max_BM = Q_BM[alpha_key]["Price"].iloc[start_idx:end_idx].values
        p_min_BM = Q_BM[complement_alpha_key]["Price"].iloc[start_idx:end_idx].values

        # Optimize for BM
        prob_bm = LpProblem("BM_Quantile_Strategy", LpMaximize)
        buy_action_bm = LpVariable.dicts("Buy_Action_BM", range(48), cat='Binary')
        sell_action_bm = LpVariable.dicts("Sell_Action_BM", range(48), cat='Binary')
        buy_bm = LpVariable.dicts("Buy_BM", range(48), lowBound=0, cat='Continuous')
        sell_bm = LpVariable.dicts("Sell_BM", range(48), lowBound=0, cat='Continuous')
        charge_bm = LpVariable.dicts("Charge_BM", range(49), lowBound=min_battery_capacity, upBound=max_battery_capacity, cat='Continuous')

        prob_bm += charge_bm[0] == current_charge, "Initial_Charge_BM"
        prob_bm += lpSum([buy_bm[t] for t in range(48)]) + lpSum([sell_bm[t] for t in range(48)]) == Total_Daily_Volume, "Total_Daily_Volume_BM"

        for t in range(48):
            prob_bm += buy_action_bm[t] + sell_action_bm[t] <= 1, f"One_Action_Per_Hour_BM_{t}"
            prob_bm += buy_bm[t] <= ramp_rate_charge * buy_action_bm[t], f"Buy_Action_Link_BM_{t}"
            prob_bm += sell_bm[t] <= ramp_rate_discharge * sell_action_bm[t], f"Sell_Action_Link_BM_{t}"
            prob_bm += charge_bm[t + 1] == charge_bm[t] + buy_bm[t] - sell_bm[t], f"Charge_Update_BM_{t}"
            prob_bm += charge_bm[t] <= max_battery_capacity, f"Max_Capacity_BM_{t}"
            prob_bm += charge_bm[t] >= min_battery_capacity, f"Min_Capacity_BM_{t}"
            prob_bm += buy_bm[t] <= max_battery_capacity - charge_bm[t], f"Charging_Constraint_BM_{t}"
            prob_bm += buy_bm[t] <= ramp_rate_charge, f"Charging_Ramp_Rate_Constraint_BM_{t}"
            prob_bm += sell_bm[t] <= charge_bm[t] - min_battery_capacity, f"Discharging_Constraint_BM_{t}"
            prob_bm += sell_bm[t] <= ramp_rate_discharge, f"Discharging_Ramp_Rate_Constraint_BM_{t}"
            prob_bm += buy_bm[t] == dam_buy_hours.get(t, 0), f"DAM_Buy_Constraint_{t}"
            prob_bm += sell_bm[t] == dam_sell_hours.get(t, 0), f"DAM_Sell_Constraint_{t}"

        for t_buy in range(48):
            for t_sell in range(t_buy + 1, min(t_buy + 17, 48)):
                prob_bm += sell_action_bm[t_sell] <= buy_action_bm[t_buy], f"Max_Time_Diff_{t_buy}_{t_sell}"
                
        prob_bm += lpSum([(battery_efficiency_discharge * p_max_BM[t] * sell_bm[t]) - ((p_min_BM[t] / battery_efficiency_charge) * buy_bm[t]) for t in range(48)])
        prob_bm.solve()

        if LpStatus[prob_bm.status] == 'Optimal':
            buy_times_bm = [t for t in range(48) if buy_bm[t].varValue > 0]
            sell_times_bm = [t for t in range(48) if sell_bm[t].varValue > 0]

            used_hours_bm = set()
            valid_trades_bm = []
            for bt in buy_times_bm:
                if bt in used_hours_bm:
                    continue
                for st in sell_times_bm:
                    if st in used_hours_bm or bt >= st:
                        continue

                    buy_price_bm = p_min_BM[bt]
                    sell_price_bm = p_max_BM[st]
                    expected_profit_bm = (battery_efficiency_discharge * sell_price_bm * sell_bm[st].varValue) - ((buy_price_bm * buy_bm[bt].varValue) / battery_efficiency_charge)

                    if expected_profit_bm > 0:
                        valid_trades_bm.append((bt, st))
                        used_hours_bm.add(bt)
                        used_hours_bm.add(st)
                        break

            for bt, st in valid_trades_bm:
                buy_price_bm = Y_r_BM.iloc[start_idx + bt]["Price"]
                sell_price_bm = Y_r_BM.iloc[start_idx + st]["Price"]
                buy_amount_bm = buy_bm[bt].varValue
                sell_amount_bm = sell_bm[st].varValue
                profit_bm = (battery_efficiency_discharge * sell_price_bm * sell_amount_bm) - ((buy_price_bm * buy_amount_bm) / battery_efficiency_charge)
                results.append(profit_bm)
                print(f"Day {start_idx // 48 + 1}, Alpha {alpha}-{complement_alpha}: Buy at {bt} ({buy_amount_bm} MW), Sell at {st} ({sell_amount_bm} MW), Profit: {profit_bm}")

        current_charge = charge_dam[48].varValue

    total_profit = sum(results)
    total_profits[(alpha, complement_alpha)] += total_profit
    print(f"Total profit for Alpha {alpha}-{complement_alpha}: {total_profit}")

print("Total profits for all quantile pairs:")
for pair, profit in total_profits.items():
    print(f"Quantile pair {pair}: {profit}")


    
    
    
