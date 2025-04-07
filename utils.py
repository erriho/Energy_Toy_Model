import numpy as np
import pandas as pd
import os
import csv

def fit_lognormal_from_percentiles(p25, p50, p75):
    mu_median = np.log(p50)
    z_75 = 0.6745
    sigma_75 = (np.log(p75) - mu_median) / z_75
    z_25 = -0.6745
    sigma_25 = (np.log(p25) - mu_median) / z_25

    sigma = (sigma_75 + sigma_25) / 2
    mu = mu_median

    return mu, sigma

def get_stationary_value(time_series, steps_used_to_compute = 50):
    if not isinstance(time_series, np.ndarray):
        print("Error: Input is not a NumPy array.")
    if steps_used_to_compute >= len(time_series):
        print("Error: Too many steps.")
    else:
        time_series_last_steps = time_series[-steps_used_to_compute:].copy()
        return np.average(time_series_last_steps)

###fixed costs management
def convert_fixed_cost(fixed_cost, timestep_duration_in_days = 1):
    return fixed_cost/365*timestep_duration_in_days #fixed costs given are in $/kW-year, should I consider it somehow?

def convert_fixed_cost_per_mwh(fixed_cost, capacity, timestep_duration_in_days = 1):
    single_iteration_fixed_cost = fixed_cost/365*timestep_duration_in_days
    return single_iteration_fixed_cost/capacity

###load, save, and update data

def load_from_plant_csv(database_path, data_to_load = ''):
    data_dict = {}
    try:
        with open(database_path, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader)
            value_index = header.index(data_to_load) 
            for row in reader:
                key = row[0]
                value = row[value_index]
                data_dict[key] = value
        return data_dict
    except FileNotFoundError:
        print(f"Error: File '{database_path}' not found.")
        return None

def initialize_metrics(plants, firms, t_max, p, k):
    metrics = {}
    metrics['prob_data'] = {}
    metrics['chosen_action_data'] = {}
    metrics['bid_data'] = {}
    metrics['plant_profits_data'] = {}
    metrics['firm_profits_data'] = {}
    for plant in plants:
        metrics['prob_data'][plant] = {}
        metrics['chosen_action_data'][plant] = {}
        metrics['bid_data'][plant] = {}
        metrics['plant_profits_data'][plant] = {}
        for t in range(t_max):
            metrics['prob_data'][plant][t] = np.copy(p[plant, :])
            metrics['chosen_action_data'][plant][t] = k[plant]
            metrics['bid_data'][plant][t] = 0
            metrics['plant_profits_data'][plant][t] = 0.0 
    for firm in firms:
        metrics['firm_profits_data'][firm] = {}
        for t in range(t_max):
            metrics['firm_profits_data'][firm][t] = 0.0
    return metrics

def update_metrics(plants, firms, t, p, k, bids, plant_profits, firm_profits, metrics):
    for plant in plants:
        metrics['prob_data'][plant][t] = np.copy(p[plant, :])
        metrics['plant_profits_data'][plant][t] = plant_profits[plant]
        metrics['chosen_action_data'][plant][t] = k[plant]
        metrics['bid_data'][plant][t] = bids[plant]
    for firm in firms:
        metrics['firm_profits_data'][firm][t] = firm_profits[firm]  
    return metrics

def prob_data_to_npy(t, plant, pdf_to_save, output_path):
    npy_filename = os.path.join(output_path, f'prob_data\plant_{plant}_pdf_at_time_{t}.npy')
    np.save(npy_filename, pdf_to_save)

def prob_data_load_from_npy(number_of_plants, t_max, output_path):
    prob_data = {}
    for plant in range(number_of_plants):
        prob_data[plant] = {}
        for t in range(t_max):
            npy_filename = os.path.join(output_path, f'prob_data\plant_{plant}_pdf_at_time_{t}.npy')
            prob_data[plant][t] = np.load(npy_filename)
    return prob_data

def save_metrics_to_csv(metrics, t_max, output_path, csv_filenames):
    print("...saving simulation results...")
    del(metrics['prob_data'])
    for metric in metrics.keys():
        csv_filename = os.path.join(output_path, f'{metric}.csv')
        csv_filenames[metric] = csv_filename
        #extracting data
        metric_data_dict = metrics[f'{metric}']
        #creating csv data for firm metrics
        if metric == 'firm_profits_data':
            firm_IDs = list(metric_data_dict.keys())
            #creating header
            columns = ['firm_ID'] + [str(t) for t in range(t_max+1)]
            #creating csv rows
            metric_data = []
            for firm in firm_IDs:
                row = [firm]
                for t in range(t_max+1):
                    row.append(metric_data_dict[firm].get(t,''))
                metric_data.append(row)
        #creating csv data for plant metrics
        else:
            plant_IDs = list(metric_data_dict.keys())
            columns = ['plant_ID'] + [str(t) for t in range(t_max+1)]
            metric_data = []
            for plant in plant_IDs:
                row = [plant]
                for t in range(t_max+1):
                    row.append(metric_data_dict[plant].get(t,''))
                metric_data.append(row)
        #saving csv file
        metric_df = pd.DataFrame(metric_data, columns = columns)
        metric_df.to_csv(csv_filename, index=False)
    print(f"Done!")
    return csv_filenames

def save_prices_to_csv(marginal_prices_through_time, t_max, output_path, csv_filenames):
    print('...saving prices timeseries in same folder...')
    csv_filename = os.path.join(output_path, 'prices.csv')
    csv_filenames['price_data'] = csv_filename
    timesteps = list(range(0, t_max+1))
    prices_df = pd.DataFrame([timesteps, marginal_prices_through_time])
    prices_df.columns = timesteps
    prices_df.index = [None, None]
    prices_df.to_csv(csv_filename, index=False, header=False)
    print('Done.')
    return csv_filenames

def load_prices_from_csv(csv_filename):
    try:
        with open(csv_filename, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            for i, row in enumerate(reader):
                if i == 1:
                    row_as_list = [float(val) for val in row]
                    break #stop once row is found.
            else: #if loop completes without break, row was not found
                print(f"Could not extract row 1.")
    except FileNotFoundError:
        print(f"File not found.")
    return row_as_list

def load_dict_data_from_csv(csv_filename, data_to_load = ''):
    data_dict = {}
    try:
        with open(csv_filename, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader) 
            for row in reader:
                ID = int(row[0]) #first column is plant_ID or firm_ID
                time_series_data = {}
                for i, value in enumerate(row[1:]):  #start from the second column
                    if data_to_load == 'chosen_action_data':
                        time_series_data[int(header[i + 1])] = int(value)
                    else:
                        time_series_data[int(header[i + 1])] = float(value)
                data_dict[ID] = time_series_data
        return data_dict
    except FileNotFoundError:
        print(f"Error: File '{csv_filename}' not found.")
        return None