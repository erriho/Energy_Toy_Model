import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

##custom
from utils import get_stationary_value, convert_fixed_cost, convert_fixed_cost_per_mwh

def saving_manager(should_save = False, save_path = '', should_show = True):
    if should_save == True:
        saving_directory = os.path.dirname(save_path)
        if not os.path.exists(saving_directory):
            os.makedirs(saving_directory)
        plt.savefig(save_path, dpi=300, transparent = True)
    if should_show == True:
        plt.show()
        if should_save == True:
            print(f'Plot saved here: {save_path}')
        else:
            print('Plot not saved. Remember to set should_save to True if you want to save it.')
    else:
        plt.close()

#SIMULATION-FOCUSED PLOTS
def prices(prices_data_list, demand_type, database_path, t_max, min_price, max_price, ER_params, costs_management = 'fixed_costs_only', should_save = False, save_path = '', should_show = True):
    fig, ax = plt.subplots()
    ax.plot(prices_data_list)
    ax.set_title(f"price(t) - {demand_type} demand")

    plants_info_df = pd.read_csv(database_path)
    if costs_management == 'fixed_costs_only':
        costs_per_mwh = convert_fixed_cost_per_mwh(plants_info_df['fixed_cost'].to_numpy(), plants_info_df['capacity'], timestep_duration_in_days=1)
    elif costs_management == 'variable_costs_only':
        costs_per_mwh = plants_info_df['variable_cost'].to_numpy()
    else:
        fixed_costs_per_mwh = convert_fixed_cost_per_mwh(plants_info_df['fixed_cost'].to_numpy(), plants_info_df['capacity'], timestep_duration_in_days=1)
        variable_costs_per_mwh = plants_info_df['variable_cost'].to_numpy()
        costs_per_mwh = fixed_costs_per_mwh + variable_costs_per_mwh
    

    ax.axhline(np.max(costs_per_mwh), 0, t_max, linestyle = '--', color = 'grey', zorder = 0)
    ax.axhline(np.min(costs_per_mwh), 0, t_max, linestyle = '--', color = 'grey', zorder = 0)
    ax.axhline(np.average(costs_per_mwh), 0, t_max, color = 'grey', zorder = 0)

    ax.set_xlabel('time')
    ax.set_ylabel('price')
    ax.set_xlim(0, t_max)
    ax.set_ylim(min_price, max_price)
    plt.tight_layout(rect=[0, 0, 0.8, 1])

    stationary_price_value = get_stationary_value(np.array(prices_data_list))
    info_text = f"Simulation parameters:\n" \
                f"$\epsilon = {ER_params[0]}, r = {ER_params[1]}, \psi = {ER_params[2]}, s1 = {ER_params[3]}$\n" \
                f"----------------------------------------------------\n" \
                f"stationary clearing price value: {stationary_price_value: .2f} $/MWh\n" \
                f"(computed over the last 50 steps)\n" \
                f"----------------------------------------------------\n" \
                f"Maximum cost per plant: {np.max(costs_per_mwh):.2f} $/MWh\n" \
                f"Minimum cost per plant: {np.min(costs_per_mwh):.2f} $/MWh\n" \
                f"Average cost per plant: {np.average(costs_per_mwh):.2f} $/MWh\n" 
    fig.text(0.8, 0.5, info_text, transform=fig.transFigure, fontsize=10, horizontalalignment = 'left', verticalalignment = 'center')
    saving_manager(should_save, save_path, should_show)

def average_plant_profits(database_path, plant_profits_data, number_of_plants, t_max, ER_params, profits_per, color = 'forestgreen', should_save = False, save_path = '', should_show = True):
    avg_profit_through_time = np.zeros(t_max)
    std_profit_through_time = np.zeros(t_max)
    if profits_per == 'plant':
        if color == 'forestgreen': color = 'green'
    elif profits_per == 'MWh':
        plants_info_df = pd.read_csv(database_path)
        plant_capacities = plants_info_df['capacity'].to_numpy()

    time = np.arange(t_max)
    for t in range(t_max):
        profits_at_time_t = np.zeros(number_of_plants)
        for plant in range(number_of_plants):
            if profits_per == 'plant':
                profits_at_time_t[plant] = plant_profits_data[plant][t] 
            elif profits_per == 'MWh':
                profits_at_time_t[plant] = plant_profits_data[plant][t]/plant_capacities[plant]
        avg_profit_through_time[t] = np.mean(profits_at_time_t)
        std_profit_through_time[t] = np.std(profits_at_time_t)

    plt.plot(time, avg_profit_through_time, color = color)
    plt.fill_between(time, avg_profit_through_time-std_profit_through_time, avg_profit_through_time+std_profit_through_time, color = color, alpha=0.1, zorder=0)

    plt.xlim(0,t_max)
    plt.ylim(0,max(avg_profit_through_time)+max(std_profit_through_time)+10)

    params_text = f"$\epsilon = {ER_params[0]}$\n$r = {ER_params[1]}$\n$\psi = {ER_params[2]}$\n$s1 = {ER_params[3]}$"
    plt.text(t_max*3/4, (max(avg_profit_through_time)+max(std_profit_through_time)+10)*7/10, params_text)

    stationary_avg_profit_value = get_stationary_value(avg_profit_through_time)
    if profits_per == 'plant':
        stationary_text = f'stationary value: {stationary_avg_profit_value: .2f} $\n(computed over the last 50 steps)'
    elif profits_per == 'MWh':
        stationary_text = f'stationary value: {stationary_avg_profit_value: .2f} $/MWh\n(computed over the last 50 steps)'
    plt.text(t_max*5/9, (stationary_avg_profit_value*3), stationary_text, fontsize=8)

    plt.xlabel('time')
    if profits_per == 'plant':
        plt.ylabel('$')
        plt.title("plants' average profit")
    elif profits_per == 'MWh':
        plt.ylabel('$/MWh')
        plt.title("plants' per MWh average profit")
    saving_manager(should_save, save_path, should_show)

def average_firm_profits(firm_profits_data, firm_portfolios_dict, structured_firm_dict, profits_per, t_max, ER_params, color = 'purple', should_save = False, save_path = '', should_show = True):
    number_of_firms = len(firm_portfolios_dict)
    avg_profit_through_time = np.zeros(t_max)
    std_profit_through_time = np.zeros(t_max)
    if profits_per == 'plant':
        number_of_plants_owned_by_firms = np.zeros(number_of_firms)
        for firm in firm_portfolios_dict.keys():
            number_of_plants_owned_by_firms[firm] = len(firm_portfolios_dict.values())
        if color == 'purple': color = 'pink'
    elif profits_per == 'MWh':
        total_capacity_of_firms = np.zeros(number_of_firms)
        for firm in firm_portfolios_dict.keys():
            total_capacity_of_firms[firm] = structured_firm_dict[firm]['total_capacity']
    
    time = np.arange(t_max)
    for t in range(t_max):
        profits_at_time_t = np.zeros(number_of_firms)
        for firm in range(number_of_firms):
            if profits_per == 'plant':
                profits_at_time_t[firm] = firm_profits_data[firm][t]/number_of_plants_owned_by_firms[firm] 
            elif profits_per == 'MWh':
                profits_at_time_t[firm] = firm_profits_data[firm][t]/total_capacity_of_firms[firm]
        avg_profit_through_time[t] = np.mean(profits_at_time_t)
        std_profit_through_time[t] = np.std(profits_at_time_t)

    plt.plot(time, avg_profit_through_time, color = color)
    plt.fill_between(time, avg_profit_through_time-std_profit_through_time, avg_profit_through_time+std_profit_through_time, color = color, alpha=0.1, zorder=0)

    plt.xlim(0,t_max)
    plt.ylim(0,max(avg_profit_through_time)+max(std_profit_through_time)+10)

    params_text = f"$\epsilon = {ER_params[0]}$\n$r = {ER_params[1]}$\n$\psi = {ER_params[2]}$\n$s1 = {ER_params[3]}$"
    plt.text(t_max*3/4, (max(avg_profit_through_time)+max(std_profit_through_time)+10)*7/10, params_text)

    stationary_avg_profit_value = get_stationary_value(avg_profit_through_time)
    if profits_per == 'plant':
        stationary_text = f'stationary value: {stationary_avg_profit_value: .2f} $/plant\n(computed over the last 50 steps)'
    elif profits_per == 'MWh':
        stationary_text = f'stationary value: {stationary_avg_profit_value: .2f} $/MWh\n(computed over the last 50 steps)'
    plt.text(t_max*5/9, (stationary_avg_profit_value*3), stationary_text, fontsize=8)

    plt.xlabel('time')
    if profits_per == 'plant':
        plt.ylabel('$/plant')
        plt.title("firm's average profit per plant")
    elif profits_per == 'MWh':
        plt.ylabel('$/MWh')
        plt.title("firm's average profit per MWh of total capacity")

    saving_manager(should_save, save_path, should_show)

def firm_profits(firm_profits_data, firm_portfolios_dict, structured_firm_dict, profits_per, t_max, ER_params, firms_color_map = 'tab10', starting_timestep_to_zoom = 'default', should_save = False, save_path = '', should_show = True):
    if starting_timestep_to_zoom == 'default': starting_timestep_to_zoom = t_max*2/3
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    number_of_firms = len(firm_portfolios_dict)
    firm_profits_series = {}
    zoomed_series = {}
    number_of_plants_owned_by_firms = np.zeros(number_of_firms)
    total_capacity_of_firms = np.zeros(number_of_firms)
    for firm in firm_portfolios_dict.keys():
        firm_profits_series[firm] = []
        zoomed_series[firm] = []
        if profits_per == 'plant':
            number_of_plants_owned_by_firms[firm] = len(firm_portfolios_dict.values())
        elif profits_per == 'MWh':
            total_capacity_of_firms[firm] = structured_firm_dict[firm]['total_capacity']

    time = np.arange(t_max)
    zoomed_time = np.arange(t_max-starting_timestep_to_zoom)+starting_timestep_to_zoom
    for t in range(t_max):
        for firm in range(number_of_firms):
            if profits_per == 'plant':
                firm_profits_series[firm].append(firm_profits_data[firm][t]/number_of_plants_owned_by_firms[firm]) 
            elif profits_per == 'MWh':
                firm_profits_series[firm].append(firm_profits_data[firm][t]/total_capacity_of_firms[firm]) 
            if t >= starting_timestep_to_zoom:
                zoomed_series[firm].append(firm_profits_series[firm][t])

    cmap = plt.get_cmap(firms_color_map)
    firm_colors = [cmap(firm) for firm in range(number_of_firms)]
    for firm in firm_portfolios_dict.keys():
        ax1.plot(time, firm_profits_series[firm], color = firm_colors[firm], label = f'{firm}')
        ax2.plot(zoomed_time, zoomed_series[firm], color = firm_colors[firm])
    
    ax1.set_xlim(0,t_max)
    ax2.set_xlim(starting_timestep_to_zoom, t_max)
    ax1.set_ylim(0, None)
    ax2.set_ylim(0, None)

    y_pos = 0.9
    for firm in firm_portfolios_dict.keys():
        firm_text = f"Firm {firm}:\n   - plants owned: {firm_portfolios_dict[firm]}\n   - total capacity: {structured_firm_dict[firm]['total_capacity']} MWh"
        fig.text(0.8, y_pos, firm_text, transform=fig.transFigure, fontsize=10, horizontalalignment = 'left', color = firm_colors[firm])
        y_pos -= 0.1

    ax1.set_xlabel('time')
    if profits_per == 'plant':
        ax1.set_ylabel('$/plant')
    elif profits_per == 'MWh':
        ax1.set_ylabel('$/MWh')
    ax1.set_title(f"firm's per {profits_per} profit")
    ax1.legend()

    ax2.set_xlabel('time')
    ax2.set_title(f"firm's per {profits_per} profit - final steps")

    params_text = f"Simulation parameters: $\epsilon = {ER_params[0]}, r = {ER_params[1]}, \psi = {ER_params[2]}, s1 = {ER_params[3]}$"
    fig.suptitle(params_text, fontstyle = 'italic')
    plt.tight_layout(rect=[0, 0, 0.8, 1])
    saving_manager(should_save, save_path, should_show)

def simulation_overview(database_path, min_price, max_price, prices_data_list, plant_profits_data, number_of_plants, firm_profits_data, firm_portfolios_dict, structured_firm_dict, t_max, ER_params, profits_per = 'MWh', costs_management = 'fixed_costs_only', colors = ['#1f77b4', 'forestgreen', 'purple'], should_save = False, save_path = '', should_show = True):

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 8))
    time = np.arange(t_max+1)
    ### ax1: prices
    ax1.plot(time, prices_data_list, color = colors[0])
    ax1.set_xlim(0, t_max)
    ax1.set_ylim(min_price, max_price)
    ax1.set_xlabel('time')
    ax1.set_ylabel('$/MWh')
    ax1.set_title(f"price")
    ax1.grid(alpha=0.15)
    stationary_price_value = get_stationary_value(np.array(prices_data_list))
    ### ax2: profits per plant
    avg_profit_through_time = np.zeros(t_max)
    std_profit_through_time = np.zeros(t_max)
    if profits_per == 'plant':
        if colors[1] == 'forestgreen': colors[1] = 'green'
    elif profits_per == 'MWh':
        plants_info_df = pd.read_csv(database_path)
        plant_capacities = plants_info_df['capacity'].to_numpy()
    time = np.arange(t_max)
    for t in range(t_max):
        profits_at_time_t = np.zeros(number_of_plants)
        for plant in range(number_of_plants):
            if profits_per == 'plant':
                profits_at_time_t[plant] = plant_profits_data[plant][t] 
            elif profits_per == 'MWh':
                profits_at_time_t[plant] = plant_profits_data[plant][t]/plant_capacities[plant]
        avg_profit_through_time[t] = np.mean(profits_at_time_t)
        std_profit_through_time[t] = np.std(profits_at_time_t)
    ax2.plot(time, avg_profit_through_time, color = colors[1])
    ax2.fill_between(time, avg_profit_through_time-std_profit_through_time, avg_profit_through_time+std_profit_through_time, color = colors[1], alpha=0.1, zorder=0)
    ax2.set_xlim(0,t_max)
    ax2.set_ylim(0,max(avg_profit_through_time)+max(std_profit_through_time)+10)
    stationary_avg_plant_profit_value = get_stationary_value(avg_profit_through_time)
    ax2.set_xlabel('time')
    if profits_per == 'plant':
        ax2.set_ylabel('$')
        ax2.set_title("plants' average profit")
    elif profits_per == 'MWh':
        ax2.set_ylabel('$/MWh')
        ax2.set_title("plants' per MWh average profit")
    ax2.grid(alpha=0.15)
    ### ax3: profits per firm 
    number_of_firms = len(firm_portfolios_dict)
    avg_profit_through_time = np.zeros(t_max)
    std_profit_through_time = np.zeros(t_max)
    if profits_per == 'plant':
        number_of_plants_owned_by_firms = np.zeros(number_of_firms)
        for firm in firm_portfolios_dict.keys():
            number_of_plants_owned_by_firms[firm] = len(firm_portfolios_dict.values())
        if colors[2] == 'purple': colors[2] = 'pink'
    elif profits_per == 'MWh':
        total_capacity_of_firms = np.zeros(number_of_firms)
        for firm in firm_portfolios_dict.keys():
            total_capacity_of_firms[firm] = structured_firm_dict[firm]['total_capacity']
    time = np.arange(t_max)
    for t in range(t_max):
        profits_at_time_t = np.zeros(number_of_firms)
        for firm in range(number_of_firms):
            if profits_per == 'plant':
                profits_at_time_t[firm] = firm_profits_data[firm][t]/number_of_plants_owned_by_firms[firm] 
            elif profits_per == 'MWh':
                profits_at_time_t[firm] = firm_profits_data[firm][t]/total_capacity_of_firms[firm]
        avg_profit_through_time[t] = np.mean(profits_at_time_t)
        std_profit_through_time[t] = np.std(profits_at_time_t)
    stationary_avg_firm_profit_value = get_stationary_value(avg_profit_through_time)
    ax3.plot(time, avg_profit_through_time, color = colors[2])
    ax3.fill_between(time, avg_profit_through_time-std_profit_through_time, avg_profit_through_time+std_profit_through_time, color = colors[2], alpha=0.1, zorder=0)
    ax3.set_xlim(0,t_max)
    ax3.set_ylim(0,max(avg_profit_through_time)+max(std_profit_through_time)+10)
    ax3.set_xlabel('time')
    if profits_per == 'plant':
        ax3.set_ylabel('$/plant')
        ax3.set_title("firm's average profit per plant")
    elif profits_per == 'MWh':
        ax3.set_ylabel('$/MWh')
        ax3.set_title("firm's average profit per MWh of total capacity")
    ax3.grid(alpha=0.15)
    ### text to show
    plants_info_df = pd.read_csv(database_path)
    if costs_management == 'fixed_costs_only':
        costs_per_mwh = convert_fixed_cost_per_mwh(plants_info_df['fixed_cost'].to_numpy(), plants_info_df['capacity'], timestep_duration_in_days=1)
    elif costs_management == 'variable_costs_only':
        costs_per_mwh = plants_info_df['variable_cost'].to_numpy()
    else:
        fixed_costs_per_mwh = convert_fixed_cost_per_mwh(plants_info_df['fixed_cost'].to_numpy(), plants_info_df['capacity'], timestep_duration_in_days=1)
        variable_costs_per_mwh = plants_info_df['variable_cost'].to_numpy()
        costs_per_mwh = fixed_costs_per_mwh + variable_costs_per_mwh
    
    params_text = f"Simulation parameters: $\epsilon = {ER_params[0]}, r = {ER_params[1]}, \psi = {ER_params[2]}, s1 = {ER_params[3]}$"
    fig.suptitle(params_text, fontstyle = 'italic')
    info_text = f"Stationary values computed over the last 50 steps: \n" \
                f" - clearing price: {stationary_price_value: .2f} $/MWh\n" \
                f" - plant profits: {stationary_avg_plant_profit_value: .2f} $ (or $/MWh)\n" \
                f" - firm profits: {stationary_avg_firm_profit_value: .2f} $/{profits_per}\n" \
                f"----------------------------------------------------\n" \
                f"Maximum cost per plant: {np.max(costs_per_mwh):.2f} $/MWh\n" \
                f"Minimum cost per plant: {np.min(costs_per_mwh):.2f} $/MWh\n" \
                f"Average cost per plant: {np.average(costs_per_mwh):.2f} $/MWh\n" 
    
    fig.text(0.5, 0.93, info_text, transform=fig.transFigure, fontsize=10, horizontalalignment = 'center', verticalalignment = 'top')
    plt.tight_layout(rect = [0, 0, 1, 0.75])  
    saving_manager(should_save, save_path, should_show)

## PLANT-FOCUSED PLOTS

def bid_probabilities_over_time(prob_data, plant_list, price_increment, t_max, J, ER_params, should_save = False, saving_directory = '.', should_show = True, debug = False):
    for plant_ID in plant_list:
        action_and_its_avg_probability_dict = {}
        for action in range(J):
            action_probabilities = []
            for t in range(t_max):
                action_probabilities.append(prob_data[plant_ID][t][action])
            label = f'{action*price_increment} $'
            average_prob = np.mean(np.array(action_probabilities))
            action_and_its_avg_probability_dict[action] = average_prob
            plt.plot(action_probabilities, label = label)
        plt.ylabel('probabilities')
        plt.ylim(0,1)
        plt.xlabel('time')
        if debug == True:
            plt.xlim(0, t_max)
        else:
            plt.xlim(1, t_max)
        plt.title(f"Simulation parameters: $\epsilon = {ER_params[0]}, r = {ER_params[1]}, \psi = {ER_params[2]}, s1 = {ER_params[3]}$\n\nProbabilities for plant {plant_ID} of choosing each bid")
        if len(range(J)) <= 5: 
            plt.legend()
        else:
            top_5_actions = sorted(action_and_its_avg_probability_dict, key=action_and_its_avg_probability_dict.get, reverse=True)[:5] 
            handles, labels = plt.gca().get_legend_handles_labels()
            legend_handles = []
            legend_labels = []
            for i, label in enumerate(range(J)):
                if label in top_5_actions:
                    legend_handles.append(handles[i])
                    legend_labels.append(labels[i])
            plt.legend(legend_handles, legend_labels)
        plt.show()
        if should_save == True:
            save_path = os.path.join(saving_directory, f'bid_probabilities_for_plant_{plant_ID}.png')
        else:
            save_path = ''
        saving_manager(should_save, save_path, should_show)

def plant_overview(database_path, plant_ID, t_max, plant_profits_data, bid_data, price_increment, prices_data_list, prob_data, ER_params, profits_per, timestep_in_days = 1, should_save = False, save_path = '', should_show = True):
    plants_info_df = pd.read_csv(database_path)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 8))  
    #1st subplot: profits
    if profits_per == 'plant':
        ax1.plot(list(plant_profits_data[plant_ID].keys()), list(plant_profits_data[plant_ID].values()), color='green')
        y_max = max(list(plant_profits_data[plant_ID].values()))
    elif profits_per == 'MWh':
        plant_capacity = plants_info_df.loc[plant_ID, 'capacity']
        ax1.plot(list(plant_profits_data[plant_ID].keys()), list(plant_profits_data[plant_ID].values())/plant_capacity, color='forestgreen')
        y_max = max(list(plant_profits_data[plant_ID].values())/plant_capacity)
    ax1.set_xlim(0, t_max)
    ax1.set_ylim(0, y_max*1.03)
    ax1.set_xlabel('time')
    if profits_per == 'plant':
        ax1.set_ylabel('$')  
        ax1.set_title(f'Profits for Plant {plant_ID} over Time')
    elif profits_per == 'MWh':
        ax1.set_ylabel('$/MWh')
        ax1.set_title(f'Profits per MWh for Plant {plant_ID} over Time')
    ax1.grid(alpha=0.15)
    #2nd subplot: bids
    ax2.plot(list(bid_data[plant_ID].keys()), list(bid_data[plant_ID].values()), color='orange', label='Bid')
    ax2.plot(prices_data_list, color='darkred', linestyle='-', linewidth=1, label='Sell Price')
    ax2.set_xlim(0, t_max)
    ax2.set_ylim(0, max(list(bid_data[plant_ID].values()))*1.03)
    ax2.set_xlabel('time')
    ax2.set_title(f'Bids for Plant {plant_ID} over Time')
    ax2.set_ylabel('$/MWh')
    ax2.grid(alpha=0.15)
    ax2.legend()
    #3rd subplot: probabilities
    pdf_to_plot = prob_data[plant_ID][t_max]
    bid_space = np.arange(len(pdf_to_plot))*price_increment
    ax3.bar(bid_space, pdf_to_plot, width = price_increment, color='navy')
    ax3.set_xlim(0, np.max(bid_space))
    ax3.set_ylim(0,1)
    ax3.set_xlabel('bids ($)')
    ax3.set_ylabel('p')
    ax3.set_title(f'Final pdf over bids for Plant {plant_ID}')
    bids_with_highest_prob = {}
    action_space = np.arange(len(pdf_to_plot))
    for action in action_space:
        if pdf_to_plot[action] >= 0.2:
            bid = action*price_increment
            bids_with_highest_prob[bid] = pdf_to_plot[action]
    sorted_bids = sorted(bids_with_highest_prob, key=bids_with_highest_prob.get, reverse=True)
    surviving_bids_number = len(sorted_bids)
    most_probable_actions_info = ''
    for i in range(surviving_bids_number):
        new_info = f"{i+1}) {sorted_bids[i]} $/MWh with probability {bids_with_highest_prob[sorted_bids[i]]:.3f}\n"
        most_probable_actions_info += new_info
        if i > 3 : break
    ax3.grid(alpha=0.15)
    #text to plot
    params_text = f"Simulation parameters: $\epsilon = {ER_params[0]}, r = {ER_params[1]}, \psi = {ER_params[2]}, s1 = {ER_params[3]}$"
    fig.suptitle(params_text, fontstyle = 'italic')
    if profits_per == 'plant':
        stationary_avg_profit_value = get_stationary_value(np.array(list(plant_profits_data[plant_ID].values())))
    elif profits_per == 'MWh':
        plant_capacity = plants_info_df.loc[plant_ID, 'capacity']
        stationary_avg_profit_value = get_stationary_value(np.array(list(plant_profits_data[plant_ID].values()))/plant_capacity)
    stationary_avg_bid_value = get_stationary_value(np.array(list(bid_data[plant_ID].values())))
    #stationary pdf
    fixed_cost = plants_info_df.loc[plant_ID, 'fixed_cost']
    fixed_cost_per_it_per_MWh = convert_fixed_cost_per_mwh(fixed_cost, timestep_in_days)
    variable_cost = plants_info_df.loc[plant_ID, 'variable_cost']
    info_text = f"Average values computed over the last 50 steps: \n" \
                f"plant profits: {stationary_avg_profit_value: .2f} $ (or $/MWh)\n" \
                f"plant bids: {stationary_avg_bid_value: .2f} $/MWh\n" \
                f"----------------------------------------------------\n" \
                f"The {surviving_bids_number} bids with the highest probability are:\n" \
                f"{most_probable_actions_info}" \
                f"----------------------------------------------------\n" \
                f"Plant {plant_ID} fixed cost: {fixed_cost:.2f} $\n" \
                f"Plant {plant_ID} fixed cost per iteration and MWh: {fixed_cost_per_it_per_MWh:.2f} $/MWh\n" \
                f"Plant {plant_ID} variable cost: {variable_cost:.2f} $/MWh\n"  
    fig.text(0.5, 0.93, info_text, transform=fig.transFigure, fontsize=10, horizontalalignment = 'center', verticalalignment = 'top')
    plt.tight_layout(rect = [0, 0, 1, 0.75])  
    saving_manager(should_save, save_path, should_show)
