import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import norm
import glob
import os
import csv
from PIL import Image
import matplotlib.animation as animation

##custom
import utils
import myplotcreator as myplot
import gifcreator

def reinforcement_function(plant_ID, psi, profits, plant_ownership_dict, firm_portfolios):
    #recovering the plant ownership
    plant_owner = plant_ownership_dict[plant_ID]
    #recovering all the plants under the same owner
    firm_plant_list = firm_portfolios[plant_owner]
    n = len(firm_plant_list)
    #computing the firm profits
    firm_profit = 0
    for plant in firm_plant_list:
        firm_profit = firm_profit + profits[plant]
    #computing reinforcement for the plant
    reinforcement = psi*profits[plant_ID] + (1-psi)*(firm_profit)/n
    return reinforcement

def compute_firm_profits(firms, firm_portfolios, plant_profits):
    firm_profits = [0.0]*len(firms)
    for firm in firms:
        for plant in firm_portfolios[firm]:
            firm_profits[firm] = firm_profits[firm] + plant_profits[plant]
    return firm_profits

def DemandBuilder(t, plants_df, demand_type = 'constant', total_capacity_fraction = None):

        if demand_type == 'constant':
            total_capacity = plants_df['capacity'].sum()
            if total_capacity_fraction == None:
                 demand = total_capacity*0.5
            else:
                 demand = total_capacity*total_capacity_fraction

        if demand_type == 'sinusoidal':
            total_capacity = plants_df['capacity'].sum()
            period = 200
            demand = 0.5*total_capacity + 1/3*total_capacity*np.sin(2*np.pi/period * t)

        if demand_type == '2periods':
            total_capacity = plants_df['capacity'].sum()
            long_period = 365*4
            short_period = 4
            demand = 0.5*total_capacity + 1/3*total_capacity*np.sin(2*np.pi/long_period * t) +  1/6*total_capacity*np.cos(2*np.pi/short_period * t)
        
        if demand_type == 'noisy-constant':
            total_capacity = plants_df['capacity'].sum()
            gaussian_noise = random.gauss(0, 1)
            demand = total_capacity*0.5 + 1/10*total_capacity*gaussian_noise
              
        return demand

class Model():
    def __init__(self, N, M, J, price_increment, ER_params, seed, costs_management = 'both_costs', timestep_in_days = float(1), run_type = 'single'):
        #Structure
        self.N = N
        self.M = M
        self.J = J
        self.price_increment = price_increment

        #Roth&Erev parameters
        self.epsilon = ER_params[0]
        self.r = ER_params[1]
        self.psi = ER_params[2]
        self.s1 = ER_params[3]

        #Time
        self.t = 0 

        self.costs_management = costs_management
        self.seed = seed
        self.run_type = run_type
        self.timestep_in_days = timestep_in_days

    def DatasetCreator(self, path, distributions):
        #Preparing file where to store data
        filepath = path + 'Norm_LogNormDataset.csv'
        f = open(filepath, 'w')
        f.write('plantID,capacity,fixed_cost,variable_cost\n')

        j=0
        while j < 50: # self.N: #loop over plants
            keep_trying = True
            while keep_trying != False:
            #sampling a capacity in the distribution of capacities
                sampled_capacity = np.random.normal(distributions['capacity']['mean'], distributions['capacity']['std_dev'])
                sampled_fixed_cost = np.random.lognormal(distributions['fixed_cost']['mu'], distributions['fixed_cost']['sigma'])
                sampled_variable_cost = np.random.lognormal(distributions['variable_cost']['mu'], distributions['variable_cost']['sigma'])
                if sampled_capacity > 10 and sampled_fixed_cost <= distributions['fixed_cost']['max'] and sampled_variable_cost <= distributions['variable_cost']['max']:
                    keep_trying = False
            #saving 
            line = "%i,%i,%.2f,%.2f\n"%(j, sampled_capacity, sampled_fixed_cost, sampled_variable_cost)
            f.write(line)
            j += 1

        f.close()
        print(f'Dataset created. Location: {filepath}')

    def FirmPortfoliosCreation(self, database_path, possession_probs):
        np.random.seed(self.seed)
        df = pd.read_csv(database_path)
        #initialization
        plantIDs = df['plantID'].tolist()
        updated_plantIDs = plantIDs.copy()
        firm_portfolios = {firm_ID: [] for firm_ID in range(self.M)}
        plant_ownership_dict_prov= {}
        plant_ownership_dict = {}
        #assigning first plant
        for firm in range(self.M):
            #selecting plant
            selected_plant = np.random.choice(updated_plantIDs)
            #inserting plant in firm's portfolio
            firm_portfolios[firm].append(selected_plant)
            #updating list of ownerless plants
            updated_plantIDs.remove(selected_plant)
        #assigning all remaning plants
        remaining_plants = len(updated_plantIDs)
        while remaining_plants > 0:
            for firm in range(self.M):
                if updated_plantIDs: #checking if the list is not empty (removing this condition brings up an error)
                    p = np.random.uniform(0,1)
                    if p < possession_probs[firm]:
                        selected_plant = np.random.choice(updated_plantIDs)
                        firm_portfolios[firm].append(selected_plant)
                        updated_plantIDs.remove(selected_plant)
                        remaining_plants = len(updated_plantIDs)
        #making a provisional dictionary 
        for firm, portfolio in firm_portfolios.items():
            for plant in portfolio:
                plant_ownership_dict_prov[plant] = firm
        #create tha final dictionary by sorting the previous one by plant ID
        plant_ownership_dict = {key: plant_ownership_dict_prov[key] for key in sorted(plant_ownership_dict_prov)}
        print("All plants were assigned to firms. No more plants left.")
        return firm_portfolios, plant_ownership_dict
        
    def run(self, database_path, t_max, plant_ownership_dict, firm_portfolios_dict, demand_type='constant', total_capacity_fraction = None, output_path = './Simulation_Results/', run_ID = None):

        np.random.seed(self.seed)
        plants_info_df = pd.read_csv(database_path)
        #initializing stuff
        plant_IDs = plants_info_df['plantID'].tolist()
        plant_capacities = plants_info_df['capacity'].tolist()
#        plant_capacities_dict = utils.load_from_plant_csv(database_path, 'capacity')
        plant_capacities_arr = plants_info_df['capacity'].to_numpy()
#        plant_fixed_costs_dict = utils.load_from_plant_csv(database_path, 'fixed_cost')
#            plant_variable_costs_dict = utils.load_from_plant_csv(database_path, 'variable_cost')
        q = np.zeros((self.N,self.J))
        p = np.zeros((self.N,self.J))
        normalization = np.zeros(self.N)
        k = np.zeros(self.N, dtype=int)
        R = np.zeros(self.N)
        profits = np.zeros(self.N)
        marginal_prices_through_time = []
        marginal_prices_through_time.append(0.0)

        #initializing action space
        actions_space = np.tile(np.arange(self.J), (self.N, 1))

        #initializing propensities and probabilities
        for i in range(self.N): #i are plants
            for j in range(self.J): #j are bids
                q[i,j] = self.s1

        simulation_data_dict = utils.initialize_metrics(plants = range(self.N), firms = range(self.M), t_max = t_max, p = p, k=k)
        #let's run the simulation
        self.t = 0        
        while self.t < t_max:
            #update time
            self.t += 1
            #print(f't={self.t}')

            #restrict action space by computing costs first
            #plant_fixed_costs_arr = plants_info_df['fixed_cost'].to_numpy()
            plant_fixed_costs_arr = utils.convert_fixed_cost(plants_info_df['fixed_cost'].to_numpy(dtype=np.float64), self.timestep_in_days)
            if self.costs_management != 'fixed_costs_only':
                plant_variable_costs_per_mwh_arr = plants_info_df['variable_cost'].to_numpy(dtype=np.float64)
            plant_activation_costs = np.zeros(self.N)
            plant_activation_costs_per_mwh = np.zeros(self.N)
            if self.costs_management == 'fixed_costs_only':
                plant_activation_costs = plant_fixed_costs_arr
            elif self.costs_management == 'variable_costs_only':
                plant_activation_costs = plant_variable_costs_per_mwh_arr*plant_capacities_arr
            else:
                plant_variable_costs_arr = plant_variable_costs_per_mwh_arr*plant_capacities_arr
                plant_activation_costs = plant_variable_costs_arr + plant_fixed_costs_arr
            
            #draw action (price)
            k = np.zeros(self.N, dtype=int)
            plant_activation_costs_per_mwh = plant_activation_costs/plant_capacities_arr
            for plant in plant_IDs:
                if plant_activation_costs_per_mwh[plant] > (self.J-1)*self.price_increment:
                    raise RuntimeError(f'Plant {plant} could not bid hoping for a profit. Simulation interrupted.')
            #reducing action space
            forbidden_actions = {}
            for plant in plant_IDs:
                forbidden_actions[plant] = []
                for j in range(self.J):
                    if j*self.price_increment < plant_activation_costs_per_mwh[plant]:
                        forbidden_actions[plant].append(j)
            #initializing probabilities
            for i in range(self.N):
                highest_forbbiden_action = forbidden_actions[i][-1]
                normalization[i] = np.sum(q[i, highest_forbbiden_action+1:])
            for i in range(self.N):
                accessible_actions = [action for action in list(range(self.J)) if action not in forbidden_actions[i]]
                for j in accessible_actions:
                    p[i,j] = q[i,j]/normalization[i]           

            for plant in plant_IDs:
                k[plant] = np.random.choice(actions_space[plant], p = p[plant])
            bids = (k*self.price_increment).tolist()
            plant_activation_costs_per_mwh_list = plant_activation_costs_per_mwh.tolist()
    
            #supply curve
            supply_df = pd.DataFrame({
                'plant_ID': plant_IDs,
                'capacity': plant_capacities,
                'plant_cost': plant_activation_costs_per_mwh_list,
            })
            if self.costs_management == 'fixed_costs_only':
                supply_df['fixed_cost'] = plant_fixed_costs_arr
            elif self.costs_management == 'variable_costs_only':
                supply_df['variable_cost'] = plant_variable_costs_per_mwh_arr
            else:
                supply_df['fixed_cost'] = plant_fixed_costs_arr
                supply_df['variable_cost'] = plant_variable_costs_per_mwh_arr
            supply_df['bid'] = bids
            sorted_by_bid_supply_df = supply_df.sort_values(by='bid')

            #demand curve
            demand = DemandBuilder(self.t, plants_info_df, demand_type, total_capacity_fraction)

            #solving the market
            dispatched_capacity = 0
            marginal_price = 0
            dispatch_schedule = []
            marginal_plants = []

            for index, row in sorted_by_bid_supply_df.iterrows():
                if dispatched_capacity >= demand:
                    break
                plant_ID = row['plant_ID']
                capacity = row['capacity']
                bid_price = row['bid']
                #dispatching the plant
                if dispatched_capacity + capacity < demand: #demand is not met by the selected plant
                    remaining_plants_df = sorted_by_bid_supply_df[sorted_by_bid_supply_df.index > index]
                    prov_dispatched_capacity = dispatched_capacity + capacity
                    marginal_price = bid_price
                    marginal_plants.append((int(plant_ID), capacity, bid_price))                      
                    for remaining_index, remaining_row in remaining_plants_df.iterrows(): #demand could be met by considering other plants bidding at the same price level
                        next_plant_bid_price = remaining_row['bid']
                        if next_plant_bid_price == bid_price:
                            next_plant_ID = remaining_row['plant_ID']
                            next_plant_capacity = remaining_row['capacity']
                            marginal_plants.append((int(next_plant_ID), next_plant_capacity, bid_price))
                            prov_dispatched_capacity += next_plant_capacity
                        else:
                            break
                    if prov_dispatched_capacity < demand:
                        dispatch_schedule.append((int(plant_ID), capacity, bid_price))
                        dispatched_capacity += capacity
                        marginal_plants = []
                    else:
                        break
                else:
                    remaining_demand = demand - dispatched_capacity
                    if marginal_plants == []:
                        marginal_price = bid_price
                        if remaining_demand >= 0:
                            marginal_plants.append((int(plant_ID), capacity, bid_price))
                    else:
                        if bid_price == marginal_price:
                            if remaining_demand >= 0:
                                marginal_plants.append((int(plant_ID), capacity, bid_price))
                        else: 
                            break

            if dispatched_capacity < demand and marginal_plants: # marginal plants exist and demand is not met
                remaining_demand = demand - dispatched_capacity
                total_marginal_capacity = sum(plant[1] for plant in marginal_plants)
                for plant_ID, capacity, bid_price in marginal_plants:
                    proportional_dispatch = (capacity/total_marginal_capacity) * remaining_demand
                    dispatch_schedule.append((int(plant_ID), min(capacity, proportional_dispatch), bid_price))
                    dispatched_capacity += min(capacity, proportional_dispatch)
            supply_df['dispatched'] = False
            supply_df['selling_price'] = 0
            supply_df['supplied_capacity'] = 0.
            supply_df['profit'] = 0.
            for plant_ID, capacity, _ in dispatch_schedule:
                supply_df.loc[supply_df['plant_ID'] == plant_ID, 'dispatched'] = True
                supply_df.loc[supply_df['plant_ID'] == plant_ID, 'supplied_capacity'] = capacity
                supply_df.loc[supply_df['plant_ID'] == plant_ID, 'selling_price'] = marginal_price
                supply_df.loc[supply_df['plant_ID'] == plant_ID, 'profit'] = ((supply_df['selling_price'] - supply_df['plant_cost'])*supply_df['supplied_capacity']).round(2)

            marginal_prices_through_time.append(marginal_price)
            
            #computing profits (if negative, 0 is displayed)
            sorted_by_plantID_supply_df = supply_df.sort_values(by='plant_ID')
            profits = sorted_by_plantID_supply_df['profit'].tolist()

            #recording simulation data
            firm_profits = compute_firm_profits(firms = range(self.M), firm_portfolios = firm_portfolios_dict, plant_profits = profits)
            simulation_data_dict = utils.update_metrics(plants = range(self.N), firms = range(self.M), t = self.t, p = p, k=k, bids=bids, plant_profits=profits, firm_profits=firm_profits, metrics = simulation_data_dict)

            #compute reinforcement
            for plant in range(self.N):
                R[plant] = reinforcement_function(plant, self.psi, profits, plant_ownership_dict, firm_portfolios_dict)
            #update propensities
            for i in range(self.N):
                for j in range(self.J):
                    if j == k[i]:
                        q[i,j] = (1-self.r)*q[i,j]+R[i]*(1-self.epsilon)
                    else:
                        q[i,j] = (1-self.r)*q[i,j]+q[i,j]*self.epsilon/(self.J-1)
        
        #creating a subfolder structure if run_type is set to multiple
        if self.run_type == 'multiple':
            output_path = output_path + fr'varying_params/run_with_params_eps_{self.epsilon}_r_{self.r}_psi_{self.psi}_s1_{self.s1}_{self.costs_management}/run_{run_ID}/'
        else:
            output_path = output_path + fr'single_run_with_eps_{self.epsilon}_r_{self.r}_psi_{self.psi}_s1_{self.s1}_{self.costs_management}/'
        #checking if output_path exists, if not creating it
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        csv_filenames = {}
        #saving simulation data
        prob_data = simulation_data_dict['prob_data']
        csv_filenames = utils.save_metrics_to_csv(metrics = simulation_data_dict, t_max = t_max, output_path = output_path, csv_filenames=csv_filenames) #be careful, this functiond deletes prob_data
        #saving prices data
        csv_filenames = utils.save_prices_to_csv(marginal_prices_through_time, t_max, output_path, csv_filenames)                
        if self.run_type == 'single':
            print('Results stored here:')
            formatted_csv_filenames = ' - '
            formatted_csv_filenames = formatted_csv_filenames+'\n - '.join([f"{key}: {value}" for key, value in csv_filenames.items()])
            print(formatted_csv_filenames)
        return csv_filenames, prob_data