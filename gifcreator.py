import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
from PIL import Image
import matplotlib.animation as animation

import utils

def bid_popularity(bid_data, price_increment, ER_params, J, plants, t_max, save_path, time_stamp_every = 10):
    #plotting single frames
    bid_space = (np.arange(J)*price_increment).tolist()
    frame_counter = 0
    for t in range(t_max):
        if t%time_stamp_every==0:
            #getting bids data
            bids_at_timestep = np.zeros(len(plants)).tolist()
            for plant in plants:
                bids_at_timestep[plant] = bid_data[plant][t]
            plt.hist(bids_at_timestep, bins=bid_space, edgecolor='black')
            plt.xlim(min(bid_space), max(bid_space)+price_increment)
            space_between_ticks = 50
            plt.xticks(range(min(bid_space), max(bid_space)+space_between_ticks, space_between_ticks), fontsize = 6)
            plt.xlabel('bid values')
            plt.ylabel('frequency')
            plt.title(f"t = {t}")
            frame_counter+=1
            storage_folder_name = 'bids_GIF_Frames_Storage'
            filename = os.path.join(storage_folder_name, f'bids_frame_{frame_counter:03d}_params_eps_{ER_params[0]}_r_{ER_params[1]}_psi_{ER_params[2]}_s1_{ER_params[3]}.png')
            directory = os.path.dirname(save_path+filename)
            # Create the directory if it doesn't exist
            if not os.path.exists(directory):
                os.makedirs(directory)
            plt.savefig(save_path+filename)
            plt.close()
            if frame_counter%10==0:
                print(f"Saved {frame_counter} frames.")
    print("I'm done creating the single frames, I will now move to making a gif out of them.")            
    #plotting gif
    frame_files = glob.glob(os.path.join(save_path, storage_folder_name,fr'*_params_eps_{ER_params[0]}_r_{ER_params[1]}_psi_{ER_params[2]}_s1_{ER_params[3]}.png'))
    frame_array = []
    for file in frame_files:
        image = Image.open(file)
        frame_array.append(image)
    fig, ax = plt.subplots()
    im = ax.imshow(frame_array[0], animated=True)
    def update(i):
        im.set_array(frame_array[i])
        return im,
    ax.axis('off')
    animation_fig = animation.FuncAnimation(fig, update, frames=len(frame_array), interval=500, blit=True, repeat_delay=1000)
    filename = f'bid_GIF_params_eps_{ER_params[0]}_r_{ER_params[1]}_psi_{ER_params[2]}_s1_{ER_params[3]}.gif'
    directory = os.path.dirname(save_path+filename)
    # Create the directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)
    animation_fig.save(save_path+filename)
    plt.close()
    print(f"GIF saved as {save_path+filename}")

def single_plant_pdf_evolution(database_path, plant_ID, prob_data, min_price, max_price, price_increment, ER_params, t_max, save_path, time_stamp_every = 10, color = 'navy', costs_management = 'fixed_costs_only'):
    
    plants_info_df = pd.read_csv(database_path)
    if costs_management == 'fixed_costs_only':
        costs_per_mwh = utils.convert_fixed_cost_per_mwh(plants_info_df['fixed_cost'].to_numpy(), plants_info_df['capacity'], timestep_duration_in_days=1)
    elif costs_management == 'variable_costs_only':
        costs_per_mwh = plants_info_df['variable_cost'].to_numpy()
    else:
        fixed_costs_per_mwh = utils.convert_fixed_cost_per_mwh(plants_info_df['fixed_cost'].to_numpy(), plants_info_df['capacity'], timestep_duration_in_days=1)
        variable_costs_per_mwh = plants_info_df['variable_cost'].to_numpy()
        costs_per_mwh = fixed_costs_per_mwh + variable_costs_per_mwh
    #creating frames
    frame_counter = 0
    for t in range(t_max):    
        if t%time_stamp_every==0:
            pdf_to_plot = prob_data[plant_ID][t]
            maximum_y_value = max(pdf_to_plot)
            bid_space = np.arange(len(pdf_to_plot))*price_increment
            middle_x_value = max(bid_space)/2 
            plt.bar(bid_space, pdf_to_plot, width = price_increment, color = color)
            plt.axvline(costs_per_mwh[plant_ID], color = 'black', linestyle = '--', zorder = 0)
            plt.xlim(min_price, max_price)
            plt.xlabel('bid price $/MWh')
            plt.title(f"Probability distribution for plant {plant_ID} at time t = {t}\nSimulation parameters: $\epsilon = {ER_params[0]}, r = {ER_params[1]}, \psi = {ER_params[2]}, s1 = {ER_params[3]}$")
            if max(pdf_to_plot) > 0.5:
                plt.ylim(0,1)
                params_text = f"$\epsilon = {ER_params[0]}$\n$r = {ER_params[1]}$\n$\psi = {ER_params[2]}$\n$s1 = {ER_params[3]}$"
                plt.text(middle_x_value*1.5, 0.75, params_text)
            else:
                plt.ylim(0,maximum_y_value*2)   
                params_text = f"$\epsilon = {ER_params[0]}, r = {ER_params[1]}, \psi = {ER_params[2]}, s1 = {ER_params[3]}$"
                plt.text(middle_x_value, maximum_y_value*1.5, params_text, ha = 'center')
            frame_counter+=1
            storage_folder_name = os.path.join('plant_pdf_GIF_Frames_Storage', f'plant_{plant_ID}')
            filename = os.path.join(storage_folder_name, f'plant_pdf_frame_{frame_counter:03d}_params_eps_{ER_params[0]}_r_{ER_params[1]}_psi_{ER_params[2]}_s1_{ER_params[3]}.png')
            directory = os.path.dirname(save_path+filename)
            # Create the directory if it doesn't exist
            if not os.path.exists(directory):
                os.makedirs(directory)
            plt.savefig(save_path+filename)
            plt.close()
            if frame_counter%10==0:
                print(f"Saved {frame_counter} frames.")
    print("I'm done creating the single frames, I will now move to making a gif out of them.")  
    #plotting gif
    frame_files = glob.glob(os.path.join(save_path, storage_folder_name,fr'*_params_eps_{ER_params[0]}_r_{ER_params[1]}_psi_{ER_params[2]}_s1_{ER_params[3]}.png'))
    frame_array = []
    for file in frame_files:
        image = Image.open(file)
        frame_array.append(image)

    fig, ax = plt.subplots()
    im = ax.imshow(frame_array[0], animated=True)
    def update(i):
        im.set_array(frame_array[i])
        return im,
    ax.axis('off')
    animation_fig = animation.FuncAnimation(fig, update, frames=len(frame_array), interval=500, blit=True, repeat_delay=1000)
    filename = f'plant_{plant_ID}_pdf_GIF_params_eps_{ER_params[0]}_r_{ER_params[1]}_psi_{ER_params[2]}_s1_{ER_params[3]}.gif'
    animation_fig.save(save_path+filename)
    plt.close()
    print(f"GIF saved as {save_path+filename}")

def pdf_evo_gif_creator(prob_data, min_price, max_price, price_increment, plants, ER_params, t_max, save_path, time_stamp_every = 10, color = 'navy'):
    #creating frames
    frame_counter = 0
    alpha_param = max(0.05, 1/len(plants))
    for t in range(1, t_max):    
        if t%time_stamp_every==0:
            maximum_y_values =  []
            for plant_ID in plants:
                pdf_to_plot = prob_data[plant_ID][t].copy()
                maximum_y_values.append(max(pdf_to_plot)) 
                bid_space = np.arange(len(pdf_to_plot))*price_increment
                plt.plot(bid_space, pdf_to_plot, color = color, alpha = alpha_param)
            maximum_y_value = max(maximum_y_values)
            plt.xlabel('bid price $/MWh')
            plt.title(f"Probability distributions at time t = {t}\nSimulation parameters: $\epsilon = {ER_params[0]}, r = {ER_params[1]}, \psi = {ER_params[2]}, s1 = {ER_params[3]}$")
            plt.xlim(min_price, max_price)
            if maximum_y_value > 0.5:
                plt.ylim(0,1)
            else:
                plt.ylim(0,maximum_y_value*2)                       
            frame_counter+=1
            storage_folder_name = 'plant_evo_pdf_GIF_Frames_Storage'
            filename = os.path.join(storage_folder_name, f'pdf_frame_{frame_counter:03d}_params_eps_{ER_params[0]}_r_{ER_params[1]}_psi_{ER_params[2]}_s1_{ER_params[3]}.png')
            directory = os.path.dirname(save_path+filename)
            # Create the directory if it doesn't exist
            if not os.path.exists(directory):
                os.makedirs(directory)
            plt.savefig(save_path+filename, dpi = 300)
            plt.close()
            if frame_counter%10==0:
                print(f"Saved {frame_counter} frames.")
    
    print("I'm done creating the single frames, I will now move to making a gif out of them.")  
    #plotting gif
    frame_files = glob.glob(save_path+fr'plant_evo_pdf_GIF_Frames_Storage/*_params_eps_{ER_params[0]}_r_{ER_params[1]}_psi_{ER_params[2]}_s1_{ER_params[3]}.png')
    frame_array = []
    for file in frame_files:
        image = Image.open(file)
        frame_array.append(image)

    fig, ax = plt.subplots()
    im = ax.imshow(frame_array[0], animated=True)
    def update(i):
        im.set_array(frame_array[i])
        return im,
    ax.axis('off')
    animation_fig = animation.FuncAnimation(fig, update, frames=len(frame_array), interval=500, blit=True, repeat_delay=1000)
    filename = f'plant_evo_pdf_GIF_params_eps_{ER_params[0]}_r_{ER_params[1]}_psi_{ER_params[2]}_s1_{ER_params[3]}.gif'
    animation_fig.save(save_path+filename, dpi = 300)
    plt.close()
    print(f"GIF saved as {save_path+filename}")