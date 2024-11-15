# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 19:21:09 2022

@author: Lizzie0930
"""
import numpy as np
import pandas as pd
import json
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
import os

def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created.")
    else:
        print(f"Folder '{folder_path}' already exists.")

#====================================================================================
#==============================  STEP ONE  ==========================================
#====================================================================================
#======= Open the JSON file, load its data and extract important info ===============
#====================================================================================
dir_list = os.listdir('malware_report/Report')
for z in range(len(dir_list)):
    with open('malware_report/Report/'+ dir_list[z]) as dat_file:
        data = json.load(dat_file)
    processes = data['behavior']['processes']
    start_time = data['info']['started']
    end_time = data['info']['ended']
    duration = data['info']['duration']
    unit_time = duration/16
    
    import time
    process_num =[]
    call_cat = []
    call_api = []
    call_time = []
    call_time_par = []
    for i in range(len(processes)):
        if len(processes[i]['calls']) >0:
            calls = processes[i]['calls']
            for call in calls:
                process_num.append(i)
                call_cat.append(call['category'])
                call_api.append(call['api'])
                call_time.append(call['time'])
                call_time_par.append(time.strftime("%Y-%m-%d, %H:%M:%S", time.localtime(int(call['time']))))
                
                
    report_df = pd.DataFrame()
    report_df['process_num'] = process_num
    report_df['call_cat'] = call_cat
    report_df['call_api'] = call_api
    report_df['call_time'] = call_time
    report_df['call_time_par'] = call_time_par
    report_df = report_df.sort_values(by=['call_time'])
    
    call_api_set = set(call_api)
    call_api_set_len = len(call_api_set)
    
    report_df.to_csv('Report.csv',index=False)   
    

    start_time = call_time[0]
    end_time = call_time[-1]
    duration = end_time-start_time
    unit_time = duration/16
    
    #====================================================================================
    #==============================  STEP TWO  ==========================================
    #====================================================================================
    # ========= Count frequency of each kind of API in each time unit ===================
    #====================================================================================
    
    report_df = pd.read_csv('Report.csv')
    
    cut_time = [0]
    call_cat_count = []
    call_api_count = []
    for j in range(16):
        for i in range(len(report_df)):
            time = report_df['call_time'][i]
            if (time > start_time + ((j+1)*unit_time) and time < start_time + ((j+2)*unit_time)) or (i == (len(report_df)-1)):
                call_cat_inte = report_df['call_cat'][cut_time[-1]:i]
                call_api_inte = report_df['call_api'][cut_time[-1]:i]
                call_cat_count.append(Counter(call_cat_inte))
                call_api_count.append(Counter(call_api_inte))
                cut_time.append(i)
                break
            elif time > start_time + ((j+1)*unit_time) and time > start_time + ((j+2)*unit_time):
                break
        if len(call_cat_count) < (j+1):
            call_cat_count.append(Counter([]))
            call_api_count.append(Counter([]))
            cut_time.append(cut_time[-1])
            
            
    #====================================================================================
    #==============================  STEP THREE  ========================================
    #====================================================================================
    #======================== Read color-mapping files ==================================
    #====================================================================================
    
    color_mapping_num = pd.read_csv('color_mapping_num.csv')
    color_mapping = pd.read_csv('color_mapping.csv')
    color_mapping = color_mapping.drop(['range_f', 'range_e'], axis=1)
    color_mapping_num = color_mapping_num.drop(['range_f', 'range_e'], axis=1)
    
    
    
    
    
    #====================================================================================
    #==============================  STEP FOUR  =========================================
    #====================================================================================
    # Mapping API frequency with predetermined color and trasform them into a numpy array 
    #====================================================================================
    
    row_array = [0 for i in range(16)]
    pic_array =[row_array for i in range(16)]
    print(pic_array)
    
    pic_array = np.array(pic_array)
            
    dict_lab = {'networking':0, 'registry':1, 'service':2, 'file':3,
           'system':4, 'message':5, 'process':6, 'LdrGetProcedureAddress':7, 'synchronisation':8,
           'NtQueryValueKey':9, 'LdrUnloadDll':10, 'NtOpenDirectoryObject':11,
           'ReadProcessMemory':12, 'CreateProcessInternalW':13,
           'NtAllocateVirtualMemory':14, 'NtReadFile':15}
    
    for i in range(len(call_api_count)):
        inter = call_api_count[i]
        print(i)
        if inter == 0 :
            continue
        for j in inter:
            print(j)
            index = 0
            cl_count = inter[j]
            print(cl_count)
            if cl_count == 0:
                index = 0
            elif cl_count in range(0,3):
                index = 1
            elif cl_count in range(3,7):
                index = 2
            elif cl_count in range(7,12):
                index = 3
            elif cl_count in range(12,18):
                index = 4
            elif cl_count in range(18,25):
                index = 5
            elif cl_count in range(25,33):
                index = 6
            elif cl_count in range(33,42):
                index = 7
            elif cl_count in range(42,100):
                index = 8
            elif cl_count in range(100,200):
                index = 9
            elif cl_count in range(200,1000):
                index = 10
            
            if j in dict_lab:
                color_num = color_mapping_num[j][index]            
                pic_array[dict_lab[j]][i] = color_num
            
            
    
    
    for i in range(len(call_cat_count)):
        inter = call_cat_count[i]
        print(i)
        if inter == 0 :
            continue
        for j in inter:
            print(j)
            index = 0
            cl_count = inter[j]
            print(cl_count)
            if cl_count == 0:
                index = 0
            elif cl_count in range(0,3):
                index = 1
            elif cl_count in range(3,7):
                index = 2
            elif cl_count in range(7,12):
                index = 3
            elif cl_count in range(12,18):
                index = 4
            elif cl_count in range(18,25):
                index = 5
            elif cl_count in range(25,33):
                index = 6
            elif cl_count in range(33,42):
                index = 7
            elif cl_count in range(42,100):
                index = 8
            elif cl_count in range(100,200):
                index = 9
            elif cl_count in range(200,1000):
                index = 10
            
            if j in dict_lab:
                color_num = color_mapping_num[j][index]            
                pic_array[dict_lab[j]][i] = color_num
    
    #====================================================================================
    #==============================  STEP FIVE  =========================================
    #====================================================================================
    #====================== Customize colormap for plotting =============================
    #====================================================================================
    
    def hex_to_rgb(hex):
      rgb = []
      for i in (0, 2, 4):
        decimal = int(hex[i:i+2], 16)
        rgb.append(decimal)
      
      return tuple(rgb)
    
    print(hex_to_rgb('FFFFFF'))
    
    viridis = cm.get_cmap('viridis', 176)
    newcolors = viridis(np.linspace(0, 1, 176))
    
    
    for i in range(16):
        for j in range(11):
            set_num = color_mapping_num.iloc[j,i]
            set_color_hex = color_mapping.iloc[j,i]
            print(set_num)
            print(set_color_hex)
            set_color_rgb = hex_to_rgb(set_color_hex[1:])
            set_color = np.array([set_color_rgb[0]/255, set_color_rgb[1]/255, set_color_rgb[2]/255, 1])
            newcolors[set_num:set_num+1, :] = set_color
            newcmp = ListedColormap(newcolors)
            
    #====================================================================================
    #==============================  STEP SIX  ==========================================
    #====================================================================================
    #==================== Plotting and save it as a png file ============================
    #====================================================================================
    data = np.random.random((16, 16))
    pixel_plot = plt.figure()
    pixel_plot = plt.imshow(
      pic_array, cmap=newcmp, interpolation='nearest', rasterized=True, vmin=0, vmax=175)
    plt.axis('off')
   
    folder_path = os.sep.join([os.path.normpath('g_pic_2'), dir_list[z].split('_')[1]])  # Replace with your folder path
    create_folder_if_not_exists(folder_path) 
    img_path = os.sep.join([os.path.normpath(folder_path), dir_list[z][:-5]+'.png'])
    plt.savefig(img_path,bbox_inches='tight', pad_inches=-0.0)  