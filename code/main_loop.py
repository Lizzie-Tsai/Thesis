import torch
import tensorflow as tf
from tensorflow.python.platform import build_info as tf_build_info

import matplotlib.pyplot as plt
import numpy as np
import random
from PIL import Image
import PIL.ImageOps

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torchvision.utils
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torchvision.models as models
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

from sklearn.neighbors import NearestNeighbors
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN
import shutil
import seaborn as sns
import pandas as pd
import math
import re
import cv2
from collections import Counter
import time
import statistics
from itertools import combinations

from clustering import DBSCAN_clustering_with_auto_emb_new, HI_clustering_with_auto_emb_new, DBSCAN_clustering_with_imgh_new, HI_clustering_with_imgh_new,DBSCAN_clustering_with_siame_emb_new, HI_clustering_with_siame_emb_new
from networks import Autoencoder_Conv, SiameseNetwork_VGG16_Based_v2, SiameseNetwork_autoencoder_Based
from utils import AutoEncoder_Training_Loop, generate_training_prototype, re_group_train_data, re_group_train_data_for_Hi, generate_testing_prototype_new
from utils import Training_Loop, Training_Loop_2, group_test, group_test_2
from dataset import SiameseNetworkDataset_1


def complete_training_loop_1(autoencoder_training_data_path, autoencoder_save_model_path,
                            autoencoder_save_img_base_path,
                            training_data_cluster_target, training_clustered_inter_dir,
                            training_clustered_best_dir,
                            train_prototype_dir, regrouped_target_dir,
                            test_cluster_target_dir, testing_clustered_inter_dir,
                            testing_clustered_best_dir,
                            save_best_si_model_dir, save_inter_si_model_dir,
                            si_epoch_num, vali_data_dir,
                            way_list, net_name, csv_save_dir, case_len, net_type, shots,
                            test_prototype_dest_dir,
                            proto_clustered_best_dest_dir, proto_clustered_inter_dest_dir,
                            result_csv_dir, transformation,
                            model_num=2, auto_epoch_num=100):
    '''
    with clustering, regrouping, pretrained network, concatencation
    representation: auoencoder Emb
    clustering: DBSCAN
    pretrained: autoencoder
    '''
    '''
    training and testing loop for model using Autoencoder Embedding for clustering.
    1. Training Autoencoder
    2. Clustering on Training data and clustering on testing data
    3. Generate Prototypes
    4. Re-group Training data
    5. Training Siamese network on Training data
    6. Testing Siamese network on Testing data
    '''
    cluster_target_folder_names = os.listdir(test_cluster_target_dir)
    cluster_target_folders = [os.sep.join([os.path.normpath(test_cluster_target_dir), i]) for i in
                              cluster_target_folder_names]
    
    #  1. Training Autoencoder
    print("====================================== TRAINING AUTOENCODER ....... ==================================")
    AutoEncoder_Training_Loop(epoch_num=auto_epoch_num,
                              data_path=autoencoder_training_data_path,
                              save_model_path=autoencoder_save_model_path,
                              save_img_base_path=autoencoder_save_img_base_path, transformation=transformation)

    # 2. Clustering on Training data
    print("====================================== CLUSTERING ON TRAINING DATA ....... ==================================")
    
    net = Autoencoder_Conv().cuda()
    net.load_state_dict(torch.load(autoencoder_save_model_path))
    
    best_cluster_path = DBSCAN_clustering_with_auto_emb_new(
        dataset_path=training_data_cluster_target,
        transformation=transformation,
        inter_dest_dir=training_clustered_inter_dir,
        best_dest_dir=training_clustered_best_dir,
        net=net, save_inter=False,
        desig_eps=15)
    train_best_clustered_path = best_cluster_path

    # 3. Generate Prototypes
    print("====================================== GENERATING TRAINING PROTOTYPES ....... ==================================")
    generate_training_prototype(target_dataset_path=best_cluster_path, dest_dataset_path=train_prototype_dir)
    
    # 4. Re-group Training data
    print("====================================== RE-GROUP TRAINING DATA ....... ==================================")
    best_cluster_path = DBSCAN_clustering_with_auto_emb_new(
        dataset_path=train_prototype_dir,
        transformation=transformation,
        inter_dest_dir=proto_clustered_inter_dest_dir,
        best_dest_dir=proto_clustered_best_dest_dir,
        net=net, save_inter=True,
        desig_eps=13)#17

    # train_best_clustered_path = "C:/Users/Lizzie0930/Desktop/git/Results_Auto/4_train_clustered_best_dest/final_clustered_dataset"
    re_group_train_data(source_dir=train_best_clustered_path, info_dir=best_cluster_path,
                        target_dir=regrouped_target_dir)
    
    # 2. Clustering on testing data
    print("====================================== CLUSTERING ON TESTING DATA ....... ==================================")
    """
        should cut testing data into known and unknown, and generate prototype only on known data
        2. pass known_cluster_target_folders to test_cluster_target_dir
        3. change generate_testing_prototype to generate_testing_prototype_new
        4. pass unknown_cluster_target_folders to vali_data_dir
    """
    
    for i in range(len(cluster_target_folders)):
        single_case_inter_dest_dir = os.sep.join(
            [os.path.normpath(testing_clustered_inter_dir), cluster_target_folder_names[i]])
        if not os.path.exists(single_case_inter_dest_dir):
            os.mkdir(single_case_inter_dest_dir)
        single_case_inter_best_dir = os.sep.join(
            [os.path.normpath(testing_clustered_best_dir), cluster_target_folder_names[i]])
        if not os.path.exists(single_case_inter_best_dir):
            os.mkdir(single_case_inter_best_dir)
        single_case_prototype_dest_dir = os.sep.join(
            [os.path.normpath(test_prototype_dest_dir), cluster_target_folder_names[i]])
        if not os.path.exists(single_case_prototype_dest_dir):
            os.mkdir(single_case_prototype_dest_dir)
        best_cluster_path = DBSCAN_clustering_with_auto_emb_new(dataset_path=cluster_target_folders[i], transformation=transformation,
                                              inter_dest_dir=single_case_inter_dest_dir,
                                              best_dest_dir=single_case_inter_best_dir, net=net,
                                              save_inter=False, desig_eps=15)
        generate_testing_prototype_new(target_dataset_path=best_cluster_path,
                                    dest_dataset_path=single_case_prototype_dest_dir)
    
    
    # train the model & validate the model and save the results
    print(
        "====================================== TRAIN THE SIAMESE NETWORK MODEL ....... ==================================")
    
    folder_dataset = datasets.ImageFolder(root=regrouped_target_dir)
    siamese_train_dataset = SiameseNetworkDataset_1(imageFolderDataset=folder_dataset,
                                                    transform=transformation)
    training_dataset = siamese_train_dataset
    best_epoch_list, best_batch_list, best_model_path_list = Training_Loop(model_num=model_num, net_name=net_name,
                                                                           dataset=training_dataset,
                                                                           save_best_model_dir=save_best_si_model_dir,
                                                                           save_inter_model_dir=save_inter_si_model_dir,
                                                                           epoch_num=si_epoch_num, way_list=way_list,
                                                                           vali_data_dir=vali_data_dir,
                                                                           proto_data_dir=test_prototype_dest_dir,
                                                                           csv_save_dir=csv_save_dir,
                                                                           case_len=case_len, net_type=net_type)
    print("best_epoch_list: ", best_epoch_list)
    print("best_batch_list: ", best_batch_list)
    print("best_model_path_list: ", best_model_path_list)
    
    print(
        "====================================== TESTING SIAMESE NETWORK MODEL ON TESTING DATA ....... ==================================")

    result_df, avg_df = group_test(model_num=model_num, way_list=way_list, case_len=case_len, epoch=20,
                                   model_path=best_model_path_list[0], batch_num=22, net_name=net_name,
                                   net_type=net_type, test_data_dir=vali_data_dir, test_data_postfix="",
                                   proto_data_dir=test_prototype_dest_dir, proto_data_postfix="")
    
    
    print(result_df)
    print(avg_df)
    result_csv_path = os.sep.join([os.path.normpath(result_csv_dir), "results.csv"])
    avg_csv_path = os.sep.join([os.path.normpath(result_csv_dir), "avg.csv"])
    result_df.to_csv(result_csv_path)
    avg_df.to_csv(avg_csv_path)

def complete_training_loop_2(autoencoder_training_data_path, autoencoder_save_model_path,
                            autoencoder_save_img_base_path,
                            training_data_cluster_target, training_clustered_inter_dir,
                            training_clustered_best_dir,
                            train_prototype_dir, regrouped_target_dir,
                            test_cluster_target_dir, testing_clustered_inter_dir,
                            testing_clustered_best_dir,
                            save_best_si_model_dir, save_inter_si_model_dir,
                            si_epoch_num, vali_data_dir,
                            way_list, net_name, csv_save_dir, case_len, net_type, shots,
                            test_prototype_dest_dir,
                            proto_clustered_best_dest_dir, proto_clustered_inter_dest_dir,
                            result_csv_dir, transformation,
                            weight_avg, weight_color, 
                            threshold_combine, threshold_cluster,
                            model_num=2, auto_epoch_num=100,):
    '''
    with clustering, regrouping, pretrained network, concatencation
    representation: auoencoder Emb
    clustering: Hi
    pretrained: autoencoder
    '''
    '''
    training and testing loop for model using Autoencoder Embedding for clustering.
    1. Training Autoencoder
    2. Clustering on Training data and clustering on testing data
    3. Generate Prototypes
    4. Re-group Training data
    5. Training Siamese network on Training data
    6. Testing Siamese network on Testing data
    '''
    cluster_target_folder_names = os.listdir(test_cluster_target_dir)
    cluster_target_folders = [os.sep.join([os.path.normpath(test_cluster_target_dir), i]) for i in
                              cluster_target_folder_names]
    
    #  1. Training Autoencoder
    print("====================================== TRAINING AUTOENCODER ....... ==================================")
    AutoEncoder_Training_Loop(epoch_num=auto_epoch_num,
                              data_path=autoencoder_training_data_path,
                              save_model_path=autoencoder_save_model_path,
                              save_img_base_path=autoencoder_save_img_base_path, transformation=transformation)
    
    # 2. Clustering on Training data
    print("====================================== CLUSTERING ON TRAINING DATA ....... ==================================")
    
    net = Autoencoder_Conv().cuda()
    net.load_state_dict(torch.load(autoencoder_save_model_path))
    
    best_cluster_path = HI_clustering_with_auto_emb_new(
        dataset_path=training_data_cluster_target,
        dest_dir=training_clustered_best_dir,
        transformation=transformation,net=net,
        weight_avg=weight_avg, weight_color=weight_color, 
        threshold_combine=threshold_combine, 
        threshold_cluster=threshold_cluster)
    train_best_clustered_path = best_cluster_path
    
    
    # 3. Generate Prototypes
    print("====================================== GENERATING TRAINING PROTOTYPES ....... ==================================")
    generate_training_prototype(target_dataset_path=best_cluster_path, dest_dataset_path=train_prototype_dir)
    
    # 4. Re-group Training data
    print("====================================== RE-GROUP TRAINING DATA ....... ==================================")
    
    best_cluster_path = HI_clustering_with_auto_emb_new(
        dataset_path=train_prototype_dir,
        dest_dir=proto_clustered_best_dest_dir,
        transformation=transformation,net=net,
        weight_avg=weight_avg, weight_color=weight_color, 
        threshold_combine=threshold_combine, 
        threshold_cluster=threshold_cluster)
    
  
    re_group_train_data_for_Hi(source_dir=train_best_clustered_path, info_dir=best_cluster_path,
                        target_dir=regrouped_target_dir)
    
    # 2. Clustering on testing data
    print("====================================== CLUSTERING ON TESTING DATA ....... ==================================")
    """
        should cut testing data into known and unknown, and generate prototype only on known data
        2. pass known_cluster_target_folders to test_cluster_target_dir
        3. change generate_testing_prototype to generate_testing_prototype_new
        4. pass unknown_cluster_target_folders to vali_data_dir
    """
    
    for i in range(len(cluster_target_folders)):
        single_case_inter_dest_dir = os.sep.join(
            [os.path.normpath(testing_clustered_inter_dir), cluster_target_folder_names[i]])
        if not os.path.exists(single_case_inter_dest_dir):
            os.mkdir(single_case_inter_dest_dir)
        single_case_inter_best_dir = os.sep.join(
            [os.path.normpath(testing_clustered_best_dir), cluster_target_folder_names[i]])
        if not os.path.exists(single_case_inter_best_dir):
            os.mkdir(single_case_inter_best_dir)
        single_case_prototype_dest_dir = os.sep.join(
            [os.path.normpath(test_prototype_dest_dir), cluster_target_folder_names[i]])
        if not os.path.exists(single_case_prototype_dest_dir):
            os.mkdir(single_case_prototype_dest_dir)
        best_cluster_path = HI_clustering_with_auto_emb_new(
                            dataset_path=cluster_target_folders[i],
                            dest_dir=single_case_inter_best_dir,
                            transformation=transformation,net=net,
                            weight_avg=weight_avg, weight_color=weight_color, 
                            threshold_combine=threshold_combine, 
                            threshold_cluster=threshold_cluster)
        
        generate_testing_prototype_new(target_dataset_path=best_cluster_path,
                                    dest_dataset_path=single_case_prototype_dest_dir)
    
    
    # train the model & validate the model and save the results
    print(
        "====================================== TRAIN THE SIAMESE NETWORK MODEL ....... ==================================")
    
    
    folder_dataset = datasets.ImageFolder(root=regrouped_target_dir)
    siamese_train_dataset = SiameseNetworkDataset_1(imageFolderDataset=folder_dataset,
                                                    transform=transformation)
    training_dataset = siamese_train_dataset
    best_epoch_list, best_batch_list, best_model_path_list = Training_Loop(model_num=model_num, net_name=net_name,
                                                                           dataset=training_dataset,
                                                                           save_best_model_dir=save_best_si_model_dir,
                                                                           save_inter_model_dir=save_inter_si_model_dir,
                                                                           epoch_num=si_epoch_num, way_list=[5],
                                                                           vali_data_dir=vali_data_dir,
                                                                           proto_data_dir=test_prototype_dest_dir,
                                                                           csv_save_dir=csv_save_dir,
                                                                           case_len=case_len, net_type=net_type)
    print("best_epoch_list: ", best_epoch_list)
    print("best_batch_list: ", best_batch_list)
    print("best_model_path_list: ", best_model_path_list)
    
    print(
        "====================================== TESTING SIAMESE NETWORK MODEL ON TESTING DATA ....... ==================================")
    
    result_df, avg_df = group_test(model_num=model_num, way_list=way_list, case_len=case_len, epoch=best_epoch_list[0],
                                   model_path=best_model_path_list[0], batch_num=best_batch_list[0], net_name=net_name,
                                   net_type=net_type, test_data_dir=vali_data_dir, test_data_postfix="",
                                   proto_data_dir=test_prototype_dest_dir, proto_data_postfix="")
       
    print(result_df)
    print(avg_df)
    result_csv_path = os.sep.join([os.path.normpath(result_csv_dir), "results.csv"])
    avg_csv_path = os.sep.join([os.path.normpath(result_csv_dir), "avg.csv"])
    result_df.to_csv(result_csv_path)
    avg_df.to_csv(avg_csv_path)

def complete_training_loop_3(autoencoder_training_data_path, autoencoder_save_model_path,
                            autoencoder_save_img_base_path,
                            training_data_cluster_target, training_clustered_inter_dir,
                            training_clustered_best_dir,
                            train_prototype_dir, regrouped_target_dir,
                            test_cluster_target_dir, testing_clustered_inter_dir,
                            testing_clustered_best_dir,
                            save_best_si_model_dir, save_inter_si_model_dir,
                            si_epoch_num, vali_data_dir,
                            way_list, net_name, csv_save_dir, case_len, net_type, shots,
                            test_prototype_dest_dir,
                            proto_clustered_best_dest_dir, proto_clustered_inter_dest_dir,
                            result_csv_dir, transformation,  
                            weight_avg, weight_color,
                            model_num=2, auto_epoch_num=100):
    '''
    with clustering, regrouping, pretrained network, concatencation
    representation: imgh
    clustering: DBSCAN
    pretrained: autoencoder
    '''
    '''
    training and testing loop for model using Autoencoder Embedding for clustering.
    1. Training Autoencoder
    2. Clustering on Training data and clustering on testing data
    3. Generate Prototypes
    4. Re-group Training data
    5. Training Siamese network on Training data
    6. Testing Siamese network on Testing data
    '''
    cluster_target_folder_names = os.listdir(test_cluster_target_dir)
    cluster_target_folders = [os.sep.join([os.path.normpath(test_cluster_target_dir), i]) for i in
                              cluster_target_folder_names]
    
    #  1. Training Autoencoder
    print("====================================== TRAINING AUTOENCODER ....... ==================================")
    AutoEncoder_Training_Loop(epoch_num=auto_epoch_num,
                              data_path=autoencoder_training_data_path,
                              save_model_path=autoencoder_save_model_path,
                              save_img_base_path=autoencoder_save_img_base_path, transformation=transformation)
    
    # 2. Clustering on Training data
    print("====================================== CLUSTERING ON TRAINING DATA ....... ==================================")
    
    net = Autoencoder_Conv().cuda()
    net.load_state_dict(torch.load(autoencoder_save_model_path))
    
    best_cluster_path = DBSCAN_clustering_with_imgh_new(
        dataset_path=training_data_cluster_target,
        transformation=transformation,
        inter_dest_dir=training_clustered_inter_dir,
        best_dest_dir=training_clustered_best_dir,
        weight_avg=weight_avg, weight_color=weight_color, 
        save_inter=False,
        desig_eps=15)
    train_best_clustered_path = best_cluster_path
    
    # 3. Generate Prototypes
    print("====================================== GENERATING TRAINING PROTOTYPES ....... ==================================")
    generate_training_prototype(target_dataset_path=best_cluster_path, dest_dataset_path=train_prototype_dir)
    
    # 4. Re-group Training data
    print("====================================== RE-GROUP TRAINING DATA ....... ==================================")
    best_cluster_path = DBSCAN_clustering_with_imgh_new(
        dataset_path=train_prototype_dir,
        transformation=transformation,
        inter_dest_dir=proto_clustered_inter_dest_dir,
        best_dest_dir=proto_clustered_best_dest_dir,
        weight_avg=weight_avg, weight_color=weight_color, 
        save_inter=True,
        desig_eps=13)#17
    
    # train_best_clustered_path = "C:/Users/Lizzie0930/Desktop/git/Results_Auto/4_train_clustered_best_dest/final_clustered_dataset"
    re_group_train_data(source_dir=train_best_clustered_path, info_dir=best_cluster_path,
                        target_dir=regrouped_target_dir)
    
    # 2. Clustering on testing data
    print("====================================== CLUSTERING ON TESTING DATA ....... ==================================")
    """
        should cut testing data into known and unknown, and generate prototype only on known data
        2. pass known_cluster_target_folders to test_cluster_target_dir
        3. change generate_testing_prototype to generate_testing_prototype_new
        4. pass unknown_cluster_target_folders to vali_data_dir
    """
    for i in range(len(cluster_target_folders)):
        single_case_inter_dest_dir = os.sep.join(
            [os.path.normpath(testing_clustered_inter_dir), cluster_target_folder_names[i]])
        if not os.path.exists(single_case_inter_dest_dir):
            os.mkdir(single_case_inter_dest_dir)
        single_case_inter_best_dir = os.sep.join(
            [os.path.normpath(testing_clustered_best_dir), cluster_target_folder_names[i]])
        if not os.path.exists(single_case_inter_best_dir):
            os.mkdir(single_case_inter_best_dir)
        single_case_prototype_dest_dir = os.sep.join(
            [os.path.normpath(test_prototype_dest_dir), cluster_target_folder_names[i]])
        if not os.path.exists(single_case_prototype_dest_dir):
            os.mkdir(single_case_prototype_dest_dir)
        best_cluster_path = DBSCAN_clustering_with_imgh_new(dataset_path=cluster_target_folders[i], transformation=transformation,
                                              inter_dest_dir=single_case_inter_dest_dir,
                                              best_dest_dir=single_case_inter_best_dir,
                                              weight_avg=weight_avg, weight_color=weight_color,
                                              save_inter=False, desig_eps=15)
        generate_testing_prototype_new(target_dataset_path=best_cluster_path,
                                    dest_dataset_path=single_case_prototype_dest_dir)
    
    
    # train the model & validate the model and save the results
    print(
        "====================================== TRAIN THE SIAMESE NETWORK MODEL ....... ==================================")
    folder_dataset = datasets.ImageFolder(root=regrouped_target_dir)
    siamese_train_dataset = SiameseNetworkDataset_1(imageFolderDataset=folder_dataset,
                                                    transform=transformation)
    training_dataset = siamese_train_dataset
    best_epoch_list, best_batch_list, best_model_path_list = Training_Loop(model_num=model_num, net_name=net_name,
                                                                           dataset=training_dataset,
                                                                           save_best_model_dir=save_best_si_model_dir,
                                                                           save_inter_model_dir=save_inter_si_model_dir,
                                                                           epoch_num=si_epoch_num, way_list=[5],
                                                                           vali_data_dir=vali_data_dir,
                                                                           proto_data_dir=test_prototype_dest_dir,
                                                                           csv_save_dir=csv_save_dir,
                                                                           case_len=case_len, net_type=net_type)
    print("best_epoch_list: ", best_epoch_list)
    print("best_batch_list: ", best_batch_list)
    print("best_model_path_list: ", best_model_path_list)
    
    print(
        "====================================== TESTING SIAMESE NETWORK MODEL ON TESTING DATA ....... ==================================")

    result_df, avg_df = group_test(model_num=model_num, way_list=way_list, case_len=case_len, epoch=best_epoch_list[0],
                                   model_path=best_model_path_list[0], batch_num=best_batch_list[0], net_name=net_name,
                                   net_type=net_type, test_data_dir=vali_data_dir, test_data_postfix="",
                                   proto_data_dir=test_prototype_dest_dir, proto_data_postfix="")
    
    
    print(result_df)
    print(avg_df)
    result_csv_path = os.sep.join([os.path.normpath(result_csv_dir), "results.csv"])
    avg_csv_path = os.sep.join([os.path.normpath(result_csv_dir), "avg.csv"])
    result_df.to_csv(result_csv_path)
    avg_df.to_csv(avg_csv_path)

def complete_training_loop_4(autoencoder_training_data_path, autoencoder_save_model_path,
                            autoencoder_save_img_base_path,
                            training_data_cluster_target, training_clustered_inter_dir,
                            training_clustered_best_dir,
                            train_prototype_dir, regrouped_target_dir,
                            test_cluster_target_dir, testing_clustered_inter_dir,
                            testing_clustered_best_dir,
                            save_best_si_model_dir, save_inter_si_model_dir,
                            si_epoch_num, vali_data_dir,
                            way_list, net_name, csv_save_dir, case_len, net_type, shots,
                            test_prototype_dest_dir,
                            proto_clustered_best_dest_dir, proto_clustered_inter_dest_dir,
                            result_csv_dir, transformation,
                            weight_avg, weight_color, threshold_cluster,
                            model_num=2, auto_epoch_num=100):
    '''
    with clustering, regrouping, pretrained network, concatencation
    representation: imgh
    clustering: Hi
    pretrained: autoencoder
    '''
    '''
    training and testing loop for model using Autoencoder Embedding for clustering.
    1. Training Autoencoder
    2. Clustering on Training data and clustering on testing data
    3. Generate Prototypes
    4. Re-group Training data
    5. Training Siamese network on Training data
    6. Testing Siamese network on Testing data
    '''
    cluster_target_folder_names = os.listdir(test_cluster_target_dir)
    cluster_target_folders = [os.sep.join([os.path.normpath(test_cluster_target_dir), i]) for i in
                              cluster_target_folder_names]
    
    #  1. Training Autoencoder
    print("====================================== TRAINING AUTOENCODER ....... ==================================")
    AutoEncoder_Training_Loop(epoch_num=auto_epoch_num,
                              data_path=autoencoder_training_data_path,
                              save_model_path=autoencoder_save_model_path,
                              save_img_base_path=autoencoder_save_img_base_path, transformation=transformation)

    
    # 2. Clustering on Training data
    print("====================================== CLUSTERING ON TRAINING DATA ....... ==================================")
    
    net = Autoencoder_Conv().cuda()
    net.load_state_dict(torch.load(autoencoder_save_model_path))
    
    best_cluster_path = HI_clustering_with_imgh_new(
        dataset_path=training_data_cluster_target,
        dest_dir=training_clustered_best_dir,
        weight_avg=weight_avg, weight_color=weight_color, 
        threshold_cluster=threshold_cluster
        )
    train_best_clustered_path = best_cluster_path

    # 3. Generate Prototypes
    print("====================================== GENERATING TRAINING PROTOTYPES ....... ==================================")
    generate_training_prototype(target_dataset_path=best_cluster_path, dest_dataset_path=train_prototype_dir)
    
    # 4. Re-group Training data
    print("====================================== RE-GROUP TRAINING DATA ....... ==================================")
    best_cluster_path = HI_clustering_with_imgh_new(
        dataset_path=train_prototype_dir,
        dest_dir=proto_clustered_best_dest_dir,
        weight_avg=weight_avg, weight_color=weight_color, 
        threshold_cluster=threshold_cluster
        )

    # train_best_clustered_path = "C:/Users/Lizzie0930/Desktop/git/Results_Auto/4_train_clustered_best_dest/final_clustered_dataset"
    re_group_train_data_for_Hi(source_dir=train_best_clustered_path, info_dir=best_cluster_path,
                                target_dir=regrouped_target_dir)
    
    # 2. Clustering on testing data
    print("====================================== CLUSTERING ON TESTING DATA ....... ==================================")
    """
        should cut testing data into known and unknown, and generate prototype only on known data
        2. pass known_cluster_target_folders to test_cluster_target_dir
        3. change generate_testing_prototype to generate_testing_prototype_new
        4. pass unknown_cluster_target_folders to vali_data_dir
    """
    for i in range(len(cluster_target_folders)):
        single_case_inter_dest_dir = os.sep.join(
            [os.path.normpath(testing_clustered_inter_dir), cluster_target_folder_names[i]])
        if not os.path.exists(single_case_inter_dest_dir):
            os.mkdir(single_case_inter_dest_dir)
        single_case_inter_best_dir = os.sep.join(
            [os.path.normpath(testing_clustered_best_dir), cluster_target_folder_names[i]])
        if not os.path.exists(single_case_inter_best_dir):
            os.mkdir(single_case_inter_best_dir)
        single_case_prototype_dest_dir = os.sep.join(
            [os.path.normpath(test_prototype_dest_dir), cluster_target_folder_names[i]])
        if not os.path.exists(single_case_prototype_dest_dir):
            os.mkdir(single_case_prototype_dest_dir)
        best_cluster_path = HI_clustering_with_imgh_new(dataset_path=cluster_target_folders[i],
                                              dest_dir=single_case_inter_best_dir, 
                                              weight_avg=weight_avg, weight_color=weight_color, 
                                              threshold_cluster=threshold_cluster)
        generate_testing_prototype_new(target_dataset_path=best_cluster_path,
                                    dest_dataset_path=single_case_prototype_dest_dir)
    

    # train the model & validate the model and save the results
    print(
        "====================================== TRAIN THE SIAMESE NETWORK MODEL ....... ==================================")
    folder_dataset = datasets.ImageFolder(root=regrouped_target_dir)
    siamese_train_dataset = SiameseNetworkDataset_1(imageFolderDataset=folder_dataset,
                                                    transform=transformation)
    training_dataset = siamese_train_dataset
    best_epoch_list, best_batch_list, best_model_path_list = Training_Loop(model_num=model_num, net_name=net_name,
                                                                           dataset=training_dataset,
                                                                           save_best_model_dir=save_best_si_model_dir,
                                                                           save_inter_model_dir=save_inter_si_model_dir,
                                                                           epoch_num=si_epoch_num, way_list=[5],
                                                                           vali_data_dir=vali_data_dir,
                                                                           proto_data_dir=test_prototype_dest_dir,
                                                                           csv_save_dir=csv_save_dir,
                                                                           case_len=case_len, net_type=net_type)
    print("best_epoch_list: ", best_epoch_list)
    print("best_batch_list: ", best_batch_list)
    print("best_model_path_list: ", best_model_path_list)

    print(
        "====================================== TESTING SIAMESE NETWORK MODEL ON TESTING DATA ....... ==================================")

    result_df, avg_df = group_test(model_num=model_num, way_list=way_list, case_len=case_len, epoch=best_epoch_list[0],
                                   model_path=best_model_path_list[0], batch_num=best_batch_list[0], net_name=net_name,
                                   net_type=net_type, test_data_dir=vali_data_dir, test_data_postfix="",
                                   proto_data_dir=test_prototype_dest_dir, proto_data_postfix="")
    
    print(result_df)
    print(avg_df)
    result_csv_path = os.sep.join([os.path.normpath(result_csv_dir), "results.csv"])
    avg_csv_path = os.sep.join([os.path.normpath(result_csv_dir), "avg.csv"])
    result_df.to_csv(result_csv_path)
    avg_df.to_csv(avg_csv_path)

def complete_training_loop_5(autoencoder_training_data_path, autoencoder_save_model_path,
                            autoencoder_save_img_base_path,
                            training_data_cluster_target, training_clustered_inter_dir,
                            training_clustered_best_dir,
                            train_prototype_dir, regrouped_target_dir,
                            test_cluster_target_dir, testing_clustered_inter_dir,
                            testing_clustered_best_dir,
                            save_best_si_model_dir, save_inter_si_model_dir,
                            si_epoch_num, vali_data_dir,
                            save_best_si_model_dir_cl_siame, save_inter_si_model_dir_cl_siame,
                            csv_save_dir_cl_siame,
                            way_list, net_name, csv_save_dir, case_len, net_type, shots,
                            test_prototype_dest_dir,
                            proto_clustered_best_dest_dir, proto_clustered_inter_dest_dir,
                            result_csv_dir, transformation,
                            model_num=2, auto_epoch_num=100):
    '''
    with clustering, regrouping, pretrained network, concatencation
    representation: siame Emb
    clustering: DBSCAN
    pretrained: autoencoder
    '''
    '''
    training and testing loop for model using Autoencoder Embedding for clustering.
    1. Training Autoencoder
    2. Clustering on Training data and clustering on testing data
    3. Generate Prototypes
    4. Re-group Training data
    5. Training Siamese network on Training data
    6. Testing Siamese network on Testing data
    '''
    cluster_target_folder_names = os.listdir(test_cluster_target_dir)
    cluster_target_folders = [os.sep.join([os.path.normpath(test_cluster_target_dir), i]) for i in
                              cluster_target_folder_names]
    
    #  1. Training Autoencoder
    print("====================================== TRAINING AUTOENCODER ....... ==================================")
    AutoEncoder_Training_Loop(epoch_num=auto_epoch_num,
                              data_path=autoencoder_training_data_path,
                              save_model_path=autoencoder_save_model_path,
                              save_img_base_path=autoencoder_save_img_base_path, transformation=transformation)
    
    
    # train the model & validate the model and save the results
    print(
        "====================================== TRAIN THE SIAMESE NETWORK MODEL FOR CLUSTERING ....... ==================================")
    folder_dataset = datasets.ImageFolder(root=training_data_cluster_target)
    siamese_train_dataset = SiameseNetworkDataset_1(imageFolderDataset=folder_dataset,
                                                    transform=transformation)
    training_dataset = siamese_train_dataset
    best_epoch_list, best_batch_list, best_model_path_list = Training_Loop(model_num=2, net_name=net_name,
                                                                           dataset=training_dataset,
                                                                           save_best_model_dir=save_best_si_model_dir_cl_siame,
                                                                           save_inter_model_dir=save_inter_si_model_dir_cl_siame,
                                                                           epoch_num=si_epoch_num, way_list=[5],
                                                                           vali_data_dir=vali_data_dir,
                                                                           proto_data_dir=test_cluster_target_dir,
                                                                           csv_save_dir=csv_save_dir_cl_siame,
                                                                           case_len=case_len, net_type=net_type)
    print("best_epoch_list: ", best_epoch_list)
    print("best_batch_list: ", best_batch_list)
    print("best_model_path_list: ", best_model_path_list)

    # 2. Clustering on Training data
    print("====================================== CLUSTERING ON TRAINING DATA ....... ==================================")
    
    net = SiameseNetwork_autoencoder_Based().cuda()
    net.load_state_dict(torch.load(best_model_path_list[0]))
    
    best_cluster_path = DBSCAN_clustering_with_siame_emb_new(
        dataset_path=training_data_cluster_target,
        transformation=transformation,
        inter_dest_dir=training_clustered_inter_dir,
        best_dest_dir=training_clustered_best_dir,
        net=net, save_inter=False,
        desig_eps=0.5)
    train_best_clustered_path = best_cluster_path

    # 3. Generate Prototypes
    print("====================================== GENERATING TRAINING PROTOTYPES ....... ==================================")
    generate_training_prototype(target_dataset_path=best_cluster_path, dest_dataset_path=train_prototype_dir)
    
    # 4. Re-group Training data
    print("====================================== RE-GROUP TRAINING DATA ....... ==================================")
    best_cluster_path = DBSCAN_clustering_with_siame_emb_new(
        dataset_path=train_prototype_dir,
        transformation=transformation,
        inter_dest_dir=proto_clustered_inter_dest_dir,
        best_dest_dir=proto_clustered_best_dest_dir,
        net=net, save_inter=True,
        desig_eps=0.4)#17

    # train_best_clustered_path = "C:/Users/Lizzie0930/Desktop/git/Results_Auto/4_train_clustered_best_dest/final_clustered_dataset"
    re_group_train_data(source_dir=train_best_clustered_path, info_dir=best_cluster_path,
                        target_dir=regrouped_target_dir)
    
    # 2. Clustering on testing data
    print("====================================== CLUSTERING ON TESTING DATA ....... ==================================")
    """
        should cut testing data into known and unknown, and generate prototype only on known data
        2. pass known_cluster_target_folders to test_cluster_target_dir
        3. change generate_testing_prototype to generate_testing_prototype_new
        4. pass unknown_cluster_target_folders to vali_data_dir
    """
    for i in range(len(cluster_target_folders)):
        single_case_inter_dest_dir = os.sep.join(
            [os.path.normpath(testing_clustered_inter_dir), cluster_target_folder_names[i]])
        if not os.path.exists(single_case_inter_dest_dir):
            os.mkdir(single_case_inter_dest_dir)
        single_case_inter_best_dir = os.sep.join(
            [os.path.normpath(testing_clustered_best_dir), cluster_target_folder_names[i]])
        if not os.path.exists(single_case_inter_best_dir):
            os.mkdir(single_case_inter_best_dir)
        single_case_prototype_dest_dir = os.sep.join(
            [os.path.normpath(test_prototype_dest_dir), cluster_target_folder_names[i]])
        if not os.path.exists(single_case_prototype_dest_dir):
            os.mkdir(single_case_prototype_dest_dir)
        best_cluster_path = DBSCAN_clustering_with_siame_emb_new(dataset_path=cluster_target_folders[i], transformation=transformation,
                                              inter_dest_dir=single_case_inter_dest_dir,
                                              best_dest_dir=single_case_inter_best_dir, net=net,
                                              save_inter=False, desig_eps=0.5)
        generate_testing_prototype_new(target_dataset_path=best_cluster_path,
                                    dest_dataset_path=single_case_prototype_dest_dir)
    
    
    # train the model & validate the model and save the results
    print(
        "====================================== TRAIN THE SIAMESE NETWORK MODEL ....... ==================================")
    folder_dataset = datasets.ImageFolder(root=regrouped_target_dir)
    siamese_train_dataset = SiameseNetworkDataset_1(imageFolderDataset=folder_dataset,
                                                    transform=transformation)
    training_dataset = siamese_train_dataset
    best_epoch_list, best_batch_list, best_model_path_list = Training_Loop(model_num=model_num, net_name=net_name,
                                                                           dataset=training_dataset,
                                                                           save_best_model_dir=save_best_si_model_dir,
                                                                           save_inter_model_dir=save_inter_si_model_dir,
                                                                           epoch_num=si_epoch_num, way_list=[5],
                                                                           vali_data_dir=vali_data_dir,
                                                                           proto_data_dir=test_prototype_dest_dir,
                                                                           csv_save_dir=csv_save_dir,
                                                                           case_len=case_len, net_type=net_type)
    print("best_epoch_list: ", best_epoch_list)
    print("best_batch_list: ", best_batch_list)
    print("best_model_path_list: ", best_model_path_list)

    print(
        "====================================== TESTING SIAMESE NETWORK MODEL ON TESTING DATA ....... ==================================")

    result_df, avg_df = group_test(model_num=model_num, way_list=way_list, case_len=case_len, epoch=best_epoch_list[0],
                                   model_path=best_model_path_list[0], batch_num=best_batch_list[0], net_name=net_name,
                                   net_type=net_type, test_data_dir=vali_data_dir, test_data_postfix="",
                                   proto_data_dir=test_prototype_dest_dir, proto_data_postfix="")
    
    print(result_df)
    print(avg_df)
    result_csv_path = os.sep.join([os.path.normpath(result_csv_dir), "results.csv"])
    avg_csv_path = os.sep.join([os.path.normpath(result_csv_dir), "avg.csv"])
    result_df.to_csv(result_csv_path)
    avg_df.to_csv(avg_csv_path)

def complete_training_loop_6(autoencoder_training_data_path, autoencoder_save_model_path,
                            autoencoder_save_img_base_path,
                            training_data_cluster_target, training_clustered_inter_dir,
                            training_clustered_best_dir,
                            train_prototype_dir, regrouped_target_dir,
                            test_cluster_target_dir, testing_clustered_inter_dir,
                            testing_clustered_best_dir,
                            save_best_si_model_dir, save_inter_si_model_dir,
                            si_epoch_num, vali_data_dir,
                            save_best_si_model_dir_cl_siame, save_inter_si_model_dir_cl_siame,
                            csv_save_dir_cl_siame,
                            way_list, net_name, csv_save_dir, case_len, net_type, shots,
                            test_prototype_dest_dir,
                            proto_clustered_best_dest_dir, proto_clustered_inter_dest_dir,
                            result_csv_dir, transformation,
                            weight_avg, weight_color, 
                            threshold_combine, threshold_cluster,
                            model_num=2, auto_epoch_num=100):
    '''
    with clustering, regrouping, pretrained network, concatencation
    representation: siame Emb
    clustering: Hi
    pretrained: autoencoder
    '''
    '''
    training and testing loop for model using Autoencoder Embedding for clustering.
    1. Training Autoencoder
    2. Clustering on Training data and clustering on testing data
    3. Generate Prototypes
    4. Re-group Training data
    5. Training Siamese network on Training data
    6. Testing Siamese network on Testing data
    '''
    cluster_target_folder_names = os.listdir(test_cluster_target_dir)
    cluster_target_folders = [os.sep.join([os.path.normpath(test_cluster_target_dir), i]) for i in
                              cluster_target_folder_names]
    
    #  1. Training Autoencoder
    print("====================================== TRAINING AUTOENCODER ....... ==================================")
    AutoEncoder_Training_Loop(epoch_num=auto_epoch_num,
                              data_path=autoencoder_training_data_path,
                              save_model_path=autoencoder_save_model_path,
                              save_img_base_path=autoencoder_save_img_base_path, transformation=transformation)
    
    
    # train the model & validate the model and save the results
    print(
        "====================================== TRAIN THE SIAMESE NETWORK MODEL FOR CLUSTERING ....... ==================================")
    folder_dataset = datasets.ImageFolder(root=training_data_cluster_target)
    siamese_train_dataset = SiameseNetworkDataset_1(imageFolderDataset=folder_dataset,
                                                    transform=transformation)
    training_dataset = siamese_train_dataset
    best_epoch_list, best_batch_list, best_model_path_list = Training_Loop(model_num=2, net_name=net_name,
                                                                           dataset=training_dataset,
                                                                           save_best_model_dir=save_best_si_model_dir_cl_siame,
                                                                           save_inter_model_dir=save_inter_si_model_dir_cl_siame,
                                                                           epoch_num=si_epoch_num, way_list=[5],
                                                                           vali_data_dir=vali_data_dir,
                                                                           proto_data_dir=test_cluster_target_dir,
                                                                           csv_save_dir=csv_save_dir_cl_siame,
                                                                           case_len=case_len, net_type=net_type)
    print("best_epoch_list: ", best_epoch_list)
    print("best_batch_list: ", best_batch_list)
    print("best_model_path_list: ", best_model_path_list)
    
    # 2. Clustering on Training data
    print("====================================== CLUSTERING ON TRAINING DATA ....... ==================================")
    
    net = SiameseNetwork_autoencoder_Based().cuda()
    net.load_state_dict(torch.load(best_model_path_list[0]))
    
    best_cluster_path = HI_clustering_with_siame_emb_new(
        dataset_path=training_data_cluster_target,
        dest_dir=training_clustered_best_dir,
        transformation=transformation,net=net,
        weight_avg=weight_avg, weight_color=weight_color, 
        threshold_combine=threshold_combine, 
        threshold_cluster=threshold_cluster)
    train_best_clustered_path = best_cluster_path

    # 3. Generate Prototypes
    print("====================================== GENERATING TRAINING PROTOTYPES ....... ==================================")
    generate_training_prototype(target_dataset_path=best_cluster_path, dest_dataset_path=train_prototype_dir)
    
    # 4. Re-group Training data
    print("====================================== RE-GROUP TRAINING DATA ....... ==================================")
    best_cluster_path = HI_clustering_with_siame_emb_new(
        dataset_path=train_prototype_dir,
        dest_dir=proto_clustered_best_dest_dir,
        transformation=transformation,net=net,
        weight_avg=weight_avg, weight_color=weight_color, 
        threshold_combine=threshold_combine, 
        threshold_cluster=threshold_cluster)
    
    
    re_group_train_data_for_Hi(source_dir=train_best_clustered_path, info_dir=best_cluster_path,
                        target_dir=regrouped_target_dir)
    
    # 2. Clustering on testing data
    print("====================================== CLUSTERING ON TESTING DATA ....... ==================================")
    """
        should cut testing data into known and unknown, and generate prototype only on known data
        2. pass known_cluster_target_folders to test_cluster_target_dir
        3. change generate_testing_prototype to generate_testing_prototype_new
        4. pass unknown_cluster_target_folders to vali_data_dir
    """
    for i in range(len(cluster_target_folders)):
        single_case_inter_dest_dir = os.sep.join(
            [os.path.normpath(testing_clustered_inter_dir), cluster_target_folder_names[i]])
        if not os.path.exists(single_case_inter_dest_dir):
            os.mkdir(single_case_inter_dest_dir)
        single_case_inter_best_dir = os.sep.join(
            [os.path.normpath(testing_clustered_best_dir), cluster_target_folder_names[i]])
        if not os.path.exists(single_case_inter_best_dir):
            os.mkdir(single_case_inter_best_dir)
        single_case_prototype_dest_dir = os.sep.join(
            [os.path.normpath(test_prototype_dest_dir), cluster_target_folder_names[i]])
        if not os.path.exists(single_case_prototype_dest_dir):
            os.mkdir(single_case_prototype_dest_dir)
        best_cluster_path = HI_clustering_with_siame_emb_new(
                            dataset_path=cluster_target_folders[i],
                            dest_dir=single_case_inter_best_dir,
                            transformation=transformation,net=net,
                            weight_avg=weight_avg, weight_color=weight_color, 
                            threshold_combine=threshold_combine, 
                            threshold_cluster=threshold_cluster)
        generate_testing_prototype_new(target_dataset_path=best_cluster_path,
                                    dest_dataset_path=single_case_prototype_dest_dir)
    

    # train the model & validate the model and save the results
    print(
        "====================================== TRAIN THE SIAMESE NETWORK MODEL ....... ==================================")
    folder_dataset = datasets.ImageFolder(root=regrouped_target_dir)
    siamese_train_dataset = SiameseNetworkDataset_1(imageFolderDataset=folder_dataset,
                                                    transform=transformation)
    training_dataset = siamese_train_dataset
    best_epoch_list, best_batch_list, best_model_path_list = Training_Loop(model_num=model_num, net_name=net_name,
                                                                           dataset=training_dataset,
                                                                           save_best_model_dir=save_best_si_model_dir,
                                                                           save_inter_model_dir=save_inter_si_model_dir,
                                                                           epoch_num=si_epoch_num, way_list=[5],
                                                                           vali_data_dir=vali_data_dir,
                                                                           proto_data_dir=test_prototype_dest_dir,
                                                                           csv_save_dir=csv_save_dir,
                                                                           case_len=case_len, net_type=net_type)
    print("best_epoch_list: ", best_epoch_list)
    print("best_batch_list: ", best_batch_list)
    print("best_model_path_list: ", best_model_path_list)

    print(
        "====================================== TESTING SIAMESE NETWORK MODEL ON TESTING DATA ....... ==================================")

    result_df, avg_df = group_test(model_num=model_num, way_list=way_list, case_len=case_len, epoch=best_epoch_list[0],
                                   model_path=best_model_path_list[0], batch_num=best_batch_list[0], net_name=net_name,
                                   net_type=net_type, test_data_dir=vali_data_dir, test_data_postfix="",
                                   proto_data_dir=test_prototype_dest_dir, proto_data_postfix="")
    
    print(result_df)
    print(avg_df)
    result_csv_path = os.sep.join([os.path.normpath(result_csv_dir), "results.csv"])
    avg_csv_path = os.sep.join([os.path.normpath(result_csv_dir), "avg.csv"])
    result_df.to_csv(result_csv_path)
    avg_df.to_csv(avg_csv_path)

def complete_training_loop_Pr1(autoencoder_training_data_path, autoencoder_save_model_path,
                            autoencoder_save_img_base_path,
                            training_data_cluster_target, training_clustered_inter_dir,
                            training_clustered_best_dir,
                            train_prototype_dir, regrouped_target_dir,
                            test_cluster_target_dir, testing_clustered_inter_dir,
                            testing_clustered_best_dir,
                            save_best_si_model_dir, save_inter_si_model_dir,
                            si_epoch_num, vali_data_dir,
                            way_list, net_name, csv_save_dir, case_len, net_type, shots,
                            test_prototype_dest_dir,
                            proto_clustered_best_dest_dir, proto_clustered_inter_dest_dir,
                            result_csv_dir, transformation,
                            model_num=2, auto_epoch_num=100):
    '''
    with clustering, regrouping, pretrained network, concatencation
    representation: auoencoder Emb
    clustering: DBSCAN
    pretrained: autoencoder
    '''
    '''
    training and testing loop for model using Autoencoder Embedding for clustering.
    1. Training Autoencoder
    2. Clustering on Training data and clustering on testing data
    3. Generate Prototypes
    4. Re-group Training data
    5. Training Siamese network on Training data
    6. Testing Siamese network on Testing data
    '''
    cluster_target_folder_names = os.listdir(test_cluster_target_dir)
    cluster_target_folders = [os.sep.join([os.path.normpath(test_cluster_target_dir), i]) for i in
                              cluster_target_folder_names]
    
    #  1. Training Autoencoder
    print("====================================== TRAINING AUTOENCODER ....... ==================================")
    AutoEncoder_Training_Loop(epoch_num=auto_epoch_num,
                              data_path=autoencoder_training_data_path,
                              save_model_path=autoencoder_save_model_path,
                              save_img_base_path=autoencoder_save_img_base_path, transformation=transformation)

    # 2. Clustering on Training data
    print("====================================== CLUSTERING ON TRAINING DATA ....... ==================================")
    
    net = Autoencoder_Conv().cuda()
    net.load_state_dict(torch.load(autoencoder_save_model_path))
    
    best_cluster_path = DBSCAN_clustering_with_auto_emb_new(
        dataset_path=training_data_cluster_target,
        transformation=transformation,
        inter_dest_dir=training_clustered_inter_dir,
        best_dest_dir=training_clustered_best_dir,
        net=net, save_inter=False,
        desig_eps=15)
    train_best_clustered_path = best_cluster_path

    # 3. Generate Prototypes
    print("====================================== GENERATING TRAINING PROTOTYPES ....... ==================================")
    generate_training_prototype(target_dataset_path=best_cluster_path, dest_dataset_path=train_prototype_dir)
    
    # 4. Re-group Training data
    print("====================================== RE-GROUP TRAINING DATA ....... ==================================")
    best_cluster_path = DBSCAN_clustering_with_auto_emb_new(
        dataset_path=train_prototype_dir,
        transformation=transformation,
        inter_dest_dir=proto_clustered_inter_dest_dir,
        best_dest_dir=proto_clustered_best_dest_dir,
        net=net, save_inter=True,
        desig_eps=13)#17

    # train_best_clustered_path = "C:/Users/Lizzie0930/Desktop/git/Results_Auto/4_train_clustered_best_dest/final_clustered_dataset"
    re_group_train_data(source_dir=train_best_clustered_path, info_dir=best_cluster_path,
                        target_dir=regrouped_target_dir)
    
    # 2. Clustering on testing data
    print("====================================== CLUSTERING ON TESTING DATA ....... ==================================")
    """
        should cut testing data into known and unknown, and generate prototype only on known data
        2. pass known_cluster_target_folders to test_cluster_target_dir
        3. change generate_testing_prototype to generate_testing_prototype_new
        4. pass unknown_cluster_target_folders to vali_data_dir
    """
    
    for i in range(len(cluster_target_folders)):
        single_case_inter_dest_dir = os.sep.join(
            [os.path.normpath(testing_clustered_inter_dir), cluster_target_folder_names[i]])
        if not os.path.exists(single_case_inter_dest_dir):
            os.mkdir(single_case_inter_dest_dir)
        single_case_inter_best_dir = os.sep.join(
            [os.path.normpath(testing_clustered_best_dir), cluster_target_folder_names[i]])
        if not os.path.exists(single_case_inter_best_dir):
            os.mkdir(single_case_inter_best_dir)
        single_case_prototype_dest_dir = os.sep.join(
            [os.path.normpath(test_prototype_dest_dir), cluster_target_folder_names[i]])
        if not os.path.exists(single_case_prototype_dest_dir):
            os.mkdir(single_case_prototype_dest_dir)
        best_cluster_path = DBSCAN_clustering_with_auto_emb_new(dataset_path=cluster_target_folders[i], transformation=transformation,
                                              inter_dest_dir=single_case_inter_dest_dir,
                                              best_dest_dir=single_case_inter_best_dir, net=net,
                                              save_inter=False, desig_eps=15)
        generate_testing_prototype_new(target_dataset_path=best_cluster_path,
                                    dest_dataset_path=single_case_prototype_dest_dir)
    
    
    # train the model & validate the model and save the results
    print(
        "====================================== TRAIN THE SIAMESE NETWORK MODEL ....... ==================================")
    
    folder_dataset = datasets.ImageFolder(root=regrouped_target_dir)
    siamese_train_dataset = SiameseNetworkDataset_1(imageFolderDataset=folder_dataset,
                                                    transform=transformation)
    training_dataset = siamese_train_dataset
    best_epoch_list, best_batch_list, best_model_path_list = Training_Loop(model_num=model_num, net_name=net_name,
                                                                           dataset=training_dataset,
                                                                           save_best_model_dir=save_best_si_model_dir,
                                                                           save_inter_model_dir=save_inter_si_model_dir,
                                                                           epoch_num=si_epoch_num, way_list=way_list,
                                                                           vali_data_dir=vali_data_dir,
                                                                           proto_data_dir=test_prototype_dest_dir,
                                                                           csv_save_dir=csv_save_dir,
                                                                           case_len=case_len, net_type=net_type)
    print("best_epoch_list: ", best_epoch_list)
    print("best_batch_list: ", best_batch_list)
    print("best_model_path_list: ", best_model_path_list)
    
    print(
        "====================================== TESTING SIAMESE NETWORK MODEL ON TESTING DATA ....... ==================================")

    result_df, avg_df = group_test(model_num=model_num, way_list=way_list, case_len=case_len, epoch=20,
                                   model_path=best_model_path_list[0], batch_num=22, net_name=net_name,
                                   net_type=net_type, test_data_dir=vali_data_dir, test_data_postfix="",
                                   proto_data_dir=test_prototype_dest_dir, proto_data_postfix="")
    
    print(result_df)
    print(avg_df)
    result_csv_path = os.sep.join([os.path.normpath(result_csv_dir), "results.csv"])
    avg_csv_path = os.sep.join([os.path.normpath(result_csv_dir), "avg.csv"])
    result_df.to_csv(result_csv_path)
    avg_df.to_csv(avg_csv_path)

def complete_training_loop_Pr2(autoencoder_training_data_path, autoencoder_save_model_path,
                            autoencoder_save_img_base_path,
                            training_data_cluster_target, training_clustered_inter_dir,
                            training_clustered_best_dir,
                            train_prototype_dir, regrouped_target_dir,
                            test_cluster_target_dir, testing_clustered_inter_dir,
                            testing_clustered_best_dir,
                            save_best_si_model_dir, save_inter_si_model_dir,
                            si_epoch_num, vali_data_dir,
                            way_list, net_name, csv_save_dir, case_len, net_type, shots,
                            test_prototype_dest_dir,
                            proto_clustered_best_dest_dir, proto_clustered_inter_dest_dir,
                            result_csv_dir, transformation,
                            model_num=2, auto_epoch_num=100):
    '''
    with clustering, regrouping, pretrained network, concatencation
    representation: auoencoder Emb
    clustering: DBSCAN
    pretrained: autoencoder
    '''
    '''
    training and testing loop for model using Autoencoder Embedding for clustering.
    1. Training Autoencoder
    2. Clustering on Training data and clustering on testing data
    3. Generate Prototypes
    4. Re-group Training data
    5. Training Siamese network on Training data
    6. Testing Siamese network on Testing data
    '''
    cluster_target_folder_names = os.listdir(test_cluster_target_dir)
    cluster_target_folders = [os.sep.join([os.path.normpath(test_cluster_target_dir), i]) for i in
                              cluster_target_folder_names]
    
    #  1. Training Autoencoder
    print("====================================== TRAINING AUTOENCODER ....... ==================================")
    AutoEncoder_Training_Loop(epoch_num=auto_epoch_num,
                              data_path=autoencoder_training_data_path,
                              save_model_path=autoencoder_save_model_path,
                              save_img_base_path=autoencoder_save_img_base_path, transformation=transformation)

    # 2. Clustering on Training data
    print("====================================== CLUSTERING ON TRAINING DATA ....... ==================================")
    
    
    net = Autoencoder_Conv().cuda()
    net.load_state_dict(torch.load(autoencoder_save_model_path))
    
    best_cluster_path = DBSCAN_clustering_with_auto_emb_new(
        dataset_path=training_data_cluster_target,
        transformation=transformation,
        inter_dest_dir=training_clustered_inter_dir,
        best_dest_dir=training_clustered_best_dir,
        net=net, save_inter=False,
        desig_eps=15)
    train_best_clustered_path = best_cluster_path

    # 3. Generate Prototypes
    print("====================================== GENERATING TRAINING PROTOTYPES ....... ==================================")
    generate_training_prototype(target_dataset_path=best_cluster_path, dest_dataset_path=train_prototype_dir)
    
    # 4. Re-group Training data
    print("====================================== RE-GROUP TRAINING DATA ....... ==================================")
    best_cluster_path = DBSCAN_clustering_with_auto_emb_new(
        dataset_path=train_prototype_dir,
        transformation=transformation,
        inter_dest_dir=proto_clustered_inter_dest_dir,
        best_dest_dir=proto_clustered_best_dest_dir,
        net=net, save_inter=True,
        desig_eps=13)#17

    # train_best_clustered_path = "C:/Users/Lizzie0930/Desktop/git/Results_Auto/4_train_clustered_best_dest/final_clustered_dataset"
    re_group_train_data(source_dir=train_best_clustered_path, info_dir=best_cluster_path,
                        target_dir=regrouped_target_dir)
    
    # 2. Clustering on testing data
    print("====================================== CLUSTERING ON TESTING DATA ....... ==================================")
    """
        should cut testing data into known and unknown, and generate prototype only on known data
        2. pass known_cluster_target_folders to test_cluster_target_dir
        3. change generate_testing_prototype to generate_testing_prototype_new
        4. pass unknown_cluster_target_folders to vali_data_dir
    """
    
    for i in range(len(cluster_target_folders)):
        single_case_inter_dest_dir = os.sep.join(
            [os.path.normpath(testing_clustered_inter_dir), cluster_target_folder_names[i]])
        if not os.path.exists(single_case_inter_dest_dir):
            os.mkdir(single_case_inter_dest_dir)
        single_case_inter_best_dir = os.sep.join(
            [os.path.normpath(testing_clustered_best_dir), cluster_target_folder_names[i]])
        if not os.path.exists(single_case_inter_best_dir):
            os.mkdir(single_case_inter_best_dir)
        single_case_prototype_dest_dir = os.sep.join(
            [os.path.normpath(test_prototype_dest_dir), cluster_target_folder_names[i]])
        if not os.path.exists(single_case_prototype_dest_dir):
            os.mkdir(single_case_prototype_dest_dir)
        best_cluster_path = DBSCAN_clustering_with_auto_emb_new(dataset_path=cluster_target_folders[i], transformation=transformation,
                                              inter_dest_dir=single_case_inter_dest_dir,
                                              best_dest_dir=single_case_inter_best_dir, net=net,
                                              save_inter=False, desig_eps=15)
        generate_testing_prototype_new(target_dataset_path=best_cluster_path,
                                    dest_dataset_path=single_case_prototype_dest_dir)
    
    
    # train the model & validate the model and save the results
    print(
        "====================================== TRAIN THE SIAMESE NETWORK MODEL ....... ==================================")
    
    folder_dataset = datasets.ImageFolder(root=regrouped_target_dir)
    siamese_train_dataset = SiameseNetworkDataset_1(imageFolderDataset=folder_dataset,
                                                    transform=transformation)
    training_dataset = siamese_train_dataset
    best_epoch_list, best_batch_list, best_model_path_list = Training_Loop_2(model_num=model_num, net_name=net_name,
                                                                           dataset=training_dataset,
                                                                           save_best_model_dir=save_best_si_model_dir,
                                                                           save_inter_model_dir=save_inter_si_model_dir,
                                                                           epoch_num=si_epoch_num, way_list=[5],
                                                                           vali_data_dir=vali_data_dir,
                                                                           proto_data_dir=test_prototype_dest_dir,
                                                                           csv_save_dir=csv_save_dir,
                                                                           case_len=case_len, net_type=net_type)
    print("best_epoch_list: ", best_epoch_list)
    print("best_batch_list: ", best_batch_list)
    print("best_model_path_list: ", best_model_path_list)

    print(
        "====================================== TESTING SIAMESE NETWORK MODEL ON TESTING DATA ....... ==================================")


    result_df, avg_df = group_test_2(model_num=model_num, way_list=way_list, case_len=case_len, epoch=best_epoch_list[0],
                                   model_path=best_model_path_list[0], batch_num=best_batch_list[0], net_name=net_name,
                                   net_type=net_type, test_data_dir=vali_data_dir, test_data_postfix="",
                                   proto_data_dir=test_prototype_dest_dir, proto_data_postfix="")
    
    print(result_df)
    print(avg_df)
    result_csv_path = os.sep.join([os.path.normpath(result_csv_dir), "results.csv"])
    avg_csv_path = os.sep.join([os.path.normpath(result_csv_dir), "avg.csv"])
    result_df.to_csv(result_csv_path)
    avg_df.to_csv(avg_csv_path)

def complete_training_loop_Pr3(autoencoder_training_data_path, autoencoder_save_model_path,
                            autoencoder_save_img_base_path,
                            training_data_cluster_target, training_clustered_inter_dir,
                            training_clustered_best_dir,
                            train_prototype_dir, regrouped_target_dir,
                            test_cluster_target_dir, testing_clustered_inter_dir,
                            testing_clustered_best_dir,
                            save_best_si_model_dir, save_inter_si_model_dir,
                            si_epoch_num, vali_data_dir,
                            way_list, net_name, csv_save_dir, case_len, net_type, shots,
                            test_prototype_dest_dir,
                            proto_clustered_best_dest_dir, proto_clustered_inter_dest_dir,
                            result_csv_dir, transformation,
                            model_num=2, auto_epoch_num=100):
    '''
    with clustering, regrouping, pretrained network, concatencation
    representation: auoencoder Emb
    clustering: DBSCAN
    pretrained: RESNET
    '''
    '''
    training and testing loop for model using Autoencoder Embedding for clustering.
    1. Training Autoencoder
    2. Clustering on Training data and clustering on testing data
    3. Generate Prototypes
    4. Re-group Training data
    5. Training Siamese network on Training data
    6. Testing Siamese network on Testing data
    '''
    cluster_target_folder_names = os.listdir(test_cluster_target_dir)
    cluster_target_folders = [os.sep.join([os.path.normpath(test_cluster_target_dir), i]) for i in
                              cluster_target_folder_names]
    
    #  1. Training Autoencoder
    print("====================================== TRAINING AUTOENCODER ....... ==================================")
    AutoEncoder_Training_Loop(epoch_num=auto_epoch_num,
                              data_path=autoencoder_training_data_path,
                              save_model_path=autoencoder_save_model_path,
                              save_img_base_path=autoencoder_save_img_base_path, transformation=transformation)

    # 2. Clustering on Training data
    print("====================================== CLUSTERING ON TRAINING DATA ....... ==================================")
    
    
    net = Autoencoder_Conv().cuda()
    net.load_state_dict(torch.load(autoencoder_save_model_path))
    
    best_cluster_path = DBSCAN_clustering_with_auto_emb_new(
        dataset_path=training_data_cluster_target,
        transformation=transformation,
        inter_dest_dir=training_clustered_inter_dir,
        best_dest_dir=training_clustered_best_dir,
        net=net, save_inter=False,
        desig_eps=15)
    train_best_clustered_path = best_cluster_path

    # 3. Generate Prototypes
    print("====================================== GENERATING TRAINING PROTOTYPES ....... ==================================")
    generate_training_prototype(target_dataset_path=best_cluster_path, dest_dataset_path=train_prototype_dir)
    
    # 4. Re-group Training data
    print("====================================== RE-GROUP TRAINING DATA ....... ==================================")
    best_cluster_path = DBSCAN_clustering_with_auto_emb_new(
        dataset_path=train_prototype_dir,
        transformation=transformation,
        inter_dest_dir=proto_clustered_inter_dest_dir,
        best_dest_dir=proto_clustered_best_dest_dir,
        net=net, save_inter=True,
        desig_eps=13)#17

    # train_best_clustered_path = "C:/Users/Lizzie0930/Desktop/git/Results_Auto/4_train_clustered_best_dest/final_clustered_dataset"
    re_group_train_data(source_dir=train_best_clustered_path, info_dir=best_cluster_path,
                        target_dir=regrouped_target_dir)
    
    # 2. Clustering on testing data
    print("====================================== CLUSTERING ON TESTING DATA ....... ==================================")
    """
        should cut testing data into known and unknown, and generate prototype only on known data
        2. pass known_cluster_target_folders to test_cluster_target_dir
        3. change generate_testing_prototype to generate_testing_prototype_new
        4. pass unknown_cluster_target_folders to vali_data_dir
    """
    
    for i in range(len(cluster_target_folders)):
        single_case_inter_dest_dir = os.sep.join(
            [os.path.normpath(testing_clustered_inter_dir), cluster_target_folder_names[i]])
        if not os.path.exists(single_case_inter_dest_dir):
            os.mkdir(single_case_inter_dest_dir)
        single_case_inter_best_dir = os.sep.join(
            [os.path.normpath(testing_clustered_best_dir), cluster_target_folder_names[i]])
        if not os.path.exists(single_case_inter_best_dir):
            os.mkdir(single_case_inter_best_dir)
        single_case_prototype_dest_dir = os.sep.join(
            [os.path.normpath(test_prototype_dest_dir), cluster_target_folder_names[i]])
        if not os.path.exists(single_case_prototype_dest_dir):
            os.mkdir(single_case_prototype_dest_dir)
        best_cluster_path = DBSCAN_clustering_with_auto_emb_new(dataset_path=cluster_target_folders[i], transformation=transformation,
                                              inter_dest_dir=single_case_inter_dest_dir,
                                              best_dest_dir=single_case_inter_best_dir, net=net,
                                              save_inter=False, desig_eps=15)
        generate_testing_prototype_new(target_dataset_path=best_cluster_path,
                                    dest_dataset_path=single_case_prototype_dest_dir)
    
    
    # train the model & validate the model and save the results
    print(
        "====================================== TRAIN THE SIAMESE NETWORK MODEL ....... ==================================")
    
    folder_dataset = datasets.ImageFolder(root=regrouped_target_dir)
    siamese_train_dataset = SiameseNetworkDataset_1(imageFolderDataset=folder_dataset,
                                                    transform=transformation)
    training_dataset = siamese_train_dataset
    best_epoch_list, best_batch_list, best_model_path_list = Training_Loop_2(model_num=model_num, net_name=net_name,
                                                                           dataset=training_dataset,
                                                                           save_best_model_dir=save_best_si_model_dir,
                                                                           save_inter_model_dir=save_inter_si_model_dir,
                                                                           epoch_num=si_epoch_num, way_list=[5],
                                                                           vali_data_dir=vali_data_dir,
                                                                           proto_data_dir=test_prototype_dest_dir,
                                                                           csv_save_dir=csv_save_dir,
                                                                           case_len=case_len, net_type=net_type)
    print("best_epoch_list: ", best_epoch_list)
    print("best_batch_list: ", best_batch_list)
    print("best_model_path_list: ", best_model_path_list)
    
    print(
        "====================================== TESTING SIAMESE NETWORK MODEL ON TESTING DATA ....... ==================================")

    result_df, avg_df = group_test_2(model_num=model_num, way_list=way_list, case_len=case_len, epoch=best_epoch_list[0],
                                   model_path=best_model_path_list[0], batch_num=best_batch_list[0], net_name=net_name,
                                   net_type=net_type, test_data_dir=vali_data_dir, test_data_postfix="",
                                   proto_data_dir=test_prototype_dest_dir, proto_data_postfix="")
    
    print(result_df)
    print(avg_df)
    result_csv_path = os.sep.join([os.path.normpath(result_csv_dir), "results.csv"])
    avg_csv_path = os.sep.join([os.path.normpath(result_csv_dir), "avg.csv"])
    result_df.to_csv(result_csv_path)
    avg_df.to_csv(avg_csv_path)

def complete_training_loop_Pr4(autoencoder_training_data_path, autoencoder_save_model_path,
                            autoencoder_save_img_base_path,
                            training_data_cluster_target, training_clustered_inter_dir,
                            training_clustered_best_dir,
                            train_prototype_dir, regrouped_target_dir,
                            test_cluster_target_dir, testing_clustered_inter_dir,
                            testing_clustered_best_dir,
                            save_best_si_model_dir, save_inter_si_model_dir,
                            si_epoch_num, vali_data_dir,
                            way_list, net_name, csv_save_dir, case_len, net_type, shots,
                            test_prototype_dest_dir,
                            proto_clustered_best_dest_dir, proto_clustered_inter_dest_dir,
                            result_csv_dir, transformation,
                            model_num=2, auto_epoch_num=100):
    '''
    with clustering, regrouping, pretrained network, concatencation
    representation: auoencoder Emb
    clustering: DBSCAN
    pretrained: Without
    '''
    '''
    training and testing loop for model using Autoencoder Embedding for clustering.
    1. Training Autoencoder
    2. Clustering on Training data and clustering on testing data
    3. Generate Prototypes
    4. Re-group Training data
    5. Training Siamese network on Training data
    6. Testing Siamese network on Testing data
    '''
    cluster_target_folder_names = os.listdir(test_cluster_target_dir)
    cluster_target_folders = [os.sep.join([os.path.normpath(test_cluster_target_dir), i]) for i in
                              cluster_target_folder_names]
    
    #  1. Training Autoencoder
    print("====================================== TRAINING AUTOENCODER ....... ==================================")
    AutoEncoder_Training_Loop(epoch_num=auto_epoch_num,
                              data_path=autoencoder_training_data_path,
                              save_model_path=autoencoder_save_model_path,
                              save_img_base_path=autoencoder_save_img_base_path, transformation=transformation)

    # 2. Clustering on Training data
    print("====================================== CLUSTERING ON TRAINING DATA ....... ==================================")
    
    
    net = Autoencoder_Conv().cuda()
    net.load_state_dict(torch.load(autoencoder_save_model_path))
    
    best_cluster_path = DBSCAN_clustering_with_auto_emb_new(
        dataset_path=training_data_cluster_target,
        transformation=transformation,
        inter_dest_dir=training_clustered_inter_dir,
        best_dest_dir=training_clustered_best_dir,
        net=net, save_inter=False,
        desig_eps=15)
    train_best_clustered_path = best_cluster_path

    # 3. Generate Prototypes
    print("====================================== GENERATING TRAINING PROTOTYPES ....... ==================================")
    generate_training_prototype(target_dataset_path=best_cluster_path, dest_dataset_path=train_prototype_dir)
    
    # 4. Re-group Training data
    print("====================================== RE-GROUP TRAINING DATA ....... ==================================")
    best_cluster_path = DBSCAN_clustering_with_auto_emb_new(
        dataset_path=train_prototype_dir,
        transformation=transformation,
        inter_dest_dir=proto_clustered_inter_dest_dir,
        best_dest_dir=proto_clustered_best_dest_dir,
        net=net, save_inter=True,
        desig_eps=13)#17

    # train_best_clustered_path = "C:/Users/Lizzie0930/Desktop/git/Results_Auto/4_train_clustered_best_dest/final_clustered_dataset"
    re_group_train_data(source_dir=train_best_clustered_path, info_dir=best_cluster_path,
                        target_dir=regrouped_target_dir)
    
    # 2. Clustering on testing data
    print("====================================== CLUSTERING ON TESTING DATA ....... ==================================")
    """
        should cut testing data into known and unknown, and generate prototype only on known data
        2. pass known_cluster_target_folders to test_cluster_target_dir
        3. change generate_testing_prototype to generate_testing_prototype_new
        4. pass unknown_cluster_target_folders to vali_data_dir
    """
    
    for i in range(len(cluster_target_folders)):
        single_case_inter_dest_dir = os.sep.join(
            [os.path.normpath(testing_clustered_inter_dir), cluster_target_folder_names[i]])
        if not os.path.exists(single_case_inter_dest_dir):
            os.mkdir(single_case_inter_dest_dir)
        single_case_inter_best_dir = os.sep.join(
            [os.path.normpath(testing_clustered_best_dir), cluster_target_folder_names[i]])
        if not os.path.exists(single_case_inter_best_dir):
            os.mkdir(single_case_inter_best_dir)
        single_case_prototype_dest_dir = os.sep.join(
            [os.path.normpath(test_prototype_dest_dir), cluster_target_folder_names[i]])
        if not os.path.exists(single_case_prototype_dest_dir):
            os.mkdir(single_case_prototype_dest_dir)
        best_cluster_path = DBSCAN_clustering_with_auto_emb_new(dataset_path=cluster_target_folders[i], transformation=transformation,
                                              inter_dest_dir=single_case_inter_dest_dir,
                                              best_dest_dir=single_case_inter_best_dir, net=net,
                                              save_inter=False, desig_eps=15)
        generate_testing_prototype_new(target_dataset_path=best_cluster_path,
                                    dest_dataset_path=single_case_prototype_dest_dir)
    
    
    # train the model & validate the model and save the results
    print(
        "====================================== TRAIN THE SIAMESE NETWORK MODEL ....... ==================================")
    
    folder_dataset = datasets.ImageFolder(root=regrouped_target_dir)
    siamese_train_dataset = SiameseNetworkDataset_1(imageFolderDataset=folder_dataset,
                                                    transform=transformation)
    training_dataset = siamese_train_dataset
    best_epoch_list, best_batch_list, best_model_path_list = Training_Loop_2(model_num=model_num, net_name=net_name,
                                                                           dataset=training_dataset,
                                                                           save_best_model_dir=save_best_si_model_dir,
                                                                           save_inter_model_dir=save_inter_si_model_dir,
                                                                           epoch_num=si_epoch_num, way_list=[5],
                                                                           vali_data_dir=vali_data_dir,
                                                                           proto_data_dir=test_prototype_dest_dir,
                                                                           csv_save_dir=csv_save_dir,
                                                                           case_len=case_len, net_type=net_type)
    print("best_epoch_list: ", best_epoch_list)
    print("best_batch_list: ", best_batch_list)
    print("best_model_path_list: ", best_model_path_list)
    
    print(
        "====================================== TESTING SIAMESE NETWORK MODEL ON TESTING DATA ....... ==================================")

    result_df, avg_df = group_test_2(model_num=model_num, way_list=way_list, case_len=case_len, epoch=best_epoch_list[0],
                                   model_path=best_model_path_list[0], batch_num=best_batch_list[0], net_name=net_name,
                                   net_type=net_type, test_data_dir=vali_data_dir, test_data_postfix="",
                                   proto_data_dir=test_prototype_dest_dir, proto_data_postfix="")
   
    print(result_df)
    print(avg_df)
    result_csv_path = os.sep.join([os.path.normpath(result_csv_dir), "results.csv"])
    avg_csv_path = os.sep.join([os.path.normpath(result_csv_dir), "avg.csv"])
    result_df.to_csv(result_csv_path)
    avg_df.to_csv(avg_csv_path)

def complete_training_loop_A1(autoencoder_training_data_path, autoencoder_save_model_path,
                            autoencoder_save_img_base_path,
                            training_data_cluster_target, training_clustered_inter_dir,
                            training_clustered_best_dir,
                            train_prototype_dir, regrouped_target_dir,
                            test_cluster_target_dir, testing_clustered_inter_dir,
                            testing_clustered_best_dir,
                            save_best_si_model_dir, save_inter_si_model_dir,
                            si_epoch_num, vali_data_dir,
                            way_list, net_name, csv_save_dir, case_len, net_type, shots,
                            test_prototype_dest_dir,
                            proto_clustered_best_dest_dir, proto_clustered_inter_dest_dir,
                            result_csv_dir, transformation,
                            model_num=2, auto_epoch_num=100):
    '''
    with clustering, regrouping, pretrained network, concatencation
    representation: auoencoder Emb
    clustering: DBSCAN
    pretrained: autoencoder
    '''
    '''
    training and testing loop for model using Autoencoder Embedding for clustering.
    1. Training Autoencoder
    2. Clustering on Training data and clustering on testing data
    3. Generate Prototypes
    4. Re-group Training data
    5. Training Siamese network on Training data
    6. Testing Siamese network on Testing data
    '''
    cluster_target_folder_names = os.listdir(test_cluster_target_dir)
    cluster_target_folders = [os.sep.join([os.path.normpath(test_cluster_target_dir), i]) for i in
                              cluster_target_folder_names]
    
    #  1. Training Autoencoder
    print("====================================== TRAINING AUTOENCODER ....... ==================================")
    AutoEncoder_Training_Loop(epoch_num=auto_epoch_num,
                              data_path=autoencoder_training_data_path,
                              save_model_path=autoencoder_save_model_path,
                              save_img_base_path=autoencoder_save_img_base_path, transformation=transformation)

    # 2. Clustering on Training data
    print("====================================== CLUSTERING ON TRAINING DATA ....... ==================================")
    
    net = Autoencoder_Conv().cuda()
    net.load_state_dict(torch.load(autoencoder_save_model_path))
    
    best_cluster_path = DBSCAN_clustering_with_auto_emb_new(
        dataset_path=training_data_cluster_target,
        transformation=transformation,
        inter_dest_dir=training_clustered_inter_dir,
        best_dest_dir=training_clustered_best_dir,
        net=net, save_inter=False,
        desig_eps=15)
    train_best_clustered_path = best_cluster_path

    # 3. Generate Prototypes
    print("====================================== GENERATING TRAINING PROTOTYPES ....... ==================================")
    generate_training_prototype(target_dataset_path=best_cluster_path, dest_dataset_path=train_prototype_dir)
    
    # 4. Re-group Training data
    print("====================================== RE-GROUP TRAINING DATA ....... ==================================")
    best_cluster_path = DBSCAN_clustering_with_auto_emb_new(
        dataset_path=train_prototype_dir,
        transformation=transformation,
        inter_dest_dir=proto_clustered_inter_dest_dir,
        best_dest_dir=proto_clustered_best_dest_dir,
        net=net, save_inter=True,
        desig_eps=13)#17

    
    re_group_train_data(source_dir=train_best_clustered_path, info_dir=best_cluster_path,
                        target_dir=regrouped_target_dir)
    
    # 2. Clustering on testing data
    print("====================================== CLUSTERING ON TESTING DATA ....... ==================================")
    """
        should cut testing data into known and unknown, and generate prototype only on known data
        2. pass known_cluster_target_folders to test_cluster_target_dir
        3. change generate_testing_prototype to generate_testing_prototype_new
        4. pass unknown_cluster_target_folders to vali_data_dir
    """
    
    for i in range(len(cluster_target_folders)):
        single_case_inter_dest_dir = os.sep.join(
            [os.path.normpath(testing_clustered_inter_dir), cluster_target_folder_names[i]])
        if not os.path.exists(single_case_inter_dest_dir):
            os.mkdir(single_case_inter_dest_dir)
        single_case_inter_best_dir = os.sep.join(
            [os.path.normpath(testing_clustered_best_dir), cluster_target_folder_names[i]])
        if not os.path.exists(single_case_inter_best_dir):
            os.mkdir(single_case_inter_best_dir)
        single_case_prototype_dest_dir = os.sep.join(
            [os.path.normpath(test_prototype_dest_dir), cluster_target_folder_names[i]])
        if not os.path.exists(single_case_prototype_dest_dir):
            os.mkdir(single_case_prototype_dest_dir)
        best_cluster_path = DBSCAN_clustering_with_auto_emb_new(dataset_path=cluster_target_folders[i], transformation=transformation,
                                              inter_dest_dir=single_case_inter_dest_dir,
                                              best_dest_dir=single_case_inter_best_dir, net=net,
                                              save_inter=False, desig_eps=15)
        generate_testing_prototype_new(target_dataset_path=best_cluster_path,
                                    dest_dataset_path=single_case_prototype_dest_dir)
    
    
    # train the model & validate the model and save the results
    print(
        "====================================== TRAIN THE SIAMESE NETWORK MODEL ....... ==================================")
    
    folder_dataset = datasets.ImageFolder(root=regrouped_target_dir)
    siamese_train_dataset = SiameseNetworkDataset_1(imageFolderDataset=folder_dataset,
                                                    transform=transformation)
    training_dataset = siamese_train_dataset
    best_epoch_list, best_batch_list, best_model_path_list = Training_Loop(model_num=model_num, net_name=net_name,
                                                                           dataset=training_dataset,
                                                                           save_best_model_dir=save_best_si_model_dir,
                                                                           save_inter_model_dir=save_inter_si_model_dir,
                                                                           epoch_num=si_epoch_num, way_list=way_list,
                                                                           vali_data_dir=vali_data_dir,
                                                                           proto_data_dir=test_prototype_dest_dir,
                                                                           csv_save_dir=csv_save_dir,
                                                                           case_len=case_len, net_type=net_type)
    print("best_epoch_list: ", best_epoch_list)
    print("best_batch_list: ", best_batch_list)
    print("best_model_path_list: ", best_model_path_list)
    
    print(
        "====================================== TESTING SIAMESE NETWORK MODEL ON TESTING DATA ....... ==================================")

    result_df, avg_df = group_test(model_num=model_num, way_list=way_list, case_len=case_len, epoch=20,
                                   model_path=best_model_path_list[0], batch_num=22, net_name=net_name,
                                   net_type=net_type, test_data_dir=vali_data_dir, test_data_postfix="",
                                   proto_data_dir=test_prototype_dest_dir, proto_data_postfix="")
    
    print(result_df)
    print(avg_df)
    result_csv_path = os.sep.join([os.path.normpath(result_csv_dir), "results.csv"])
    avg_csv_path = os.sep.join([os.path.normpath(result_csv_dir), "avg.csv"])
    result_df.to_csv(result_csv_path)
    avg_df.to_csv(avg_csv_path)

def complete_training_loop_A2(autoencoder_training_data_path, autoencoder_save_model_path,
                            autoencoder_save_img_base_path,
                            training_data_cluster_target, training_clustered_inter_dir,
                            training_clustered_best_dir,
                            train_prototype_dir, regrouped_target_dir,
                            test_cluster_target_dir, testing_clustered_inter_dir,
                            testing_clustered_best_dir,
                            save_best_si_model_dir, save_inter_si_model_dir,
                            si_epoch_num, vali_data_dir,
                            way_list, net_name, csv_save_dir, case_len, net_type, shots,
                            test_prototype_dest_dir,
                            proto_clustered_best_dest_dir, proto_clustered_inter_dest_dir,
                            result_csv_dir, transformation,
                            model_num=2, auto_epoch_num=100):
    '''
    with clustering, regrouping, (X)pretrained network, (X)concatencation
    representation: auoencoder Emb
    clustering: DBSCAN
    pretrained: autoencoder
    '''
    '''
    training and testing loop for model using Autoencoder Embedding for clustering.
    1. Training Autoencoder
    2. Clustering on Training data and clustering on testing data
    3. Generate Prototypes
    4. Re-group Training data
    5. Training Siamese network on Training data
    6. Testing Siamese network on Testing data
    '''
    cluster_target_folder_names = os.listdir(test_cluster_target_dir)
    cluster_target_folders = [os.sep.join([os.path.normpath(test_cluster_target_dir), i]) for i in
                              cluster_target_folder_names]
    
    #  1. Training Autoencoder
    print("====================================== TRAINING AUTOENCODER ....... ==================================")
    AutoEncoder_Training_Loop(epoch_num=auto_epoch_num,
                              data_path=autoencoder_training_data_path,
                              save_model_path=autoencoder_save_model_path,
                              save_img_base_path=autoencoder_save_img_base_path)

    # 2. Clustering on Training data
    print("====================================== CLUSTERING ON TRAINING DATA ....... ==================================")
    
    net = Autoencoder_Conv().cuda()
    net.load_state_dict(torch.load(autoencoder_save_model_path))
    
    best_cluster_path = DBSCAN_clustering_with_auto_emb_new(
        dataset_path=training_data_cluster_target,
        transformation=transformation,
        inter_dest_dir=training_clustered_inter_dir,
        best_dest_dir=training_clustered_best_dir,
        net=net, save_inter=False,
        desig_eps=15)
    train_best_clustered_path = best_cluster_path

    # 3. Generate Prototypes
    print("====================================== GENERATING TRAINING PROTOTYPES ....... ==================================")
    generate_training_prototype(target_dataset_path=best_cluster_path, dest_dataset_path=train_prototype_dir)
    
    # 4. Re-group Training data
    print("====================================== RE-GROUP TRAINING DATA ....... ==================================")
    best_cluster_path = DBSCAN_clustering_with_auto_emb_new(
        dataset_path=train_prototype_dir,
        transformation=transformation,
        inter_dest_dir=proto_clustered_inter_dest_dir,
        best_dest_dir=proto_clustered_best_dest_dir,
        net=net, save_inter=True,
        desig_eps=13)#17

    # train_best_clustered_path = "C:/Users/Lizzie0930/Desktop/git/Results_Auto/4_train_clustered_best_dest/final_clustered_dataset"
    re_group_train_data(source_dir=train_best_clustered_path, info_dir=best_cluster_path,
                        target_dir=regrouped_target_dir)
    
    # 2. Clustering on testing data
    print("====================================== CLUSTERING ON TESTING DATA ....... ==================================")
    """
        should cut testing data into known and unknown, and generate prototype only on known data
        2. pass known_cluster_target_folders to test_cluster_target_dir
        3. change generate_testing_prototype to generate_testing_prototype_new
        4. pass unknown_cluster_target_folders to vali_data_dir
    """
    for i in range(len(cluster_target_folders)):
        single_case_inter_dest_dir = os.sep.join(
            [os.path.normpath(testing_clustered_inter_dir), cluster_target_folder_names[i]])
        if not os.path.exists(single_case_inter_dest_dir):
            os.mkdir(single_case_inter_dest_dir)
        single_case_inter_best_dir = os.sep.join(
            [os.path.normpath(testing_clustered_best_dir), cluster_target_folder_names[i]])
        if not os.path.exists(single_case_inter_best_dir):
            os.mkdir(single_case_inter_best_dir)
        single_case_prototype_dest_dir = os.sep.join(
            [os.path.normpath(test_prototype_dest_dir), cluster_target_folder_names[i]])
        if not os.path.exists(single_case_prototype_dest_dir):
            os.mkdir(single_case_prototype_dest_dir)
        best_cluster_path = DBSCAN_clustering_with_auto_emb_new(dataset_path=cluster_target_folders[i], transformation=transformation,
                                              inter_dest_dir=single_case_inter_dest_dir,
                                              best_dest_dir=single_case_inter_best_dir, net=net,
                                              save_inter=False, desig_eps=15)
        generate_testing_prototype_new(target_dataset_path=best_cluster_path,
                                    dest_dataset_path=single_case_prototype_dest_dir)
    

    # train the model & validate the model and save the results
    print(
        "====================================== TRAIN THE SIAMESE NETWORK MODEL ....... ==================================")
    folder_dataset = datasets.ImageFolder(root=regrouped_target_dir)
    siamese_train_dataset = SiameseNetworkDataset_1(imageFolderDataset=folder_dataset,
                                                    transform=transformation)
    training_dataset = siamese_train_dataset
    best_epoch_list, best_batch_list, best_model_path_list = Training_Loop(model_num=model_num, net_name=net_name,
                                                                           dataset=training_dataset,
                                                                           save_best_model_dir=save_best_si_model_dir,
                                                                           save_inter_model_dir=save_inter_si_model_dir,
                                                                           epoch_num=si_epoch_num, way_list=way_list,
                                                                           vali_data_dir=vali_data_dir,
                                                                           proto_data_dir=test_prototype_dest_dir,
                                                                           csv_save_dir=csv_save_dir,
                                                                           case_len=case_len, net_type=net_type)
    print("best_epoch_list: ", best_epoch_list)
    print("best_batch_list: ", best_batch_list)
    print("best_model_path_list: ", best_model_path_list)

    print(
        "====================================== TESTING SIAMESE NETWORK MODEL ON TESTING DATA ....... ==================================")

    result_df, avg_df = group_test(model_num=model_num, way_list=way_list, case_len=case_len, epoch=best_epoch_list[0],
                                   model_path=best_model_path_list[0], batch_num=best_batch_list[0], net_name=net_name,
                                   net_type=net_type, test_data_dir=vali_data_dir, test_data_postfix="",
                                   proto_data_dir=test_prototype_dest_dir, proto_data_postfix="")
    
    print(result_df)
    print(avg_df)
    result_csv_path = os.sep.join([os.path.normpath(result_csv_dir), "results.csv"])
    avg_csv_path = os.sep.join([os.path.normpath(result_csv_dir), "avg.csv"])
    result_df.to_csv(result_csv_path)
    avg_df.to_csv(avg_csv_path)

def complete_training_loop_A3(autoencoder_training_data_path, autoencoder_save_model_path,
                            autoencoder_save_img_base_path,
                            training_data_cluster_target, training_clustered_inter_dir,
                            training_clustered_best_dir,
                            train_prototype_dir, regrouped_target_dir,
                            test_cluster_target_dir, testing_clustered_inter_dir,
                            testing_clustered_best_dir,
                            save_best_si_model_dir, save_inter_si_model_dir,
                            si_epoch_num, vali_data_dir,
                            way_list, net_name, csv_save_dir, case_len, net_type, shots,
                            test_prototype_dest_dir,
                            proto_clustered_best_dest_dir, proto_clustered_inter_dest_dir,
                            result_csv_dir, transformation,
                            model_num=2, auto_epoch_num=100):
    '''
    with (X)clustering, (X)regrouping, pretrained network, (X)concatencation
    representation: auoencoder Emb
    clustering: DBSCAN
    pretrained: autoencoder
    '''
    '''
    training and testing loop for model using Autoencoder Embedding for clustering.
    1. Training Autoencoder
    2. Clustering on Training data and clustering on testing data
    3. Generate Prototypes
    4. Re-group Training data
    5. Training Siamese network on Training data
    6. Testing Siamese network on Testing data
    '''
    cluster_target_folder_names = os.listdir(test_cluster_target_dir)
    cluster_target_folders = [os.sep.join([os.path.normpath(test_cluster_target_dir), i]) for i in
                              cluster_target_folder_names]
    
    #  1. Training Autoencoder
    print("====================================== TRAINING AUTOENCODER ....... ==================================")
    AutoEncoder_Training_Loop(epoch_num=auto_epoch_num,
                              data_path=autoencoder_training_data_path,
                              save_model_path=autoencoder_save_model_path,
                              save_img_base_path=autoencoder_save_img_base_path)

    # 2. Clustering on Training data
    print("====================================== CLUSTERING ON TRAINING DATA ....... ==================================")
    
    net = Autoencoder_Conv().cuda()
    net.load_state_dict(torch.load(autoencoder_save_model_path))
    
    best_cluster_path = DBSCAN_clustering_with_auto_emb_new(
        dataset_path=training_data_cluster_target,
        transformation=transformation,
        inter_dest_dir=training_clustered_inter_dir,
        best_dest_dir=training_clustered_best_dir,
        net=net, save_inter=False,
        desig_eps=15)
    train_best_clustered_path = best_cluster_path

    # 3. Generate Prototypes
    print("====================================== GENERATING TRAINING PROTOTYPES ....... ==================================")
    generate_training_prototype(target_dataset_path=best_cluster_path, dest_dataset_path=train_prototype_dir)
    
    # 4. Re-group Training data
    print("====================================== RE-GROUP TRAINING DATA ....... ==================================")
    best_cluster_path = DBSCAN_clustering_with_auto_emb_new(
        dataset_path=train_prototype_dir,
        transformation=transformation,
        inter_dest_dir=proto_clustered_inter_dest_dir,
        best_dest_dir=proto_clustered_best_dest_dir,
        net=net, save_inter=True,
        desig_eps=13)#17

    # train_best_clustered_path = "C:/Users/Lizzie0930/Desktop/git/Results_Auto/4_train_clustered_best_dest/final_clustered_dataset"
    re_group_train_data(source_dir=train_best_clustered_path, info_dir=best_cluster_path,
                        target_dir=regrouped_target_dir)
    
    # 2. Clustering on testing data
    print("====================================== CLUSTERING ON TESTING DATA ....... ==================================")
    """
        should cut testing data into known and unknown, and generate prototype only on known data
        2. pass known_cluster_target_folders to test_cluster_target_dir
        3. change generate_testing_prototype to generate_testing_prototype_new
        4. pass unknown_cluster_target_folders to vali_data_dir
    """
    for i in range(len(cluster_target_folders)):
        single_case_inter_dest_dir = os.sep.join(
            [os.path.normpath(testing_clustered_inter_dir), cluster_target_folder_names[i]])
        if not os.path.exists(single_case_inter_dest_dir):
            os.mkdir(single_case_inter_dest_dir)
        single_case_inter_best_dir = os.sep.join(
            [os.path.normpath(testing_clustered_best_dir), cluster_target_folder_names[i]])
        if not os.path.exists(single_case_inter_best_dir):
            os.mkdir(single_case_inter_best_dir)
        single_case_prototype_dest_dir = os.sep.join(
            [os.path.normpath(test_prototype_dest_dir), cluster_target_folder_names[i]])
        if not os.path.exists(single_case_prototype_dest_dir):
            os.mkdir(single_case_prototype_dest_dir)
        best_cluster_path = DBSCAN_clustering_with_auto_emb_new(dataset_path=cluster_target_folders[i], transformation=transformation,
                                              inter_dest_dir=single_case_inter_dest_dir,
                                              best_dest_dir=single_case_inter_best_dir, net=net,
                                              save_inter=False, desig_eps=15)
        generate_testing_prototype_new(target_dataset_path=best_cluster_path,
                                    dest_dataset_path=single_case_prototype_dest_dir)
    

    # train the model & validate the model and save the results
    print(
        "====================================== TRAIN THE SIAMESE NETWORK MODEL ....... ==================================")
    folder_dataset = datasets.ImageFolder(root=regrouped_target_dir)
    siamese_train_dataset = SiameseNetworkDataset_1(imageFolderDataset=folder_dataset,
                                                    transform=transformation)
    training_dataset = siamese_train_dataset
    best_epoch_list, best_batch_list, best_model_path_list = Training_Loop(model_num=model_num, net_name=net_name,
                                                                           dataset=training_dataset,
                                                                           save_best_model_dir=save_best_si_model_dir,
                                                                           save_inter_model_dir=save_inter_si_model_dir,
                                                                           epoch_num=si_epoch_num, way_list=way_list,
                                                                           vali_data_dir=vali_data_dir,
                                                                           proto_data_dir=test_prototype_dest_dir,
                                                                           csv_save_dir=csv_save_dir,
                                                                           case_len=case_len, net_type=net_type)
    print("best_epoch_list: ", best_epoch_list)
    print("best_batch_list: ", best_batch_list)
    print("best_model_path_list: ", best_model_path_list)

    print(
        "====================================== TESTING SIAMESE NETWORK MODEL ON TESTING DATA ....... ==================================")

    result_df, avg_df = group_test(model_num=model_num, way_list=way_list, case_len=case_len, epoch=best_epoch_list[0],
                                   model_path=best_model_path_list[0], batch_num=best_batch_list[0], net_name=net_name,
                                   net_type=net_type, test_data_dir=vali_data_dir, test_data_postfix="",
                                   proto_data_dir=test_prototype_dest_dir, proto_data_postfix="")
    
    print(result_df)
    print(avg_df)
    result_csv_path = os.sep.join([os.path.normpath(result_csv_dir), "results.csv"])
    avg_csv_path = os.sep.join([os.path.normpath(result_csv_dir), "avg.csv"])
    result_df.to_csv(result_csv_path)
    avg_df.to_csv(avg_csv_path)

def complete_training_loop_A4(autoencoder_training_data_path, autoencoder_save_model_path,
                            autoencoder_save_img_base_path,
                            training_data_cluster_target, training_clustered_inter_dir,
                            training_clustered_best_dir,
                            train_prototype_dir, regrouped_target_dir,
                            test_cluster_target_dir, testing_clustered_inter_dir,
                            testing_clustered_best_dir,
                            save_best_si_model_dir, save_inter_si_model_dir,
                            si_epoch_num, vali_data_dir,
                            way_list, net_name, csv_save_dir, case_len, net_type, shots,
                            test_prototype_dest_dir,
                            proto_clustered_best_dest_dir, proto_clustered_inter_dest_dir,
                            result_csv_dir, transformation,
                            model_num=2, auto_epoch_num=100):
    '''
    with (X)clustering, (X)regrouping, (X)pretrained network, concatencation
    representation: auoencoder Emb
    clustering: DBSCAN
    pretrained: autoencoder
    '''
    '''
    training and testing loop for model using Autoencoder Embedding for clustering.
    1. Training Autoencoder
    2. Clustering on Training data and clustering on testing data
    3. Generate Prototypes
    4. Re-group Training data
    5. Training Siamese network on Training data
    6. Testing Siamese network on Testing data
    '''
    cluster_target_folder_names = os.listdir(test_cluster_target_dir)
    cluster_target_folders = [os.sep.join([os.path.normpath(test_cluster_target_dir), i]) for i in
                              cluster_target_folder_names]
    
    #  1. Training Autoencoder
    print("====================================== TRAINING AUTOENCODER ....... ==================================")
    AutoEncoder_Training_Loop(epoch_num=auto_epoch_num,
                              data_path=autoencoder_training_data_path,
                              save_model_path=autoencoder_save_model_path,
                              save_img_base_path=autoencoder_save_img_base_path)

    # 2. Clustering on Training data
    print("====================================== CLUSTERING ON TRAINING DATA ....... ==================================")
    
    net = Autoencoder_Conv().cuda()
    net.load_state_dict(torch.load(autoencoder_save_model_path))
    
    best_cluster_path = DBSCAN_clustering_with_auto_emb_new(
        dataset_path=training_data_cluster_target,
        transformation=transformation,
        inter_dest_dir=training_clustered_inter_dir,
        best_dest_dir=training_clustered_best_dir,
        net=net, save_inter=False,
        desig_eps=15)
    train_best_clustered_path = best_cluster_path

    # 3. Generate Prototypes
    print("====================================== GENERATING TRAINING PROTOTYPES ....... ==================================")
    generate_training_prototype(target_dataset_path=best_cluster_path, dest_dataset_path=train_prototype_dir)
    
    # 4. Re-group Training data
    print("====================================== RE-GROUP TRAINING DATA ....... ==================================")
    best_cluster_path = DBSCAN_clustering_with_auto_emb_new(
        dataset_path=train_prototype_dir,
        transformation=transformation,
        inter_dest_dir=proto_clustered_inter_dest_dir,
        best_dest_dir=proto_clustered_best_dest_dir,
        net=net, save_inter=True,
        desig_eps=13)#17

    re_group_train_data(source_dir=train_best_clustered_path, info_dir=best_cluster_path,
                        target_dir=regrouped_target_dir)
    
    # 2. Clustering on testing data
    print("====================================== CLUSTERING ON TESTING DATA ....... ==================================")
    """
        should cut testing data into known and unknown, and generate prototype only on known data
        2. pass known_cluster_target_folders to test_cluster_target_dir
        3. change generate_testing_prototype to generate_testing_prototype_new
        4. pass unknown_cluster_target_folders to vali_data_dir
    """
    for i in range(len(cluster_target_folders)):
        single_case_inter_dest_dir = os.sep.join(
            [os.path.normpath(testing_clustered_inter_dir), cluster_target_folder_names[i]])
        if not os.path.exists(single_case_inter_dest_dir):
            os.mkdir(single_case_inter_dest_dir)
        single_case_inter_best_dir = os.sep.join(
            [os.path.normpath(testing_clustered_best_dir), cluster_target_folder_names[i]])
        if not os.path.exists(single_case_inter_best_dir):
            os.mkdir(single_case_inter_best_dir)
        single_case_prototype_dest_dir = os.sep.join(
            [os.path.normpath(test_prototype_dest_dir), cluster_target_folder_names[i]])
        if not os.path.exists(single_case_prototype_dest_dir):
            os.mkdir(single_case_prototype_dest_dir)
        best_cluster_path = DBSCAN_clustering_with_auto_emb_new(dataset_path=cluster_target_folders[i], transformation=transformation,
                                              inter_dest_dir=single_case_inter_dest_dir,
                                              best_dest_dir=single_case_inter_best_dir, net=net,
                                              save_inter=False, desig_eps=15)
        generate_testing_prototype_new(target_dataset_path=best_cluster_path,
                                    dest_dataset_path=single_case_prototype_dest_dir)
    

    # train the model & validate the model and save the results
    print(
        "====================================== TRAIN THE SIAMESE NETWORK MODEL ....... ==================================")
    folder_dataset = datasets.ImageFolder(root=regrouped_target_dir)
    siamese_train_dataset = SiameseNetworkDataset_1(imageFolderDataset=folder_dataset,
                                                    transform=transformation)
    training_dataset = siamese_train_dataset
    best_epoch_list, best_batch_list, best_model_path_list = Training_Loop(model_num=model_num, net_name=net_name,
                                                                           dataset=training_dataset,
                                                                           save_best_model_dir=save_best_si_model_dir,
                                                                           save_inter_model_dir=save_inter_si_model_dir,
                                                                           epoch_num=si_epoch_num, way_list=way_list,
                                                                           vali_data_dir=vali_data_dir,
                                                                           proto_data_dir=test_prototype_dest_dir,
                                                                           csv_save_dir=csv_save_dir,
                                                                           case_len=case_len, net_type=net_type)
    print("best_epoch_list: ", best_epoch_list)
    print("best_batch_list: ", best_batch_list)
    print("best_model_path_list: ", best_model_path_list)

    print(
        "====================================== TESTING SIAMESE NETWORK MODEL ON TESTING DATA ....... ==================================")

    result_df, avg_df = group_test(model_num=model_num, way_list=way_list, case_len=case_len, epoch=best_epoch_list[0],
                                   model_path=best_model_path_list[0], batch_num=best_batch_list[0], net_name=net_name,
                                   net_type=net_type, test_data_dir=vali_data_dir, test_data_postfix="",
                                   proto_data_dir=test_prototype_dest_dir, proto_data_postfix="")
    
    print(result_df)
    print(avg_df)
    result_csv_path = os.sep.join([os.path.normpath(result_csv_dir), "results.csv"])
    avg_csv_path = os.sep.join([os.path.normpath(result_csv_dir), "avg.csv"])
    result_df.to_csv(result_csv_path)
    avg_df.to_csv(avg_csv_path)

def complete_training_loop_A5(autoencoder_training_data_path, autoencoder_save_model_path,
                            autoencoder_save_img_base_path,
                            training_data_cluster_target, training_clustered_inter_dir,
                            training_clustered_best_dir,
                            train_prototype_dir, regrouped_target_dir,
                            test_cluster_target_dir, testing_clustered_inter_dir,
                            testing_clustered_best_dir,
                            save_best_si_model_dir, save_inter_si_model_dir,
                            si_epoch_num, vali_data_dir,
                            way_list, net_name, csv_save_dir, case_len, net_type, shots,
                            test_prototype_dest_dir,
                            proto_clustered_best_dest_dir, proto_clustered_inter_dest_dir,
                            result_csv_dir, transformation,
                            model_num=2, auto_epoch_num=100):
    '''
    with clustering, regrouping, pretrained network, (X)concatencation
    representation: auoencoder Emb
    clustering: DBSCAN
    pretrained: autoencoder
    '''
    '''
    training and testing loop for model using Autoencoder Embedding for clustering.
    1. Training Autoencoder
    2. Clustering on Training data and clustering on testing data
    3. Generate Prototypes
    4. Re-group Training data
    5. Training Siamese network on Training data
    6. Testing Siamese network on Testing data
    '''
    cluster_target_folder_names = os.listdir(test_cluster_target_dir)
    cluster_target_folders = [os.sep.join([os.path.normpath(test_cluster_target_dir), i]) for i in
                              cluster_target_folder_names]
    
    #  1. Training Autoencoder
    print("====================================== TRAINING AUTOENCODER ....... ==================================")
    AutoEncoder_Training_Loop(epoch_num=auto_epoch_num,
                              data_path=autoencoder_training_data_path,
                              save_model_path=autoencoder_save_model_path,
                              save_img_base_path=autoencoder_save_img_base_path)

    # 2. Clustering on Training data
    print("====================================== CLUSTERING ON TRAINING DATA ....... ==================================")
    
    net = Autoencoder_Conv().cuda()
    net.load_state_dict(torch.load(autoencoder_save_model_path))
    
    best_cluster_path = DBSCAN_clustering_with_auto_emb_new(
        dataset_path=training_data_cluster_target,
        transformation=transformation,
        inter_dest_dir=training_clustered_inter_dir,
        best_dest_dir=training_clustered_best_dir,
        net=net, save_inter=False,
        desig_eps=15)
    train_best_clustered_path = best_cluster_path

    # 3. Generate Prototypes
    print("====================================== GENERATING TRAINING PROTOTYPES ....... ==================================")
    generate_training_prototype(target_dataset_path=best_cluster_path, dest_dataset_path=train_prototype_dir)
    
    # 4. Re-group Training data
    print("====================================== RE-GROUP TRAINING DATA ....... ==================================")
    best_cluster_path = DBSCAN_clustering_with_auto_emb_new(
        dataset_path=train_prototype_dir,
        transformation=transformation,
        inter_dest_dir=proto_clustered_inter_dest_dir,
        best_dest_dir=proto_clustered_best_dest_dir,
        net=net, save_inter=True,
        desig_eps=13)#17

    # train_best_clustered_path = "C:/Users/Lizzie0930/Desktop/git/Results_Auto/4_train_clustered_best_dest/final_clustered_dataset"
    re_group_train_data(source_dir=train_best_clustered_path, info_dir=best_cluster_path,
                        target_dir=regrouped_target_dir)
    
    # 2. Clustering on testing data
    print("====================================== CLUSTERING ON TESTING DATA ....... ==================================")
    """
        should cut testing data into known and unknown, and generate prototype only on known data
        2. pass known_cluster_target_folders to test_cluster_target_dir
        3. change generate_testing_prototype to generate_testing_prototype_new
        4. pass unknown_cluster_target_folders to vali_data_dir
    """
    for i in range(len(cluster_target_folders)):
        single_case_inter_dest_dir = os.sep.join(
            [os.path.normpath(testing_clustered_inter_dir), cluster_target_folder_names[i]])
        if not os.path.exists(single_case_inter_dest_dir):
            os.mkdir(single_case_inter_dest_dir)
        single_case_inter_best_dir = os.sep.join(
            [os.path.normpath(testing_clustered_best_dir), cluster_target_folder_names[i]])
        if not os.path.exists(single_case_inter_best_dir):
            os.mkdir(single_case_inter_best_dir)
        single_case_prototype_dest_dir = os.sep.join(
            [os.path.normpath(test_prototype_dest_dir), cluster_target_folder_names[i]])
        if not os.path.exists(single_case_prototype_dest_dir):
            os.mkdir(single_case_prototype_dest_dir)
        best_cluster_path = DBSCAN_clustering_with_auto_emb_new(dataset_path=cluster_target_folders[i], transformation=transformation,
                                              inter_dest_dir=single_case_inter_dest_dir,
                                              best_dest_dir=single_case_inter_best_dir, net=net,
                                              save_inter=False, desig_eps=15)
        generate_testing_prototype_new(target_dataset_path=best_cluster_path,
                                    dest_dataset_path=single_case_prototype_dest_dir)
    

    # train the model & validate the model and save the results
    print(
        "====================================== TRAIN THE SIAMESE NETWORK MODEL ....... ==================================")
    folder_dataset = datasets.ImageFolder(root=regrouped_target_dir)
    siamese_train_dataset = SiameseNetworkDataset_1(imageFolderDataset=folder_dataset,
                                                    transform=transformation)
    training_dataset = siamese_train_dataset
    best_epoch_list, best_batch_list, best_model_path_list = Training_Loop(model_num=model_num, net_name=net_name,
                                                                           dataset=training_dataset,
                                                                           save_best_model_dir=save_best_si_model_dir,
                                                                           save_inter_model_dir=save_inter_si_model_dir,
                                                                           epoch_num=si_epoch_num, way_list=way_list,
                                                                           vali_data_dir=vali_data_dir,
                                                                           proto_data_dir=test_prototype_dest_dir,
                                                                           csv_save_dir=csv_save_dir,
                                                                           case_len=case_len, net_type=net_type)
    print("best_epoch_list: ", best_epoch_list)
    print("best_batch_list: ", best_batch_list)
    print("best_model_path_list: ", best_model_path_list)

    print(
        "====================================== TESTING SIAMESE NETWORK MODEL ON TESTING DATA ....... ==================================")

    result_df, avg_df = group_test(model_num=model_num, way_list=way_list, case_len=case_len, epoch=best_epoch_list[0],
                                   model_path=best_model_path_list[0], batch_num=best_batch_list[0], net_name=net_name,
                                   net_type=net_type, test_data_dir=vali_data_dir, test_data_postfix="",
                                   proto_data_dir=test_prototype_dest_dir, proto_data_postfix="")
    
    print(result_df)
    print(avg_df)
    result_csv_path = os.sep.join([os.path.normpath(result_csv_dir), "results.csv"])
    avg_csv_path = os.sep.join([os.path.normpath(result_csv_dir), "avg.csv"])
    result_df.to_csv(result_csv_path)
    avg_df.to_csv(avg_csv_path)

def complete_training_loop_A6(autoencoder_training_data_path, autoencoder_save_model_path,
                            autoencoder_save_img_base_path,
                            training_data_cluster_target, training_clustered_inter_dir,
                            training_clustered_best_dir,
                            train_prototype_dir, regrouped_target_dir,
                            test_cluster_target_dir, testing_clustered_inter_dir,
                            testing_clustered_best_dir,
                            save_best_si_model_dir, save_inter_si_model_dir,
                            si_epoch_num, vali_data_dir,
                            way_list, net_name, csv_save_dir, case_len, net_type, shots,
                            test_prototype_dest_dir,
                            proto_clustered_best_dest_dir, proto_clustered_inter_dest_dir,
                            result_csv_dir, transformation,
                            model_num=2, auto_epoch_num=100):
    '''
    with clustering, regrouping, (X)pretrained network, concatencation
    representation: auoencoder Emb
    clustering: DBSCAN
    pretrained: autoencoder
    '''
    '''
    training and testing loop for model using Autoencoder Embedding for clustering.
    1. Training Autoencoder
    2. Clustering on Training data and clustering on testing data
    3. Generate Prototypes
    4. Re-group Training data
    5. Training Siamese network on Training data
    6. Testing Siamese network on Testing data
    '''
    cluster_target_folder_names = os.listdir(test_cluster_target_dir)
    cluster_target_folders = [os.sep.join([os.path.normpath(test_cluster_target_dir), i]) for i in
                              cluster_target_folder_names]
    
    #  1. Training Autoencoder
    print("====================================== TRAINING AUTOENCODER ....... ==================================")
    AutoEncoder_Training_Loop(epoch_num=auto_epoch_num,
                              data_path=autoencoder_training_data_path,
                              save_model_path=autoencoder_save_model_path,
                              save_img_base_path=autoencoder_save_img_base_path)

    # 2. Clustering on Training data
    print("====================================== CLUSTERING ON TRAINING DATA ....... ==================================")
    
    net = Autoencoder_Conv().cuda()
    net.load_state_dict(torch.load(autoencoder_save_model_path))
    
    best_cluster_path = DBSCAN_clustering_with_auto_emb_new(
        dataset_path=training_data_cluster_target,
        transformation=transformation,
        inter_dest_dir=training_clustered_inter_dir,
        best_dest_dir=training_clustered_best_dir,
        net=net, save_inter=False,
        desig_eps=15)
    train_best_clustered_path = best_cluster_path

    # 3. Generate Prototypes
    print("====================================== GENERATING TRAINING PROTOTYPES ....... ==================================")
    generate_training_prototype(target_dataset_path=best_cluster_path, dest_dataset_path=train_prototype_dir)
    
    # 4. Re-group Training data
    print("====================================== RE-GROUP TRAINING DATA ....... ==================================")
    best_cluster_path = DBSCAN_clustering_with_auto_emb_new(
        dataset_path=train_prototype_dir,
        transformation=transformation,
        inter_dest_dir=proto_clustered_inter_dest_dir,
        best_dest_dir=proto_clustered_best_dest_dir,
        net=net, save_inter=True,
        desig_eps=13)#17

    # train_best_clustered_path = "C:/Users/Lizzie0930/Desktop/git/Results_Auto/4_train_clustered_best_dest/final_clustered_dataset"
    re_group_train_data(source_dir=train_best_clustered_path, info_dir=best_cluster_path,
                        target_dir=regrouped_target_dir)
    
    # 2. Clustering on testing data
    print("====================================== CLUSTERING ON TESTING DATA ....... ==================================")
    """
        should cut testing data into known and unknown, and generate prototype only on known data
        2. pass known_cluster_target_folders to test_cluster_target_dir
        3. change generate_testing_prototype to generate_testing_prototype_new
        4. pass unknown_cluster_target_folders to vali_data_dir
    """
    for i in range(len(cluster_target_folders)):
        single_case_inter_dest_dir = os.sep.join(
            [os.path.normpath(testing_clustered_inter_dir), cluster_target_folder_names[i]])
        if not os.path.exists(single_case_inter_dest_dir):
            os.mkdir(single_case_inter_dest_dir)
        single_case_inter_best_dir = os.sep.join(
            [os.path.normpath(testing_clustered_best_dir), cluster_target_folder_names[i]])
        if not os.path.exists(single_case_inter_best_dir):
            os.mkdir(single_case_inter_best_dir)
        single_case_prototype_dest_dir = os.sep.join(
            [os.path.normpath(test_prototype_dest_dir), cluster_target_folder_names[i]])
        if not os.path.exists(single_case_prototype_dest_dir):
            os.mkdir(single_case_prototype_dest_dir)
        best_cluster_path = DBSCAN_clustering_with_auto_emb_new(dataset_path=cluster_target_folders[i], transformation=transformation,
                                              inter_dest_dir=single_case_inter_dest_dir,
                                              best_dest_dir=single_case_inter_best_dir, net=net,
                                              save_inter=False, desig_eps=15)
        generate_testing_prototype_new(target_dataset_path=best_cluster_path,
                                    dest_dataset_path=single_case_prototype_dest_dir)
    

    # train the model & validate the model and save the results
    print(
        "====================================== TRAIN THE SIAMESE NETWORK MODEL ....... ==================================")
    folder_dataset = datasets.ImageFolder(root=regrouped_target_dir)
    siamese_train_dataset = SiameseNetworkDataset_1(imageFolderDataset=folder_dataset,
                                                    transform=transformation)
    training_dataset = siamese_train_dataset
    best_epoch_list, best_batch_list, best_model_path_list = Training_Loop(model_num=model_num, net_name=net_name,
                                                                           dataset=training_dataset,
                                                                           save_best_model_dir=save_best_si_model_dir,
                                                                           save_inter_model_dir=save_inter_si_model_dir,
                                                                           epoch_num=si_epoch_num, way_list=way_list,
                                                                           vali_data_dir=vali_data_dir,
                                                                           proto_data_dir=test_prototype_dest_dir,
                                                                           csv_save_dir=csv_save_dir,
                                                                           case_len=case_len, net_type=net_type)
    print("best_epoch_list: ", best_epoch_list)
    print("best_batch_list: ", best_batch_list)
    print("best_model_path_list: ", best_model_path_list)

    print(
        "====================================== TESTING SIAMESE NETWORK MODEL ON TESTING DATA ....... ==================================")

    result_df, avg_df = group_test(model_num=model_num, way_list=way_list, case_len=case_len, epoch=best_epoch_list[0],
                                   model_path=best_model_path_list[0], batch_num=best_batch_list[0], net_name=net_name,
                                   net_type=net_type, test_data_dir=vali_data_dir, test_data_postfix="",
                                   proto_data_dir=test_prototype_dest_dir, proto_data_postfix="")
    
    print(result_df)
    print(avg_df)
    result_csv_path = os.sep.join([os.path.normpath(result_csv_dir), "results.csv"])
    avg_csv_path = os.sep.join([os.path.normpath(result_csv_dir), "avg.csv"])
    result_df.to_csv(result_csv_path)
    avg_df.to_csv(avg_csv_path)

def complete_training_loop_A7(autoencoder_training_data_path, autoencoder_save_model_path,
                            autoencoder_save_img_base_path,
                            training_data_cluster_target, training_clustered_inter_dir,
                            training_clustered_best_dir,
                            train_prototype_dir, regrouped_target_dir,
                            test_cluster_target_dir, testing_clustered_inter_dir,
                            testing_clustered_best_dir,
                            save_best_si_model_dir, save_inter_si_model_dir,
                            si_epoch_num, vali_data_dir,
                            way_list, net_name, csv_save_dir, case_len, net_type, shots,
                            test_prototype_dest_dir,
                            proto_clustered_best_dest_dir, proto_clustered_inter_dest_dir,
                            result_csv_dir, transformation,
                            model_num=2, auto_epoch_num=100):
    '''
    with (X)clustering, (X)regrouping, pretrained network, concatencation
    representation: auoencoder Emb
    clustering: DBSCAN
    pretrained: autoencoder
    '''
    '''
    training and testing loop for model using Autoencoder Embedding for clustering.
    1. Training Autoencoder
    2. Clustering on Training data and clustering on testing data
    3. Generate Prototypes
    4. Re-group Training data
    5. Training Siamese network on Training data
    6. Testing Siamese network on Testing data
    '''
    cluster_target_folder_names = os.listdir(test_cluster_target_dir)
    cluster_target_folders = [os.sep.join([os.path.normpath(test_cluster_target_dir), i]) for i in
                              cluster_target_folder_names]
    
    #  1. Training Autoencoder
    print("====================================== TRAINING AUTOENCODER ....... ==================================")
    AutoEncoder_Training_Loop(epoch_num=auto_epoch_num,
                              data_path=autoencoder_training_data_path,
                              save_model_path=autoencoder_save_model_path,
                              save_img_base_path=autoencoder_save_img_base_path)

    # 2. Clustering on Training data
    print("====================================== CLUSTERING ON TRAINING DATA ....... ==================================")
    
    net = Autoencoder_Conv().cuda()
    net.load_state_dict(torch.load(autoencoder_save_model_path))
    
    best_cluster_path = DBSCAN_clustering_with_auto_emb_new(
        dataset_path=training_data_cluster_target,
        transformation=transformation,
        inter_dest_dir=training_clustered_inter_dir,
        best_dest_dir=training_clustered_best_dir,
        net=net, save_inter=False,
        desig_eps=15)
    train_best_clustered_path = best_cluster_path

    # 3. Generate Prototypes
    print("====================================== GENERATING TRAINING PROTOTYPES ....... ==================================")
    generate_training_prototype(target_dataset_path=best_cluster_path, dest_dataset_path=train_prototype_dir)
    
    # 4. Re-group Training data
    print("====================================== RE-GROUP TRAINING DATA ....... ==================================")
    best_cluster_path = DBSCAN_clustering_with_auto_emb_new(
        dataset_path=train_prototype_dir,
        transformation=transformation,
        inter_dest_dir=proto_clustered_inter_dest_dir,
        best_dest_dir=proto_clustered_best_dest_dir,
        net=net, save_inter=True,
        desig_eps=13)#17

    # train_best_clustered_path = "C:/Users/Lizzie0930/Desktop/git/Results_Auto/4_train_clustered_best_dest/final_clustered_dataset"
    re_group_train_data(source_dir=train_best_clustered_path, info_dir=best_cluster_path,
                        target_dir=regrouped_target_dir)
    
    # 2. Clustering on testing data
    print("====================================== CLUSTERING ON TESTING DATA ....... ==================================")
    """
        should cut testing data into known and unknown, and generate prototype only on known data
        2. pass known_cluster_target_folders to test_cluster_target_dir
        3. change generate_testing_prototype to generate_testing_prototype_new
        4. pass unknown_cluster_target_folders to vali_data_dir
    """
    for i in range(len(cluster_target_folders)):
        single_case_inter_dest_dir = os.sep.join(
            [os.path.normpath(testing_clustered_inter_dir), cluster_target_folder_names[i]])
        if not os.path.exists(single_case_inter_dest_dir):
            os.mkdir(single_case_inter_dest_dir)
        single_case_inter_best_dir = os.sep.join(
            [os.path.normpath(testing_clustered_best_dir), cluster_target_folder_names[i]])
        if not os.path.exists(single_case_inter_best_dir):
            os.mkdir(single_case_inter_best_dir)
        single_case_prototype_dest_dir = os.sep.join(
            [os.path.normpath(test_prototype_dest_dir), cluster_target_folder_names[i]])
        if not os.path.exists(single_case_prototype_dest_dir):
            os.mkdir(single_case_prototype_dest_dir)
        best_cluster_path = DBSCAN_clustering_with_auto_emb_new(dataset_path=cluster_target_folders[i], transformation=transformation,
                                              inter_dest_dir=single_case_inter_dest_dir,
                                              best_dest_dir=single_case_inter_best_dir, net=net,
                                              save_inter=False, desig_eps=15)
        generate_testing_prototype_new(target_dataset_path=best_cluster_path,
                                    dest_dataset_path=single_case_prototype_dest_dir)
    

    # train the model & validate the model and save the results
    print(
        "====================================== TRAIN THE SIAMESE NETWORK MODEL ....... ==================================")
    folder_dataset = datasets.ImageFolder(root=regrouped_target_dir)
    siamese_train_dataset = SiameseNetworkDataset_1(imageFolderDataset=folder_dataset,
                                                    transform=transformation)
    training_dataset = siamese_train_dataset
    best_epoch_list, best_batch_list, best_model_path_list = Training_Loop(model_num=model_num, net_name=net_name,
                                                                           dataset=training_dataset,
                                                                           save_best_model_dir=save_best_si_model_dir,
                                                                           save_inter_model_dir=save_inter_si_model_dir,
                                                                           epoch_num=si_epoch_num, way_list=way_list,
                                                                           vali_data_dir=vali_data_dir,
                                                                           proto_data_dir=test_prototype_dest_dir,
                                                                           csv_save_dir=csv_save_dir,
                                                                           case_len=case_len, net_type=net_type)
    print("best_epoch_list: ", best_epoch_list)
    print("best_batch_list: ", best_batch_list)
    print("best_model_path_list: ", best_model_path_list)

    print(
        "====================================== TESTING SIAMESE NETWORK MODEL ON TESTING DATA ....... ==================================")

    result_df, avg_df = group_test(model_num=model_num, way_list=way_list, case_len=case_len, epoch=best_epoch_list[0],
                                   model_path=best_model_path_list[0], batch_num=best_batch_list[0], net_name=net_name,
                                   net_type=net_type, test_data_dir=vali_data_dir, test_data_postfix="",
                                   proto_data_dir=test_prototype_dest_dir, proto_data_postfix="")
    
    print(result_df)
    print(avg_df)
    result_csv_path = os.sep.join([os.path.normpath(result_csv_dir), "results.csv"])
    avg_csv_path = os.sep.join([os.path.normpath(result_csv_dir), "avg.csv"])
    result_df.to_csv(result_csv_path)
    avg_df.to_csv(avg_csv_path)

def complete_training_loop_A8(autoencoder_training_data_path, autoencoder_save_model_path,
                            autoencoder_save_img_base_path,
                            training_data_cluster_target, training_clustered_inter_dir,
                            training_clustered_best_dir,
                            train_prototype_dir, regrouped_target_dir,
                            test_cluster_target_dir, testing_clustered_inter_dir,
                            testing_clustered_best_dir,
                            save_best_si_model_dir, save_inter_si_model_dir,
                            si_epoch_num, vali_data_dir,
                            way_list, net_name, csv_save_dir, case_len, net_type, shots,
                            test_prototype_dest_dir,
                            proto_clustered_best_dest_dir, proto_clustered_inter_dest_dir,
                            result_csv_dir, transformation,
                            model_num=2, auto_epoch_num=100):
    '''
    with (X)clustering, (X)regrouping, (X)pretrained network, (X)concatencation
    representation: auoencoder Emb
    clustering: DBSCAN
    pretrained: autoencoder
    '''
    '''
    training and testing loop for model using Autoencoder Embedding for clustering.
    1. Training Autoencoder
    2. Clustering on Training data and clustering on testing data
    3. Generate Prototypes
    4. Re-group Training data
    5. Training Siamese network on Training data
    6. Testing Siamese network on Testing data
    '''
    cluster_target_folder_names = os.listdir(test_cluster_target_dir)
    cluster_target_folders = [os.sep.join([os.path.normpath(test_cluster_target_dir), i]) for i in
                              cluster_target_folder_names]
    
    #  1. Training Autoencoder
    print("====================================== TRAINING AUTOENCODER ....... ==================================")
    AutoEncoder_Training_Loop(epoch_num=auto_epoch_num,
                              data_path=autoencoder_training_data_path,
                              save_model_path=autoencoder_save_model_path,
                              save_img_base_path=autoencoder_save_img_base_path)

    # 2. Clustering on Training data
    print("====================================== CLUSTERING ON TRAINING DATA ....... ==================================")
    
    net = Autoencoder_Conv().cuda()
    net.load_state_dict(torch.load(autoencoder_save_model_path))
    
    best_cluster_path = DBSCAN_clustering_with_auto_emb_new(
        dataset_path=training_data_cluster_target,
        transformation=transformation,
        inter_dest_dir=training_clustered_inter_dir,
        best_dest_dir=training_clustered_best_dir,
        net=net, save_inter=False,
        desig_eps=15)
    train_best_clustered_path = best_cluster_path

    # 3. Generate Prototypes
    print("====================================== GENERATING TRAINING PROTOTYPES ....... ==================================")
    generate_training_prototype(target_dataset_path=best_cluster_path, dest_dataset_path=train_prototype_dir)
    
    # 4. Re-group Training data
    print("====================================== RE-GROUP TRAINING DATA ....... ==================================")
    best_cluster_path = DBSCAN_clustering_with_auto_emb_new(
        dataset_path=train_prototype_dir,
        transformation=transformation,
        inter_dest_dir=proto_clustered_inter_dest_dir,
        best_dest_dir=proto_clustered_best_dest_dir,
        net=net, save_inter=True,
        desig_eps=13)#17

    # train_best_clustered_path = "C:/Users/Lizzie0930/Desktop/git/Results_Auto/4_train_clustered_best_dest/final_clustered_dataset"
    re_group_train_data(source_dir=train_best_clustered_path, info_dir=best_cluster_path,
                        target_dir=regrouped_target_dir)
    
    # 2. Clustering on testing data
    print("====================================== CLUSTERING ON TESTING DATA ....... ==================================")
    """
        should cut testing data into known and unknown, and generate prototype only on known data
        2. pass known_cluster_target_folders to test_cluster_target_dir
        3. change generate_testing_prototype to generate_testing_prototype_new
        4. pass unknown_cluster_target_folders to vali_data_dir
    """
    for i in range(len(cluster_target_folders)):
        single_case_inter_dest_dir = os.sep.join(
            [os.path.normpath(testing_clustered_inter_dir), cluster_target_folder_names[i]])
        if not os.path.exists(single_case_inter_dest_dir):
            os.mkdir(single_case_inter_dest_dir)
        single_case_inter_best_dir = os.sep.join(
            [os.path.normpath(testing_clustered_best_dir), cluster_target_folder_names[i]])
        if not os.path.exists(single_case_inter_best_dir):
            os.mkdir(single_case_inter_best_dir)
        single_case_prototype_dest_dir = os.sep.join(
            [os.path.normpath(test_prototype_dest_dir), cluster_target_folder_names[i]])
        if not os.path.exists(single_case_prototype_dest_dir):
            os.mkdir(single_case_prototype_dest_dir)
        best_cluster_path = DBSCAN_clustering_with_auto_emb_new(dataset_path=cluster_target_folders[i], transformation=transformation,
                                              inter_dest_dir=single_case_inter_dest_dir,
                                              best_dest_dir=single_case_inter_best_dir, net=net,
                                              save_inter=False, desig_eps=15)
        generate_testing_prototype_new(target_dataset_path=best_cluster_path,
                                    dest_dataset_path=single_case_prototype_dest_dir)
    

    # train the model & validate the model and save the results
    print(
        "====================================== TRAIN THE SIAMESE NETWORK MODEL ....... ==================================")
    folder_dataset = datasets.ImageFolder(root=regrouped_target_dir)
    siamese_train_dataset = SiameseNetworkDataset_1(imageFolderDataset=folder_dataset,
                                                    transform=transformation)
    training_dataset = siamese_train_dataset
    best_epoch_list, best_batch_list, best_model_path_list = Training_Loop(model_num=model_num, net_name=net_name,
                                                                           dataset=training_dataset,
                                                                           save_best_model_dir=save_best_si_model_dir,
                                                                           save_inter_model_dir=save_inter_si_model_dir,
                                                                           epoch_num=si_epoch_num, way_list=way_list,
                                                                           vali_data_dir=vali_data_dir,
                                                                           proto_data_dir=test_prototype_dest_dir,
                                                                           csv_save_dir=csv_save_dir,
                                                                           case_len=case_len, net_type=net_type)
    print("best_epoch_list: ", best_epoch_list)
    print("best_batch_list: ", best_batch_list)
    print("best_model_path_list: ", best_model_path_list)

    print(
        "====================================== TESTING SIAMESE NETWORK MODEL ON TESTING DATA ....... ==================================")

    result_df, avg_df = group_test(model_num=model_num, way_list=way_list, case_len=case_len, epoch=best_epoch_list[0],
                                   model_path=best_model_path_list[0], batch_num=best_batch_list[0], net_name=net_name,
                                   net_type=net_type, test_data_dir=vali_data_dir, test_data_postfix="",
                                   proto_data_dir=test_prototype_dest_dir, proto_data_postfix="")
    
    print(result_df)
    print(avg_df)
    result_csv_path = os.sep.join([os.path.normpath(result_csv_dir), "results.csv"])
    avg_csv_path = os.sep.join([os.path.normpath(result_csv_dir), "avg.csv"])
    result_df.to_csv(result_csv_path)
    avg_df.to_csv(avg_csv_path)

def complete_training_loop_RN(autoencoder_training_data_path, autoencoder_save_model_path,
                            autoencoder_save_img_base_path,
                            training_data_cluster_target, training_clustered_inter_dir,
                            training_clustered_best_dir,
                            train_prototype_dir, regrouped_target_dir,
                            test_cluster_target_dir, testing_clustered_inter_dir,
                            testing_clustered_best_dir,
                            save_best_si_model_dir, save_inter_si_model_dir,
                            si_epoch_num, vali_data_dir,
                            way_list, net_name, csv_save_dir, case_len, net_type, shots,
                            test_prototype_dest_dir,
                            proto_clustered_best_dest_dir, proto_clustered_inter_dest_dir,
                            result_csv_dir, transformation,
                            model_num=2, auto_epoch_num=100):
    '''
    with (X)clustering, (X)regrouping, (X)pretrained network, (X)concatencation
    representation: auoencoder Emb
    clustering: DBSCAN
    pretrained: autoencoder
    '''
    '''
    training and testing loop for model using Autoencoder Embedding for clustering.
    1. Training Autoencoder
    2. Clustering on Training data and clustering on testing data
    3. Generate Prototypes
    4. Re-group Training data
    5. Training Siamese network on Training data
    6. Testing Siamese network on Testing data
    '''
    cluster_target_folder_names = os.listdir(test_cluster_target_dir)
    cluster_target_folders = [os.sep.join([os.path.normpath(test_cluster_target_dir), i]) for i in
                              cluster_target_folder_names]
    
    #  1. Training Autoencoder
    print("====================================== TRAINING AUTOENCODER ....... ==================================")
    AutoEncoder_Training_Loop(epoch_num=auto_epoch_num,
                              data_path=autoencoder_training_data_path,
                              save_model_path=autoencoder_save_model_path,
                              save_img_base_path=autoencoder_save_img_base_path)

    # 2. Clustering on Training data
    print("====================================== CLUSTERING ON TRAINING DATA ....... ==================================")
    
    net = Autoencoder_Conv().cuda()
    net.load_state_dict(torch.load(autoencoder_save_model_path))
    
    best_cluster_path = DBSCAN_clustering_with_auto_emb_new(
        dataset_path=training_data_cluster_target,
        transformation=transformation,
        inter_dest_dir=training_clustered_inter_dir,
        best_dest_dir=training_clustered_best_dir,
        net=net, save_inter=False,
        desig_eps=15)
    train_best_clustered_path = best_cluster_path

    # 3. Generate Prototypes
    print("====================================== GENERATING TRAINING PROTOTYPES ....... ==================================")
    generate_training_prototype(target_dataset_path=best_cluster_path, dest_dataset_path=train_prototype_dir)
    
    # 4. Re-group Training data
    print("====================================== RE-GROUP TRAINING DATA ....... ==================================")
    best_cluster_path = DBSCAN_clustering_with_auto_emb_new(
        dataset_path=train_prototype_dir,
        transformation=transformation,
        inter_dest_dir=proto_clustered_inter_dest_dir,
        best_dest_dir=proto_clustered_best_dest_dir,
        net=net, save_inter=True,
        desig_eps=13)#17

    # train_best_clustered_path = "C:/Users/Lizzie0930/Desktop/git/Results_Auto/4_train_clustered_best_dest/final_clustered_dataset"
    re_group_train_data(source_dir=train_best_clustered_path, info_dir=best_cluster_path,
                        target_dir=regrouped_target_dir)
    
    # 2. Clustering on testing data
    print("====================================== CLUSTERING ON TESTING DATA ....... ==================================")
    """
        should cut testing data into known and unknown, and generate prototype only on known data
        2. pass known_cluster_target_folders to test_cluster_target_dir
        3. change generate_testing_prototype to generate_testing_prototype_new
        4. pass unknown_cluster_target_folders to vali_data_dir
    """
    for i in range(len(cluster_target_folders)):
        single_case_inter_dest_dir = os.sep.join(
            [os.path.normpath(testing_clustered_inter_dir), cluster_target_folder_names[i]])
        if not os.path.exists(single_case_inter_dest_dir):
            os.mkdir(single_case_inter_dest_dir)
        single_case_inter_best_dir = os.sep.join(
            [os.path.normpath(testing_clustered_best_dir), cluster_target_folder_names[i]])
        if not os.path.exists(single_case_inter_best_dir):
            os.mkdir(single_case_inter_best_dir)
        single_case_prototype_dest_dir = os.sep.join(
            [os.path.normpath(test_prototype_dest_dir), cluster_target_folder_names[i]])
        if not os.path.exists(single_case_prototype_dest_dir):
            os.mkdir(single_case_prototype_dest_dir)
        best_cluster_path = DBSCAN_clustering_with_auto_emb_new(dataset_path=cluster_target_folders[i], transformation=transformation,
                                              inter_dest_dir=single_case_inter_dest_dir,
                                              best_dest_dir=single_case_inter_best_dir, net=net,
                                              save_inter=False, desig_eps=15)
        generate_testing_prototype_new(target_dataset_path=best_cluster_path,
                                    dest_dataset_path=single_case_prototype_dest_dir)
    

    # train the model & validate the model and save the results
    print(
        "====================================== TRAIN THE SIAMESE NETWORK MODEL ....... ==================================")
    folder_dataset = datasets.ImageFolder(root=regrouped_target_dir)
    siamese_train_dataset = SiameseNetworkDataset_1(imageFolderDataset=folder_dataset,
                                                    transform=transformation)
    training_dataset = siamese_train_dataset
    best_epoch_list, best_batch_list, best_model_path_list = Training_Loop(model_num=model_num, net_name=net_name,
                                                                           dataset=training_dataset,
                                                                           save_best_model_dir=save_best_si_model_dir,
                                                                           save_inter_model_dir=save_inter_si_model_dir,
                                                                           epoch_num=si_epoch_num, way_list=way_list,
                                                                           vali_data_dir=vali_data_dir,
                                                                           proto_data_dir=test_prototype_dest_dir,
                                                                           csv_save_dir=csv_save_dir,
                                                                           case_len=case_len, net_type=net_type)
    print("best_epoch_list: ", best_epoch_list)
    print("best_batch_list: ", best_batch_list)
    print("best_model_path_list: ", best_model_path_list)

    print(
        "====================================== TESTING SIAMESE NETWORK MODEL ON TESTING DATA ....... ==================================")

    result_df, avg_df = group_test(model_num=model_num, way_list=way_list, case_len=case_len, epoch=best_epoch_list[0],
                                   model_path=best_model_path_list[0], batch_num=best_batch_list[0], net_name=net_name,
                                   net_type=net_type, test_data_dir=vali_data_dir, test_data_postfix="",
                                   proto_data_dir=test_prototype_dest_dir, proto_data_postfix="")
    
    print(result_df)
    print(avg_df)
    result_csv_path = os.sep.join([os.path.normpath(result_csv_dir), "results.csv"])
    avg_csv_path = os.sep.join([os.path.normpath(result_csv_dir), "avg.csv"])
    result_df.to_csv(result_csv_path)
    avg_df.to_csv(avg_csv_path)

