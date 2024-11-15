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

import pandas as pd
import numpy as np
import matplotlib as mpl
import json
import os
import matplotlib.pylab as plt
import csv
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from sklearn.cluster import DBSCAN
import shutil
from PIL import Image
import imagehash
import torchvision.transforms as T

from dataset import DBSCANDataset

def get_hashes(img):
    phash = imagehash.phash(img,hash_size=8)
    chash = imagehash.colorhash(img)
    return phash, chash

def clustering_family(fam_path, dest, weight_avg, weight_color, threshold_cluster):
    images = []
    img_paths = []
    
    for img in os.listdir(fam_path):
        image_path = os.path.join(fam_path, img)
        image = Image.open(image_path)
        images.append(image)
        img_paths.append(image_path)
    
    distance_matrix_avg = np.zeros((len(images),len(images)), dtype=int)
    distance_matrix_col = np.zeros((len(images),len(images)), dtype=int)
    
    for img1 in range(0,len(images)):
        hash_avg1, hash_col1 = get_hashes(images[img1])
        for img2 in range(0,len(images)):
            hash_avg2, hash_col2 = get_hashes(images[img2])
            dff1 = abs(hash_avg1-hash_avg2)
            dff2 = abs(hash_col1-hash_col2)
            distance_matrix_avg[img1][img2] = dff1
            distance_matrix_col[img1][img2] = dff2
            
    combined_distance_matrix = (weight_avg * distance_matrix_avg) + (weight_color * distance_matrix_col)

    if len(images) <= 1:
        sub_families = {}
        sub_families[1] = []
        sub_families[1].append(img_paths[0])
    else:
        
        # Perform hierarchical clustering
        Z = linkage(combined_distance_matrix, method='complete', metric='euclidean')  # Complete linkage for agglomerative clustering
        
        # Determine clusters based on a threshold distance or number of clusters
        clusters = fcluster(Z, threshold_cluster, criterion='distance')
        sub_families = {}
        for img_path, cluster in zip(img_paths, clusters):
            if cluster not in sub_families:
                sub_families[cluster] = []
            sub_families[cluster].append(img_path)
    
        
        
    for cluster, images in sub_families.items():
        fam_name = os.path.basename(fam_path)
        sub_family_folder = dest + '\\' + fam_name + str(cluster)
        if not os.path.isdir(sub_family_folder):
            os.makedirs(sub_family_folder)
            
        
        for image_path in images:
            image_filename = os.path.basename(image_path)
            new_image_path = os.path.join(sub_family_folder, image_filename)
            # Check whether it is called from Testing data.
            if fam_path == dest: # If yes (fam_path == dest), move the file
                shutil.move(image_path, new_image_path)
            else: # Otherwise, copy the file instead (reserve the original images)
                shutil.copy(image_path, new_image_path)
    
def save_avg_img(avg_img,fam_name,dest,test):
    # display the plot 
    fig_size = (160/72, 160/72) #saving the figure size
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_axes([0, 0, 1, 1], frameon=False, xticks=[], yticks=[])
    plt.axis('off')
    ax.imshow(avg_img.astype(np.uint8))
    if test == False:
        dest_path = os.path.join(dest,'1_avg')
    else:
        dest_path = dest

    if not os.path.isdir(dest_path):
        os.makedirs(dest_path)
    plt.savefig(dest_path+'\\' +fam_name + '.png',dpi=72)
    plt.close(fig)

def get_avg_img(fam_path,dest,test=False):
    images = []
    fam_name = os.path.basename(fam_path)
    for img in os.listdir(fam_path):
        image_path = os.path.join(fam_path, img)
        image = Image.open(image_path)
        images.append(image)
    # Compute the average of the images
    average_image = np.zeros_like(images[0], dtype=float)
    for image in images:
        average_image += np.array(image, dtype=float)
    average_image /= len(images)
    save_avg_img(average_image,fam_name,dest,test)

def combine_subfamilies_imgh(dest, weight_avg, weight_color, threshold_combine):
    avg_path = os.path.join(dest,'1_avg')
    images = []
    img_paths = []

    for avg_img in os.listdir(avg_path):
        image_path = os.path.join(avg_path, avg_img)
        image = Image.open(image_path)
        images.append(image)
        img_paths.append(image_path)

    distance_matrix_avg = np.zeros((len(images),len(images)), dtype=int)
    distance_matrix_col = np.zeros((len(images),len(images)), dtype=int)
    for img1 in range(0,len(images)):
        hash_avg1, hash_col1 = get_hashes(images[img1])
        for img2 in range(0,len(images)):
            hash_avg2, hash_col2 = get_hashes(images[img2])
            dff1 = abs(hash_avg1-hash_avg2)
            dff2 = abs(hash_col1-hash_col2)
            distance_matrix_avg[img1][img2] = dff1
            distance_matrix_col[img1][img2] = dff2

    combined_distance_matrix = (weight_avg * distance_matrix_avg) + (weight_color * distance_matrix_col)
    # Perform hierarchical clustering
    Z = linkage(combined_distance_matrix, method='complete')  # Complete linkage for agglomerative clustering

    # Determine clusters based on a threshold distance or number of clusters
    clusters = fcluster(Z, threshold_combine, criterion='distance')
    sub_families = {}
    for img_path, cluster in zip(img_paths, clusters):
        if cluster not in sub_families:
            sub_families[cluster] = []
        sub_families[cluster].append(img_path)

    for cluster, images in sub_families.items():
        if len(images) > 1:
            
            folder_name = "-".join([os.path.splitext(os.path.basename(image))[0] for image in images])
            # Create the output folder for the combined sub-family
            combined_folder_path = os.path.join(dest, folder_name)
            os.makedirs(combined_folder_path, exist_ok=True)
        
            # Move and rename the images to the combined sub-family folder
            for image in images:
                image_name = os.path.splitext(os.path.basename(image))[0] 
                image_folder = os.path.join(dest, image_name)
                
                for img in os.listdir(image_folder):
                    new_image_path = os.path.join(combined_folder_path, img)
                    shutil.copy(image, new_image_path)
                # Delete the original folder
                shutil.rmtree(image_folder)

    print('Original number of subfamilies: ',combined_distance_matrix.shape[0])
    print('Combined number of subfamilies: ',np.unique(clusters).shape[0])

def combine_subfamilies_auto_emb(dest, threshold_combine, transformation, net):
    images = []
    img_paths = []

    images_group_path = os.path.normpath(dest)
    folder_dataset = datasets.ImageFolder(root=images_group_path)
    DBSCAN_dataset = DBSCANDataset(imageFolderDataset=folder_dataset,
                                    transform=transformation)
    DBSCAN_dataloader = DataLoader(DBSCAN_dataset, batch_size=1, shuffle=False)
    images_group_len = len(DBSCAN_dataset)
    print(images_group_len)
    embeddings_list = []
    for j, (img_0, label_0, path_0) in enumerate(DBSCAN_dataloader, 0):
        path = path_0[0].split('\\')[-1]
        images.append(img_0)
        img_paths.append(path_0[0])
        
        for k, (img_1, label_1, path_1) in enumerate(DBSCAN_dataloader, 0):
            path = path_1[0].split('\\')[-1]
            output1, output2 = torch.squeeze(net.encoder(img_0.cuda())), torch.squeeze(net.encoder(img_1.cuda()))
        embeddings_list.append(output1.detach().cpu().numpy().flatten())
    
    combined_distance_matrix = np.array(embeddings_list)

    # Perform hierarchical clustering
    Z = linkage(combined_distance_matrix, method='complete')  # Complete linkage for agglomerative clustering

    # Determine clusters based on a threshold distance or number of clusters
    clusters = fcluster(Z, threshold_combine, criterion='distance')
    sub_families = {}
    for img_path, cluster in zip(img_paths, clusters):
        if cluster not in sub_families:
            sub_families[cluster] = []
        sub_families[cluster].append(img_path)

    for cluster, images in sub_families.items():
        if len(images) > 1:
            
            folder_name = "-".join([os.path.splitext(os.path.basename(image))[0] for image in images])
            # Create the output folder for the combined sub-family
            combined_folder_path = os.path.join(dest, folder_name)
            os.makedirs(combined_folder_path, exist_ok=True)
        
            # Move and rename the images to the combined sub-family folder
            for image in images:
                image_name = os.path.splitext(os.path.basename(image))[0] 
                image_folder = os.path.join(dest, image_name)
                for img in os.listdir(image_folder):
                    new_image_path = os.path.join(combined_folder_path, img)
                    shutil.copy(image, new_image_path)
                # Delete the original folder
                shutil.rmtree(image_folder)

    print('Original number of subfamilies: ',combined_distance_matrix.shape[0])
    print('Combined number of subfamilies: ',np.unique(clusters).shape[0])

def combine_subfamilies_siame_emb(dest, threshold_combine, transformation, net):
    images = []
    img_paths = []

    images_group_path = os.path.normpath(dest)
    folder_dataset = datasets.ImageFolder(root=images_group_path)
    DBSCAN_dataset = DBSCANDataset(imageFolderDataset=folder_dataset,
                                    transform=transformation)
    DBSCAN_dataloader = DataLoader(DBSCAN_dataset, batch_size=1, shuffle=False)
    images_group_len = len(DBSCAN_dataset)
    print(images_group_len)
    embeddings_list = []
    for j, (img_0, label_0, path_0) in enumerate(DBSCAN_dataloader, 0):
        path = path_0[0].split('\\')[-1]
        images.append(img_0)
        img_paths.append(path_0[0])
        for k, (img_1, label_1, path_1) in enumerate(DBSCAN_dataloader, 0):
            path = path_1[0].split('\\')[-1]
            output1, output2 = net(img_0.cuda(), img_1.cuda())
        embeddings_list.append(output1.detach().cpu().numpy().flatten())
    
    combined_distance_matrix = np.array(embeddings_list)

    # Perform hierarchical clustering
    Z = linkage(combined_distance_matrix, method='complete')  # Complete linkage for agglomerative clustering

    # Determine clusters based on a threshold distance or number of clusters
    clusters = fcluster(Z, threshold_combine, criterion='distance')
    sub_families = {}
    for img_path, cluster in zip(img_paths, clusters):
        if cluster not in sub_families:
            sub_families[cluster] = []
        sub_families[cluster].append(img_path)

    for cluster, images in sub_families.items():
        if len(images) > 1:
            folder_name = "-".join([os.path.splitext(os.path.basename(image))[0] for image in images])
            # Create the output folder for the combined sub-family
            combined_folder_path = os.path.join(dest, folder_name)
            os.makedirs(combined_folder_path, exist_ok=True)
        
            # Move and rename the images to the combined sub-family folder
            for image in images:
                image_name = os.path.splitext(os.path.basename(image))[0] 
                image_folder = os.path.join(dest, image_name)
                for img in os.listdir(image_folder):
                    new_image_path = os.path.join(combined_folder_path, img)
                    shutil.copy(image, new_image_path)
                # Delete the original folder
                shutil.rmtree(image_folder)

    print('Original number of subfamilies: ',combined_distance_matrix.shape[0])
    print('Combined number of subfamilies: ',np.unique(clusters).shape[0])

def DBSCAN_clustering_with_auto_emb_new(dataset_path, transformation, inter_dest_dir, best_dest_dir, net=None, save_inter=True,
                      desig_eps=None):
    images_groups = os.listdir(dataset_path)
    prototypes = {class_idx: [] for class_idx in images_groups}
    for i in range(len(images_groups)):
        images_group_path = os.path.sep.join([os.path.normpath(dataset_path), images_groups[i]])
        folder_dataset = datasets.ImageFolder(root=images_group_path)
        DBSCAN_dataset = DBSCANDataset(imageFolderDataset=folder_dataset,
                                       transform=transformation)
        DBSCAN_dataloader = DataLoader(DBSCAN_dataset, batch_size=1, shuffle=False)
        images_group_len = len(DBSCAN_dataset)
        print(images_group_len)
        embeddings_list = []
        for j, (img_0, label_0, path_0) in enumerate(DBSCAN_dataloader, 0):
            path = path_0[0].split('\\')[-1]
            for k, (img_1, label_1, path_1) in enumerate(DBSCAN_dataloader, 0):
                path = path_1[0].split('\\')[-1]
                output1, output2 = torch.squeeze(net.encoder(img_0.cuda())), torch.squeeze(net.encoder(img_1.cuda()))
            embeddings_list.append(output1.detach().cpu().numpy().flatten())

        embeddings_array = np.array(embeddings_list)

        cluster_results = []
        for m in range(1, 100):
            EPS = float(m)
            # DBSCAN
            clustering = DBSCAN(eps=EPS, min_samples=2)
            SC_result = clustering.fit_predict(embeddings_array)
            print("Clustering Results ", SC_result)
            cluster = SC_result
            Labels = list(set(SC_result))
            Labels_len = len(list(set(SC_result)))
            noise_len = Counter(SC_result)[-1]
            cluster_results.append((EPS, Labels, Labels_len, noise_len))
            # print(SC_result)
            print("eps: ", EPS, "Labels: ", Labels, "Labels_len: ", Labels_len, "SC_result_len: ",
                  len(SC_result), "noise_len: ", noise_len)
            if save_inter:
                clustered_path = os.sep.join([os.path.normpath(inter_dest_dir), f"{EPS}eps"])
                if not os.path.exists(clustered_path):
                    os.mkdir(clustered_path)
                try:
                    Labels_copy = Labels.copy()
                    Labels_copy.remove(-1)
                    Labels_drop_len = len(Labels_copy)
                except:
                    Labels_drop_len = Labels_len
                for q in range(Labels_drop_len):
                    subclass_path = os.path.sep.join(
                        [os.path.normpath(clustered_path), f'{images_groups[i]}_{q}'])
                    if not os.path.exists(subclass_path):
                        os.mkdir(subclass_path)
                noise_count = 1
                for q in range(len(SC_result)):
                    image, label, original_path = DBSCAN_dataset[q]
                    path = original_path.split('\\')[-1]
                    if SC_result[q] != -1:
                        target_path = os.path.sep.join([os.path.normpath(clustered_path),
                                                        f'{images_groups[i]}_{Labels.index(SC_result[q])}',
                                                        path])
                        shutil.copyfile(original_path, target_path)
                    else:
                        print("noise count: ", noise_count)
                        subclass_path = os.path.sep.join([os.path.normpath(clustered_path),
                                                          f'{images_groups[i]}_{Labels_drop_len - 1 + noise_count}_noise'])
                        if not os.path.exists(subclass_path):
                            os.mkdir(subclass_path)
                        target_path = os.path.sep.join([os.path.normpath(clustered_path),
                                                        f'{images_groups[i]}_{Labels_drop_len - 1 + noise_count}_noise',
                                                        path])
                        shutil.copyfile(original_path, target_path)
                        noise_count += 1

                # sort cluster and keep track of label
                sorted_cluster_labels = [i[0] for i in
                                         sorted(enumerate(cluster), key=lambda x: x[1], reverse=True)]

        max_index = [i[2] - (i[3] / 3) for i in cluster_results].index(
            max([i[2] - (i[3] / 3) for i in cluster_results]))
        if desig_eps:
            EPS = desig_eps
        else:
            EPS = cluster_results[max_index][0]
        clustered_path = os.sep.join([os.path.normpath(best_dest_dir), f"final_clustered_dataset"])
        best_cluster_path = clustered_path
        if not os.path.exists(clustered_path):
            os.mkdir(clustered_path)
        # DBSCAN
        clustering = DBSCAN(eps=EPS, min_samples=1)
        SC_result = clustering.fit_predict(embeddings_array)
        #=====================================================
        # Calculate cluster centroids as prototypes
        unique_clusters = np.unique(SC_result)
        cluster_prototypes = []
        for cluster_id in unique_clusters:
            if cluster_id == -1:
                # Handle noise points (not assigned to any cluster) as needed
                continue

            cluster_points = embeddings_array[SC_result == cluster_id]
            cluster_centroid = np.mean(cluster_points, axis=0)
            cluster_prototypes.append(cluster_centroid)

        prototypes[images_groups[i]] = cluster_prototypes
        print("PROTOTYPES　EMBEDDING:", prototypes)
        # =====================================================

        cluster = SC_result
        Labels = list(set(SC_result))
        Labels_len = len(list(set(SC_result)))
        try:
            Labels_copy = Labels.copy()
            Labels_copy.remove(-1)
            Labels_drop_len = len(Labels_copy)
        except:
            Labels_drop_len = Labels_len
        for j in range(Labels_drop_len):
            subclass_path = os.path.sep.join(
                [os.path.normpath(clustered_path), f'{images_groups[i]}_{j}_{EPS}eps'])
            if not os.path.exists(subclass_path):
                os.mkdir(subclass_path)
        noise_count = 1
        for j in range(len(SC_result)):
            image, label, original_path = DBSCAN_dataset[j]
            path = original_path.split('\\')[-1]
            if SC_result[j] != -1:
                target_path = os.path.sep.join([os.path.normpath(clustered_path),
                                                f'{images_groups[i]}_{Labels.index(SC_result[j])}_{EPS}eps',
                                                path])
                shutil.copyfile(original_path, target_path)
            else:
                print("noise count: ", noise_count)
                subclass_path = os.path.sep.join([os.path.normpath(clustered_path),
                                                  f'{images_groups[i]}_{Labels_drop_len - 1 + noise_count}_{EPS}eps_noise'])
                if not os.path.exists(subclass_path):
                    os.mkdir(subclass_path)
                target_path = os.path.sep.join([os.path.normpath(clustered_path),
                                                f'{images_groups[i]}_{Labels_drop_len - 1 + noise_count}_{EPS}eps_noise',
                                                path])
                shutil.copyfile(original_path, target_path)
                noise_count += 1

        # sort cluster and keep track of label
        sorted_cluster_labels = [i[0] for i in sorted(enumerate(cluster), key=lambda x: x[1], reverse=True)]

    return best_cluster_path

def DBSCAN_clustering_with_siame_emb_new(dataset_path, transformation, inter_dest_dir, best_dest_dir, net=None, save_inter=True,
                      desig_eps=None):
    images_groups = os.listdir(dataset_path)
    prototypes = {class_idx: [] for class_idx in images_groups}
    for i in range(len(images_groups)):
        images_group_path = os.path.sep.join([os.path.normpath(dataset_path), images_groups[i]])
        folder_dataset = datasets.ImageFolder(root=images_group_path)
        DBSCAN_dataset = DBSCANDataset(imageFolderDataset=folder_dataset,
                                       transform=transformation)
        DBSCAN_dataloader = DataLoader(DBSCAN_dataset, batch_size=1, shuffle=False)
        images_group_len = len(DBSCAN_dataset)
        print(images_group_len)
        embeddings_list = []
        for j, (img_0, label_0, path_0) in enumerate(DBSCAN_dataloader, 0):
            path = path_0[0].split('\\')[-1]
            for k, (img_1, label_1, path_1) in enumerate(DBSCAN_dataloader, 0):
                path = path_1[0].split('\\')[-1]
                output1, output2 = net(img_0.cuda(), img_1.cuda())
            embeddings_list.append(output1.detach().cpu().numpy().flatten())

        embeddings_array = np.array(embeddings_list)

        cluster_results = []
        for m in range(1, 100):
            EPS = float(m)
            # DBSCAN
            clustering = DBSCAN(eps=EPS, min_samples=2)
            SC_result = clustering.fit_predict(embeddings_array)
            print("Clustering Results ", SC_result)
            cluster = SC_result
            Labels = list(set(SC_result))
            Labels_len = len(list(set(SC_result)))
            noise_len = Counter(SC_result)[-1]
            cluster_results.append((EPS, Labels, Labels_len, noise_len))
            print("eps: ", EPS, "Labels: ", Labels, "Labels_len: ", Labels_len, "SC_result_len: ",
                  len(SC_result), "noise_len: ", noise_len)
            if save_inter:
                clustered_path = os.sep.join([os.path.normpath(inter_dest_dir), f"{EPS}eps"])
                if not os.path.exists(clustered_path):
                    os.mkdir(clustered_path)
                try:
                    Labels_copy = Labels.copy()
                    Labels_copy.remove(-1)
                    Labels_drop_len = len(Labels_copy)
                except:
                    Labels_drop_len = Labels_len
                for q in range(Labels_drop_len):
                    subclass_path = os.path.sep.join(
                        [os.path.normpath(clustered_path), f'{images_groups[i]}_{q}'])
                    if not os.path.exists(subclass_path):
                        os.mkdir(subclass_path)
                noise_count = 1
                for q in range(len(SC_result)):
                    image, label, original_path = DBSCAN_dataset[q]
                    path = original_path.split('\\')[-1]
                    if SC_result[q] != -1:
                        target_path = os.path.sep.join([os.path.normpath(clustered_path),
                                                        f'{images_groups[i]}_{Labels.index(SC_result[q])}',
                                                        path])
                        shutil.copyfile(original_path, target_path)
                    else:
                        print("noise count: ", noise_count)
                        subclass_path = os.path.sep.join([os.path.normpath(clustered_path),
                                                          f'{images_groups[i]}_{Labels_drop_len - 1 + noise_count}_noise'])
                        if not os.path.exists(subclass_path):
                            os.mkdir(subclass_path)
                        target_path = os.path.sep.join([os.path.normpath(clustered_path),
                                                        f'{images_groups[i]}_{Labels_drop_len - 1 + noise_count}_noise',
                                                        path])
                        shutil.copyfile(original_path, target_path)
                        noise_count += 1

                # sort cluster and keep track of label
                sorted_cluster_labels = [i[0] for i in
                                         sorted(enumerate(cluster), key=lambda x: x[1], reverse=True)]

        max_index = [i[2] - (i[3] / 3) for i in cluster_results].index(
            max([i[2] - (i[3] / 3) for i in cluster_results]))
        if desig_eps:
            EPS = desig_eps
        else:
            EPS = cluster_results[max_index][0]
        clustered_path = os.sep.join([os.path.normpath(best_dest_dir), f"final_clustered_dataset"])
        best_cluster_path = clustered_path
        if not os.path.exists(clustered_path):
            os.mkdir(clustered_path)
        # DBSCAN
        clustering = DBSCAN(eps=EPS, min_samples=1)
        SC_result = clustering.fit_predict(embeddings_array)
        #=====================================================
        # Calculate cluster centroids as prototypes
        unique_clusters = np.unique(SC_result)
        cluster_prototypes = []
        for cluster_id in unique_clusters:
            if cluster_id == -1:
                # Handle noise points (not assigned to any cluster) as needed
                continue

            cluster_points = embeddings_array[SC_result == cluster_id]
            cluster_centroid = np.mean(cluster_points, axis=0)
            cluster_prototypes.append(cluster_centroid)

        prototypes[images_groups[i]] = cluster_prototypes
        print("PROTOTYPES　EMBEDDING:", prototypes)
        # =====================================================

        cluster = SC_result
        Labels = list(set(SC_result))
        Labels_len = len(list(set(SC_result)))
        try:
            Labels_copy = Labels.copy()
            Labels_copy.remove(-1)
            Labels_drop_len = len(Labels_copy)
        except:
            Labels_drop_len = Labels_len
        for j in range(Labels_drop_len):
            subclass_path = os.path.sep.join(
                [os.path.normpath(clustered_path), f'{images_groups[i]}_{j}_{EPS}eps'])
            if not os.path.exists(subclass_path):
                os.mkdir(subclass_path)
        noise_count = 1
        for j in range(len(SC_result)):
            image, label, original_path = DBSCAN_dataset[j]
            path = original_path.split('\\')[-1]
            if SC_result[j] != -1:
                target_path = os.path.sep.join([os.path.normpath(clustered_path),
                                                f'{images_groups[i]}_{Labels.index(SC_result[j])}_{EPS}eps',
                                                path])
                shutil.copyfile(original_path, target_path)
            else:
                print("noise count: ", noise_count)
                subclass_path = os.path.sep.join([os.path.normpath(clustered_path),
                                                  f'{images_groups[i]}_{Labels_drop_len - 1 + noise_count}_{EPS}eps_noise'])
                if not os.path.exists(subclass_path):
                    os.mkdir(subclass_path)
                target_path = os.path.sep.join([os.path.normpath(clustered_path),
                                                f'{images_groups[i]}_{Labels_drop_len - 1 + noise_count}_{EPS}eps_noise',
                                                path])
                shutil.copyfile(original_path, target_path)
                noise_count += 1

        # sort cluster and keep track of label
        sorted_cluster_labels = [i[0] for i in sorted(enumerate(cluster), key=lambda x: x[1], reverse=True)]

    return best_cluster_path

def DBSCAN_clustering_with_imgh_new(dataset_path, transformation, inter_dest_dir, best_dest_dir, weight_avg, weight_color, save_inter=True,
                      desig_eps=None):
    transform_pil = T.ToPILImage()
    images_groups = os.listdir(dataset_path)
    prototypes = {class_idx: [] for class_idx in images_groups}
    for i in range(len(images_groups)):
        images_group_path = os.path.sep.join([os.path.normpath(dataset_path), images_groups[i]])
        folder_dataset = datasets.ImageFolder(root=images_group_path)
        DBSCAN_dataset = DBSCANDataset(imageFolderDataset=folder_dataset,
                                       transform=transformation)
        DBSCAN_dataloader = DataLoader(DBSCAN_dataset, batch_size=1, shuffle=False)
        images_group_len = len(DBSCAN_dataset)
        print(images_group_len)
        embeddings_list = []
        distance_matrix_avg = np.zeros((images_group_len,images_group_len), dtype=int)
        distance_matrix_col = np.zeros((images_group_len,images_group_len), dtype=int)
        for j, (img_0, label_0, path_0) in enumerate(DBSCAN_dataloader, 0):
            path = path_0[0].split('\\')[-1]
            hash_avg1, hash_col1 = get_hashes(transform_pil(img_0[0]))
            for k, (img_1, label_1, path_1) in enumerate(DBSCAN_dataloader, 0):
                path = path_1[0].split('\\')[-1]
                hash_avg2, hash_col2 = get_hashes(transform_pil(img_1[0]))
                dff1 = abs(hash_avg1-hash_avg2)
                dff2 = abs(hash_col1-hash_col2)
                distance_matrix_avg[j][k] = dff1
                distance_matrix_col[j][k] = dff2

        embeddings_list = (weight_avg * distance_matrix_avg) + (weight_color * distance_matrix_col)

        embeddings_array = np.array(embeddings_list)

        cluster_results = []
        for m in range(1, 100):
            EPS = float(m)
            # DBSCAN
            clustering = DBSCAN(eps=EPS, min_samples=2)
            SC_result = clustering.fit_predict(embeddings_array)
            print("Clustering Results ", SC_result)
            cluster = SC_result
            Labels = list(set(SC_result))
            Labels_len = len(list(set(SC_result)))
            noise_len = Counter(SC_result)[-1]
            cluster_results.append((EPS, Labels, Labels_len, noise_len))
            print("eps: ", EPS, "Labels: ", Labels, "Labels_len: ", Labels_len, "SC_result_len: ",
                  len(SC_result), "noise_len: ", noise_len)
            if save_inter:
                clustered_path = os.sep.join([os.path.normpath(inter_dest_dir), f"{EPS}eps"])
                if not os.path.exists(clustered_path):
                    os.mkdir(clustered_path)
                try:
                    Labels_copy = Labels.copy()
                    Labels_copy.remove(-1)
                    Labels_drop_len = len(Labels_copy)
                except:
                    Labels_drop_len = Labels_len
                for q in range(Labels_drop_len):
                    subclass_path = os.path.sep.join(
                        [os.path.normpath(clustered_path), f'{images_groups[i]}_{q}'])
                    if not os.path.exists(subclass_path):
                        os.mkdir(subclass_path)
                noise_count = 1
                for q in range(len(SC_result)):
                    image, label, original_path = DBSCAN_dataset[q]
                    path = original_path.split('\\')[-1]
                    if SC_result[q] != -1:
                        target_path = os.path.sep.join([os.path.normpath(clustered_path),
                                                        f'{images_groups[i]}_{Labels.index(SC_result[q])}',
                                                        path])
                        shutil.copyfile(original_path, target_path)
                    else:
                        print("noise count: ", noise_count)
                        subclass_path = os.path.sep.join([os.path.normpath(clustered_path),
                                                          f'{images_groups[i]}_{Labels_drop_len - 1 + noise_count}_noise'])
                        if not os.path.exists(subclass_path):
                            os.mkdir(subclass_path)
                        target_path = os.path.sep.join([os.path.normpath(clustered_path),
                                                        f'{images_groups[i]}_{Labels_drop_len - 1 + noise_count}_noise',
                                                        path])
                        shutil.copyfile(original_path, target_path)
                        noise_count += 1

                # sort cluster and keep track of label
                sorted_cluster_labels = [i[0] for i in
                                         sorted(enumerate(cluster), key=lambda x: x[1], reverse=True)]

        max_index = [i[2] - (i[3] / 3) for i in cluster_results].index(
            max([i[2] - (i[3] / 3) for i in cluster_results]))
        if desig_eps:
            EPS = desig_eps
        else:
            EPS = cluster_results[max_index][0]
        clustered_path = os.sep.join([os.path.normpath(best_dest_dir), f"final_clustered_dataset"])
        best_cluster_path = clustered_path
        if not os.path.exists(clustered_path):
            os.mkdir(clustered_path)
        # DBSCAN
        clustering = DBSCAN(eps=EPS, min_samples=1)
        SC_result = clustering.fit_predict(embeddings_array)
        #=====================================================
        # Calculate cluster centroids as prototypes
        unique_clusters = np.unique(SC_result)
        cluster_prototypes = []
        for cluster_id in unique_clusters:
            if cluster_id == -1:
                # Handle noise points (not assigned to any cluster) as needed
                continue

            cluster_points = embeddings_array[SC_result == cluster_id]
            cluster_centroid = np.mean(cluster_points, axis=0)
            cluster_prototypes.append(cluster_centroid)

        prototypes[images_groups[i]] = cluster_prototypes
        print("PROTOTYPES EMBEDDING:", prototypes)
        # =====================================================

        cluster = SC_result
        Labels = list(set(SC_result))
        Labels_len = len(list(set(SC_result)))
        try:
            Labels_copy = Labels.copy()
            Labels_copy.remove(-1)
            Labels_drop_len = len(Labels_copy)
        except:
            Labels_drop_len = Labels_len
        for j in range(Labels_drop_len):
            subclass_path = os.path.sep.join(
                [os.path.normpath(clustered_path), f'{images_groups[i]}_{j}_{EPS}eps'])
            if not os.path.exists(subclass_path):
                os.mkdir(subclass_path)
        noise_count = 1
        for j in range(len(SC_result)):
            image, label, original_path = DBSCAN_dataset[j]
            path = original_path.split('\\')[-1]
            if SC_result[j] != -1:
                target_path = os.path.sep.join([os.path.normpath(clustered_path),
                                                f'{images_groups[i]}_{Labels.index(SC_result[j])}_{EPS}eps',
                                                path])
                shutil.copyfile(original_path, target_path)
            else:
                print("noise count: ", noise_count)
                subclass_path = os.path.sep.join([os.path.normpath(clustered_path),
                                                  f'{images_groups[i]}_{Labels_drop_len - 1 + noise_count}_{EPS}eps_noise'])
                if not os.path.exists(subclass_path):
                    os.mkdir(subclass_path)
                target_path = os.path.sep.join([os.path.normpath(clustered_path),
                                                f'{images_groups[i]}_{Labels_drop_len - 1 + noise_count}_{EPS}eps_noise',
                                                path])
                shutil.copyfile(original_path, target_path)
                noise_count += 1

        # sort cluster and keep track of label
        sorted_cluster_labels = [i[0] for i in sorted(enumerate(cluster), key=lambda x: x[1], reverse=True)]

    return best_cluster_path

def imgh_hi_clustering_family(fam_path, dest, weight_avg, weight_color, threshold_cluster):
    
    images = []
    img_paths = []
    fam_names = os.listdir(fam_path)
    fam_paths = [os.sep.join([os.path.normpath(fam_path),i]) for i in fam_names]
    for fam_root in fam_paths:
        for img in os.listdir(fam_root):
            image_path = os.path.join(fam_root, img)
            image = Image.open(image_path)
            images.append(image)
            img_paths.append(image_path)

    distance_matrix_avg = np.zeros((len(images),len(images)), dtype=int)
    distance_matrix_col = np.zeros((len(images),len(images)), dtype=int)
    
    for img1 in range(0,len(images)):
        hash_avg1, hash_col1 = get_hashes(images[img1])
        for img2 in range(0,len(images)):
            hash_avg2, hash_col2 = get_hashes(images[img2])
            dff1 = abs(hash_avg1-hash_avg2)
            dff2 = abs(hash_col1-hash_col2)
            distance_matrix_avg[img1][img2] = dff1
            distance_matrix_col[img1][img2] = dff2

    combined_distance_matrix = (weight_avg * distance_matrix_avg) + (weight_color * distance_matrix_col)

    if len(images) <= 1:
        sub_families = {}
        sub_families[1] = []
        sub_families[1].append(img_paths[0])
    else:
        
        # Perform hierarchical clustering
        Z = linkage(combined_distance_matrix, method='complete', metric='euclidean')  # Complete linkage for agglomerative clustering
        
        # Determine clusters based on a threshold distance or number of clusters
        clusters = fcluster(Z, threshold_cluster, criterion='distance')
        sub_families = {}
        for img_path, cluster in zip(img_paths, clusters):
            if cluster not in sub_families:
                sub_families[cluster] = []
            sub_families[cluster].append(img_path)
    
        
    for cluster, images in sub_families.items():
        fam_name = os.path.basename(fam_path)
        sub_family_folder = dest + '\\' + fam_name + str(cluster)
        if not os.path.isdir(sub_family_folder):
            os.makedirs(sub_family_folder)
            
        for image_path in images:
            image_filename = os.path.basename(image_path)
            new_image_path = os.path.join(sub_family_folder, image_filename)
            # Check whether it is called from Testing data.
            if fam_path == dest: # If yes (fam_path == dest), move the file
                shutil.move(image_path, new_image_path)
            else: # Otherwise, copy the file instead (reserve the original images)
                shutil.copy(image_path, new_image_path)

def auto_emb_hi_clustering_family(fam_path, dest, threshold_cluster, transformation, net=None):
    images = []
    img_paths = []

    images_group_path = os.path.normpath(fam_path)
    folder_dataset = datasets.ImageFolder(root=images_group_path)
    DBSCAN_dataset = DBSCANDataset(imageFolderDataset=folder_dataset,
                                    transform=transformation)
    DBSCAN_dataloader = DataLoader(DBSCAN_dataset, batch_size=1, shuffle=False)
    images_group_len = len(DBSCAN_dataset)
    print(images_group_len)
    embeddings_list = []
    for j, (img_0, label_0, path_0) in enumerate(DBSCAN_dataloader, 0):
        path = path_0[0].split('\\')[-1]
        images.append(img_0)
        img_paths.append(path_0[0])
        for k, (img_1, label_1, path_1) in enumerate(DBSCAN_dataloader, 0):
            path = path_1[0].split('\\')[-1]
            output1, output2 = torch.squeeze(net.encoder(img_0.cuda())), torch.squeeze(net.encoder(img_1.cuda()))
        embeddings_list.append(output1.detach().cpu().numpy().flatten())
    
    combined_distance_matrix = np.array(embeddings_list)

    if len(images) <= 1:
        sub_families = {}
        sub_families[1] = []
        sub_families[1].append(img_paths[0])
    else:
        
        # Perform hierarchical clustering
        Z = linkage(combined_distance_matrix, method='complete', metric='euclidean')  # Complete linkage for agglomerative clustering
        
        # Determine clusters based on a threshold distance or number of clusters
        clusters = fcluster(Z, threshold_cluster, criterion='distance')
        sub_families = {}
        for img_path, cluster in zip(img_paths, clusters):
            if cluster not in sub_families:
                sub_families[cluster] = []
            sub_families[cluster].append(img_path)
    
        
    for cluster, images in sub_families.items():
        fam_name = os.path.basename(fam_path)
        sub_family_folder = dest + '\\' + fam_name + str(cluster)
        if not os.path.isdir(sub_family_folder):
            os.makedirs(sub_family_folder)
            
        for image_path in images:
            image_filename = os.path.basename(image_path)
            new_image_path = os.path.join(sub_family_folder, image_filename)
            # Check whether it is called from Testing data.
            if fam_path == dest: # If yes (fam_path == dest), move the file
                shutil.move(image_path, new_image_path)
            else: # Otherwise, copy the file instead (reserve the original images)
                shutil.copy(image_path, new_image_path)

def siame_emb_hi_clustering_family(fam_path, dest, threshold_cluster, transformation, net=None):
    images = []
    img_paths = []

    images_group_path = os.path.normpath(fam_path)
    folder_dataset = datasets.ImageFolder(root=images_group_path)
    DBSCAN_dataset = DBSCANDataset(imageFolderDataset=folder_dataset,
                                    transform=transformation)
    DBSCAN_dataloader = DataLoader(DBSCAN_dataset, batch_size=1, shuffle=False)
    images_group_len = len(DBSCAN_dataset)
    print(images_group_len)
    embeddings_list = []
    for j, (img_0, label_0, path_0) in enumerate(DBSCAN_dataloader, 0):
        path = path_0[0].split('\\')[-1]
        images.append(img_0)
        img_paths.append(path_0[0])
        for k, (img_1, label_1, path_1) in enumerate(DBSCAN_dataloader, 0):
            path = path_1[0].split('\\')[-1]
            output1, output2 = net(img_0.cuda(), img_1.cuda())
        embeddings_list.append(output1.detach().cpu().numpy().flatten())
    
    combined_distance_matrix = np.array(embeddings_list)

    if len(images) <= 1:
        sub_families = {}
        sub_families[1] = []
        sub_families[1].append(img_paths[0])
    else:
        
        # Perform hierarchical clustering
        Z = linkage(combined_distance_matrix, method='complete', metric='euclidean')  # Complete linkage for agglomerative clustering
        
        # Determine clusters based on a threshold distance or number of clusters
        clusters = fcluster(Z, threshold_cluster, criterion='distance')
        sub_families = {}
        for img_path, cluster in zip(img_paths, clusters):
            if cluster not in sub_families:
                sub_families[cluster] = []
            sub_families[cluster].append(img_path)
    
        
    for cluster, images in sub_families.items():
        fam_name = os.path.basename(fam_path)
        sub_family_folder = dest + '\\' + fam_name + str(cluster)
        if not os.path.isdir(sub_family_folder):
            os.makedirs(sub_family_folder)
            
        for image_path in images:
            image_filename = os.path.basename(image_path)
            new_image_path = os.path.join(sub_family_folder, image_filename)
            # Check whether it is called from Testing data.
            if fam_path == dest: # If yes (fam_path == dest), move the file
                shutil.move(image_path, new_image_path)
            else: # Otherwise, copy the file instead (reserve the original images)
                shutil.copy(image_path, new_image_path)

def HI_clustering_with_imgh_new(dataset_path, dest_dir, weight_avg, weight_color, threshold_cluster):
    # Remove subdirectory and its contents in the train_dest path
    if os.path.isdir(dest_dir):
        shutil.rmtree(dest_dir)

    # Prceed all families (folders)
    print('[Info] Families Clustering...')
    for fam in os.listdir(dataset_path):
        fam_path = os.path.join(dataset_path,fam)
        imgh_hi_clustering_family(fam_path, dest_dir, weight_avg, weight_color, threshold_cluster)
    
    return dest_dir

   

def HI_clustering_with_auto_emb_new(dataset_path, dest_dir, weight_avg, weight_color, threshold_combine, threshold_cluster, transformation, net=None):
    # Remove subdirectory and its contents in the train_dest path
    if os.path.isdir(dest_dir):
        shutil.rmtree(dest_dir)

    # Prceed all families (folders)
    print('[Info] Families Clustering...')
    for fam in os.listdir(dataset_path):
        fam_path = os.sep.join([dataset_path,fam])
        #print('family:',fam)
        auto_emb_hi_clustering_family(fam_path, dest_dir, threshold_cluster, transformation, net=net)
        # clustering_family(fam_path,dest_dir)

    return dest_dir
    

def HI_clustering_with_siame_emb_new(dataset_path, dest_dir, weight_avg, weight_color, threshold_combine, threshold_cluster, transformation, net=None):
    # Remove subdirectory and its contents in the train_dest path
    if os.path.isdir(dest_dir):
        shutil.rmtree(dest_dir)

    # Prceed all families (folders)
    print('[Info] Families Clustering...')
    for fam in os.listdir(dataset_path):
        fam_path = os.path.join(dataset_path,fam)
        #print('family:',fam)
        siame_emb_hi_clustering_family(fam_path, dest_dir, threshold_cluster, transformation, net=net)
        # auto_emb_hi_clustering_family(fam_path, dest_dir, threshold_cluster, transformation, net=net)
        # clustering_family(fam_path,dest_dir)
    
    return dest_dir

    
