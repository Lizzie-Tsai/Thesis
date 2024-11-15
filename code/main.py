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

from main_loop import complete_training_loop_1, complete_training_loop_2, complete_training_loop_3, complete_training_loop_4, complete_training_loop_5, complete_training_loop_6
from main_loop import complete_training_loop_Pr2, complete_training_loop_Pr3, complete_training_loop_Pr4

CLUSTERTARGET = "C:/Users/Lizzie0930/Desktop/git/multichannel_123grams/train"
COPYPATH = "C:/Users/Lizzie0930/Desktop/git/multichannel_123grams/train_copy"
# prepare_clustering_dataset(CLUSTERTARGET, COPYPATH)

# Pr4
# Resize the images and transform to tensors
transformation = transforms.Compose([transforms.Resize((100, 100)),
                                     transforms.ToTensor()
                                   ])
TRAINING_DATA_BASEPATH = "C:/Users/Lizzie0930/Desktop/final_folder/data/training"
TESTING_DATA_BASEPATH = "C:/Users/Lizzie0930/Desktop/final_folder/data/testing/9_prepare_data_for_method_1_split3_5shots_5_10_35cases"
RESULT_BASEPATH = "C:/Users/Lizzie0930/Desktop/final_folder/final_result/Results_Auto_Pr4"
# os.sep.join([os.path.normpath(RESULT_BASEPATH), '1_autoencoder_model', 'autoencoder_model.pth'])
complete_training_loop_Pr4(autoencoder_training_data_path=os.sep.join([os.path.normpath(TRAINING_DATA_BASEPATH), 'split3_train_regrouped_for_auto']),#
                        autoencoder_save_model_path=os.sep.join([os.path.normpath(RESULT_BASEPATH), '1_autoencoder_model', 'autoencoder_model.pth']),
                        autoencoder_save_img_base_path=os.sep.join([os.path.normpath(RESULT_BASEPATH), '2_autoencoder_img']),
                        training_data_cluster_target=os.sep.join([os.path.normpath(TRAINING_DATA_BASEPATH), 'split3_train_regrouped_copy']),#
                        training_clustered_inter_dir=os.sep.join([os.path.normpath(RESULT_BASEPATH), '3_train_clustered_inter_dest']),
                        training_clustered_best_dir=os.sep.join([os.path.normpath(RESULT_BASEPATH), '4_train_clustered_best_dest']),
                        train_prototype_dir=os.sep.join([os.path.normpath(RESULT_BASEPATH), '5_train_prototype']),
                        proto_clustered_best_dest_dir=os.sep.join([os.path.normpath(RESULT_BASEPATH), '14_train_proto_clustered_best_dest']),
                        proto_clustered_inter_dest_dir=os.sep.join([os.path.normpath(RESULT_BASEPATH), '13_train_proto_clustered_inter_dest']),
                        regrouped_target_dir=os.sep.join([os.path.normpath(RESULT_BASEPATH), '6_regrouped_train']),
                        test_cluster_target_dir=os.sep.join([os.path.normpath(TESTING_DATA_BASEPATH), 'known_with_case']),#
                        testing_clustered_inter_dir=os.sep.join([os.path.normpath(RESULT_BASEPATH), '7_test_clustered_inter_dest']),
                        testing_clustered_best_dir=os.sep.join([os.path.normpath(RESULT_BASEPATH), '8_test_clustered_best_dest']),
                        save_inter_si_model_dir=os.sep.join([os.path.normpath(RESULT_BASEPATH), '9_si_inter_model']),
                        save_best_si_model_dir=os.sep.join([os.path.normpath(RESULT_BASEPATH), '10_si_best_model']),
                        si_epoch_num=150, way_list=[5,6,7,8,9,10], net_name="SiameseNetwork", net_type="vgg",
                        case_len=35, shots=5, model_num=2, auto_epoch_num=150,
                        vali_data_dir= os.sep.join([os.path.normpath(TESTING_DATA_BASEPATH), 'unknown_with_case']),#
                        csv_save_dir=os.sep.join([os.path.normpath(RESULT_BASEPATH), '11_validation_results_in_train_csv']),
                        test_prototype_dest_dir=os.sep.join([os.path.normpath(RESULT_BASEPATH), '12_test_prototype_dest']),
                        result_csv_dir=os.sep.join([os.path.normpath(RESULT_BASEPATH), '15_test_results_after_train_csv']),
                        transformation = transformation)