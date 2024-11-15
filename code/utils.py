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

from dataset import AutoEncoderDataset, SiameseNetworkDataset_1, PrototypeDataset
from networks import Autoencoder_Conv, SiameseNetwork, SiameseNetwork_VGG16_Based_v2, SiameseNetwork_resnet18_Based, SiameseNetwork_only_autoencoder, SiameseNetwork_autoencoder_Based

# Define the Contrastive Loss Function
class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
      # Calculate the euclidian distance and calculate the contrastive loss
      euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)

      loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                    (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))


      return loss_contrastive
    
def AutoEncoder_Training_Loop(epoch_num, data_path, save_model_path, save_img_base_path, transformation):
    # Resize the images and transform to tensors
    transformation = transformation
    # Locate the dataset and load it into the AutoEncoderDataset
    folder_dataset = datasets.ImageFolder(root=data_path)
    autoencoder_dataset = AutoEncoderDataset(imageFolderDataset=folder_dataset,
                                                   transform=transformation)
    autoencoder_dataloader = DataLoader(autoencoder_dataset, batch_size=64, shuffle=False)

    model = Autoencoder_Conv()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=1e-3,
                                 weight_decay=1e-5)
    # Point to training loop video
    num_epochs = 1
    outputs = []
    for epoch in range(epoch_num):
        for (img, _, _) in autoencoder_dataloader:
            # img = img.reshape(-1, 28*28) # -> use for Autoencoder_Linear
            # print("img: ", img[0].shape)
            recon = model(img)
            # print("recon: ", recon[0].shape)
            loss = criterion(recon, img)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch:{epoch + 1}, Loss:{loss.item():.4f}')
        outputs.append((epoch, img, recon))
    # save the model
    torch.save(model.state_dict(), save_model_path)

    for k in range(0, epoch_num, 5):
        plt.figure(figsize=(9, 2))
        # plt.gray()
        imgs = outputs[k][1].detach().numpy()
        recon = outputs[k][2].detach().numpy()
        for i, item in enumerate(imgs):
            # print("item.shape: ", item.shape)
            if i >= 9: break
            plt.subplot(2, 9, i + 1)
            plt.imshow(np.transpose(item, (1, 2, 0)))

        for i, item in enumerate(recon):
            # print("item.shape: ", item.shape)
            if i >= 9: break
            plt.subplot(2, 9, 9 + i + 1)  # row_length + i + 1
            plt.imshow(np.transpose(item, (1, 2, 0)))

        save_img_path = os.sep.join([os.path.normpath(save_img_base_path), f"Epoch_{k}.png"])

        plt.savefig(save_img_path)

def generate_training_prototype(target_dataset_path, dest_dataset_path):
    '''
    calculate average image from each subclass of training dataset to generate
    testing prototype.
    '''
    # FOLDERPOSTFIX = "_3_round"
    TARGETFOLDER = target_dataset_path
    DESTFOLDER = dest_dataset_path
    if not os.path.exists(DESTFOLDER):
        os.mkdir(DESTFOLDER)
    class_folder_name = os.listdir(TARGETFOLDER)
    print(class_folder_name)
    for i in range(len(class_folder_name)):
        # Folder containing the images
        image_folder = TARGETFOLDER + f'/{class_folder_name[i]}'

        # Get a list of image filenames in the folder
        image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg') or f.endswith('.png')]

        # Initialize variables to hold sum of pixel values
        sum_image = None

        # Loop through each image
        for image_file in image_files:
            image_path = os.path.join(image_folder, image_file)
            img = cv2.imread(image_path)

            if sum_image is None:
                sum_image = np.zeros_like(img, dtype=np.float64)

            # Accumulate pixel values
            sum_image += img.astype(np.float64)

        # Calculate the average pixel values
        average_image = (sum_image / len(image_files)).astype(np.uint8)

        class_path = DESTFOLDER + f'/classes/class'
        if not os.path.exists(class_path):
            os.makedirs(class_path)
        # Save the average image
        cv2.imwrite(DESTFOLDER + f'/classes/class/{class_folder_name[i]}_average_image.png', average_image)

def re_group_train_data(source_dir, info_dir, target_dir):
    target_dir = target_dir
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
    class_list = os.listdir(source_dir)
    print("class_list: ", class_list)
    group_list = os.listdir(info_dir)
    print("group_list: ", group_list)
    for i in range(len(group_list)):
        group_list_path = os.sep.join([os.path.normpath(info_dir), group_list[i]])
        class_img_in_group = os.listdir(group_list_path)
        classes_in_group = ["_".join(i.split("_")[0:-2]) for i in class_img_in_group]
        new_class_name = "-".join(classes_in_group)
        if len(new_class_name) > 100:
            new_class_name = new_class_name[:100]
        print("new_class_name: ", new_class_name)
        new_class_path = os.sep.join([os.path.normpath(target_dir), new_class_name])
        if int(group_list[i].split("_")[1]) != len(group_list)-1:
            if not os.path.exists(new_class_path):
                os.mkdir(new_class_path)
            for j in range(len(classes_in_group)):
                source_class_dir = os.sep.join([os.path.normpath(source_dir), classes_in_group[j]])
                source_class_img = os.listdir(source_class_dir)
                source_class_img_paths = [os.sep.join([os.path.normpath(source_class_dir), i]) for i in source_class_img]
                target_class_img_paths = [os.sep.join([os.path.normpath(new_class_path), i]) for i in source_class_img]
                for k in range(len(source_class_img_paths)):
                    shutil.copyfile(source_class_img_paths[k], target_class_img_paths[k])
        else:
            for j in range(len(classes_in_group)):
                source_class_dir = os.sep.join([os.path.normpath(source_dir), classes_in_group[j]])
                source_class_img = os.listdir(source_class_dir)
                source_class_img_paths = [os.sep.join([os.path.normpath(source_class_dir), i]) for i in source_class_img]
                target_class_path = os.sep.join([os.path.normpath(target_dir), classes_in_group[j]])
                if not os.path.exists(target_class_path):
                    os.mkdir(target_class_path)
                target_class_img_paths = [os.sep.join([os.path.normpath(target_class_path), i]) for i in source_class_img]
                for k in range(len(source_class_img_paths)):
                    shutil.copyfile(source_class_img_paths[k], target_class_img_paths[k])

def re_group_train_data_for_Hi(source_dir, info_dir, target_dir):
    target_dir = target_dir
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
    class_list = os.listdir(source_dir)
    print("class_list: ", class_list)
    group_list = os.listdir(info_dir)
    print("group_list: ", group_list)
    for i in range(len(group_list)):
        group_list_path = os.sep.join([os.path.normpath(info_dir), group_list[i]])
        class_img_in_group = os.listdir(group_list_path)
        classes_in_group = ["_".join(i.split("_")[0:-2]) for i in class_img_in_group]
        new_class_name = "-".join(classes_in_group)
        if len(new_class_name) > 100:
            new_class_name = new_class_name[:100]
        print("new_class_name: ", new_class_name)
        new_class_path = os.sep.join([os.path.normpath(target_dir), new_class_name])
        if int(group_list[i][7:]) != len(group_list)-1:
            if not os.path.exists(new_class_path):
                os.mkdir(new_class_path)
            for j in range(len(classes_in_group)):
                source_class_dir = os.sep.join([os.path.normpath(source_dir), classes_in_group[j]])
                source_class_img = os.listdir(source_class_dir)
                source_class_img_paths = [os.sep.join([os.path.normpath(source_class_dir), i]) for i in source_class_img]
                target_class_img_paths = [os.sep.join([os.path.normpath(new_class_path), i]) for i in source_class_img]
                for k in range(len(source_class_img_paths)):
                    shutil.copyfile(source_class_img_paths[k], target_class_img_paths[k])
        else:
            for j in range(len(classes_in_group)):
                source_class_dir = os.sep.join([os.path.normpath(source_dir), classes_in_group[j]])
                source_class_img = os.listdir(source_class_dir)
                source_class_img_paths = [os.sep.join([os.path.normpath(source_class_dir), i]) for i in source_class_img]
                target_class_path = os.sep.join([os.path.normpath(target_dir), classes_in_group[j]])
                if not os.path.exists(target_class_path):
                    os.mkdir(target_class_path)
                target_class_img_paths = [os.sep.join([os.path.normpath(target_class_path), i]) for i in source_class_img]
                for k in range(len(source_class_img_paths)):
                    shutil.copyfile(source_class_img_paths[k], target_class_img_paths[k])

def generate_testing_prototype_new(target_dataset_path, dest_dataset_path):
    '''
    calculate average image from each subclass of testing and validation dataset
    to generate testing and validation prototype, only {shots} prototypes can be
    obtained.
    '''
    TARGETFOLDER = target_dataset_path
    DESTFOLDER = dest_dataset_path
    if not os.path.exists(DESTFOLDER):
        os.mkdir(DESTFOLDER)
    class_folder_name = os.listdir(TARGETFOLDER)
    print(class_folder_name)
    for i in range(len(class_folder_name)):
        class_name = class_folder_name[i].split("_")[0]
        # Folder containing the images
        image_folder = TARGETFOLDER + f'/{class_folder_name[i]}'

        # Get a list of image filenames in the folder
        image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg') or f.endswith('.png')]

        # Initialize variables to hold sum of pixel values
        sum_image = None

        # Loop through each image
        for image_file in image_files:
            image_path = os.path.join(image_folder, image_file)
            img = cv2.imread(image_path)

            if sum_image is None:
                sum_image = np.zeros_like(img, dtype=np.float64)

            # Accumulate pixel values
            sum_image += img.astype(np.float64)

        # Calculate the average pixel values
        average_image = (sum_image / len(image_files)).astype(np.uint8)

        class_path = DESTFOLDER + f'/{class_folder_name[i]}'
        if not os.path.exists(class_path):
            os.mkdir(class_path)
        # Save the average image
        cv2.imwrite(DESTFOLDER + f'/{class_folder_name[i]}/{class_folder_name[i]}_average_image.png', average_image)
        save_img_path = DESTFOLDER + f'/{class_folder_name[i]}/{class_folder_name[i]}_average_image.png'

def Training_Loop(model_num, net_name, dataset, save_best_model_dir, save_inter_model_dir, epoch_num, way_list,
                  vali_data_dir, proto_data_dir, csv_save_dir, case_len, net_type):
    # Load the training dataset
    train_dataloader = DataLoader(dataset,
                            shuffle=True,
                            batch_size=64)
    if net_name == "SiameseNetwork": # 100*100?
        net = SiameseNetwork().cuda()
    elif net_name == "SiameseNetwork_VGG16_Based_v2": # 64*64
        net = SiameseNetwork_VGG16_Based_v2().cuda()
    elif net_name == "SiameseNetwork_resnet18_Based": # 100*100
        net = SiameseNetwork_resnet18_Based().cuda()
    elif net_name == "SiameseNetwork_autoencoder_Based": # 64*64
        net = SiameseNetwork_autoencoder_Based().cuda()

    criterion = ContrastiveLoss()
    optimizer = optim.Adam(net.parameters(), lr = 0.0005 )

    counter = []
    loss_history = []
    iteration_number= 0
    batch = 0
    # Iterate through the epochs
    result_df_list = []
    avg_df_list = []
    for epoch in range(epoch_num):

        # Iterate over batches
        for i, (img0, img1, label, label_1, label_2, _, _) in enumerate(train_dataloader, 0):
            batch = i

            # Send the images and labels to CUDA
            img0, img1, label = img0.cuda(), img1.cuda(), label.cuda()

            # Zero the gradients
            optimizer.zero_grad()

            # Pass in the two images into the network and obtain two outputs
            output1, output2 = net(img0, img1)

            # Pass the outputs of the networks and label into the loss function
            loss_contrastive = criterion(output1, output2, label)

            # Calculate the backpropagation
            loss_contrastive.backward()

            # Optimize
            optimizer.step()

            # Every 10 batches print out the loss
            if i % 10 == 0 :
                print(f"Epoch number {epoch}\n Current loss {loss_contrastive.item()}\n")
                iteration_number += 10
                counter.append(iteration_number)
                loss_history.append(loss_contrastive.item())

        if epoch % 20 == 0 and epoch/20 >= 1:
            model_postfix = f"_{epoch}ep_{batch}batch"
            save_inter_model_path = os.sep.join([os.path.normpath(save_inter_model_dir),f"model{model_postfix}.pth"])
            torch.save(net.state_dict(), save_inter_model_path)
            result_df, avg_df = group_test(model_num=model_num, way_list=way_list, case_len=case_len, epoch=epoch,
                                           model_path=save_inter_model_path, batch_num=batch,
                                           net_name=net_name, net_type=net_type,
                                           test_data_dir=vali_data_dir,
                                           test_data_postfix="",
                                           proto_data_dir=proto_data_dir,
                                           proto_data_postfix="")
            result_df_list.append(result_df)
            avg_df_list.append(avg_df)
            
    result_final_df = pd.concat(result_df_list, 1)
    result_final_df_save_path = os.sep.join([os.path.normpath(csv_save_dir), "final_result.csv"])
    result_final_df.to_csv(result_final_df_save_path)
    avg_final_df = pd.concat(avg_df_list, 1)
    avg_final_df_save_path = os.sep.join([os.path.normpath(csv_save_dir), "final_avg.csv"])
    avg_final_df.to_csv(avg_final_df_save_path)
    best_epoch_list = []
    best_batch_list = []
    target_path_list = []
    for i in range(len(way_list)):
        col_list = [j for j in range(len(avg_final_df.columns)) if j%len(way_list)==i]
        avg_final_df_way = avg_final_df.iloc[:, col_list]
        best_result = avg_final_df_way.idxmax(axis=1)
        print("best_result: ", best_result[0])
        print(str(best_result[0]))
        best_info = re.match(r"(\d+)EPOCH(\d+)BATCH-(\d+)WAYS", str(best_result[0]))
        best_epoch = best_info.group(1)
        best_batch = best_info.group(2)
        model_file_name = f"model_{best_epoch}ep_{best_batch}batch.pth"
        source_path = os.sep.join([os.path.normpath(save_inter_model_dir), model_file_name])
        target_model_file_name = f"{way_list[i]}ways_best_model_{best_epoch}ep_{best_batch}batch.pth"
        if i == 0:
            target_path = os.sep.join([os.path.normpath(save_best_model_dir), target_model_file_name])
            shutil.copyfile(source_path, target_path)
        best_epoch_list.append(best_epoch)
        best_batch_list.append(best_batch)
        target_path_list.append(target_path)
    return best_epoch_list, best_batch_list, target_path_list

def Training_Loop_2(model_num, net_name, dataset, save_best_model_dir, save_inter_model_dir, epoch_num, way_list,
                  vali_data_dir, proto_data_dir, csv_save_dir, case_len, net_type):
    # Load the training dataset
    train_dataloader = DataLoader(dataset,
                            shuffle=True,
                            batch_size=64)
    if net_name == "SiameseNetwork": # 100*100?
        net = SiameseNetwork().cuda()
    elif net_name == "SiameseNetwork_VGG16_Based_v2": # 64*64
        net = SiameseNetwork_VGG16_Based_v2().cuda()
    elif net_name == "SiameseNetwork_resnet18_Based": # 100*100
        net = SiameseNetwork_resnet18_Based().cuda()
    elif net_name == "SiameseNetwork_autoencoder_Based": # 64*64
        net = SiameseNetwork_autoencoder_Based().cuda()

    criterion = ContrastiveLoss()
    optimizer = optim.Adam(net.parameters(), lr = 0.0005 )

    counter = []
    loss_history = []
    iteration_number= 0
    batch = 0
    # Iterate through the epochs
    result_df_list = []
    avg_df_list = []
    for epoch in range(epoch_num):

        # Iterate over batches
        for i, (img0, img1, label, label_1, label_2, _, _) in enumerate(train_dataloader, 0):
            batch = i

            # Send the images and labels to CUDA
            img0, img1, label = img0.cuda(), img1.cuda(), label.cuda()

            # Zero the gradients
            optimizer.zero_grad()

            # Pass in the two images into the network and obtain two outputs
            output1, output2 = net(img0, img1)

            # Pass the outputs of the networks and label into the loss function
            loss_contrastive = criterion(output1, output2, label)

            # Calculate the backpropagation
            loss_contrastive.backward()

            # Optimize
            optimizer.step()

            # Every 10 batches print out the loss
            if i % 10 == 0 :
                print(f"Epoch number {epoch}\n Current loss {loss_contrastive.item()}\n")
                iteration_number += 10
                counter.append(iteration_number)
                loss_history.append(loss_contrastive.item())

        if epoch % 20 == 0 and epoch/20 >= 1:
            model_postfix = f"_{epoch}ep_{batch}batch"
            save_inter_model_path = os.sep.join([os.path.normpath(save_inter_model_dir),f"model{model_postfix}.pth"])
            torch.save(net.state_dict(), save_inter_model_path)
            result_df, avg_df = group_test_2(model_num=model_num, way_list=way_list, case_len=case_len, epoch=epoch,
                                           model_path=save_inter_model_path, batch_num=batch,
                                           net_name=net_name, net_type=net_type,
                                           test_data_dir=vali_data_dir,
                                           test_data_postfix="",
                                           proto_data_dir=proto_data_dir,
                                           proto_data_postfix="")
            result_df_list.append(result_df)
            avg_df_list.append(avg_df)
            
    result_final_df = pd.concat(result_df_list, 1)
    result_final_df_save_path = os.sep.join([os.path.normpath(csv_save_dir), "final_result.csv"])
    result_final_df.to_csv(result_final_df_save_path)
    avg_final_df = pd.concat(avg_df_list, 1)
    avg_final_df_save_path = os.sep.join([os.path.normpath(csv_save_dir), "final_avg.csv"])
    avg_final_df.to_csv(avg_final_df_save_path)
    best_epoch_list = []
    best_batch_list = []
    target_path_list = []
    for i in range(len(way_list)):
        col_list = [j for j in range(len(avg_final_df.columns)) if j%len(way_list)==i]
        avg_final_df_way = avg_final_df.iloc[:, col_list]
        best_result = avg_final_df_way.idxmax(axis=1)
        print("best_result: ", best_result[0])
        print(str(best_result[0]))
        best_info = re.match(r"(\d+)EPOCH(\d+)BATCH-(\d+)WAYS", str(best_result[0]))
        best_epoch = best_info.group(1)
        best_batch = best_info.group(2)
        model_file_name = f"model_{best_epoch}ep_{best_batch}batch.pth"
        source_path = os.sep.join([os.path.normpath(save_inter_model_dir), model_file_name])
        target_model_file_name = f"{way_list[i]}ways_best_model_{best_epoch}ep_{best_batch}batch.pth"
        if i == 0:
            target_path = os.sep.join([os.path.normpath(save_best_model_dir), target_model_file_name])
            shutil.copyfile(source_path, target_path)
        best_epoch_list.append(best_epoch)
        best_batch_list.append(best_batch)
        target_path_list.append(target_path)
    return best_epoch_list, best_batch_list, target_path_list

def group_test(model_num=1, way_list=[5, 6], case_len=10, epoch=0, model_path="", batch_num=0,
               net_name="", net_type="",
               test_data_dir="", test_data_postfix="",
               proto_data_dir="", proto_data_postfix=""):
    WAYLIST = way_list
    CASELEN = case_len
    NET_TYPE = net_type
    result_df = pd.DataFrame()
    avg_df = pd.DataFrame()
    for j in range(len(WAYLIST)):
        WAYS = WAYLIST[j]
        cases_result = []
        for k in range(CASELEN):
            CASE = k+1
            print(f'{WAYS} WAYS CASE {CASE}')
            TESTINGDATAPATH = os.sep.join([os.path.normpath(test_data_dir),f"test_{WAYS}_{CASE}{test_data_postfix}"])
            PROTODATAPATH =  os.sep.join([os.path.normpath(proto_data_dir),f"test_{WAYS}_{CASE}{proto_data_postfix}"])
            print("TESTINGDATAPATH: ", TESTINGDATAPATH)
            print("PROTODATAPATH: ", PROTODATAPATH)
            MODELPATH = model_path
            EPOCH = epoch

            # Resize the images and transform to tensors
            transformation = transforms.Compose([transforms.Resize((64,64)),
                                                 transforms.ToTensor()
                                                ])
            transformation_1 = transforms.Compose([transforms.Resize((100,100)),
                                                 transforms.ToTensor()
                                                ])

            # Locate the test dataset and load it into the SiameseNetworkDataset
            folder_dataset_test = datasets.ImageFolder(root=TESTINGDATAPATH)
            siamese_test_dataset = SiameseNetworkDataset_1(imageFolderDataset=folder_dataset_test,
                                                    transform=transformation)


            # Locate the test dataset and load it into the SiameseNetworkDataset
            if model_num != 1:
                pr_data_path = PROTODATAPATH
                pr_folder_dataset_test = datasets.ImageFolder(root=PROTODATAPATH)
                pr_dataset = PrototypeDataset(imageFolderDataset=pr_folder_dataset_test,
                                                        transform=transformation)
            else:
                pr_data_path = ""
                pr_dataset = ""

            result_dict = Testing_Loop_0(model_num=model_num, dataset=siamese_test_dataset, net_name=net_name, model_path=MODELPATH,
                         testing_data_path=TESTINGDATAPATH, pr_dataset=pr_dataset, pr_data_path=pr_data_path)
            cases_result.append(result_dict["accuracy"])
        col_name = f'{EPOCH}EPOCH{batch_num}BATCH-{WAYS}WAYS'
        result_df[col_name] = cases_result
        avg_df[col_name] = [np.average(cases_result)]
    print(result_df)
    print(avg_df)
    return result_df, avg_df

def group_test_2(model_num=1, way_list=[5, 6], case_len=10, epoch=0, model_path="", batch_num=0,
               net_name="", net_type="",
               test_data_dir="", test_data_postfix="",
               proto_data_dir="", proto_data_postfix=""):
    WAYLIST = way_list
    CASELEN = case_len
    NET_TYPE = net_type
    result_df = pd.DataFrame()
    avg_df = pd.DataFrame()
    for j in range(len(WAYLIST)):
        WAYS = WAYLIST[j]
        cases_result = []
        for k in range(CASELEN):
            CASE = k+1
            print(f'{WAYS} WAYS CASE {CASE}')
            TESTINGDATAPATH = os.sep.join([os.path.normpath(test_data_dir),f"test_{WAYS}_{CASE}{test_data_postfix}"])
            PROTODATAPATH =  os.sep.join([os.path.normpath(proto_data_dir),f"test_{WAYS}_{CASE}{proto_data_postfix}"])
            print("TESTINGDATAPATH: ", TESTINGDATAPATH)
            print("PROTODATAPATH: ", PROTODATAPATH)
            MODELPATH = model_path
            EPOCH = epoch

            # Resize the images and transform to tensors
            transformation_1 = transforms.Compose([transforms.Resize((64,64)),
                                                 transforms.ToTensor()
                                                ])
            transformation = transforms.Compose([transforms.Resize((100,100)),
                                                 transforms.ToTensor()
                                                ])

            # Locate the test dataset and load it into the SiameseNetworkDataset
            folder_dataset_test = datasets.ImageFolder(root=TESTINGDATAPATH)
            siamese_test_dataset = SiameseNetworkDataset_1(imageFolderDataset=folder_dataset_test,
                                                    transform=transformation)


            # Locate the test dataset and load it into the SiameseNetworkDataset
            if model_num != 1:
                pr_data_path = PROTODATAPATH
                pr_folder_dataset_test = datasets.ImageFolder(root=PROTODATAPATH)
                pr_dataset = PrototypeDataset(imageFolderDataset=pr_folder_dataset_test,
                                                        transform=transformation)
            else:
                pr_data_path = ""
                pr_dataset = ""

            result_dict = Testing_Loop_1(model_num=model_num, dataset=siamese_test_dataset, net_name=net_name, model_path=MODELPATH,
                         testing_data_path=TESTINGDATAPATH, pr_dataset=pr_dataset, pr_data_path=pr_data_path)
            cases_result.append(result_dict["accuracy"])
        col_name = f'{EPOCH}EPOCH{batch_num}BATCH-{WAYS}WAYS'
        result_df[col_name] = cases_result
        avg_df[col_name] = [np.average(cases_result)]
    print(result_df)
    print(avg_df)
    return result_df, avg_df

def Testing_Loop_0(model_num, dataset, net_name, model_path, testing_data_path, pr_dataset, pr_data_path=""):
    test_dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    auto_net = SiameseNetwork_only_autoencoder().cuda()
    auto_net.eval()

    if net_name == "SiameseNetwork":
        net = SiameseNetwork().cuda()
    elif net_name == "SiameseNetwork_VGG16_Based_v2":
        net = SiameseNetwork_VGG16_Based_v2().cuda()
    elif net_name == "SiameseNetwork_resnet18_Based":
        net = SiameseNetwork_resnet18_Based().cuda()
    elif net_name == "SiameseNetwork_autoencoder_Based":
        net = SiameseNetwork_autoencoder_Based().cuda()

    net.load_state_dict(torch.load(model_path))
    net.eval()

    with torch.inference_mode():
        if model_num == 1:
            print("INTO MODEL NUM 1")
            correct_count = 0
            total = 0
            class_list = os.listdir(testing_data_path)
            print(class_list)
            for j, (x0, _, _, label1, _, L1, _) in enumerate(test_dataloader, 0):
                prob_list = []
                a_prob_list = []
                a_proto_vect = []
                
                pr_dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
                dataiter = iter(pr_dataloader)
                while True:
                    try:
                        x1, _, _, label2, _, L2, _ = next(dataiter)
                    except StopIteration:
                        print("error")
                        pr_dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
                        dataiter = iter(pr_dataloader)
                        continue
                    if L2[0].split("\\")[-2] == class_list[i]:
                        # Concatenate the two images together
                        output1, output2 = net(x0.cuda(), x1.cuda())
                        euclidean_distance = F.pairwise_distance(output1, output2)
                        prob_list.append((euclidean_distance.item(),
                                            (L1[0].split("\\")[-2].split('_')[0], L2[0].split("\\")[-2].split('_')[0])))
                        
                        a_output1, a_output2 = auto_net(x0.cuda(), x1.cuda())
                        a_euclidean_distance = F.pairwise_distance(a_output1, a_output2)
                        a_prob_list.append((a_euclidean_distance.item(),
                                            (L1[0].split("\\")[-2].split('_')[0], L2[0].split("\\")[-2].split('_')[0])))
                        a_proto_vect.append(a_output2.detach().cpu().numpy().flatten())
                        a_proto_array = np.array(a_proto_vect)
                        break

                print("=== original ===")
                print(f"model_num: {model_num} j: {j} prob_list: ", prob_list)
                score = [prob_list[i][0] for i in range(len(prob_list))]
                print(f"j: {j} score: ", score)
                min_score = min(score)
                min_score_index = score.index(min_score)
                print(f"model_num: {model_num}  j: {j} min_index: ", min_score_index)

                print("=== auto ===")
                print(f"model_num: {model_num} j: {j} a_prob_list: ", a_prob_list)
                a_score = [a_prob_list[i][0] for i in range(len(a_prob_list))]
                print(f"j: {j} score: ", a_score)
                a_min_score = min(a_score)
                a_min_score_index = a_score.index(a_min_score)
                print(f"model_num: {model_num}  j: {j} min_index: ", a_min_score_index)

                print("=== choose net ===")
                clustering = DBSCAN(eps=8, min_samples=2)
                SC_result = clustering.fit_predict(a_proto_array)
                if len(list(set(list(SC_result).remove(-1)))) > 0:
                    use_auto = True
                else:
                    use_auto = False
                

                if use_auto:
                    print("Choose Auto")
                    if a_prob_list[a_min_score_index][1][0] == a_prob_list[a_min_score_index][1][1]:
                        correct_count += 1
                    print(f"model_num: {model_num}  j: {j} accu_correct_count: ", correct_count)
                    total = j + 1
                    
                else:
                    print("Choose Original")
                    if prob_list[min_score_index][1][0] == prob_list[min_score_index][1][1]:
                        correct_count += 1
                    print(f"model_num: {model_num}  j: {j} accu_correct_count: ", correct_count)
                    total = j + 1

            print("correct_count: ", correct_count)
            print("total: ", total)
            print("accuracy: ", correct_count / total)

        else:
            correct_count = 0
            total = 0
            class_list = os.listdir(pr_data_path)
            print(class_list)
            for j, (x0, _, _, label1, _, L1, _) in enumerate(test_dataloader, 0):
                prob_list = []
                a_prob_list = []
                a_proto_vect = []
                a_proto_label = []
                a_dist = []
                c_prob_list = []
                s_prob_list = []
                for i in range(len(class_list)):
                    # print("i: ", i)
                    pr_dataloader = DataLoader(pr_dataset, batch_size=1, shuffle=True)
                    dataiter = iter(pr_dataloader)
                    while True:
                        try:
                            x1, label2, L2 = next(dataiter)
                        except StopIteration:
                            print("error")
                            pr_dataloader = DataLoader(pr_dataset, batch_size=1, shuffle=True)
                            dataiter = iter(pr_dataloader)
                            continue
                        if L2[0].split("\\")[-2] == class_list[i]:
                            # Concatenate the two images together
                            # concatenated = torch.cat((x0, x1), 0)
                            output1, output2 = net(x0.cuda(), x1.cuda())
                            euclidean_distance = F.pairwise_distance(output1, output2)
                            prob_list.append((euclidean_distance.item(),
                                              (L1[0].split("\\")[-2].split('_')[0], L2[0].split("\\")[-2].split('_')[0])))
                            
                            a_output1, a_output2 = auto_net(x0.cuda(), x1.cuda())
                            # print("a_output2: ", a_output2)
                            a_euclidean_distance = F.pairwise_distance(a_output1, a_output2)
                            a_prob_list.append((a_euclidean_distance.item(),
                                                (L1[0].split("\\")[-2].split('_')[0], L2[0].split("\\")[-2].split('_')[0])))
                            a_dist.append(a_euclidean_distance.item())
                            a_proto_label.append(L2[0].split("\\")[-2].split('_')[0])
                            a_proto_vect.append(a_output2.detach().cpu().numpy().flatten())
                            a_proto_array = np.array(a_proto_vect)

                            '''
                            c_output1 = torch.cat((F.normalize(output1.unsqueeze(0), p=1, dim=1), 
                                                   F.normalize(a_output1.unsqueeze(0), p=1, dim=1)), dim=1)
                            c_output2 = torch.cat((F.normalize(output2.unsqueeze(0), p=1, dim=1), 
                                                   F.normalize(a_output2.unsqueeze(0), p=1, dim=1)), dim=1)
                            # print("F.normalize(output1.unsqueeze(0), p=2, dim=1): ", F.normalize(output1.unsqueeze(0), p=2, dim=1))
                            # print(len(F.normalize(output1.unsqueeze(0), p=2, dim=1)))
                            # print("F.normalize(output2.unsqueeze(0), p=2, dim=1): ", F.normalize(output2.unsqueeze(0), p=2, dim=1))
                            # print(len(F.normalize(output2.unsqueeze(0), p=2, dim=1)))
                            # print("c_output1: ", c_output1)
                            # print(len(c_output1))
                            
                            c_output1 = torch.cat((minmax_NormalizeTensor(output1), 
                                                   minmax_NormalizeTensor(a_output1)), dim=0)
                            c_output2 = torch.cat((minmax_NormalizeTensor(output2), 
                                                   minmax_NormalizeTensor(a_output2)), dim=0)
                            c_euclidean_distance = F.pairwise_distance(c_output1, c_output2)
                            '''
                            
                            c_output1 = torch.cat((F.normalize(output1.unsqueeze(0), p=1, dim=1), 
                                                   F.normalize(a_output1.unsqueeze(0), p=1, dim=1)), dim=1)
                            c_output2 = torch.cat((F.normalize(output2.unsqueeze(0), p=1, dim=1), 
                                                   F.normalize(a_output2.unsqueeze(0), p=1, dim=1)), dim=1)
                            
                            c_euclidean_distance = F.pairwise_distance(c_output1, c_output2)
                            c_prob_list.append((c_euclidean_distance.item(),
                                              (L1[0].split("\\")[-2].split('_')[0], L2[0].split("\\")[-2].split('_')[0])))
                            
                            s_distance_1 = F.pairwise_distance(F.normalize(output1.unsqueeze(0), p=1, dim=1),
                                                            F.normalize(output2.unsqueeze(0), p=1, dim=1))
                            s_distance_2 = F.pairwise_distance(F.normalize(a_output1.unsqueeze(0), p=1, dim=1),
                                                            F.normalize(a_output2.unsqueeze(0), p=1, dim=1))
                            s_euclidean_distance = torch.tensor([s_distance_1*2.5, s_distance_2])
                
                            s_prob_list.append((sum(s_euclidean_distance).item(),
                                              (L1[0].split("\\")[-2].split('_')[0], L2[0].split("\\")[-2].split('_')[0])))
                            break

                print("=== original ===")
                print(f"model_num: {model_num} j: {j} prob_list: ", prob_list)
                score = [prob_list[i][0] for i in range(len(prob_list))]
                print(f"j: {j} score: ", score)
                min_score = min(score)
                min_score_index = score.index(min_score)
                print(f"model_num: {model_num}  j: {j} min_index: ", min_score_index)

                print("=== auto ===")
                print(f"model_num: {model_num} j: {j} a_prob_list: ", a_prob_list)
                a_score = [a_prob_list[i][0] for i in range(len(a_prob_list))]
                print(f"j: {j} score: ", a_score)
                a_min_score = min(a_score)
                a_min_score_index = a_score.index(a_min_score)
                print(f"model_num: {model_num}  j: {j} min_index: ", a_min_score_index)

                print("=== concat ===")
                print(f"model_num: {model_num} j: {j} c_prob_list: ", c_prob_list)
                c_score = [c_prob_list[i][0] for i in range(len(c_prob_list))]
                print(f"j: {j} score: ", c_score)
                c_min_score = min(c_score)
                c_min_score_index = c_score.index(c_min_score)
                print(f"model_num: {model_num}  j: {j} min_index: ", c_min_score_index)

                print("=== sep_concat ===")
                print(f"model_num: {model_num} j: {j} s_prob_list: ", s_prob_list)
                s_score = [s_prob_list[i][0] for i in range(len(s_prob_list))]
                print(f"j: {j} score: ", s_score)
                s_min_score = min(s_score)
                s_min_score_index = s_score.index(s_min_score)
                print(f"model_num: {model_num}  j: {j} min_index: ", s_min_score_index)


                print("=== choose net ===")
                # use_auto = False
                '''
                # method 2
                print("a_dist: ", a_dist)
                if all([value > 15 for value in a_dist]):
                    use_auto = False
                else:
                    use_auto = True
                '''
                '''
                # method 3
                print("a_dist: ", a_dist)
                if all([value > 15 for value in a_dist]):
                    use_auto = True
                else:
                    use_auto = False
                '''
                '''
                # not used
                pair_differences = [(y - x) for x, y in combinations(a_dist, 2)]
                print("pair_differences: ", pair_differences)
                if all([value > 5 for value in pair_differences]):
                    use_auto = True
                else:
                    use_auto = False
                '''
                '''
                # method 4 
                a_dist.sort()
                pair_difference = a_dist[:2][1] - a_dist[:2][0]
                print("pair_differences: ", pair_difference)
                if pair_difference > 3:
                    use_auto = True
                else:
                    use_auto = False
                '''
                # method 5 
                # concat autoencoder embedding and siamese net embedding
                '''
                # method 1
                clustering = DBSCAN(eps=15, min_samples=1)
                SC_result = list(clustering.fit_predict(a_proto_array))
                print("SC_result: ", SC_result)
                print("a_proto_label: ", a_proto_label)
                print("zip(SC_result, a_proto_label): ", list(zip(SC_result, a_proto_label)))
                print(f"pr: {len(list(set(SC_result)))} act: {len(list(set(a_proto_label)))} zip: {len(list(set(list(zip(SC_result, a_proto_label)))))}")
                if len(list(set(SC_result))) == len(list(set(list(zip(SC_result, a_proto_label))))):
                    use_auto = True
                else:
                    use_auto = False
                '''
                print("Choose sep_concat")
                try:
                    right = re.sub(re.match(r'.*?(\d+)$', s_prob_list[s_min_score_index][1][1]).group(1), '', s_prob_list[s_min_score_index][1][1])
                except:
                    right = s_prob_list[s_min_score_index][1][1]

                print("right: ", right)
                if s_prob_list[s_min_score_index][1][0] == right:
                    correct_count += 1
                print(f"model_num: {model_num}  j: {j} accu_correct_count: ", correct_count)
                total = j + 1

            print("correct_count: ", correct_count)
            print("total: ", total)
            print("accuracy: ", correct_count/total)
    return {"correct_count": correct_count, "total": total, "accuracy": correct_count/total}

def Testing_Loop_1(model_num, dataset, net_name, model_path, testing_data_path, pr_dataset, pr_data_path=""):
    test_dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    auto_net = SiameseNetwork_only_autoencoder().cuda()
    auto_net.eval()

    if net_name == "SiameseNetwork":
        net = SiameseNetwork().cuda()
    elif net_name == "SiameseNetwork_VGG16_Based_v2":
        net = SiameseNetwork_VGG16_Based_v2().cuda()
    elif net_name == "SiameseNetwork_resnet18_Based":
        net = SiameseNetwork_resnet18_Based().cuda()
    elif net_name == "SiameseNetwork_autoencoder_Based":
        net = SiameseNetwork_autoencoder_Based().cuda()

    net.load_state_dict(torch.load(model_path))
    net.eval()

    with torch.inference_mode():
        if model_num == 1:
            print("INTO MODEL NUM 1")
            correct_count = 0
            total = 0
            class_list = os.listdir(testing_data_path)
            print(class_list)
            for j, (x0, _, _, label1, _, L1, _) in enumerate(test_dataloader, 0):
                prob_list = []
                a_prob_list = []
                a_proto_vect = []
                pr_dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
                dataiter = iter(pr_dataloader)
                while True:
                    try:
                        x1, _, _, label2, _, L2, _ = next(dataiter)
                    except StopIteration:
                        print("error")
                        pr_dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
                        dataiter = iter(pr_dataloader)
                        continue
                    if L2[0].split("\\")[-2] == class_list[i]:
                        
                        output1, output2 = net(x0.cuda(), x1.cuda())
                        euclidean_distance = F.pairwise_distance(output1, output2)
                        prob_list.append((euclidean_distance.item(),
                                            (L1[0].split("\\")[-2].split('_')[0], L2[0].split("\\")[-2].split('_')[0])))
                        
                        a_output1, a_output2 = auto_net(x0.cuda(), x1.cuda())
                        a_euclidean_distance = F.pairwise_distance(a_output1, a_output2)
                        a_prob_list.append((a_euclidean_distance.item(),
                                            (L1[0].split("\\")[-2].split('_')[0], L2[0].split("\\")[-2].split('_')[0])))
                        a_proto_vect.append(a_output2.detach().cpu().numpy().flatten())
                        a_proto_array = np.array(a_proto_vect)
                        break

                print("=== original ===")
                print(f"model_num: {model_num} j: {j} prob_list: ", prob_list)
                score = [prob_list[i][0] for i in range(len(prob_list))]
                print(f"j: {j} score: ", score)
                min_score = min(score)
                min_score_index = score.index(min_score)
                print(f"model_num: {model_num}  j: {j} min_index: ", min_score_index)

                print("=== auto ===")
                print(f"model_num: {model_num} j: {j} a_prob_list: ", a_prob_list)
                a_score = [a_prob_list[i][0] for i in range(len(a_prob_list))]
                print(f"j: {j} score: ", a_score)
                a_min_score = min(a_score)
                a_min_score_index = a_score.index(a_min_score)
                print(f"model_num: {model_num}  j: {j} min_index: ", a_min_score_index)

                print("=== choose net ===")
                clustering = DBSCAN(eps=8, min_samples=2)
                SC_result = clustering.fit_predict(a_proto_array)
                if len(list(set(list(SC_result).remove(-1)))) > 0:
                    use_auto = True
                else:
                    use_auto = False
                

                if use_auto:
                    print("Choose Auto")
                    if a_prob_list[a_min_score_index][1][0] == a_prob_list[a_min_score_index][1][1]:
                        correct_count += 1
                    print(f"model_num: {model_num}  j: {j} accu_correct_count: ", correct_count)
                    total = j + 1
                    
                else:
                    print("Choose Original")
                    if prob_list[min_score_index][1][0] == prob_list[min_score_index][1][1]:
                        correct_count += 1
                    print(f"model_num: {model_num}  j: {j} accu_correct_count: ", correct_count)
                    total = j + 1

            print("correct_count: ", correct_count)
            print("total: ", total)
            print("accuracy: ", correct_count / total)

        else:
            correct_count = 0
            total = 0
            class_list = os.listdir(pr_data_path)
            print(class_list)
            for j, (x0, _, _, label1, _, L1, _) in enumerate(test_dataloader, 0):
                prob_list = []
                a_prob_list = []
                a_proto_vect = []
                a_proto_label = []
                a_dist = []
                c_prob_list = []
                s_prob_list = []
                for i in range(len(class_list)):
                    # print("i: ", i)
                    pr_dataloader = DataLoader(pr_dataset, batch_size=1, shuffle=True)
                    dataiter = iter(pr_dataloader)
                    while True:
                        try:
                            x1, label2, L2 = next(dataiter)
                        except StopIteration:
                            print("error")
                            pr_dataloader = DataLoader(pr_dataset, batch_size=1, shuffle=True)
                            dataiter = iter(pr_dataloader)
                            continue
                        if L2[0].split("\\")[-2] == class_list[i]:
                            # Concatenate the two images together
                            output1, output2 = net(x0.cuda(), x1.cuda())
                            euclidean_distance = F.pairwise_distance(output1, output2)
                            prob_list.append((euclidean_distance.item(),
                                              (L1[0].split("\\")[-2].split('_')[0], L2[0].split("\\")[-2].split('_')[0])))
                            
                            a_output1, a_output2 = auto_net(x0.cuda(), x1.cuda())
                            a_euclidean_distance = F.pairwise_distance(a_output1, a_output2)
                            a_prob_list.append((a_euclidean_distance.item(),
                                                (L1[0].split("\\")[-2].split('_')[0], L2[0].split("\\")[-2].split('_')[0])))
                            a_dist.append(a_euclidean_distance.item())
                            a_proto_label.append(L2[0].split("\\")[-2].split('_')[0])
                            a_proto_vect.append(a_output2.detach().cpu().numpy().flatten())
                            a_proto_array = np.array(a_proto_vect)

                            '''
                            c_output1 = torch.cat((F.normalize(output1.unsqueeze(0), p=1, dim=1), 
                                                   F.normalize(a_output1.unsqueeze(0), p=1, dim=1)), dim=1)
                            c_output2 = torch.cat((F.normalize(output2.unsqueeze(0), p=1, dim=1), 
                                                   F.normalize(a_output2.unsqueeze(0), p=1, dim=1)), dim=1)
                            # print("F.normalize(output1.unsqueeze(0), p=2, dim=1): ", F.normalize(output1.unsqueeze(0), p=2, dim=1))
                            # print(len(F.normalize(output1.unsqueeze(0), p=2, dim=1)))
                            # print("F.normalize(output2.unsqueeze(0), p=2, dim=1): ", F.normalize(output2.unsqueeze(0), p=2, dim=1))
                            # print(len(F.normalize(output2.unsqueeze(0), p=2, dim=1)))
                            # print("c_output1: ", c_output1)
                            # print(len(c_output1))
                            
                            c_output1 = torch.cat((minmax_NormalizeTensor(output1), 
                                                   minmax_NormalizeTensor(a_output1)), dim=0)
                            c_output2 = torch.cat((minmax_NormalizeTensor(output2), 
                                                   minmax_NormalizeTensor(a_output2)), dim=0)
                            c_euclidean_distance = F.pairwise_distance(c_output1, c_output2)
                            '''
                            
                            c_output1 = torch.cat((F.normalize(output1, p=1, dim=1), 
                                                   F.normalize(a_output1.unsqueeze(0), p=1, dim=1)), dim=1)
                            c_output2 = torch.cat((F.normalize(output2, p=1, dim=1), 
                                                   F.normalize(a_output2.unsqueeze(0), p=1, dim=1)), dim=1)
                            
                            c_euclidean_distance = F.pairwise_distance(c_output1, c_output2)
                            c_prob_list.append((c_euclidean_distance.item(),
                                              (L1[0].split("\\")[-2].split('_')[0], L2[0].split("\\")[-2].split('_')[0])))
                            
                            s_distance_1 = F.pairwise_distance(F.normalize(output1.unsqueeze(0), p=1, dim=1),
                                                            F.normalize(output2.unsqueeze(0), p=1, dim=1))
                            s_distance_2 = F.pairwise_distance(F.normalize(a_output1.unsqueeze(0), p=1, dim=1),
                                                            F.normalize(a_output2.unsqueeze(0), p=1, dim=1))
                            s_euclidean_distance = torch.tensor([s_distance_1*2.5, s_distance_2])
                           
                            s_prob_list.append((sum(s_euclidean_distance).item(),
                                              (L1[0].split("\\")[-2].split('_')[0], L2[0].split("\\")[-2].split('_')[0])))
                            break

                print("=== original ===")
                print(f"model_num: {model_num} j: {j} prob_list: ", prob_list)
                score = [prob_list[i][0] for i in range(len(prob_list))]
                print(f"j: {j} score: ", score)
                min_score = min(score)
                min_score_index = score.index(min_score)
                print(f"model_num: {model_num}  j: {j} min_index: ", min_score_index)

                print("=== auto ===")
                print(f"model_num: {model_num} j: {j} a_prob_list: ", a_prob_list)
                a_score = [a_prob_list[i][0] for i in range(len(a_prob_list))]
                print(f"j: {j} score: ", a_score)
                a_min_score = min(a_score)
                a_min_score_index = a_score.index(a_min_score)
                print(f"model_num: {model_num}  j: {j} min_index: ", a_min_score_index)

                print("=== concat ===")
                print(f"model_num: {model_num} j: {j} c_prob_list: ", c_prob_list)
                c_score = [c_prob_list[i][0] for i in range(len(c_prob_list))]
                print(f"j: {j} score: ", c_score)
                c_min_score = min(c_score)
                c_min_score_index = c_score.index(c_min_score)
                print(f"model_num: {model_num}  j: {j} min_index: ", c_min_score_index)

                print("=== sep_concat ===")
                print(f"model_num: {model_num} j: {j} s_prob_list: ", s_prob_list)
                s_score = [s_prob_list[i][0] for i in range(len(s_prob_list))]
                print(f"j: {j} score: ", s_score)
                s_min_score = min(s_score)
                s_min_score_index = s_score.index(s_min_score)
                print(f"model_num: {model_num}  j: {j} min_index: ", s_min_score_index)


                print("=== choose net ===")
                # use_auto = False
                '''
                # method 2
                print("a_dist: ", a_dist)
                if all([value > 15 for value in a_dist]):
                    use_auto = False
                else:
                    use_auto = True
                '''
                '''
                # method 3
                print("a_dist: ", a_dist)
                if all([value > 15 for value in a_dist]):
                    use_auto = True
                else:
                    use_auto = False
                '''
                '''
                # not used
                pair_differences = [(y - x) for x, y in combinations(a_dist, 2)]
                print("pair_differences: ", pair_differences)
                if all([value > 5 for value in pair_differences]):
                    use_auto = True
                else:
                    use_auto = False
                '''
                '''
                # method 4 
                a_dist.sort()
                pair_difference = a_dist[:2][1] - a_dist[:2][0]
                print("pair_differences: ", pair_difference)
                if pair_difference > 3:
                    use_auto = True
                else:
                    use_auto = False
                '''
               
                '''
                # method 1
                clustering = DBSCAN(eps=15, min_samples=1)
                SC_result = list(clustering.fit_predict(a_proto_array))
                print("SC_result: ", SC_result)
                print("a_proto_label: ", a_proto_label)
                print("zip(SC_result, a_proto_label): ", list(zip(SC_result, a_proto_label)))
                print(f"pr: {len(list(set(SC_result)))} act: {len(list(set(a_proto_label)))} zip: {len(list(set(list(zip(SC_result, a_proto_label)))))}")
                if len(list(set(SC_result))) == len(list(set(list(zip(SC_result, a_proto_label))))):
                    use_auto = True
                else:
                    use_auto = False
                '''
                
                print("Choose sep_concat")
                try:
                    right = re.sub(re.match(r'.*?(\d+)$', s_prob_list[s_min_score_index][1][1]).group(1), '', s_prob_list[s_min_score_index][1][1])
                except:
                    right = s_prob_list[s_min_score_index][1][1]

                print("right: ", right)
                if s_prob_list[s_min_score_index][1][0] == right:
                    correct_count += 1
                print(f"model_num: {model_num}  j: {j} accu_correct_count: ", correct_count)
                total = j + 1

            print("correct_count: ", correct_count)
            print("total: ", total)
            print("accuracy: ", correct_count/total)
    return {"correct_count": correct_count, "total": total, "accuracy": correct_count/total}
