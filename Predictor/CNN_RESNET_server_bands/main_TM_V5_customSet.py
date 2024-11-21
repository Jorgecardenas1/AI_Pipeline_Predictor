"""Version 3
Verison 4:  full spectra output
Version 5: top values with FWHM and one hot encode"""

from __future__ import print_function

import os

#from Utilities.SaveAnimation import Video

from druida import Stack
from druida import setup

from druida.DataManager import datamanager
#from druidaHFSS.modules import tools
from druida.tools import utils
from torch.utils.data import Dataset

import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

import torch
import torch.nn as nn
import torch.optim as optimizer
from torch.utils.data import Dataset
import torchvision.transforms.functional as F

import torch.nn.functional as f

from torchsummary import summary
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torchvision
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import glob
from tqdm.notebook import tqdm
import argparse
import json
import cv2
from matplotlib.collections import LineCollection

from PIL import Image

from scipy.signal import find_peaks,peak_widths

# Clip 

from torch import nn
from transformers import CLIPTokenizer, CLIPTextModel,CLIPTextConfig
from transformers import BertTokenizer, BertModel, BertConfig,BertForMaskedLM
from typing import List

#torch.set_printoptions(profile="full")
#torch.manual_seed(999)

#RESNET
from torchvision.models import resnet50, ResNet50_Weights
from torcheval.metrics.functional import r2_score
from torchvision.utils import save_image


# Arguments
parser = argparse.ArgumentParser()
# boxImagesPath="\\data\\francisco_pizarro\\jorge-cardenas\\data\\MetasufacesData\\Images Jorge Cardenas 512\\"
# DataPath="\\data\\francisco_pizarro\\jorge-cardenas\\data\\MetasufacesData\\Exports\\output\\"
# simulationData="\\data\\francisco_pizarro\\jorge-cardenas\\data\\MetasufacesData\\DBfiles\\"

boxImagesPath="../../../data/MetasurfacesData/Images-512-Bands/"
#boxImagesPath="../../../data/MetasufacesData/Images-512-Suband/"
DataPath="../../../data/MetasurfacesData/Exports/output/"
simulationData="../../../data/MetasurfacesData/DBfiles/"
validationImages="../../../data/MetasurfacesData/testImages/"

#Design parameters
Substrates={"Rogers RT/duroid 5880 (tm)":0}
Materials={"copper":0,"pec":1}
Surfacetypes={"Reflective":0,"Transmissive":1}
Bands={"75-78":0}
TargetGeometries={"circ":[1,0,0],"box":[0,1,0], "cross":[0,0,1]}

def arguments():

    parser.add_argument("run_name",type=str)
    parser.add_argument("epochs",type=int)
    parser.add_argument("batch_size",type=int)
    parser.add_argument("workers",type=int)
    parser.add_argument("gpu_number",type=int)
    parser.add_argument("device",type=str)
    parser.add_argument("learning_rate",type=float)
    parser.add_argument("condition_len",type=int) #This defines the length of our conditioning vector
    parser.add_argument("metricType",type=str) #This defines the length of our conditioning vector
    parser.add_argument("cond_channel",type=int) #This defines the length of our conditioning vector
    parser.add_argument("cond_channel",type=int) #This defines the length of our conditioning vector
    parser.add_argument("resnet_arch",type=str) #This defines the length of our conditioning vector
    parser.add_argument("pred_size",type=int) #This defines the length of our conditioning vector

    parser.run_name = "Predictor Training"
    parser.epochs = 25
    parser.batch_size = 128 #original 128
    parser.workers=1
    parser.gpu_number=1
    parser.image_size = 64
    parser.dataset_path = os.path.normpath('/content/drive/MyDrive/Training_Data/Training_lite/')
    parser.device = "cpu"
    parser.learning_rate = 1e-3

    parser.metricType='AbsorbanceTM' #this is to be modified when training for different metrics.
    parser.cond_channel=3 #this is to be modified when training for different metrics.
    parser.condition_len=6 #this is to be modified when training for different metrics.
    parser.resnet_arch="resnet152" #this is to be modified when training for different metrics.
    parser.pred_size=1#this is to be modified when training for different metrics.

    #if parser.pred_size==1:
    #    parser.condition_len=parser.condition_len+1 #this is to be modified when training for different metrics.


    categories=["box", "circle", "cross"]


# Data pre-processing
def join_simulationData():
    df = pd.DataFrame()
    for file in glob.glob(simulationData+"*.csv"): 
        df2 = pd.read_csv(file)
        df = pd.concat([df, df2], ignore_index=True)
    df.to_csv('out.csv',index=False)



def get_data_with_labels(image_size, randomResize, imagesPath, dataset_path,batch_size, drop_last):

    transforms = torchvision.transforms.Compose([
        #torchvision.transforms.Resize(resize),  # args.image_size + 1/4 *args.image_size
        torchvision.transforms.RandomResizedCrop(image_size, scale=(randomResize, randomResize)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    df = pd.read_csv(dataset_path)
    #'path', 'class','class_target', 'freq','band'
    image_names = df["path"].values.tolist()
    targets = df["class_target"]
    targets = targets.replace("circ", 0)
    targets = targets.replace("cross",1 )
    targets = targets.values.tolist()
    classes = df["class"]

    frequencies = df["freq"].values.tolist()

    cursomDataset = customDataset(imagesPath,image_names, targets, classes,frequencies,transforms)
    dataloader = DataLoader(cursomDataset, batch_size=batch_size, shuffle=True, drop_last=drop_last)


    return dataloader


class customDataset(Dataset):
    def __init__(self, image_paths,image_names, targets, classes,frequencies, transforms):
        self.image_paths = image_paths
        self.transform = transforms
        self.names = image_names
        self.classes = classes
        self.frequencies = frequencies
        self.targets = targets

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):

        image_filepath = self.names[idx]
        path =  self.image_paths+'/'+ self.classes[idx] + '/' + image_filepath
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = F.to_pil_image(image)
        type_class = self.targets[idx]
        if "\\" in self.names:
            class_name=self.names[idx].split('_')[0].split('\\')

        else:
            class_name=self.names[idx].split('_')[0].split('/')

        image = self.transform(image)
        
        return image, type_class ,os.path.basename(image_filepath),class_name[-1], self.frequencies[idx]


# Load Model
def get_net_resnet(device,hiden_num=1000,dropout=0.1,features=3000, Y_prediction_size=601):
    model = Stack.Predictor_RESNET(parser.resnet_arch,conditional=True, cond_input_size=parser.condition_len, 
                                   cond_channels=parser.cond_channel, 
                                ngpu=1, image_size=parser.image_size ,
                                output_size=8, channels=3,
                                features_num=features,hiden_num=hiden_num, #Its working with hiden nums. Features in case and extra linear layer
                                dropout=dropout, 
                                Y_prediction_size=Y_prediction_size) #size of the output vector in this case frenquency points
    

    opt = optimizer.Adam(model.parameters(), lr=parser.learning_rate, betas=(0.5, 0.999),weight_decay=1e-4)
    criterion=nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.999)
    model.train()
    return model, opt, criterion , scheduler

def get_net_CNN(device,hiden_num=1000,dropout=0.1,features=3000, Y_prediction_size=601):
   
    model =  Stack.Predictor_CNN(cond_input_size=parser.condition_len, 
                               ngpu=1, image_size=parser.image_size ,
                               output_size=8, channels=3,
                               features_num=features,hiden_num=hiden_num, #Its working with hiden nums. Features in case and extra linear layer
                               dropout=dropout, 
                               Y_prediction_size=Y_prediction_size) #size of the output vector in this case frenquency points


    #opt = optimizer.Adam(model.parameters(), lr=parser.learning_rate, betas=(0.5, 0.999),weight_decay=1e-4)
    opt  = optimizer.SGD(model.parameters(), lr =parser.learning_rate, momentum=0.9,weight_decay=1e-4)
    criterion=nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=1.0004)

    return model, opt, criterion , scheduler





# Conditioning
def set_conditioning(target,path,categories,frequencies,df):
        

    arr=[] #intermediate array to build the batch of conditioning
    values_array=torch.tensor((), dtype=torch.float) # the conditioning tensor

    for idx,name in enumerate(path):

        series=name.split('_')[-2]
        batch=name.split('_')[4]

        if batch=="v2":
            batch=name.split('_')[5]      
        
        iteration=series.split('-')[-1]
        row=df[(df['sim_id']==batch) & (df['iteration']==int(iteration))  ]

        #getting values from design parameters
        target_val=target[idx]
        category=categories[idx]
        geometry=TargetGeometries[category]

        """"
        surface type: reflective, transmissive
        layers: conductor and conductor material / Substrate information
        """
       
        
        surfacekey=row["type"].values[0]
        surfacetype=Surfacetypes[surfacekey]
        
        layers=row["layers"].values[0]
        layers= layers.replace("'", '"')
        layer=json.loads(layers)
        
        materialconductor=Materials[layer['conductor']['material']]
        materialsustrato=Substrates[layer['substrate']['material']]
        
        

        """Ojo esto es delicado porque los indices dependen de como el dataloader 
        cargue las carpetas de imagenes"""
        if (target_val==2): #is cross. Because an added variable to the desing 
            sustratoHeight= json.loads(row["paramValues"].values[0])
            sustratoHeight= sustratoHeight[-2]
            #sustratoHeight = Height[str(sustratoHeight)]

            substrateWidth = json.loads(row["paramValues"].values[0])[-1] # from the simulation crosses have this additional free param
        else:
        
            sustratoHeight= json.loads(row["paramValues"].values[0])
            sustratoHeight= sustratoHeight[-1]
            #sustratoHeight = Height[str(sustratoHeight)]
            substrateWidth = 5 # 5 mm size
        
        val_arr=torch.Tensor(geometry)
        val_arr=torch.cat((val_arr,torch.Tensor([sustratoHeight,substrateWidth,frequencies[idx]])),0)

        arr.append(val_arr)

    #embedding = torch.stack(arr)
    values_array = torch.stack(arr)
    """ Values array solo pouede llenarse con n+umero y no con textos"""
    # values_array = torch.Tensor(values_array)

    return values_array


def encoders(dictionary):
    index = []
    for x,y in dictionary.items():
        index.append([y])

    index = np.asarray(index)
    enc = OneHotEncoder()
    enc.fit(index)
    return enc

def epoch_train(epoch,model,dataloader,validation_dataloader,device,opt,scheduler,criterion,clipEmbedder,df):
    i=0 #iteration
    running_loss = 0. 
    epoch_loss = 0.
    acc_train=0.0
    score=0.0
    
    print('Epoch {}/{}'.format(epoch, parser.epochs - 1))
    print('-' * 10)
    
    for data in tqdm(dataloader):
        
        inputs, classes, names, classes_types , frequencies= data
        
        opt.zero_grad()

        inputs = inputs.to(device) #images
        classes =classes.to(device) #categories
        frequencies = frequencies.to(device)
        eval_name = ""

        a, aa = [] ,[]

        """lookup for data corresponding to every image in training batch"""
        for index,name in enumerate(names):

            series=name.split('_')[-2]# getting series and batches names 
            band_name=name.split('_')[-1].split('.')[0]# Getting the band for the corresponding image
            batch=name.split('_')[4]
            version_batch=1
            frequency =  frequencies[index]

            if batch=="v2":
                version_batch=2
                batch=name.split('_')[5]

            for name in glob.glob(DataPath+batch+'/files/'+'/'+parser.metricType+'*'+series+'.csv'): 
                
                #loading the absorption data
                train = pd.read_csv(name)
                eval_name = name

                # # the band is divided in chunks 
                if Bands[str(band_name)]==0:
                
                    train=train.loc[1:100]

                elif Bands[str(band_name)]==1:
                    
                    train=train.loc[101:200]
                
                
                #preparing data from spectra for each image
                data=np.array(train.values.T)
                values=data[1]
                values = np.around(values, decimals=2, out=None)

                all_frequencies=data[0]
                all_frequencies = np.array([(float(i)-min(all_frequencies))/(max(all_frequencies)-min(all_frequencies)) for i in all_frequencies])

                #batching
                aa.append(torch.from_numpy(values))
                a.append(torch.tensor(values[frequency], dtype=torch.float))

       
        """Creating a conditioning vector"""

        array=set_conditioning(classes, names, classes_types,frequencies,df)
        array = torch.nn.functional.normalize(array, p=2.0, dim = 1)

        """Prediction"""
        y_predicted=model(input_=inputs, conditioning=array.to(device) ,b_size=inputs.shape[0])
        y_predicted=y_predicted.to(device)
        y_predicted = torch.flatten(y_predicted)
        y_truth = torch.stack(a).to(device)

        """this just in case required depending on changes applied to output vector"""
        #y_truth =  torch.unsqueeze(y_truth, 1)
        #y_truth = torch.nn.functional.normalize(y_truth, p=1.0, dim=1, eps=1e-12, out=None)

        
        loss_per_batch,running_loss, epoch_loss, acc_train,score = metrics(criterion,
                                                                    y_predicted,
                                                                    y_truth, opt,
                                                                    running_loss,
                                                                    epoch_loss,
                                                                    acc_train,
                                                                    train=True)
        acc_train = acc_train+score

        i += 1

        if i % 1000 ==  999:    # print every X mini-batches

            print(f'[{epoch + 1}, {i :5d}] loss: {loss_per_batch/y_truth.size(0):.3f} running loss:  {running_loss/10000:.3f} iter loss:  {running_loss/i:.3f}')
            print(f'accuracy: {acc_train/i :.3f} ',f' score: {score :.3f} ')
            running_loss=0.0           

    """"Evaluation"""
    with torch.no_grad():
        
        predicted = []
        a, aa = [] ,[]

        inputs, classes, names, classes_types , frequencies= data = next(iter(validation_dataloader))
        inputs = inputs.to(device) #images
        classes =classes.to(device) #categories
        frequencies = frequencies.to(device)
        
        
        for index,name in enumerate(names):

            series=name.split('_')[-2]# getting series and batches names 
            band_name=name.split('_')[-1].split('.')[0]# Getting the band for the corresponding image
            batch=name.split('_')[4]
            version_batch=1
            frequency =  frequencies[index]

            if batch=="v2":
                version_batch=2
                batch=name.split('_')[5]

            for name in glob.glob(DataPath+batch+'/files/'+'/'+parser.metricType+'*'+series+'.csv'): 
                
                #loading the absorption data
                train = pd.read_csv(name)

                # # the band is divided in chunks 
                if Bands[str(band_name)]==0:
                
                    train=train.loc[1:100]

                elif Bands[str(band_name)]==1:
                    
                    train=train.loc[101:200]

                data=np.array(train.values.T)
                values=data[1]
                values = np.around(values, decimals=2, out=None)

                aa.append(torch.from_numpy(values))
        
        for freq in range(0,100):

            array=set_conditioning(classes, names, classes_types,torch.full((parser.batch_size,), freq),df)
            array = torch.nn.functional.normalize(array, p=2.0, dim = 1)
            
            y_predicted=model(input_=inputs, conditioning=array.to(device) ,b_size=inputs.shape[0])
            predicted.append(y_predicted)

        predict =  torch.stack(predicted).to(device)
        predict = torch.transpose(predict[:,:,0],0,1)   
        aa =  torch.stack(aa).to(device)

        f = plt.figure(figsize=(10,50))
        fig, axs= plt.subplots(10, 1,figsize=(4,20),) 
        plt.subplots_adjust( bottom=0.1, top=0.9, wspace=0.4,hspace=0.4)
        counter=0
        for ax in axs:
            ax.plot(range(0,100), predict[counter].cpu().detach().numpy(),color="salmon")
            ax.plot(range(0,100), aa[counter].cpu().detach().numpy(),"-0", color='royalblue')
            counter=counter+1

        plt.savefig("./output_20Nov/"+batch+"_"+series+"_"+str(epoch)+".png")
        plt.clf()
        del predict, y_predicted, predicted



    #update the learning rate
    scheduler.step()
    print("learning_rate: ",scheduler.get_last_lr())

    return i,epoch_loss,acc_train,score

def find_peaks_values(values, all_frequencies):
        peaks = find_peaks(values, threshold=0.00001)[0] #indexes of peaks
        results_half = peak_widths(values, peaks, rel_height=0.5) #4 arrays: widths, y position, initial and final x
        results_half = results_half[0]
        data = values[peaks]
        fre_peaks = all_frequencies[peaks]
        length_output=3

        if len(peaks)>length_output:
            data = data[0:length_output]
            fre_peaks = fre_peaks[0:length_output]
            results_half = results_half[0:length_output]

        elif len(peaks)==0:

            data = np.zeros(length_output)
            fre_peaks = all_frequencies[0:length_output]
            results_half = np.zeros(length_output)

        else:

            difference = length_output-len(peaks)

            for idnx in range(difference):
                data = np.append(data, 0)
                fequencies = np.where(values<0.1)
                fequencies = np.squeeze(fequencies)
                fre_peaks = np.append(fre_peaks,all_frequencies[fequencies[idnx]])
                results_half = np.append(results_half,0)

        return data, fre_peaks,results_half, peaks

def mape_loss_fn(output, target):
    # MAPE loss
    return torch.mean(torch.abs((target - output) / target))

#here we calculate loss for the training process
def metrics(criterion,y_predicted,y_truth, opt,running_loss,epoch_loss,acc_train,train=True):

    
    loss_per_batch=0
    #apply criterion
    loss = criterion(y_predicted.float(), y_truth.float())
    if train:
        loss.backward()
        loss_per_batch=loss.item()
        opt.step()
    else:
        loss_per_batch=loss.item()

    score = r2_score(y_predicted, y_truth)

    acc_train+= 0

    #Loss
    running_loss +=loss_per_batch
    #print(running_loss)
    if train:
        epoch_loss+=loss_per_batch

    return loss_per_batch,running_loss, epoch_loss, acc_train,score


def train(opt,scheduler,criterion,model, clipEmbedder,device, PATH):
    #### #File reading conf

    loss_values, valid_loss_list, acc,acc_val = [],[],[],[]

    # Load simulated data
    df = pd.read_csv("out.csv")

    #vdataloader = utils.get_data_with_labels(parser.image_size, parser.image_size,1,
    #                                        validationImages,parser.batch_size, 
    #                                        drop_last=True,
    #                                        filter="30-40")
    
    #dataloader = utils.get_data_with_labels(parser.image_size,parser.image_size,1, 
    #                                            boxImagesPath,parser.batch_size,
    #                                            drop_last=True,
    #                                            filter="30-40")#filter disabled
    
    dataloader = get_data_with_labels(parser.image_size,
                                      1, 
                                      boxImagesPath,
                                      "outImages.csv",
                                      parser.batch_size,
                                      drop_last=True)

    validation_dataloader = get_data_with_labels(parser.image_size,
                                      1,
                                      validationImages,
                                      "outImagesValidation.csv",
                                      10,
                                      drop_last=True)
    for epoch in range(parser.epochs):
        #load images data
       
        model.train() #in case required for validation

        total,epoch_loss,acc_train,score_train=epoch_train(epoch,model,dataloader,validation_dataloader,device,opt,scheduler,criterion,clipEmbedder,df)

        print("epoch_loss: ",epoch_loss/total)
        loss_values.append(epoch_loss/total )
        print("mean Acc per epoch",acc_train/total)
        acc.append(acc_train/total)

        torch.save(model.state_dict(), PATH)

    
    return loss_values,acc,valid_loss_list,acc_val,score_train



def main():

    os.environ["PYTORCH_USE_CUDA_DSA"] = "1"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    arguments()
    join_simulationData()  
    
    fwd_test, opt, criterion,scheduler=get_net_resnet(device,hiden_num=1000,dropout=0.3,features=1000, Y_prediction_size=parser.pred_size)
    #fwd_test, opt, criterion,scheduler=get_net_CNN(device,hiden_num=1500,dropout=0.3,features=1500, Y_prediction_size=parser.pred_size)
    fwd_test = fwd_test.to(device)

    print(fwd_test)

    """option of word embedding"""
    Bert=None

    date="_RESNET152_Bands_20Nov_1e-3_25epc_ADAM_64_MSE_batch128_0.999_1out"
    PATH = 'trainedModelTM_abs_'+date+'.pth'

    loss_values,acc,valid_loss_list,acc_val,score_train=train(opt,
                                                            scheduler,
                                                            criterion,
                                                            fwd_test,
                                                            Bert,
                                                            device,
                                                            PATH )

    torch.save(fwd_test.state_dict(), PATH)

    try:
        np.savetxt('output_20Nov/loss_Train_TM_'+date+'.out', loss_values, delimiter=',')
    except:
        np.savetxt('output_20Nov/loss_Train_TM_'+date+'.out', [], delimiter=',')

    try:
        np.savetxt('output_20Nov/acc_Train_TM_'+date+'.out', acc, delimiter=',')
    except:
        np.savetxt('output_20Nov/acc_Train_TM_'+date+'.out', [], delimiter=',')
    
    try:
        np.savetxt('output_20Nov/loss_Valid_TM_'+date+'.out', valid_loss_list, delimiter=',')
    except:
        np.savetxt('output_20Nov/loss_Valid_TM_'+date+'.out', [], delimiter=',')
    
    try:
        np.savetxt('output_20Nov/acc_val_'+date+'.out', acc_val, delimiter=',')
    except:
        np.savetxt('output_20Nov/acc_val_'+date+'.out', [], delimiter=',')

    try:
        np.savetxt('output_20Nov/score_train_'+date+'.out', score_train, delimiter=',')
    except:
        np.savetxt('output_20Nov/score_train_'+date+'.out', [], delimiter=',')

if __name__ == "__main__":
    main()