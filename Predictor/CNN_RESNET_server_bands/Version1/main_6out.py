"""Version 3
Verison 4:  full spectra output
Version 5: top values with FWHM and one hot encode"""

from __future__ import print_function

import os

#from Utilities.SaveAnimation import Video

from druida import Stack
from druida import setup

from druida.DataManager import datamanager
from druida.tools import utils

import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

import torch
import torch.nn as nn
import torch.optim as optimizer

import torch.nn.functional as f

from torchsummary import summary
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import glob
from tqdm.notebook import tqdm
import argparse
import json
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

boxImagesPath="../../../data/MetasufacesData/Images-512-Bands/"
#boxImagesPath="../../../data/MetasufacesData/Images-512-Suband/"
DataPath="../../../data/MetasufacesData/Exports/output/"
simulationData="../../../data/MetasufacesData/DBfiles/"
validationImages="../../../data/MetasufacesData/testImages/"

#Design parameters
Substrates={"Rogers RT/duroid 5880 (tm)":0}
Materials={"copper":0,"pec":1}
Surfacetypes={"Reflective":0,"Transmissive":1}
#TargetGeometries={"circ":0,"box":1, "cross":2}
Bands={"30-40":0,"40-50":1, "50-60":2,"60-70":3,"70-80":4, "80-90":5}
TargetGeometries={"circ":[1,0,0],"box":[0,1,0], "cross":[0,0,1]}

#Bands={"30-40":[1,0,0,0,0,0],"40-50":[0,1,0,0,0,0], 
#       "50-60":[0,0,1,0,0,0],"60-70":[0,0,0,1,0,0],"70-80":[0,0,0,0,1,0], 
#       "80-90":[1,0,0,0,0,1]}
#Height={"0.252":[1,0,0,0],"0.508":[0,1,0,0], 
#       "0.787":[0,0,1,0],"1.575":[0,0,0,1]}

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
    parser.epochs = 200
    parser.batch_size = 32
    parser.workers=1
    parser.gpu_number=1
    parser.image_size = 128
    parser.dataset_path = os.path.normpath('/content/drive/MyDrive/Training_Data/Training_lite/')
    parser.device = "cpu"
    parser.learning_rate =5e-5
    parser.metricType='AbsorbanceTM' #this is to be modified when training for different metrics.
    parser.cond_channel=3 #this is to be modified when training for different metrics.
    parser.condition_len=4 #this is to be modified when training for different metrics.
    parser.resnet_arch="resnet152" #this is to be modified when training for different metrics.
    parser.pred_size=6#this is to be modified when training for different metrics.

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
    #opt  = optimizer.SGD(model.parameters(), lr =parser.learning_rate, momentum=0.9,weight_decay=1e-4)
    criterion=nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.99)
    model.train()
    return model, opt, criterion , scheduler

def get_net_CNN(device,hiden_num=1000,dropout=0.1,features=3000, Y_prediction_size=601):
   
    model =  Stack.Predictor_CNN(cond_input_size=parser.condition_len, 
                               ngpu=1, image_size=parser.image_size ,
                               output_size=8, channels=3,
                               features_num=features,hiden_num=hiden_num, #Its working with hiden nums. Features in case and extra linear layer
                               dropout=dropout, 
                               Y_prediction_size=Y_prediction_size) #size of the output vector in this case frenquency points


    opt = optimizer.Adam(model.parameters(), lr=parser.learning_rate, betas=(0.5, 0.999),weight_decay=1e-4)
    #opt  = optimizer.SGD(model.parameters(), lr =parser.learning_rate, momentum=0.9,weight_decay=1e-4)
    criterion=nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.99)

    return model, opt, criterion , scheduler





# Conditioning
def set_conditioning(bands_batch,freqx_val,target,path,categories,clipEmbedder,df,device):
        

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
        band=bands_batch[idx]
        

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
        

        """this to apply for one hot encoding"""
        # materialsustrato=torch.Tensor(substrate_encoder.transform(np.array(Substrates[layer['substrate']['material']]).reshape(-1, 1)).toarray()).squeeze(0)
        # materialconductor=torch.Tensor(materials_encoder.transform(np.array(Materials[layer['conductor']['material']]).reshape(-1, 1)).toarray()).squeeze(0)
        # surface=torch.Tensor(surfaceType_encoder.transform(np.array(Surfacetypes[surfacekey]).reshape(-1, 1)).toarray()).squeeze(0)
        # band=torch.Tensor(bands_encoder.transform(np.array(band).reshape(-1, 1)).toarray()).squeeze(0)
  

        #values_array = torch.cat((torch.Tensor(geometry),torch.Tensor(surfacetype),torch.Tensor(materialconductor),torch.Tensor(materialsustrato),torch.Tensor([sustratoHeight]),torch.Tensor(band)),0) #concat side
        #if parser.pred_size==1:
        #    arr.append(torch.Tensor([geometry,surfacetype,materialconductor,materialsustrato,sustratoHeight, substrateWidth,band,freqx_val]))
       
        #else:
        val_arr=torch.Tensor(geometry)
        val_arr=torch.cat((val_arr,torch.Tensor([band])),0)
        #val_arr = torch.Tensor([band,freqx_val])
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

def epoch_train(epoch,model,dataloader,device,opt,scheduler,criterion,clipEmbedder,df):
    i=0 #iteration
    running_loss = 0. 
    epoch_loss = 0.
    acc_train=0.0
    bands_batch=[]

    print('Epoch {}/{}'.format(epoch, parser.epochs - 1))
    print('-' * 10)
    
    for data in tqdm(dataloader):
        
        inputs, classes, names, classes_types = data
        
        #sending to CUDA
        inputs = inputs.to(device) #images
        classes = classes.to(device) #categories
        #Loading data
        a, bands_batch, max_freqs = []    ,[],[]    


        """lookup for data corresponding to every image in training batch"""
        for name in names:

            series=name.split('_')[-2]# getting series and batches names 
            band_name=name.split('_')[-1].split('.')[0]# Getting the band for the corresponding image
            batch=name.split('_')[4]
            version_batch=1
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

                elif Bands[str(band_name)]==2:
                    if version_batch==1:
                        train=train.loc[201:300]
                    else:
                        train=train.loc[1:100]
                elif Bands[str(band_name)]==3:
                    if version_batch==1:
                        train=train.loc[301:400]
                    else:
                        train=train.loc[101:200]

                elif Bands[str(band_name)]==4:
                    if version_batch==1: 
                        train=train.loc[401:500]
                    else:
                        train=train.loc[201:300]

                elif Bands[str(band_name)]==5:

                    train=train.loc[501:600]
                
                

                #preparing data from spectra for each image
                data=np.array(train.values.T)
                values=data[1]
                all_frequencies=data[0]
                all_frequencies = np.array([(float(i)-min(all_frequencies))/(max(all_frequencies)-min(all_frequencies)) for i in all_frequencies])

                values[values<0.05]=0
                #get top freqencies for top values 
                data, fre_peaks,results_half, peaks = find_peaks_values(values,all_frequencies)
                #creating the labels tensor with top values and its frequencies

                labels_tensor=torch.cat((torch.from_numpy(data),torch.from_numpy(fre_peaks)),0)
                #labels_tensor=torch.from_numpy(values)
                
                #batching
                a.append(labels_tensor)

                #saving bands and frequencies 
                bands_batch.append(Bands[band_name])
                max_freqs.append(all_frequencies[peaks])
                


        """Creating a conditioning vector"""
        #for item in all_frequencies:

            #freq_indx = list(all_frequencies).index(item)
        opt.zero_grad()

        array=set_conditioning(bands_batch,
                                        None,
                                        classes, 
                                        names, 
                                        classes_types,
                                        clipEmbedder,
                                        df,
                                        device)
        # the prediction 
        array = torch.nn.functional.normalize(array, p=2.0, dim=1, eps=1e-5, out=None)
        y_predicted=model(input_=inputs, conditioning=array.to(device) ,b_size=inputs.shape[0])
        y_predicted=y_predicted.to(device)
        y_truth = torch.stack(a).to(device)
        #y_truth =  torch.unsqueeze(y_truth, 1)

        
        loss_per_batch,running_loss, epoch_loss, acc_train,score = metrics(criterion,
                                                                    y_predicted,
                                                                    y_truth, opt,
                                                                    running_loss,
                                                                    epoch_loss,
                                                                    acc_train,
                                                                    train=True)
        
        i += 1

        if i % 100 ==  99:    # print every X mini-batches

            print(f'[{epoch + 1}, {i :5d}] loss: {loss_per_batch/y_truth.size(0):.3f} running loss:  {epoch_loss/i:.3f}')
            print(f' score: {score :.3f} ')
            running_loss=0.0           

    #update the learning rate
    scheduler.step()

    return i,epoch_loss,acc_train,score

def find_peaks_values(values, all_frequencies):
        peaks = find_peaks(values, threshold=0.00001)[0] #indexes of peaks
        results_half = peak_widths(values, peaks, rel_height=0.5) #4 arrays: widths, y position, initial and final x
        results_half = results_half[0]
        data = values[peaks]
        fre_peaks =all_frequencies[peaks]
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

    # Metrics
    # compute the R2 score
    #cos = nn.CosineSimilarity(dim=0, eps=1e-6)
    #sim = cos(y_predicted, y_truth).mean()
    #mape = mape_loss_fn(y_predicted, y_truth)
    #print(mape)
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
    
    dataloader = utils.get_data_with_labels(parser.image_size,parser.image_size,1, 
                                                boxImagesPath,parser.batch_size,
                                                drop_last=True,
                                                filter="30-40")#filter disabled
    for epoch in range(parser.epochs):
        #load images data
       
        model.train() #in case required for validation

        total,epoch_loss,acc_train,score_train=epoch_train(epoch,model,dataloader,device,opt,scheduler,criterion,clipEmbedder,df)

        print("learning_rate: ",scheduler.get_last_lr())
        print("epoch_loss: ",epoch_loss/total)
        loss_values.append(epoch_loss/total )
        acc.append(acc_train/total)

        torch.save(model.state_dict(), PATH)

    
    return loss_values,acc,valid_loss_list,acc_val,score_train



def main():

    os.environ["PYTORCH_USE_CUDA_DSA"] = "1"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    arguments()
    join_simulationData()  

    fwd_test, opt, criterion,scheduler=get_net_resnet(device,hiden_num=1000,dropout=0.4,features=1000, Y_prediction_size=parser.pred_size)
    #fwd_test, opt, criterion,scheduler=get_net_CNN(device,hiden_num=1000,dropout=0.4,features=1000, Y_prediction_size=parser.pred_size)

    fwd_test = fwd_test.to(device)

    print(fwd_test)

    """option of word embedding"""
    Bert=None

    date="_RESNET152_Bands_7sep_5e-4_200epc_ADAM_6out"
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
        np.savetxt('output/loss_Train_TM_'+date+'.out', loss_values, delimiter=',')
    except:
        np.savetxt('output/loss_Train_TM_'+date+'.out', [], delimiter=',')

    try:
        np.savetxt('output/acc_Train_TM_'+date+'.out', acc, delimiter=',')
    except:
        np.savetxt('output/acc_Train_TM_'+date+'.out', [], delimiter=',')
    
    try:
        np.savetxt('output/loss_Valid_TM_'+date+'.out', valid_loss_list, delimiter=',')
    except:
        np.savetxt('output/loss_Valid_TM_'+date+'.out', [], delimiter=',')
    
    try:
        np.savetxt('output/acc_val_'+date+'.out', acc_val, delimiter=',')
    except:
        np.savetxt('output/acc_val_'+date+'.out', [], delimiter=',')

    try:
        np.savetxt('output/score_train_'+date+'.out', score_train, delimiter=',')
    except:
        np.savetxt('output/score_train_'+date+'.out', [], delimiter=',')

if __name__ == "__main__":
    main()