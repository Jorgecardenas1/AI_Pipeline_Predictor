from __future__ import print_function

import os

#from Utilities.SaveAnimation import Video

from druida import Stack
from druida import setup

from druida.DataManager import datamanager
#from druidaHFSS.modules import tools
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
import torchvision
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import glob
from tqdm.notebook import tqdm
import argparse
import json
from PIL import Image


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


Substrates={"Rogers RT/duroid 5880 (tm)":0}
Materials={"copper":0,"pec":1}
Surfacetypes={"Reflective":0,"Transmissive":1}
TargetGeometries={"circ":0,"box":1, "cross":2}
Bands={"30-40":0,"40-50":1, "50-60":2,"60-70":3,"70-80":4, "80-90":5}



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

    parser.run_name = "Predictor Training"
    parser.epochs = 100
    parser.batch_size = 70
    parser.workers=1
    parser.gpu_number=1
    parser.image_size = 64
    parser.dataset_path = os.path.normpath('/content/drive/MyDrive/Training_Data/Training_lite/')
    parser.device = "cpu"
    parser.learning_rate =2e-5
    parser.metricType='AbsorbanceTM' #this is to be modified when training for different metrics.
    parser.cond_channel=3 #this is to be modified when training for different metrics.
    parser.condition_len=6 #this is to be modified when training for different metrics.
    parser.resnet_arch="resnet152" #this is to be modified when training for different metrics.

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
    
    #torch.nn.init.xavier_uniform_(model.fc.weight) #Fill the input Tensor with values using a Xavier uniform distribution.


    opt = optimizer.Adam(model.parameters(), lr=parser.learning_rate, betas=(0.5, 0.999),weight_decay=1e-4)
    #criterion = nn.CrossEntropyLoss()
    #criterion = nn.L1Loss()
    criterion=nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.99)

    return model, opt, criterion , scheduler


class CLIPTextEmbedder(nn.Module):
    """
    ## CLIP Text Embedder
    """

    def __init__(self, version: str = "openai/clip-vit-large-patch14", device="cuda:0", max_length: int = 50):
        """
        :param version: is the model version
        :param device: is the device
        :param max_length: is the max length of the tokenized prompt
        """
        super().__init__()
        # Load the tokenizer
        self.tokenizer = CLIPTokenizer.from_pretrained(version,device_map = device)
        # Load the CLIP transformer
        self.transformer = CLIPTextModel.from_pretrained(version,device_map = device).eval()
        
        self.transformer.eval()

        self.device = device
        self.max_length = max_length

    def forward(self, prompts: List[str]):
        """
        :param prompts: are the list of prompts to embed
        """
        # Tokenize the prompts
        batch_encoding = self.tokenizer(prompts, truncation=False, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt").to(self.device)
        # Get token ids
        tokens = batch_encoding["input_ids"]
        # Get CLIP embeddings
        return self.transformer(input_ids=tokens).last_hidden_state
    

class BERTTextEmbedde(nn.Module):
    def __init__(self, version: str = "'bert-base-uncased", device="cuda:0", max_length: int = 50):
        """
        :param version: is the model version
        :param device: is the device
        :param max_length: is the max length of the tokenized prompt
        """
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForMaskedLM.from_pretrained('bert-base-uncased', output_hidden_states=True)
        self.model.eval()

        self.device = device
        self.max_length = max_length

    def forward(self, prompts: List[str]):
        """
        :param prompts: are the list of prompts to embed
        """
        tokens = self.tokenizer.tokenize(prompts)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)
        segments_ids = [1] * len(tokens)

        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])

        with torch.no_grad():

            outputs = self.model(tokens_tensor, segments_tensors)

        hidden_states = outputs[1]
        
        # print(hidden_states[0].size())
        # print ("Number of layers:", len(hidden_states), "  (initial embeddings + 12 BERT layers)")
        # layer_i = 0

        # print ("Number of batches:", len(hidden_states[layer_i]))
        # batch_i = 0

        # print ("Number of tokens:", len(hidden_states[layer_i][batch_i]))
        # token_i = 0

        # print ("Number of hidden units:", len(hidden_states[layer_i][batch_i][token_i]))

        return hidden_states

        




# Conditioning
def set_conditioning(bands_batch,freq_val,target,path,categories,clipEmbedder,df,device):
     #one hot encoders
    substrate_encoder=encoders(Substrates)
    materials_encoder=encoders(Materials)
    surfaceType_encoder=encoders(Surfacetypes)
    TargetGeometries_encoder=encoders(TargetGeometries)
    bands_encoder=encoders(Bands)

    arr=[]
    values_array=torch.tensor((), dtype=torch.float)

    for idx,name in enumerate(path):
        series=name.split('_')[-2]
        batch=name.split('_')[4]
        iteration=series.split('-')[-1]
        row=df[(df['sim_id']==batch) & (df['iteration']==int(iteration))  ]


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
            
        else:
        
            sustratoHeight= json.loads(row["paramValues"].values[0])
            sustratoHeight= sustratoHeight[-1]
        

        """this to apply for one hot encoding"""
        # materialsustrato=torch.Tensor(substrate_encoder.transform(np.array(Substrates[layer['substrate']['material']]).reshape(-1, 1)).toarray()).squeeze(0)
        # materialconductor=torch.Tensor(materials_encoder.transform(np.array(Materials[layer['conductor']['material']]).reshape(-1, 1)).toarray()).squeeze(0)
        # surface=torch.Tensor(surfaceType_encoder.transform(np.array(Surfacetypes[surfacekey]).reshape(-1, 1)).toarray()).squeeze(0)
        # band=torch.Tensor(bands_encoder.transform(np.array(band).reshape(-1, 1)).toarray()).squeeze(0)
  

        #values_array = torch.cat((torch.Tensor(geometry),torch.Tensor(surfacetype),torch.Tensor(materialconductor),torch.Tensor(materialsustrato),torch.Tensor([sustratoHeight]),torch.Tensor(band)),0) #concat side
        
        arr.append(torch.Tensor([geometry,surfacetype,materialconductor,materialsustrato,sustratoHeight,band]))
    
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
    break_=False
    
    for data in tqdm(dataloader):
        
        inputs, classes, names, classes_types = data
        #sending to CUDA
        inputs = inputs.to(device)
        classes = classes.to(device)
        
        #Loading data
        a = []        
        opt.zero_grad()
        bands_batch=[]
        max_freqs = []
        """lookup for data corresponding to every image in training batch"""
        for name in names:

            series=name.split('_')[-2]#
            band_name=name.split('_')[-1].split('.')[0]#
            batch=name.split('_')[4]

            for name in glob.glob(DataPath+batch+'/files/'+'/'+parser.metricType+'*'+series+'.csv'): 
                #loading the absorption data
                train = pd.read_csv(name)

                # # the band is divided in chunks 
                if Bands[str(band_name)]==0:
                    
                    train=train.loc[1:100]

                elif Bands[str(band_name)]==1:
                    
                    train=train.loc[101:200]

                elif Bands[str(band_name)]==2:
                    
                    train=train.loc[201:300]

                elif Bands[str(band_name)]==3:
                    
                    train=train.loc[301:400]

                elif Bands[str(band_name)]==4:
                    
                    train=train.loc[401:500]

                elif Bands[str(band_name)]==5:

                    train=train.loc[501:600]
                
                #train=train.loc[401:500]

                values=np.array(train.values.T)
                values=np.around(values, decimals=2, out=None)


                tops, indx = torch.topk(torch.from_numpy(values[1]), 1, largest=True)
                           
                max_val = tops
                max_indx = indx
                all_frequencies=torch.from_numpy(values[0])

                #a.append([max_val,all_frequencies[max_indx]])
                labels_tensor=torch.cat((tops,all_frequencies[max_indx]),0)
                a.append(labels_tensor)

                bands_batch.append(Bands[band_name])
                max_freqs.append(all_frequencies[max_indx])
                

                #Creating the batch of maximum frequencies

        """Creating a conditioning vector"""

        array=set_conditioning(bands_batch,
                                         max_freqs,
                                         classes, 
                                         names, 
                                         classes_types,
                                         clipEmbedder,
                                         df,
                                         device)
        #embedded=embedded.view(parser.batch_size,parser.condition_len)
        #embedded = embedded.mean(1)
        """showing embedding image"""

        # plot =  array.clone().detach().cpu()

        # l1 = nn.Linear(parser.condition_len, parser.image_size*parser.image_size*parser.cond_channel, bias=True)           
        # x2 = l1(plot) #Size must be taken care = 800 in this case
        # m = nn.Tanh()
        # x2 = m(x2)
        # x2 = x2.reshape(int(parser.batch_size),parser.cond_channel,parser.image_size,parser.image_size)
        # save_image(x2[0], str(i)+'onehot_array.png')
        # save_image(inputs[0], str(i)+'_image.png')

        if array.shape[1]==parser.condition_len:

            y_predicted=model(input_=inputs, conditioning=array.to(device) ,b_size=inputs.shape[0])

            y_predicted=y_predicted.to(device)
            y_truth = torch.stack(a).to(device)
            #y_truth =  torch.unsqueeze(y_truth, 1)
            #y_truth = torch.nn.functional.normalize(y_truth, p=1.0, dim=1, eps=1e-12, out=None)

            loss_per_batch,running_loss, epoch_loss, acc_train,score = metrics(criterion,
                                                                        y_predicted,
                                                                        y_truth, opt,
                                                                        running_loss,
                                                                        epoch_loss,
                                                                        acc_train,
                                                                        train=True)
            i += 1

            if i % 100 ==  99:    # print every 2000 mini-batches
            
                #print(y_predicted,y_truth)

                print(f'[{epoch + 1}, {i :5d}] loss: {loss_per_batch/y_truth.size(0):.3f} running loss:  {running_loss/100:.3f}')
                print(f'accuracy: {acc_train/i :.3f} ')
                print(f'Score: {score :.3f} ')
                running_loss=0.0
            #if i % 1000 ==  999:
                

    scheduler.step()
    print("learning_rate: ",scheduler.get_last_lr())

    return i,epoch_loss,acc_train,score




def metrics(criterion,y_predicted,y_truth, opt,running_loss,epoch_loss,acc_train,train=True):

    loss_per_batch=0

    errD_real = criterion(y_predicted.float(), y_truth.float())  

    if train:
        errD_real.backward()
        loss_per_batch=errD_real.item()
        opt.step()
    else:
        loss_per_batch=errD_real.item()

    # Metrics
    # compute the R2 score

    score = r2_score(y_predicted, y_truth)
    acc_train+= score.cpu().numpy() 

    #Loss
    running_loss +=loss_per_batch

    if train:
        epoch_loss+=loss_per_batch

    return loss_per_batch,running_loss, epoch_loss, acc_train,score


def train(opt,scheduler,criterion,model, clipEmbedder,device, PATH):
    #### #File reading conf

    loss_values, valid_loss_list = [], []
    acc=[]
    acc_val=[]

    df = pd.read_csv("out.csv")
    


    dataloader = utils.get_data_with_labels(parser.image_size,parser.image_size,1, 
                                            boxImagesPath,parser.batch_size,
                                            drop_last=True,
                                            filter="30-40")#filter disabled
    
    vdataloader = utils.get_data_with_labels(parser.image_size, parser.image_size,1,
                                            validationImages,parser.batch_size, 
                                            drop_last=True,
                                            filter="30-40")

    for epoch in range(parser.epochs):

        model.train()
        total,epoch_loss,acc_train,score_train=epoch_train(epoch,model,dataloader,device,opt,scheduler,criterion,clipEmbedder,df)

        print("learning_rate: ",scheduler.get_last_lr())
        loss_values.append(epoch_loss/total )
        print("mean Acc per epoch",acc_train/total)
        acc.append(acc_train/total)
        #print("train acc",acc)
            
        """validation"""

        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        torch.save(model.state_dict(), PATH)

    
    return loss_values,acc,valid_loss_list,acc_val,score_train



def main():

    os.environ["PYTORCH_USE_CUDA_DSA"] = "1"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    arguments()
    join_simulationData()  

    fwd_test, opt, criterion,scheduler=get_net_resnet(device,hiden_num=1000,dropout=0.1,features=1000, Y_prediction_size=2)
    fwd_test = fwd_test.to(device)

    print(fwd_test)

    """option of embedding"""
    ClipEmbedder=CLIPTextEmbedder(version= "openai/clip-vit-large-patch14",device=device, max_length = parser.batch_size)
    Bert=BERTTextEmbedde(device=device, max_length = parser.batch_size)


    date="_RESNET152_Bands_11May_2e-5_100epc_h1000_f1000_64_MSE_arrayCond_1TOP1Freq"
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