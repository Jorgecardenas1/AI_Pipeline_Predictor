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

import torch
import torch.nn as nn
import torch.optim as optimizer

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



# Clip 
from typing import List

from torch import nn
from transformers import CLIPTokenizer, CLIPTextModel

torch.set_printoptions(profile="full")
torch.manual_seed(999)

#RESNET
from torchvision.models import resnet50, ResNet50_Weights




# Arguments
parser = argparse.ArgumentParser()
# boxImagesPath="\\data\\francisco_pizarro\\jorge-cardenas\\data\\MetasufacesData\\Images Jorge Cardenas 512\\"
# DataPath="\\data\\francisco_pizarro\\jorge-cardenas\\data\\MetasufacesData\\Exports\\output\\"
# simulationData="\\data\\francisco_pizarro\\jorge-cardenas\\data\\MetasufacesData\\DBfiles\\"

boxImagesPath="../../../data/MetasufacesData/Images-512-Bands/"
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
    parser.add_argument("condition_len",type=float) #This defines the length of our conditioning vector
    parser.add_argument("metricType",type=float) #This defines the length of our conditioning vector

    parser.run_name = "Predictor Training"
    parser.epochs = 100
    parser.batch_size = 20
    parser.workers=0
    parser.gpu_number=1
    parser.image_size = 512
    parser.dataset_path = os.path.normpath('/content/drive/MyDrive/Training_Data/Training_lite/')
    parser.device = "cpu"
    parser.learning_rate =3e-5
    parser.condition_len = 768
    parser.metricType='AbsorbanceTM' #this is to be modified when training for different metrics.

    categories=["box", "circle", "cross"]


# Images loading
    

def load_images():
    pass

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# Data pre-processing
def join_simulationData():
    df = pd.DataFrame()
    for file in glob.glob(simulationData+"*.csv"): 
        df2 = pd.read_csv(file)
        df = pd.concat([df, df2], ignore_index=True)
    
    df.to_csv('out.csv',index=False)
    

# Load Model
def loadModel(device):

    fwd_test = Stack.Predictor_CNN(cond_input_size=parser.condition_len, 
                                ngpu=1, image_size=parser.image_size ,
                                output_size=8, channels=3,
                                features_num=3000,hiden_num=1000, #Its working with hiden nums. Features in case and extra linear layer
                                dropout=0.1, 
                                Y_prediction_size=601) #size of the output vector in this case frenquency points
    
    fwd_test.apply(weights_init)

    """using weigth decay regularization"""
    opt = optimizer.Adam(fwd_test.parameters(), lr=parser.learning_rate, betas=(0.5, 0.999),weight_decay=1e-4)
    #criterion = nn.CrossEntropyLoss()
    criterion=nn.MSELoss()

    return fwd_test, opt, criterion

def get_net_resnet(device,hiden_num=1000,dropout=0.1,features=3000, Y_prediction_size=601):
    model = Stack.Predictor_RESNET(cond_input_size=parser.condition_len, 
                                ngpu=1, image_size=parser.image_size ,
                                output_size=8, channels=3,
                                features_num=features,hiden_num=hiden_num, #Its working with hiden nums. Features in case and extra linear layer
                                dropout=dropout, 
                                Y_prediction_size=Y_prediction_size) #size of the output vector in this case frenquency points
    
    #torch.nn.init.xavier_uniform_(model.fc.weight) #Fill the input Tensor with values using a Xavier uniform distribution.


    opt = optimizer.Adam(model.parameters(), lr=parser.learning_rate, betas=(0.5, 0.999),weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    #criterion = nn.L1Loss()
    #criterion=nn.MSELoss()
    return model, opt, criterion 


class CLIPTextEmbedder(nn.Module):
    """
    ## CLIP Text Embedder
    """

    def __init__(self, version: str = "openai/clip-vit-large-patch14", device="cuda:0", max_length: int = 77):
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

        self.device = device

        print(self.device)
        self.max_length = max_length

    def forward(self, prompts: List[str]):
        """
        :param prompts: are the list of prompts to embed
        """
        # Tokenize the prompts
        batch_encoding = self.tokenizer(prompts, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt").to(self.device)
        # Get token ids
        tokens = batch_encoding["input_ids"]
        # Get CLIP embeddings
        return self.transformer(input_ids=tokens).last_hidden_state


# Conditioning
def set_conditioning(bands_batch,target,path,categories,clipEmbedder,df,device):
    
    arr=[]

    for idx,name in enumerate(path):
        #print(name)
        series=name.split('_')[-2]#.split('.')[0]
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
        surfacetype=row["type"].values[0]
        surfacetype=Surfacetypes[surfacetype]
        
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
        
        arr.append([geometry,surfacetype,materialconductor,materialsustrato,sustratoHeight,band,1,1,1,1])

        datos=" ".join([str(element) for element in  [geometry,surfacetype,materialconductor,materialsustrato,sustratoHeight,band,1,1,1,1]])
        embedding=clipEmbedder(prompts=(datos))
        
    return arr, embedding



def train(opt,criterion,model, clipEmbedder,device, PATH):
    #### #File reading conf

    a = []

    loss_per_batch=0
    loss_per_val_batch=0
    loss_values, valid_loss_list = [], []
    acc=[]
    acc_val=[]

    df = pd.read_csv("out.csv")
    
    dataloader = utils.get_data_with_labels(parser.image_size,parser.image_size,1, boxImagesPath,parser.batch_size,drop_last=True)
    vdataloader = utils.get_data_with_labels(parser.image_size, parser.image_size,1, validationImages,parser.batch_size, drop_last=True)

    for epoch in range(parser.epochs):



        i=0 #iteration
        i_val=0 #running over validation set

        running_loss = 0. 
        epoch_loss = 0.
        running_vloss = 0.0 #over validation set
        total_correct = 0
        acc_validation=0.0
        acc_train=0.0
        
        total_samples=0
        total_samples_val=0.0
        bands_batch=[]

        print('Epoch {}/{}'.format(epoch, parser.epochs - 1))
        print('-' * 10)

        model.train()

        for data in tqdm(dataloader):
            
            inputs, classes, names, classes_types = data

            #sending to CUDA
            inputs = inputs.to(device)
            classes = classes.to(device)
            
            opt.zero_grad()

            #Loading data
            a = []
            bands_batch=[]
            """lookup for data corresponding to every image in training batch"""
            for name in names:
                series=name.split('_')[-2]#
                band_name=name.split('_')[-1].split('.')[0]#

                batch=name.split('_')[4]

                for name in glob.glob(DataPath+batch+'/files/'+'/'+parser.metricType+'*'+series+'.csv'): 
                    
                    #loading the absorption data
                    train = pd.read_csv(name)
                    
                    # the band is divided in chunks 
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
                    
                    
                    values=np.array(train.values.T)
                    a.append(values[1])

                    bands_batch.append(Bands[str(band_name)])

                    
            a=np.array(a) 

            """Creating a conditioning vector"""
            
            _, embedded=set_conditioning(bands_batch,classes, names, classes_types,clipEmbedder,df,device)
            
            conditioningTensor = torch.nn.functional.normalize(embedded, p=2.0, dim = 1)
            #conditioningArray=torch.FloatTensor(array)
            
            if embedded.shape[2]==parser.condition_len:
                pass
            
                y_predicted=model(input_=inputs, conditioning=conditioningTensor.to(device) ,b_size=inputs.shape[0])
                y_predicted=torch.nn.functional.normalize(y_predicted, p=2.0, dim = 1)
                y_predicted=y_predicted.to(device)
                
                y_truth = torch.tensor(a).to(device)
                
                errD_real = criterion(y_predicted.float(), y_truth.float())  
                
                errD_real.backward()
                loss_per_batch=errD_real.item()
                opt.step()
    
                # Metrics
                # Accuracy
                
                #predicted = torch.max(y_predicted, 1) #indice del máximo  
                vals, idx_pred = y_predicted.topk(50,dim=1)  
                vals, idx_truth = y_truth.topk(50, dim=1)  
                
                total_truths=0

                for idx,val in enumerate(idx_pred):
                    for item in val:
                        if item in idx_truth[idx]:
                            total_truths+=1

                #print(total_truths)    
                total_samples=idx_truth.size(0)*50

                acc_train+=total_truths/total_samples
                #print(acc_train)

                #Loss
                running_loss +=loss_per_batch*y_truth.size(0)
                epoch_loss+=loss_per_batch*y_truth.size(0)

                i += 1

                if i % 100 ==  99:    # print every 2000 mini-batches
                    print(f'[{epoch + 1}, {i :5d}] loss: {loss_per_batch/y_truth.size(0):.3f} running loss:  {running_loss/100:.3f}')
                    print(f'accuracy: {acc_train/i :.3f} ')
                    running_loss=0.0

        loss_values.append(epoch_loss/i )
        print("mean Acc per epoch",acc_train/len(dataloader))
        acc.append(acc_train/len(dataloader))
            #print("train acc",acc)
            

        """validation"""

        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        torch.save(model.state_dict(), PATH)

        model.eval()


        with torch.no_grad():
            for vdata in tqdm(vdataloader):
                images, classes, names, classes_types  = vdata
                

                images = images.to(device)
                classes = classes.to(device)

            
                a = [] #array with truth values
                bands_batch =[]

                """lookup for data corresponding to every image in training batch"""
                for name in names:
                    series=name.split('_')[-2]#
                    band_name=name.split('_')[-1].split('.')[0]#

                    batch=name.split('_')[4]

                    for name in glob.glob(DataPath+batch+'/files/'+'/'+parser.metricType+'*'+series+'.csv'): 
                        
                        #loading the absorption data
                        train = pd.read_csv(name)
                        
                        # the band is divided in chunks 
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
                        
                        
                        values=np.array(train.values.T)
                        a.append(values[1])

                        bands_batch.append(band_name)

                        
                a=np.array(a) 

                """Creating a conditioning vector"""
                
                _, embedded=set_conditioning(bands_batch,classes, names, classes_types,clipEmbedder,df,device)

                conditioningTensor = torch.nn.functional.normalize(embedded, p=2.0, dim = 1)

                y_predicted=model(input_=inputs, conditioning=conditioningTensor.to(device) ,b_size=inputs.shape[0])
                y_predicted=torch.nn.functional.normalize(y_predicted, p=2.0, dim = 1)

                #Scaling and normalizing

                y_predicted=y_predicted.to(device)
                y_truth = torch.tensor(a).to(device)

                loss_per_val_batch = criterion(y_predicted.float(), y_truth.float())


                #predicted = torch.max(y_predicted, 1) #indice del máximo  
                vals, idx_pred = y_predicted.topk(50,dim=1)  
                vals, idx_truth = y_truth.topk(50, dim=1) 

                total_correct += (idx_pred == idx_truth).sum().item()
            
                total_samples_val += y_truth.size(0)*50
                acc_validation = total_correct / total_samples_val

            #Loss
                running_vloss += loss_per_val_batch.item()*y_truth.size(0)
                i_val+=1

            valid_loss_list.append(running_vloss/i_val)
            acc_val.append(acc_validation)
    
    return loss_values,acc,valid_loss_list,acc_val





def main():

    os.environ["PYTORCH_USE_CUDA_DSA"] = "1"

    print("Access main")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    arguments()
    join_simulationData()  

    fwd_test, opt, criterion=get_net_resnet(device,hiden_num=500,dropout=0.1,features=1000, Y_prediction_size=100)
    fwd_test = fwd_test.to(device)
    print(fwd_test)

    ClipEmbedder=CLIPTextEmbedder(version= "openai/clip-vit-large-patch14",device=device, max_length = parser.batch_size)


    date="_RESNET_Bands_16Abr_3e-5_100epc_h500_f1000_512_CE"
    PATH = 'trainedModelTM_abs_'+date+'.pth'

    loss_values,acc,valid_loss_list,acc_val=train(opt,criterion,fwd_test,ClipEmbedder,device,PATH )


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

if __name__ == "__main__":
    main()