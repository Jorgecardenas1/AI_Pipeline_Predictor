
from __future__ import print_function

import os

#from Utilities.SaveAnimation import Video

from druida import Stack
from druida import setup

from druida.DataManager import datamanager
from druidaHFSS.modules import tools
from druida.tools import utils


#import lightning as L
import matplotlib
import matplotlib.pyplot as plt
import matplotlib_inline.backend_inline
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torch.optim as optimizer
from torcheval.metrics.functional import r2_score
from torchvision.utils import save_image
from transformers import BertTokenizer, BertModel, BertConfig,BertForMaskedLM
from typing import List

import torchvision

from torchvision import transforms

import glob
from tqdm.notebook import tqdm
import random
import numpy as np
import pandas as pd

from IPython.display import HTML
import json

import argparse

torch.manual_seed(1919)

# Path to the folder where the datasets are/should be downloaded (e.g. CIFAR10)
DATASET_PATH = os.environ.get("PATH_DATASETS", "data/")
# Path to the folder where the pretrained models are saved
CHECKPOINT_PATH = os.environ.get("PATH_CHECKPOINT", "saved_models/VisionTransformers/")

os.environ["PYTORCH_USE_CUDA_DSA"] = "1"

# Setting the seed
#L.seed_everything(42)

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
#print("Device:", device)


parser = argparse.ArgumentParser()

boxImagesPath="../../../data/MetasufacesData/Images-512-Bands/"
#boxImagesPath="../../../data/MetasufacesData/Images-512-Suband/"
DataPath="../../../data/MetasufacesData/Exports/output/"
simulationData="../../../data/MetasufacesData/DBfiles/"
validationImages="../../../data/MetasufacesData/testImages/"


Substrates={"Rogers RT/duroid 5880 (tm)":0}
Materials={"copper":0,"pec":1}
Surfacetypes={"Reflective":0,"Transmissive":1}
TargetGeometries={"circ":0,"box":1, "cross":2}
metricType=['AbsorbanceTM','AbsorbanceTE' ]
Bands={"30-40":0,"40-50":1, "50-60":2,"60-70":3,"70-80":4, "80-90":5}

categories=["box", "circle", "cross"]

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
    parser.add_argument("patch_size",type=int)
    parser.add_argument("image_size",type=int)
    parser.add_argument("regression",type=int)

    parser.run_name = "Predictor Training"
    parser.epochs = 50
    parser.batch_size = 50
    parser.workers=1
    parser.gpu_number=1
    parser.image_size = 64
    parser.dataset_path = os.path.normpath('/content/drive/MyDrive/Training_Data/Training_lite/')
    parser.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    parser.learning_rate = 5e-5
    parser.condition_len = 6
    parser.metricType='AbsorbanceTM' #this is to be modified when training for different metrics.
    parser.patch_size=16
    parser.regression=True


    model_kwargs={
            "batch_size":parser.batch_size,
            "embed_dim":  3 * (parser.patch_size)**2 ,
            "hidden_dim":  2*(3* (parser.patch_size)**2),
            "num_heads": 8, #probar menos
            "num_layers": 4, #probar menos
            "patch_size": parser.patch_size,
            "num_channels": 3,
            "con_num_channels": 3,
            "num_patches": (parser.image_size//parser.patch_size)**2,
            "num_classes": 2,
            "dropout": 0.2,
            "image_size":parser.image_size,
            "conditionalIn":True,
            "conditionalLen":6,
            "regression":parser.regression

        }


    return model_kwargs




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
        inputs = inputs.to(device)
        classes = classes.to(device)
        
        #Loading data
        a = []        
        opt.zero_grad()
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


                values=np.around(values, decimals=2, out=None)

                max_val = np.max(values[1])
                max_indx = np.argmax(values[1])
                all_frequencies=values[0]

                a.append([max_val,all_frequencies[max_indx]])
                bands_batch.append(Bands[band_name])

                #Creating the batch of maximum frequencies

        a=np.array(a) 

        """Creating a conditioning vector"""
        
        conditioningArray, embedded=set_conditioning(bands_batch,all_frequencies,classes, names, classes_types,clipEmbedder,df,device)
        
        #print(embedded)
        #embedded=embedded.view(parser.batch_size,parser.condition_len)
        #embedded = embedded.mean(1)

        """showing embedding image"""

        # plot =  conditioningArray.clone().detach().cpu()
        # l1 = nn.Linear(parser.condition_len, parser.image_size*parser.image_size*3, bias=False)           
        # x2 = l1(plot) #Size must be taken care = 800 in this case
        # x2 = x2.reshape(int(parser.batch_size),3,parser.image_size,parser.image_size)
        # x2 = torchvision.transforms.Normalize([0.1, ], [.1 ],[0.4,])(x2)
        # save_image(x2[0], str(i)+'_No.png')
        # save_image(inputs[0], str(i)+'_image.png')


        y_predicted = model(inputs,condition=conditioningArray.to(device))
        y_predicted=y_predicted.to(device)
                                    
        y_truth = torch.tensor(a).to(device)
        #print(y_predicted[1])
        #print(y_truth)
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
    
def train(model,device, PATH):
    #### #File reading conf

    df = pd.read_csv("out.csv")

    """using weigth decay regularization"""
    opt = optimizer.Adam(model.parameters(), lr=parser.learning_rate, betas=(0.9, 0.98),weight_decay=1e-4)
    
    if parser.regression:
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()
        
    #Bert=BERTTextEmbedde(device=device, max_length = parser.batch_size)
    Bert=[] #just empty when not used
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.98)

    loss_values, valid_loss_list = [], []
    acc=[]
    acc_val=[]


    # prepare model for training
    dataloader = utils.get_data_with_labels(parser.image_size, parser.image_size,1, boxImagesPath,parser.batch_size, drop_last=True,filter="30-40")
    #vdataloader = utils.get_data_with_labels(parser.image_size, parser.image_size,1, validationImages,parser.batch_size, drop_last=True)

    for epoch in range(parser.epochs):
            
        total,epoch_loss,acc_train,score_train=epoch_train(epoch,model,dataloader,device,opt,scheduler,criterion,Bert,df)

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



           
# Conditioning
def set_conditioning(bands_batch,freq_val,target,path,categories,Embedder,df,device):
    
    arr=[]
    values_array=[]
    intermediate_array=[]
    for idx,name in enumerate(path):

        series=name.split('_')[-2]
        batch=name.split('_')[4]
        iteration=series.split('-')[-1]
        row=df[(df['sim_id']==batch) & (df['iteration']==int(iteration))  ]
        #print(batch)
        #print(iteration)

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
        

        """If using text embedding then .--"""
        datos = ""
        datos=", ".join([str(element) for element in  ["Geometry is:"+str(geometry),"Surface type is:"+str(surfacetype),"Material conductor is:"+str(materialconductor),"Substrate is:"+str(materialsustrato),"with a height of "+str(sustratoHeight),"the band:"+str(band),"the frequency:"+str(freq_val[idx])]])
        datos = "[CLS] " + datos + " [SEP]"
        
        intermediate_array.append(torch.Tensor([geometry,surfacetype,materialconductor,materialsustrato,sustratoHeight,band]))
    
        embedding=0
        #embedding=Embedder(prompts=datos)   

        """clip"""
        #embedding=embedding[:,0:50:,:]
        """Bert"""
        #embedding=embedding[0][:,0:50:,:]

        #arr.append( embedding)
    
    #embedding = torch.cat(arr, dim=0)
    values_array = torch.stack(intermediate_array)

    """ Values array solo pouede llenarse con n+umero y no con textos"""
    return values_array, embedding



def join_simulationData():
    df = pd.DataFrame()
    for file in glob.glob(simulationData+"*.csv"): 
        df2 = pd.read_csv(file)
        df = pd.concat([df, df2], ignore_index=True)
    
    df.to_csv('out.csv',index=False)

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    print("Access main")

    model_kwargs=arguments()

    join_simulationData()  

    vision_transformer = Stack.VisionTransformer(**model_kwargs )
    vision_transformer.to(device)
    vision_transformer.train()
    print(vision_transformer)

    date="14May_tunedLRADAM_MSE_5e-5_6Cond_out1T1F"
    PATH = 'VITtrainedModelTM_abs_'+date+'.pth'

    loss_values,acc,valid_loss_list,acc_val,score_train=train(vision_transformer,device, PATH)
    
    torch.save(vision_transformer.state_dict(), PATH)

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