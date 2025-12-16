import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import random
import numpy as np
import copy
import h5py
import sys
import math
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader,TensorDataset,Dataset
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, recall_score, precision_score, roc_curve,confusion_matrix,matthews_corrcoef,precision_recall_curve
from sklearn import metrics
from torch.backends import cudnn
import scipy.stats

task_name=sys.argv[1]
model_name=sys.argv[2]
data_dir=sys.argv[3]
layer=int(sys.argv[4])
random_seed=int(sys.argv[5])
output_dim=int(sys.argv[6])

task_id="default"
regression=sys.argv[7]
if regression=="True":
    regression=True
else:
    regression=False

num_targets=output_dim
seed = random_seed
cudnn.benchmark = False            # if benchmark=True, deterministic will be False
cudnn.deterministic = True
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)           
torch.cuda.manual_seed(seed)       
torch.cuda.manual_seed_all(seed)  
'''devices'''
device=torch.device('cuda')




save_dir=f"{data_dir}/{model_name}/checkpoint"
if os.path.exists(save_dir):
    pass
else:
    os.makedirs(save_dir)


'''dataset'''

# dataset
class H5Dataset(Dataset):
    def __init__(self, dataset_type, data_dir ,output_dim=None,task_name='exp'):
        super().__init__()
        # check dataset_type is train, test or dev
        if dataset_type not in ["train", "test", "dev"]:
            raise ValueError("dataset_type must be either 'train', 'test' or 'dev'")
        # load data
        if task_name=='exp':
            # self.data_x_all = np.load(data_x_path + dataset_type + ".npy")
            self.data_x_all = np.load(data_dir+ f"/{model_name}" + f"/{dataset_type}" + f"_seq-embedding-layer_{layer}.npy")
            # self.data_x_all = h5py.File(data_dir+ f"/{model_name}" + f"/{dataset_type}" + f"_seq-embedding-layer_{layer}.h5", "r")["embedding"][:]
            self.data_y_all = np.load(f"{data_dir}" + f"/{dataset_type}" + "_target.npy")
        self.data_shape=self.data_x_all.shape
    def __len__(self):
        return self.data_y_all.shape[0]
    def __getitem__(self, idx):
        data_x = self.data_x_all[idx]
        data_y = self.data_y_all[idx]
        return torch.tensor(data_x), torch.tensor(data_y)

train_dataset=H5Dataset("train",data_dir)
valid_dataset=H5Dataset("dev",data_dir)
test_dataset=H5Dataset("test",data_dir)



def testing(test_dataloader, model, num_targets):
    model.eval().cuda()    
    y_pred=np.empty((len(test_dataloader.dataset),num_targets))
    begin=0
    with torch.no_grad():
        for one_batch in test_dataloader:
            
            DNA ,targets = map(lambda x: x.to(device), one_batch)
            DNA=DNA.to(torch.float32)
            target_pred = model(DNA)
            num=DNA.shape[0]
            targets=targets.to(torch.float32)
            target_pred=target_pred.to(torch.float32)
            y_pred[begin:begin+num,:]=target_pred.cpu().detach().numpy()
            begin+=num
    return y_pred

def validating(valid_dataloader, model, loss_fn):
    num_batches = len(valid_dataloader)
    model.eval().cuda()
    test_loss = 0
    step=0
    print("validating")    
    with torch.no_grad():
        for one_batch in valid_dataloader:
            
            DNA ,targets = map(lambda x: x.to(device), one_batch)
            DNA=DNA.to(torch.float32)
            target_pred = model(DNA)
            target_pred=target_pred.to(torch.float32)
            targets=targets.to(torch.float32)
            
            loss_lm = loss_fn(target_pred, targets)
            test_loss += loss_lm.detach().item()
            step+=1
    test_loss /= num_batches
    return test_loss

class DownstreamModel(nn.Module):
    def __init__(self,in_channels):
        super(DownstreamModel,self).__init__()
        self.batchnorm=nn.BatchNorm1d(in_channels)
        self.conv1=nn.Conv1d(in_channels,256,1)
        self.conv2=nn.Conv1d(256,256,7,padding='same')
        self.act=nn.GELU()
        self.linear1=nn.Linear(in_features=256,out_features=512)
        self.drop=nn.Dropout(0.2)
        self.linear2=nn.Linear(in_features=512,out_features=256)
        self.linear3=nn.Linear(in_features=256,out_features=output_dim)
    def forward(self,x):
        x=x.permute(0,2,1)
        #convolutions
        x=self.batchnorm(x)
        x=self.conv1(x)
        x=self.conv2(x)
        x=self.act(x)
        #pooling
        x=torch.nn.functional.max_pool1d(x,kernel_size=x.shape[2])
        #flatten
        x=x.view(x.shape[0],-1)
        x=self.linear1(x)
        x=self.act(x)
        x=self.drop(x)        
        x=self.linear2(x)
        x=self.act(x)
        x=self.drop(x)  
        x=self.linear3(x)
        if (output_dim==1) and (not regression):
            x=torch.sigmoid(x)
        return x



class Conv_block(nn.Module):
    def __init__(self,in_channels, out_channels,conv_kernel_size,conv_padding,pool=True,pool_size=0):
        super(Conv_block,self).__init__()
        conv_layer=nn.Conv1d(in_channels, out_channels, kernel_size=conv_kernel_size,padding=conv_padding).float()
        batchnorm=nn.BatchNorm1d(out_channels).float()
        gelu=nn.GELU()
        self.blk=nn.Sequential(
            conv_layer,
            batchnorm,
            gelu
        )
        self.pool_layer=nn.MaxPool1d(pool_size,pool_size)
        self.pool=pool
    def forward(self,X):
        X=self.blk(X)
        if self.pool:
            X=self.pool_layer(X)
        return X


batch_size=128
train_dataloader = DataLoader(train_dataset,shuffle=True, batch_size=batch_size,num_workers=8)
valid_dataloader = DataLoader(valid_dataset,shuffle=True, batch_size=batch_size,num_workers=8)
test_dataloader = DataLoader(test_dataset,shuffle=False, batch_size=batch_size, num_workers=0)

model=DownstreamModel(in_channels=valid_dataset.data_shape[2])
# model=Basenji_Basset()
model.to(device)

learning_rate=1e-3
epochs=80
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
if regression:
    loss_fn=torch.nn.MSELoss()
else:
    if output_dim==1:
        loss_fn=torch.nn.BCELoss()
    else:
        loss_fn=torch.nn.CrossEntropyLoss()

patience=10
count=0
min_valid_loss=1000
final_data={}
for epoch in range(epochs):
    model.train()
    train_epoch_loss = 0
    train_num_batches = len(train_dataloader)
    step=0
    for one_batch in train_dataloader:
        DNA ,targets = map(lambda x: x.to(device), one_batch)
        DNA=DNA.to(torch.float32)
        target_pred = model(DNA)
        target_pred=target_pred.to(torch.float32)
        targets=targets.to(torch.float32)
        
                
        loss_lm = loss_fn(target_pred, targets)
        train_epoch_loss += loss_lm.detach().item()
        print(loss_lm.detach().item())
        loss_lm.backward()
        optimizer.step()
        optimizer.zero_grad()
        step += 1
    #train epoch loss
    train_epoch_loss /= train_num_batches
    print(f"train_loss\t{train_epoch_loss}")
    #validating
    valid_loss = validating(valid_dataloader, model, loss_fn)
    print(f"valid_loss\t{valid_loss}")
    current_state={'epoch': epoch,
                    'model_state_dict': copy.deepcopy(model.state_dict()),
                    'optimizer_state_dict': copy.deepcopy(optimizer.state_dict()),
                    'step': step,
                    'train_loss': train_epoch_loss,
                    'valid_loss': valid_loss,
                    }
    
    if valid_loss<min_valid_loss:
        print('update')
        min_valid_loss=valid_loss
        final_data=copy.deepcopy(current_state)
        count=0
    
    count+=1
    if count==patience:
        break


test_model=DownstreamModel(in_channels=valid_dataset.data_shape[2])
test_model.to(device)
test_model.load_state_dict(final_data['model_state_dict'])

y_pred=testing(test_dataloader,test_model,output_dim)

def evaluate_test(test_true, test_pred):
    corr_list = []
    pear_list=[]
    for x in range(53):
        corr,_ = scipy.stats.spearmanr(test_true[:,x], test_pred[:,x])
        pearson,_ = scipy.stats.pearsonr(test_true[:,x], test_pred[:,x])
        corr_list.append(corr)
        pear_list.append(pearson)
    return corr_list,pear_list





y_test=test_dataset.data_y_all
spearman_list,pearson_list=evaluate_test(y_test,y_pred)
spearman_list=[str(elem) for elem in spearman_list]
pearson_list=[str(elem) for elem in pearson_list]

with open(f"{save_dir}/{task_id}-metrics-{layer}.txt",'a') as f:
    
    spearman_v='\t'.join(spearman_list)
    pearson_v='\t'.join(pearson_list)
    f.write(f"{model_name}\t{random_seed}\tspearmanR\t{spearman_v}\n")
    f.write(f"{model_name}\t{random_seed}\tpearsonR\t{pearson_v}\n")

np.save(save_dir+f"/{task_id}-test-{model_name}_{layer}_{random_seed}_pred.npy",y_pred)
torch.save(final_data,f"{save_dir}/{task_id}-checkpoint-{model_name}_{layer}_{random_seed}.pt")
