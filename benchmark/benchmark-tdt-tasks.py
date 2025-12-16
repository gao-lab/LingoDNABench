import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import random
import numpy as np
import copy
import sys
import math
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader,TensorDataset
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, recall_score, precision_score, roc_curve,confusion_matrix,matthews_corrcoef,precision_recall_curve
from sklearn import metrics
from torch.backends import cudnn


task_name=sys.argv[1]
model_name=sys.argv[2]
data_dir=sys.argv[3]
layer=int(sys.argv[4])
random_seed=int(sys.argv[5])
output_dim=int(sys.argv[6])

regression=sys.argv[7]
if regression=="True":
    regression=True
else:
    regression=False
multi_seq=sys.argv[8]
if multi_seq=="True":
    multi_seq=True
else:
    multi_seq=False
# model_name="MS4-512-80"
# data_dir="/lustre/grp/gglab/liangyx/data/dnalingo_dev/benchmark_dataset/CRE_activity/HepG2"
# layer=11
# task_name="CRE_HepG2"
# regression=True
# multi_seq=False

# output_dim=1
# random_seed=42
num_targets=output_dim
seed = random_seed
cudnn.benchmark = False            # if benchmark=True, deterministic will be False
cudnn.deterministic = True
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)            # 为CPU设置随机种子
torch.cuda.manual_seed(seed)       # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed)   # 为所有GPU设置随机种子
'''devices'''
device=torch.device('cuda')




save_dir=f"{data_dir}/{model_name}/checkpoint"
if os.path.exists(save_dir):
    pass
else:
    os.mkdir(save_dir)


'''dataset'''
if multi_seq:
    pass
else:
    if regression:
        train_label=np.loadtxt(data_dir+'/train_label.txt',dtype=float)
        valid_label=np.loadtxt(data_dir+'/dev_label.txt',dtype=float)
        test_label=np.loadtxt(data_dir+'/test_label.txt',dtype=float)
    else:
        train_label=np.loadtxt(data_dir+'/train_label.txt',dtype=int)
        valid_label=np.loadtxt(data_dir+'/dev_label.txt',dtype=int)
        test_label=np.loadtxt(data_dir+'/test_label.txt',dtype=int)
    train_label=torch.as_tensor(train_label)
    valid_label=torch.as_tensor(valid_label)
    test_label=torch.as_tensor(test_label)
    train_data=np.load(f"{data_dir}/{model_name}/train_data_0-embedding-layer_{layer}.npy")
    valid_data=np.load(f"{data_dir}/{model_name}/dev_data_0-embedding-layer_{layer}.npy")
    test_data=np.load(f"{data_dir}/{model_name}/test_data_0-embedding-layer_{layer}.npy")
    train_data=torch.as_tensor(train_data)
    valid_data=torch.as_tensor(valid_data)
    test_data=torch.as_tensor(test_data)
    if len(valid_label.shape)==1:
        train_dataset=TensorDataset(train_data,train_label.unsqueeze(1))
        valid_dataset=TensorDataset(valid_data,valid_label.unsqueeze(1))
        test_dataset=TensorDataset(test_data,test_label.unsqueeze(1))
    else:
        train_dataset=TensorDataset(train_data,train_label)
        valid_dataset=TensorDataset(valid_data,valid_label)
        test_dataset=TensorDataset(test_data,test_label)

def testing(test_dataloader, model, num_targets):
    model.eval().cuda()    
    # y_test=np.empty((len(test_dataloader.dataset),num_targets))
    y_pred=np.empty((len(test_dataloader.dataset),num_targets))
    begin=0
    with torch.no_grad():
        for one_batch in test_dataloader:
            if multi_seq:
                pass
            else:
                DNA ,targets = map(lambda x: x.to(device), one_batch)
                DNA=DNA.to(torch.float32)
                target_pred = model(DNA)
                num=DNA.shape[0]
                targets=targets.to(torch.float32)
                target_pred=target_pred.to(torch.float32)
                if not output_dim==1:
                    targets=targets.to(torch.int32)
                    targets=torch.tensor(targets.squeeze(1),dtype=torch.long)
                    target_pred = nn.Softmax(dim=1)(target_pred)
                
                # y_test[begin:begin+num,:]=targets.cpu().detach().numpy()
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
            if multi_seq:
                pass
            else:
                DNA ,targets = map(lambda x: x.to(device), one_batch)
                DNA=DNA.to(torch.float32)
                target_pred = model(DNA)
                target_pred=target_pred.to(torch.float32)
                targets=targets.to(torch.float32)
                if not output_dim==1:
                    targets=targets.to(torch.int32)
                    targets=torch.tensor(targets.squeeze(1),dtype=torch.long)
                    # target_pred = nn.Softmax(dim=1)(target_pred)
                
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

batch_size=64
train_dataloader = DataLoader(train_dataset,shuffle=True, batch_size=batch_size,)
valid_dataloader = DataLoader(valid_dataset,shuffle=True, batch_size=batch_size)
test_dataloader = DataLoader(test_dataset,shuffle=False, batch_size=batch_size, num_workers=0)
if multi_seq:
    pass
else:
    model=DownstreamModel(in_channels=valid_data.shape[2])
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
        
        if multi_seq:
            pass
        else:
            DNA ,targets = map(lambda x: x.to(device), one_batch)
            DNA=DNA.to(torch.float32)
            target_pred = model(DNA)
            target_pred=target_pred.to(torch.float32)
            targets=targets.to(torch.float32)
            if not output_dim==1:
                targets=targets.to(torch.int32)
                targets=torch.tensor(targets.squeeze(1),dtype=torch.long)
                # target_pred = nn.Softmax(dim=1)(target_pred)
                
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


test_model=DownstreamModel(in_channels=valid_data.shape[2])
test_model.to(device)
test_model.load_state_dict(final_data['model_state_dict'])

y_pred=testing(test_dataloader,test_model,output_dim)


if regression:
    y_test=test_label.detach().cpu().numpy()
    import scipy.stats
    pearson_v=scipy.stats.pearsonr(y_test.reshape(-1),y_pred.reshape(-1))[0]
    spearman_v=scipy.stats.spearmanr(y_test.reshape(-1),y_pred.reshape(-1))[0]
    with open(f"{save_dir}/metrics-{layer}.txt",'a') as f:
        print(f"{model_name}\t{random_seed}\t{spearman_v}\t{pearson_v}\n")
        f.write(f"{model_name}\t{random_seed}\t{spearman_v}\t{pearson_v}\n")
else:
    if output_dim==1:
        y_test=test_label.flatten().cpu().detach().numpy()
        y_pred=y_pred.reshape(-1)
        with open(f"{save_dir}/metrics-{layer}.txt",'a') as f:
            
            auc=roc_auc_score(y_test,y_pred) 
            print(auc)
            y_prediction=y_pred
            y_true=y_test
            precision, recall, _ = precision_recall_curve(y_true, y_prediction)
            pr_auc = metrics.auc(recall, precision)
            fpr, tpr, thresholds = roc_curve(y_true,y_prediction)
            youden = tpr-fpr
            cutoff = thresholds[np.argmax(youden)]
            y_prediction[y_prediction<cutoff]=0
            y_prediction[y_prediction>=cutoff]=1
            recall=recall_score(y_true,y_prediction)
            accuracy=accuracy_score(y_true,y_prediction)
            precision=precision_score(y_true,y_prediction)
            mcc=matthews_corrcoef(y_true,y_prediction)
            f1=f1_score(y_true,y_prediction)
            f.write(f"{model_name}\t{random_seed}\t{cutoff}\t{accuracy}\t{precision}\t{recall}\t{f1}\t{auc}\t{pr_auc}\t{mcc}\n")
    else:
        y_test=test_label.detach().cpu().numpy()
        auc=roc_auc_score(y_test,y_pred,multi_class='ovo')
        with open(f"{save_dir}/metrics-{layer}.txt",'a') as f:
            f.write(f"{model_name}\t{random_seed}\t{auc}\n")

# np.save(save_dir+f"/test-label.npy",y_test)
np.save(save_dir+f"/test-{model_name}_{layer}_{random_seed}_pred.npy",y_pred)
torch.save(final_data,f"{save_dir}/checkpoint-{model_name}_{layer}_{random_seed}.pt")
