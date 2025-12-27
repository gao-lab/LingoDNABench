import pandas as pd
import numpy as np
import os
import Bio
from Bio import SeqIO
import subprocess
import sys
from multiprocessing.pool import Pool
feature_name=sys.argv[1]
data_dir=sys.argv[2]
split_dir=sys.argv[3]
num_targets=int(sys.argv[4])
split_size=int(sys.argv[5])
def split_data(data,split_size,save_dir,elem,feature_name,types='labels'):
    begin=0
    cnt=0
    total_len=data.shape[0]
    while begin<total_len:
        print(cnt)
        final=min(begin+split_size,total_len)
        temp=data[begin:final,]
        np.save(os.path.join(save_dir,f"{feature_name}_{elem}_{types}_{cnt}.npy"),temp)
        begin+=split_size
        cnt+=1

def split_seq(data,split_size,save_dir,elem,feature_name):
    begin=0
    cnt=0
    total_len=data.shape[0]
    while begin<total_len:
        print(cnt)
        final=min(begin+split_size,total_len)
        temp=data[begin:final]
        np.savetxt(os.path.join(save_dir,f"{feature_name}_{elem}_{cnt}.seq"),np.array(temp),delimiter='\t',fmt="%s")
        begin+=split_size
        cnt+=1

for elem in ['train','valid','test']:
    seqs=[]
    names=[]
    for record in SeqIO.parse(os.path.join(data_dir,f"{feature_name}_{elem}.fasta"),'fasta'):
        seqs.append(str(record.seq).upper())
        names.append(str(record.id).split('::')[0])
    for i in range(len(names)):
        names[i]=names[i].split(',')
    total=[str(i) for i in range(num_targets)]
    y_dataset=pd.DataFrame(np.zeros((len(names),len(total))),columns=total)
    
    print("processsing features...")
    for index, row in y_dataset.iterrows():
        for t in names[index]:
            y_dataset.at[index,t]=1
    labels=y_dataset.to_numpy()
    with open(os.path.join(data_dir,f"{feature_name}_{elem}.seq"),'w') as seq_f:
        for seq in seqs:
            seq_f.write(f"{seq}\n")

    print("spliting...")
    """label"""
    split_data(labels,split_size,split_dir,elem,feature_name,"labels")
    split_seq(seqs,split_size,split_dir,elem,feature_name)
