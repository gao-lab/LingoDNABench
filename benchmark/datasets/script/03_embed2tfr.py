import tensorflow as tf
import numpy as np
import os
from typing import List
import sys


idx=int(sys.argv[1])
elem=sys.argv[2]
task_name=sys.argv[3]
model_name=sys.argv[4]
layer=int(sys.argv[5])
input_dir=sys.argv[6]
label_dir=sys.argv[7]
save_dir=sys.argv[8]


X_file=os.path.join(input_dir,f"{task_name}_{elem}_{idx}-embedding-layer_{layer}.npy")
Y_file=os.path.join(label_dir,f"{task_name}_{elem}_labels_{idx}.npy") 
save_file=os.path.join(save_dir,f"{task_name}-{elem}-data-layer_{layer}-{idx}.tfrecords")


def save_tfrecords(data, label, desfile):
    with tf.io.TFRecordWriter(desfile) as writer:
        for i in range(len(data)):
            features = tf.train.Features(
                feature = {
                    "data":tf.train.Feature(bytes_list = tf.train.BytesList(value = [data[i].astype(np.float32).tobytes()])),
                    "label":tf.train.Feature(bytes_list = tf.train.BytesList(value = [label[i].astype(np.float32).tobytes()]))
                }
            )
            example = tf.train.Example(features = features)
            serialized = example.SerializeToString()
            writer.write(serialized)




data=np.load(X_file)
label=np.load(Y_file)
save_tfrecords(data,label,save_file)
os.remove(X_file)





