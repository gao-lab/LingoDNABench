import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import glob
import numpy as np
# from tensorflow.compat.v1 import data
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
import random
from sklearn.metrics import roc_auc_score
from sklearn import metrics
import sys
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, recall_score, precision_score, roc_curve,confusion_matrix,matthews_corrcoef,precision_recall_curve
from sklearn import metrics
from utils.models import Conv_Block_GELU,Dense_Block,StochasticReverseComplement,StochasticShift
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)



def seed_tendsorflow(seed=42):
    os.environ['PYTHONHASHSEED']=str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['TF_DETERMINISTIC_OPS']='1'
random_seed=42
seed_tendsorflow(random_seed)
'''function for making dataset'''
def file_to_records(filename):
    return tf.data.TFRecordDataset(filename, compression_type='')


model_name=sys.argv[1]
layer=int(sys.argv[2])

embedding_len=510
model_dim=int(sys.argv[3])
dataset_dir=sys.argv[4]
checkpoint_dir=sys.argv[5]
eval_dir=sys.argv[6]


class EmbeddingDataset:
    def __init__(self, file_list, batch_size, data_mode, tfr_pattern=None):
        self.data_dir = file_list
        self.batch_size = batch_size
        self.data_mode = data_mode
        self.tfr_pattern = tfr_pattern
        #make dataset
        self.make_dataset()
    #decode data
    def generate_parser(self):
        def parse_proto(example_protos):
            features={
                'data':tf.io.FixedLenFeature([], tf.string),
                'label':tf.io.FixedLenFeature([], tf.string)
            }
            parsed_features = tf.io.parse_single_example(example_protos, features=features)
            #decode embedding and reshape
            embedding = tf.io.decode_raw(parsed_features['data'], tf.float32)
            embedding = tf.reshape(embedding, [embedding_len, model_dim])
            labels = tf.io.decode_raw(parsed_features['label'], tf.float32)
            # labels = tf.cast(labels, tf.float32)
            #return data
            return embedding, labels
        return parse_proto
    def make_dataset(self):
        #data path
        #print(tfr_files)
        dataset = tf.data.Dataset.from_tensor_slices(self.data_dir)
        if self.data_mode == 'train':
            dataset = dataset.repeat()
            dataset = dataset.interleave(file_to_records,cycle_length=len(self.data_dir),num_parallel_calls=tf.data.AUTOTUNE)
            #shuffle
            dataset = dataset.shuffle(buffer_size=1024,reshuffle_each_iteration=True)
        elif self.data_mode == 'valid':
            #dataset = dataset.flat_map(file_to_records)
            dataset = dataset.interleave(file_to_records,cycle_length=len(self.data_dir),num_parallel_calls=tf.data.AUTOTUNE)
        else:
            dataset = dataset.interleave(file_to_records,cycle_length=1,num_parallel_calls=1,deterministic=True)
        dataset = dataset.map(self.generate_parser())
        dataset = dataset.batch(self.batch_size)  
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        #hold
        self.dataset = dataset



class Basenji_Basset(tf.keras.Model):
    def __init__(self,target_num):
        super().__init__()
        self.conv_dna = Conv_Block_GELU(288,17,True,3)
        self.conv1=Conv_Block_GELU(288,5,pool=False)
        self.conv2=Conv_Block_GELU(323,5,pool=True,pool_size=2)
        self.conv3=Conv_Block_GELU(363,5,pool=False)
        self.conv4=Conv_Block_GELU(407,5,pool=True,pool_size=2)
        self.conv5=Conv_Block_GELU(456,5,pool=False)
        self.conv6=Conv_Block_GELU(512,5,pool=True,pool_size=2)
        self.conv_last=Conv_Block_GELU(256,1,pool=False)
        self.dense=Dense_Block(768,0.2)
        self.dense2=tf.keras.layers.Dense(units=target_num,activation=tf.nn.sigmoid)
    def call(self, X):
        X=self.conv_dna(X)
        X=self.conv4(self.conv3(self.conv2(self.conv1(X))))
        X=self.conv6(self.conv5(X))
        X=self.conv_last(X)
        X=self.dense(X)
        X=self.dense2(X)
        return X

batch_size=128
epoch_num = 25
steps_per_epoch = 1428717//batch_size
valid_per_epoch = 277799//batch_size
data_dir=f"{dataset_dir}/{model_name}/tfrecords"
train_list=[os.path.join(data_dir,f"TFBS-train-data-layer_{layer}-{xx}.tfrecords") for xx in range(58)]
valid_list=[os.path.join(data_dir,f"TFBS-valid-data-layer_{layer}-{xx}.tfrecords") for xx in range(12)]
test_list=[os.path.join(data_dir,f"TFBS-test-data-layer_{layer}-{xx}.tfrecords") for xx in range(15)]
train_dataset = EmbeddingDataset(train_list, batch_size, data_mode="train").dataset
valid_dataset = EmbeddingDataset(valid_list, batch_size, data_mode="valid" ).dataset
test_dataset = EmbeddingDataset(test_list, batch_size, data_mode="test").dataset
test_dataset=valid_dataset
num_targets=3572
model=Basenji_Basset(num_targets)
model.build(input_shape=(None,embedding_len,model_dim))
print(model.summary())
learning_rate=1e-3

loss_fn = tf.keras.losses.BinaryCrossentropy()

opt=tf.keras.optimizers.Adam()
model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=opt,
    metrics=[tf.keras.metrics.AUC()]
)


check_point_file = f"{checkpoint_dir}/TFBS-{model_name}-embedding-basenji-basset.h5"
verbose_state = 1
stopper=keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, verbose=verbose_state, mode='min')
check_point=keras.callbacks.ModelCheckpoint(check_point_file,monitor='val_loss',verbose=verbose_state,save_best_only=True,mode='min',save_weights_only=True)
history=model.fit(train_dataset, epochs=epoch_num, steps_per_epoch=steps_per_epoch,validation_data=valid_dataset, validation_steps=valid_per_epoch, callbacks=[stopper,check_point])

model.load_weights(check_point_file)
test_num=369087
y_pred=np.empty((test_num,num_targets))
y_test=np.empty((test_num,num_targets))
begin=0
for i, record in enumerate(test_dataset):
    temp=record[1].numpy()
    num=temp.shape[0]
    y_test[begin:begin+num,:]=record[1].numpy()
    y_pred[begin:begin+num,:]=model(record[0].numpy(),training=False)
    begin+=num


with open(f"{eval_dir}/TFBS-Result-{model_name}.txt",'w') as f:
    f.write("cutoff\taccuracy\tprecision\trecall\tF1_score\tAUC\tMCC\n")
    for i in range(num_targets):
        try:
            auc=roc_auc_score(y_test[:,i],y_pred[:,i]) 
            y_prediction=y_pred[:,i]
            y_true=y_test[:,i]
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
            f.write(f"{cutoff}\t{accuracy}\t{precision}\t{recall}\t{f1}\t{auc}\t{mcc}\n")
        except:
            pass
