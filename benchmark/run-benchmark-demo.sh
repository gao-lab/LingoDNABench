model_name=DNABERT-2-117M
# default applications
data_dir="./dataset/promoter/prom_300_tata"
layer=-1 # the last layer
random_seed=42
output_dim=1
regression=False #classification application
python benchmark-default-tasks.py  $model_name $data_dir $layer  $random_seed $output_dim $regression
 

# Exon PSI

layer=-1 # the last layer
random_seed=42
output_dim=56
regression=True #regression application
python benchmark-Exon_PSI.py  $model_name $data_dir $layer  $random_seed $output_dim $regression
 

# Bulk RNA-seq (expression)
layer=-1 # the last layer
random_seed=42
output_dim=53
regression=True #regression application
python benchmark-exp.py  $model_name $data_dir $layer  $random_seed $output_dim $regression

# Long range interaction
layer=-1 # the last layer
random_seed=42
output_dim=1
regression=False #regression application
python benchmark-PPI-EPI.py  $model_name $data_dir $layer  $random_seed $output_dim 

# variant effect prediction
python variant_effect_prediction-zero-shot.py $data_dir $model_name $layer


#TFBS
python TFBS_510-embedding-all_model.py 
#DNA accessibility
python CA_510-embedding-all_model.py
#Histone modification
python HM_510-embedding-all_model.py

