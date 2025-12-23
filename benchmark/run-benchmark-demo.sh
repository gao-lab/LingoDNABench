model_name=nucleotide-transformer-2.5b-multi-species
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
python TFBS_510-embedding-all_model.py nucleotide-transformer-2.5b-multi-species -1 $((510 / 6 + 1 )) 2560 ../dataset/TFBS ../dataset/TFBS/checkpoint ../dataset/TFBS/eval 
#DNA accessibility
python CA_510-embedding-all_model.py nucleotide-transformer-2.5b-multi-species -1 $((510 / 6 + 1 )) 2560 ../dataset/DNA_accessibility ../dataset/DNA_accessibility/checkpoint ../dataset/DNA_accessibility/eval 
#Histone modification
python HM_510-embedding-all_model.py nucleotide-transformer-2.5b-multi-species -1 $((510 / 6 + 1 )) 2560 ../dataset/HM ../dataset/HM/checkpoint ../dataset/HM/eval 

