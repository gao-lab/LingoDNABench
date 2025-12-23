## Extracting embedding

```
data_dir="../benchmark/datasets/promoter/prom_300_tata"
epoch=1
for dir in `find $data_dir -mindepth 2 -maxdepth 2 -type d|grep expression`  
do
    if [ ! -d "$dir/$model_name" ]; then
        mkdir $dir/$model_name
    fi
    for file in `find $dir -type f -name "*data*"` 
        do
        echo "$file $dir"
        python get-embedding-pretrain_loss.py  $file $dir/$model_name $((embedding_len)) -1 $epoch
    done
done

```
