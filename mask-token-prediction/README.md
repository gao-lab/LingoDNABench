## Evaluating intra-species mutual information learned in gLM
To assess whether gLMs can capture varying levels of mutual information across human genomic regions. For each nucleotide position, we iteratively masked every nucleotide position and measured the predicted probability assigned to the original nucleotide across four models: RandomWeight, RandomSeq, human-genome gLM, and multi-species gLM. Higher probabilities indicate stronger capture of position-specific statistical dependencies.
```
model_type=RandomSeq
python mask-token-prediction.py exon.txt exon-${model_type}_prob.txt $model_type
python mask-token-prediction.py alu.txt alu-${model_type}_prob.txt $model_type

model_type=RandomWeight
python mask-token-prediction.py exon.txt exon-${model_type}_prob.txt $model_type
python mask-token-prediction.py alu.txt alu-${model_type}_prob.txt $model_type

model_type=human-genome
python mask-token-prediction.py exon.txt exon-${model_type}_prob.txt $model_type
python mask-token-prediction.py alu.txt alu-${model_type}_prob.txt $model_type

model_type=multi-species
python mask-token-prediction.py exon.txt exon-${model_type}_prob.txt $model_type
python mask-token-prediction.py alu.txt alu-${model_type}_prob.txt $model_type
```

