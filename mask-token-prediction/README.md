## Evaluating intra-species mutual information

```
model_type=randomseq
python mask-token-prediction.py exon.txt exon-${model_type}_prob.txt $model_type
python mask-token-prediction.py alu.txt alu-${model_type}_prob.txt $model_type
```

