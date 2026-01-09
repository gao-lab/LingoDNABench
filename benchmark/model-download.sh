
model_dir=$1
#DNABERT
huggingface-cli download --resume-download zhihan1996/DNA_bert_3   --local-dir ${model_dir}/DNA_bert_3

#DNABERT-2
huggingface-cli download --resume-download zhihan1996/DNABERT-2-117M   --local-dir ${model_dir}/DNABERT-2

#Nucleotide-Transformer
huggingface-cli download --resume-download InstaDeepAI/nucleotide-transformer-2.5b-multi-species   --local-dir ${model_dir}/nucleotide-transformer-2.5b-multi-species

#Caduceus
huggingface-cli download --resume-download kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16   --local-dir ${model_dir}/caduceus-ps_seqlen-131k_d_model-256_n_layer-16

#HyneaDNA
huggingface-cli download --resume-download LongSafari/hyenadna-large-1m-seqlen-hf   --local-dir ${model_dir}/hyenadna-large-1m-seqlen-hf

#OmniNA
huggingface-cli download --resume-download XLS/OmniNA-220m   --local-dir ${model_dir}/OmniNA-220m

#DeepGene
git clone https://github.com/wds-seu/DeepGene.git

#Evo
huggingface-cli download --resume-download arcinstitute/evo2_7b --local-dir ${model_dir}/evo2_7b

#GENERator

huggingface-cli download --resume-download GenerTeam/GENERator-eukaryote-3b-base --local-dir ${model_dir}/GENERator-eukaryote-3b-base

#lucaone
huggingface-cli download --resume-download LucaGroup/LucaOne-default-step36M --local-dir ${model_dir}/lucaone

#GPN-MSA
git clone https://github.com/songlab-cal/gpn.git

#GPN-MSA model checkpoint
huggingface-cli download --resume-download songlab/gpn-msa-sapiens   --local-dir ${model_dir}/gpn-msa-sapiens

#enformer
huggingface-cli download --resume-download EleutherAI/enformer-official-rough   --local-dir ${model_dir}/Enformer
