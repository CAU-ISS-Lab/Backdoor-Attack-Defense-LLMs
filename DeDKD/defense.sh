defense=(BeDKD)
defense_id=0
dataset_path=# your poisoned data path
victim_model=# your poisoned model path
TOKENIZERS_PARALLELISM=false  CUDA_VISIBLE_DEVICES=0 python main.py configs/bert/config.json ${defense[${defense_id}]} ${dataset_path} ${victim_model}