# 请按照您的真实环境修改 set_env.sh 路径
source /usr/local/Ascend/ascend-toolkit/set_env.sh
mkdir -p ./dataset/GLM-Z1-Rumination-32B-0414/

python ./preprocess_data.py \
        --input ./dataset/train-00000-of-00001-a09b74b3ef9c3b56.parquet \
        --tokenizer-name-or-path ./model_from_hf/GLM-Z1-Rumination-32B-0414/ \
        --output-prefix ./dataset/GLM-Z1-Rumination-32B-0414/alpaca \
        --workers 4 \
        --log-interval 1000 \
        --tokenizer-type PretrainedFromHF
