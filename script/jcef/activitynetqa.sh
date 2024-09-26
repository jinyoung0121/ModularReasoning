CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.run --nproc_per_node 4 --master_port=8245 jcef.py \
    --options \
    exp_name internvl_internlm \
    dataset.dataset_name ActivityNetQA \
    dataset.data_path datas/ActivityNet-QA \
    dataset.split test \
    dataset.fps 1 \
    dataset.start_sample 0 \
    batch_size 5 \
    log_freq 1 \
    video_context datas/ActivityNet-QA/internvl_brief_allframe.json \
    question_type oe \
    mode jcef