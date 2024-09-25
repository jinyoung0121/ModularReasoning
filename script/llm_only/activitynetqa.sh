CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.run --nproc_per_node 2 --master_port=4616 llm_only.py \
    --options \
    exp_name navie_internlm \
    dataset.dataset_name ActivityNetQA \
    dataset.data_path datas/ActivityNet-QA \
    dataset.split test \
    dataset.fps 1 \
    dataset.start_sample 0 \
    internlm.max_batch_size 8 \
    batch_size 5 \
    log_freq 1 \
    video_context datas/internvl_brief.json \
    question_type oe \
    mode llm_only