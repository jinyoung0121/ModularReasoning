CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.run --nproc_per_node 2 --master_port=1111 ours_baseline3.py \
    --options \
    exp_name internvl_internlm \
    dataset.dataset_name NExTQA \
    dataset.data_path datas/NExT-QA \
    dataset.split val \
    dataset.version multiplechoice \
    dataset.fps 1 \
    dataset.start_sample 0 \
    dataset.grounding_data_path datas/NExT-GQA \
    batch_size 100 \
    log_freq 10 \
    video_context datas/internvl_brief.json \
    mode ours_baseline \
    eval_grounding True 