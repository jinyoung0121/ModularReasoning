CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.run --nproc_per_node 4 --master_port=1111 ours_baseline3.py \
    --options \
    exp_name internvl_baseline3 \
    dataset.dataset_name NExTGQA \
    dataset.data_path datas/NExT-GQA \
    dataset.video_path datas/NExT-QA \
    dataset.split test \
    dataset.version multiplechoice \
    dataset.fps 1 \
    dataset.start_sample 0 \
    eval_grounding True \
    batch_size 100 \
    log_freq 10 \
    video_context datas/internvl_brief.json \
    is_video False \
    is_image True \
    vlm_type internvl \
    mode ours_baseline