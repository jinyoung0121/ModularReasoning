CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.run --nproc_per_node 4 --master_port=9245 ours_baseline2.py \
    --options \
    exp_name internvl_baseline2 \
    dataset.dataset_name ActivityNetQA \
    dataset.data_path datas/ActivityNet-QA \
    dataset.split test \
    dataset.fps 1 \
    dataset.start_sample 0 \
    batch_size 5 \
    log_freq 1 \
    video_context datas/internvl_brief.json \
    is_video False \
    is_image True \
    vlm_type internvl \
    question_type oe \
    mode ours_baseline