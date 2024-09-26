CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node 4 --master_port=9245 morevqa.py \
    --options \
    exp_name internvl_brief_internlm \
    dataset.dataset_name NExTQA \
    dataset.data_path datas/NExT-QA \
    dataset.split val_temporal \
    dataset.version multiplechoice \
    dataset.fps 1 \
    dataset.start_sample 0 \
    batch_size 100 \
    log_freq 10 \
    video_context datas/NExT-QA/nextqa/internvl_brief_val.json \
    is_video False \
    is_image True \
    vlm_type internvl \
    mode morevqa