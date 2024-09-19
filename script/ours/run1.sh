CUDA_VISIBLE_DEVICES=0,1,2,3 python ours.py \
    --options \
    exp_name ours \
    dataset.dataset_name NExTQA \
    dataset.data_path datas/NExT-QA \
    dataset.split val \
    dataset.version multiplechoice \
    dataset.fps 1 \
    dataset.batch_size 20 \
    dataset.start_sample 0 \
    log_freq 1 \
    mode ours_baseline