CUDA_VISIBLE_DEVICES=6,7 python -m torch.distributed.run --nproc_per_node 2 --master_port=4616 llm_only.py \
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
    video_context datas/ActivityNet-QA/internlmxcomposer2_brief_test.json \
    vlm_type internlmxcomposer \
    question_type oe \
    mode llm_only


CUDA_VISIBLE_DEVICES=6,7 python -m torch.distributed.run --nproc_per_node 2 --master_port=5245 jcef.py \
    --options \
    exp_name internlmxcomposer2_internlm \
    dataset.dataset_name ActivityNetQA \
    dataset.data_path datas/ActivityNet-QA \
    dataset.split test \
    dataset.fps 1 \
    dataset.start_sample 0 \
    batch_size 5 \
    log_freq 1 \
    internlm.max_batch_size 1 \
    video_context datas/ActivityNet-QA/internlmxcomposer2_brief_allframe_test.json \
    is_video False \
    is_image True \
    vlm_type internlmxcomposer \
    question_type oe \
    mode jcef