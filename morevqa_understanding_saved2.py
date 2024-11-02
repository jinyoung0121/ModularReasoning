import os
import json
import torch
import logging
import time
import datetime
import pathlib
import torch.multiprocessing as mp
from configs import config
from engine import Global_planning, Program_generation, Understanding_generation, Stage1, Stage2, Stage3, Stage4_image, Stage4_video
import util
from datasets import get_dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

def load_video_context(config, video_id, vlm_answer):
    
    with open(config.video_context_vid, 'r') as f:
        datas_vid = json.load(f)
    with open(config.video_context, 'r') as f:
        datas = json.load(f)
    contexts = []
    # context formatting
    # add global video caption
    # contexts.append(datas_vid[video_id])
    # add frame caption
    for frame_idx, caption in zip(datas[video_id]['frame_idx'], datas[video_id]['captions']):
        contexts.append(f"[frame{frame_idx:>4}]caption: {caption}")
    results = '[global information]' + '\n' + '\n'.join(contexts)
    if vlm_answer:
        results += '\n' + '[local information]'
    if vlm_answer['video']:
        results += '\n' + vlm_answer['video']
    if vlm_answer['image']:
        results += '\n' + vlm_answer['image']
    return results

def main():
    mp.set_start_method('spawn')
    util.init_distributed_mode(config)
    util.setup_logger()
    device = torch.device(config.device)
    util.setup_seeds(config)

    now = datetime.datetime.now()
    time_str = now.strftime("%Y-%m-%d_%H-%M")

    if config.save:
        results_dir = pathlib.Path(config.results_dir)
        results_dir = results_dir / config.dataset.dataset_name / config.dataset.split / config.mode / f"{config.exp_name}_{time_str}"
        results_dir.mkdir(parents=True, exist_ok=True)

    config.eval_grounding = config.get('eval_grounding', False)
    config.dataset.eval_grounding = config.eval_grounding
    
    dataset = get_dataset(config.dataset)
    config.dataset.num_options = dataset.num_options
    
    if config.distributed:
        sampler = DistributedSampler(dataset, num_replicas=util.get_world_size(), rank=util.get_rank())
    else:
        sampler = None
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers,
                            pin_memory=True, drop_last=False, collate_fn=util.my_collate)

    if config.distributed:
        dataloader.sampler.set_epoch(-1)
    
    # load saved result
    with open('/hub_data1/jinyoungkim/ModularReasoning/results/NExTQA/val/morevqa_understanding/s1s2s3s4_16frame_newprompt_frameOR_viclip_internlm7b_revised_2024-10-24_16-13/external_memory.json', 'r') as f:
        saved_data = json.load(f)
    saved_data = {item["sample_id"]: {k: v for k, v in item.items()} for item in saved_data}
    
    # initialize information list
    image_ids = []

    start_time = time.time()
    logging.info('Start run')
    metric_logger = util.MetricLogger(delimiter='  ')
    header = '[Retrieve baseline]'
    for i, batch in enumerate(metric_logger.log_every(dataloader, 1, header)):
        inner_start_time = time.time()
        logging.info(f'Start inner run [{i + 1:>3}/{len(dataloader):>3}]')
        if i == 2:
            # initialize External Memory
            logging.info('Initialize External Memory')
            EXTERNAL_MEMORY = []
            for sample_id, frames, query in zip(batch['sample_id'], batch['image'], batch['query']):
                image_ids.append(sample_id)
    
    import pdb; pdb.set_trace()



        
if __name__  == '__main__':
    main()