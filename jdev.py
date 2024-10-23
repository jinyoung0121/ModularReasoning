import os
import json
import torch
import logging
import time
import datetime
import pathlib
import torch.multiprocessing as mp
from configs import config
import util
from datasets import get_dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from engine.step_interpreters import InternLM, InternLM2, Qwen, load_model

def load_video_context(config, video_id, vlm_answer):
    with open(config.video_context_vid, 'r') as f:
        datas_vid = json.load(f)
    with open(config.video_context, 'r') as f:
        datas = json.load(f)
    contexts = []
    # context formatting
    # add global video caption
    contexts.append(datas_vid[video_id])
    # add frame caption
    for frame_idx, caption in zip(datas[video_id]['frame_idx'], datas[video_id]['captions']):
        contexts.append(f"[frame{frame_idx:>4}]caption: {caption}")
    if vlm_answer:
        return '\n'.join(contexts) + '\n' + vlm_answer
    return '\n'.join(contexts)

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

    # initialize information list
    all_results = []
    all_answers = []
    all_possible_answers = []
    all_queries = []
    all_query_types = []
    all_ids = []
    all_sample_ids = []

    # load model
    if config.llm_type == 'internlm':
        model = InternLM(config, device=device)
        model = load_model(model, device, config)
    elif config.llm_type == 'qwen':
        model = Qwen(config, device=device)
    else:
        raise Exception('Invalid LLM type')
    model.eval()

    start_time = time.time()
    logging.info('Start run')
    metric_logger = util.MetricLogger(delimiter='  ')
    header = '[JDEV]'
    for i, batch in enumerate(metric_logger.log_every(dataloader, 1, header)):
        inner_start_time = time.time()
        logging.info(f'Start inner run [{i + 1:>3}/{len(dataloader):>3}]')

        # update information
        all_answers += batch['answer']
        all_possible_answers += batch['possible_answers']
        all_queries += batch['query']
        all_query_types += batch['query_type']
        all_ids += batch['video_id']
        all_sample_ids += batch['sample_id']

        # Final prediction (without using VLM answer : set to None)
        logging.info('Start final prediction')
        Final_input = {'video_context': [load_video_context(config, video_id, None) for video_id in batch['video_id']], 'question': batch['query'], 'option': batch['possible_answers']}
        # convert data
        Final_input= [dict(zip(Final_input.keys(), values)) for values in zip(*Final_input.values())]
        Final_predictions = model.generate(Final_input, prompt_type='final', num_options=config.dataset.num_options)
        
        # update information
        all_results += Final_predictions
        
        # compute metric
        try:
            accuracy, all_corrections, _ = dataset.accuracy(all_results, all_answers, all_possible_answers, all_query_types)
            metric_logger.update(accuracy=accuracy)
            if config.question_type == 'oe': # consider wups
                wups0, wups9, all_wups, _, _ = dataset.report_wups(all_results, all_answers, all_possible_answers, all_query_types)
                metric_logger.update(wups0=wups0)
                metric_logger.update(wups9=wups9)
        except Exception as e:
            print(f'Error computing accuracy: {e}')

        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        metric_stats = {k: "{:.4f}".format(meter.global_avg2) for k, meter in metric_logger.meters.items()}
        logging.info(metric_stats)
        if config.save:
            if util.is_main_process():
                # log accuracy
                with open(os.path.join(results_dir, 'metric.txt'), mode='a', encoding='utf-8') as f:
                    header = f'[{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")}] Accuracy at Batch [{i + 1:>3}/{len(dataloader):>3}] : '
                    f.write(header + json.dumps(metric_stats) + '\n')
            # log information
            final_datas = {'sample_id': all_sample_ids, 'video_id': all_ids, 'query': all_queries, 'query_type': all_query_types,
                           'answer': all_answers, 'possible_answer': all_possible_answers, "result": all_results, 'correction': all_corrections}
            if config.question_type == 'oe':
                final_datas.update({'wups': all_wups})
            final_datas = list(map(lambda x: dict(zip(final_datas.keys(), x)), zip(*final_datas.values())))
            util.save_result(final_datas, results_dir, 'results', remove_duplicate='sample_id')

        inner_total_time = time.time() - inner_start_time
        inner_total_time_str = str(datetime.timedelta(seconds=int(inner_total_time)))
        logging.info(f'End inner run [{i + 1:>3}/{len(dataloader):>3}]\nElapsed time: {inner_total_time_str}')

    # compute metric
    try:
        accuracy, all_corrections, accuryacy_add = dataset.accuracy(all_results, all_answers, all_possible_answers, all_query_types)
        metric_logger.update(accuracy=accuracy)
        if config.question_type == 'oe': # consider wups
            wups0, wups9, all_wups, wups0_add , wups9_add = dataset.report_wups(all_results, all_answers, all_possible_answers, all_query_types)
            metric_logger.update(wups0=wups0)
            metric_logger.update(wups9=wups9)
    except Exception as e:
        print(f'Error computing accuracy: {e}')

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    metric_stats = {k: "{:.4f}".format(meter.global_avg2) for k, meter in metric_logger.meters.items()}
    logging.info(metric_stats)
    if config.save:
        if util.is_main_process():
            # log accuracy
            with open(os.path.join(results_dir, 'metric.txt'), mode='a', encoding='utf-8') as f:
                header = f'[{datetime.datetime.now().strftime("%Y-%m-%d_%H:%M")}] Final Accuracy : '
                f.write(header + json.dumps(metric_stats) + '\n')
                header = f'[{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")}] Detailed Accuracy : '
                f.write(header + '\n' + json.dumps(accuryacy_add) + '\n')
                if config.question_type == 'oe': # consider wups
                    header = f'[{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")}] Detailed WUPS0 : '
                    f.write(header + '\n' + json.dumps(wups0_add) + '\n')
                    header = f'[{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")}] Detailed WUPS9 : '
                    f.write(header + '\n' + json.dumps(wups9_add) + '\n')
        # log information
        final_datas = {'sample_id': all_sample_ids, 'video_id': all_ids, 'query': all_queries, 'query_type': all_query_types,
                        'answer': all_answers, 'possible_answer': all_possible_answers, "result": all_results, 'correction': all_corrections}
        if config.question_type == 'oe':
            final_datas.update({'wups': all_wups})
        final_datas = list(map(lambda x: dict(zip(final_datas.keys(), x)), zip(*final_datas.values())))
        util.save_result(final_datas, results_dir, 'results', remove_duplicate='sample_id')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logging.info(f"End run\nElapsed time: {total_time_str}")
       
if __name__  == '__main__':
    main()