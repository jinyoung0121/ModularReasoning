import os
import json
import torch
import logging
import time
import datetime
import pathlib
import torch.multiprocessing as mp
from configs import config
from engine import Program_generation, Stage1, Stage2, FinalPrediction
import util
from datasets import get_dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

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
    all_memories = []
    all_s2_prog = []

    start_time = time.time()
    logging.info('Start run')
    metric_logger = util.MetricLogger(delimiter='  ')
    header = '[Ours baseline]'
    for i, batch in enumerate(metric_logger.log_every(dataloader, 1, header)):
        inner_start_time = time.time()
        logging.info(f'Start inner run [{i + 1:>3}/{len(dataloader):>3}]')
        
        # initialize External Memory
        logging.info('Initialize External Memory')
        EXTERNAL_MEMORY = []
        for sample_id, frames, query in zip(batch['sample_id'], batch['image'], batch['query']):
            EXTERNAL_MEMORY.append({'sample_id':sample_id,
                                    'original_question': query,
                                    'num_frames': frames.size(0),
                                    'question': query,
                                    'frame_ids': [idx for idx in range(frames.size(0))],
                                    'phrases': ['',''],
                                    'conjunction': 'none',
                                    'require_ocr': False,
                                    'qa_type': '',
                                    'error': None,
                                    'VLM_answers': None,})
        # update information
        all_answers += batch['answer']
        all_possible_answers += batch['possible_answers']
        all_queries += batch['query']
        all_query_types += batch['query_type']
        all_ids += batch['video_id']
        all_sample_ids += batch['sample_id']
        
        # Stage1 program generation
        logging.info('Start Stage1 program generation')
        S1_input = [{'question': memory['question']} for memory in EXTERNAL_MEMORY]
        S1_programs = Program_generation(config, device=device, data=S1_input, prompt_type='stage1')
        
        # Stage1 processing then update External Memory
        logging.info('Start stage1 processing')
        S1_input = [{'program': program} for program in S1_programs]
        EXTERNAL_MEMORY = Stage1(config, EXTERNAL_MEMORY, data=S1_input, device=device)

        # Stage2 program generation
        logging.info('Start stage2 program generation')
        S2_input = [{'phrases': memory['phrases'], 'conjunction': memory['conjunction'], 'image': image, 'video_path': video_path}\
                        for memory, image, video_path in zip(EXTERNAL_MEMORY, batch['image'], batch['video_path'])]
        S2_programs = Program_generation(config, device=device, data=S2_input, prompt_type='stage2')
        
        # Stage2 processing then update External Memory
        logging.info('Start stage2 processing')
        S2_input = [{'program': program, 'image': image, 'video_path': video_path}\
                        for program, image, video_path in zip(S2_programs, batch['image'], batch['video_path'])]

        
        EXTERNAL_MEMORY = Stage2(config, EXTERNAL_MEMORY, data=S2_input, device=device)
            
        # Final prediction
        logging.info('Start final prediction')
        Final_input = [{'question': memory['question'], 'option': option, 'video_path': video_path, 'frame_ids': memory['frame_ids']}\
                        for memory, option, video_path in zip(EXTERNAL_MEMORY, batch['possible_answers'], batch['video_path'])]
        Final_predictions = FinalPrediction(config, device=device, data=Final_input, model_type=config.vlm_type)
        
        # update information
        all_results += Final_predictions
        all_memories += EXTERNAL_MEMORY
        
        # update s2 program (RETRIEVE)
        s2_prog_list = [i.split('\n') for i in S2_programs]
        all_s2_prog += s2_prog_list
        
        
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
                    header = f'[{datetime.datetime.now().strftime("%Y-%m-%d_%H:%M")}] Accuracy at Batch [{i + 1:>3}/{len(dataloader):>3}] : '
                    f.write(header + json.dumps(metric_stats) + '\n')
            # log information
            final_datas = {'sample_id': all_sample_ids, 'video_id': all_ids, 'query': all_queries, 'query_type': all_query_types,
                           'answer': all_answers, 'possible_answer': all_possible_answers, "result": all_results, 'correction': all_corrections}
            if config.question_type == 'oe':
                final_datas.update({'wups': all_wups})
            final_datas = list(map(lambda x: dict(zip(final_datas.keys(), x)), zip(*final_datas.values())))
            util.save_result(final_datas, results_dir, 'results', remove_duplicate='sample_id',)
            util.save_result(all_memories, results_dir, 'external_memory', remove_duplicate='sample_id',)
            
            s2_prog_save = {'sample_id': all_sample_ids, 'video_id': all_ids, 's2_prog': all_s2_prog}
            s2_prog_save = list(map(lambda x: dict(zip(s2_prog_save.keys(), x)), zip(*s2_prog_save.values())))
            util.save_result(s2_prog_save, results_dir, 's2_program', remove_duplicate='sample_id',)
            

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
        util.save_result(final_datas, results_dir, 'results', remove_duplicate='sample_id',)
        util.save_result(all_memories, results_dir, 'external_memory', remove_duplicate='sample_id',)
        
        s2_prog_save = {'sample_id': all_sample_ids, 'video_id': all_ids, 's2_prog': all_s2_prog}
        s2_prog_save = list(map(lambda x: dict(zip(s2_prog_save.keys(), x)), zip(*s2_prog_save.values())))
        util.save_result(s2_prog_save, results_dir, 's2_program', remove_duplicate='sample_id',)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logging.info(f"End run\nElapsed time: {total_time_str}")
    
if __name__  == '__main__':
    main()