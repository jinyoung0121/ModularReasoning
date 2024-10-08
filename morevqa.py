import os
import json
import torch
import logging
import time
import datetime
import pathlib
import torch.multiprocessing as mp
from configs import config
from engine import Program_generation, Module1, Module2, Module3
import util
from datasets import get_dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

def load_video_context(config, video_id, vlm_answer):
    with open(config.video_context, 'r') as f:
        datas = json.load(f)
    contexts = []
    # context formatting
    for frame_idx, caption in zip(datas[video_id]['frame_idx'], datas[video_id]['captions']):
        contexts.append(f"[frame{frame_idx:>4}]caption: {caption}.")
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

    config.eval_grounding = config.get('eval_grounding', False)
    config.dataset.eval_grounding = config.eval_grounding
    
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
    all_num_frames = []

    all_m3_prog = []
    all_m2_prog = []
    all_m1_prog = []
    all_timespan = []
    
    start_time = time.time()
    logging.info('Start run')
    metric_logger = util.MetricLogger(delimiter='  ')
    header = '[MoReVQA]'
    for i, batch in enumerate(metric_logger.log_every(dataloader, 1, header)):
        inner_start_time = time.time()
        logging.info('*'* 90)
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
                                    'event_queue': [],
                                    'conjunction': 'none',
                                    'require_ocr': False,
                                    'qa_type': '',
                                    'error': None,
                                    'VLM_answers': None,})
            all_num_frames.append(frames.size(0))
        
        # update information
        all_answers += batch['answer']
        all_possible_answers += batch['possible_answers']
        all_queries += batch['query']
        all_query_types += batch['query_type']
        all_ids += batch['video_id']
        all_sample_ids += batch['sample_id']

        if config.eval_grounding:
            all_timespan += batch['timespan']
    
        # Module1 program generation
        logging.info('Start module1 program generation')
        M1_input = [{'question': memory['question']} for memory in EXTERNAL_MEMORY]
        M1_programs = Program_generation(config, device=device, data=M1_input, prompt_type='module1')
        
        # Module1 processing then update External Memory
        logging.info('Start module1 processing')
        M1_input = [{'program': program} for program in M1_programs]
        EXTERNAL_MEMORY = Module1(config, EXTERNAL_MEMORY, data=M1_input, device=device)
        
        # Module2 program generation
        logging.info('Start module2 program generation')
        M2_input = [{'event_queue': memory['event_queue'], 'conjunction': memory['conjunction'], 'image': image} for memory, image in zip(EXTERNAL_MEMORY, batch['image'])]
        M2_programs = Program_generation(config, device=device, data=M2_input, prompt_type='module2')
        
        # Module2 processing then update External Memory
        logging.info('Start module2 processing')
        M2_input = [{'program': program, 'image': image} for program, image in zip(M2_programs, batch['image'])]
        EXTERNAL_MEMORY = Module2(config, EXTERNAL_MEMORY, data=M2_input, device=device)
        
        # Module3 program generation
        logging.info('Start module3 program generation')
        M3_input = [{'question': memory['question'], 'frame_ids': memory['frame_ids'], 'require_ocr': memory['require_ocr'], 'qa_type': memory['qa_type']} for memory in EXTERNAL_MEMORY]
        M3_programs = Program_generation(config, device=device, data=M3_input, prompt_type='module3')
        
        # Module3 processing than update External Memory
        logging.info('Start module3 processing')
        M3_input = [{'program': program, 'image': image} for program, image in zip(M3_programs, batch['image'])]
        EXTERNAL_MEMORY = Module3(config, EXTERNAL_MEMORY, data=M3_input, device=device)
        
        # Final prediction
        logging.info('Start final prediction')
        Final_input = [{'video_context': load_video_context(config, video_id, memory['VLM_answers']), 'question': question, 'option': option } \
                                for video_id, question, option, memory in zip(batch['video_id'], batch['query'], batch['possible_answers'], EXTERNAL_MEMORY)]
        Final_predictions = Program_generation(config, device=device, data=Final_input, prompt_type='final')
        
        # update information
        all_results += Final_predictions
        all_memories += EXTERNAL_MEMORY

        # update m1, m2, m3 program (intermediate generated program)
        m3_prog_list = [i.split('\n') for i in M3_programs]
        m2_prog_list = [i.split('\n') for i in M2_programs]
        m1_prog_list = [i.split('\n') for i in M1_programs]
        all_m3_prog += m3_prog_list
        all_m2_prog += m2_prog_list
        all_m1_prog += m1_prog_list

        # compute metric
        try:
            logging.info('Start evaluating QA')
            accuracy, all_corrections, _ = dataset.accuracy(all_results, all_answers, all_possible_answers, all_query_types)
            metric_logger.update(accuracy=accuracy)
            if config.question_type == 'oe': # consider wups
                wups0, wups9, all_wups, _, _ = dataset.report_wups(all_results, all_answers, all_possible_answers, all_query_types)
                metric_logger.update(wups0=wups0)
                metric_logger.update(wups9=wups9)
            if config.eval_grounding:
                logging.info('Start evaluating grounding')
                ground_result = dataset.grounding(all_results, all_answers, all_possible_answers, all_query_types,
                                                all_memories, all_timespan, all_num_frames) 
                # return {'mIoU':mIoU, 'mIoP': mIoP, 'IoU': IoU, 'IoP':IoP, 'cnt_empty':cnt_empty, 'acc':acc}
                metric_logger.update(acc3=ground_result['acc'][0.3])
                metric_logger.update(acc5=ground_result['acc'][0.5])
                metric_logger.update(mIoU=ground_result['mIoU'])
                metric_logger.update(mIoP=ground_result['mIoP'])
                metric_logger.update(IoU3=ground_result['IoU'][0.3])
                metric_logger.update(IoU5=ground_result['IoU'][0.5])
                metric_logger.update(IoP3=ground_result['IoP'][0.3])
                metric_logger.update(IoP5=ground_result['IoP'][0.5])            
                metric_logger.update(cnt_empty=ground_result['cnt_empty'])   
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
            util.save_result(final_datas, results_dir, 'results', remove_duplicate='sample_id')
            util.save_result(all_memories, results_dir, 'external_memory', remove_duplicate='sample_id')

            m1_prog_save = {'sample_id': all_sample_ids, 'video_id': all_ids, 'query': all_queries, 'm1_prog': all_m1_prog}
            m1_prog_save = list(map(lambda x: dict(zip(m1_prog_save.keys(), x)), zip(*m1_prog_save.values())))
            util.save_result(m1_prog_save, results_dir, 'm1_program', remove_duplicate='sample_id',)
            
            m2_prog_save = {'sample_id': all_sample_ids, 'video_id': all_ids, 'query': all_queries, 'm2_prog': all_m2_prog}
            m2_prog_save = list(map(lambda x: dict(zip(m2_prog_save.keys(), x)), zip(*m2_prog_save.values())))
            util.save_result(m2_prog_save, results_dir, 'm2_program', remove_duplicate='sample_id',)

            m3_prog_save = {'sample_id': all_sample_ids, 'video_id': all_ids, 'query': all_queries, 'm3_prog': all_m3_prog}
            m3_prog_save = list(map(lambda x: dict(zip(m3_prog_save.keys(), x)), zip(*m3_prog_save.values())))
            util.save_result(m3_prog_save, results_dir, 'm3_program', remove_duplicate='sample_id',)

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
        if config.eval_grounding:
            logging.info('Start evaluating grounding')
            ground_result = dataset.grounding(all_results, all_answers, all_possible_answers, all_query_types,
                                            all_memories, all_timespan, all_num_frames) 
            # return {'mIoU':mIoU, 'mIoP': mIoP, 'IoU': IoU, 'IoP':IoP, 'cnt_empty':cnt_empty, 'acc':acc}
            metric_logger.update(acc3=ground_result['acc'][0.3])
            metric_logger.update(acc5=ground_result['acc'][0.5])
            metric_logger.update(mIoU=ground_result['mIoU'])
            metric_logger.update(mIoP=ground_result['mIoP'])
            metric_logger.update(IoU3=ground_result['IoU'][0.3])
            metric_logger.update(IoU5=ground_result['IoU'][0.5])
            metric_logger.update(IoP3=ground_result['IoP'][0.3])
            metric_logger.update(IoP5=ground_result['IoP'][0.5])            
            metric_logger.update(cnt_empty=ground_result['cnt_empty'])
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
        util.save_result(all_memories, results_dir, 'external_memory', remove_duplicate='sample_id')

        m1_prog_save = {'sample_id': all_sample_ids, 'video_id': all_ids, 'query': all_queries, 'm1_prog': all_m1_prog}
        m1_prog_save = list(map(lambda x: dict(zip(m1_prog_save.keys(), x)), zip(*m1_prog_save.values())))
        util.save_result(m1_prog_save, results_dir, 'm1_program', remove_duplicate='sample_id',)
        
        m2_prog_save = {'sample_id': all_sample_ids, 'video_id': all_ids, 'query': all_queries, 'm2_prog': all_m2_prog}
        m2_prog_save = list(map(lambda x: dict(zip(m2_prog_save.keys(), x)), zip(*m2_prog_save.values())))
        util.save_result(m2_prog_save, results_dir, 'm2_program', remove_duplicate='sample_id',)

        m3_prog_save = {'sample_id': all_sample_ids, 'video_id': all_ids, 'query': all_queries, 'm3_prog': all_m3_prog}
        m3_prog_save = list(map(lambda x: dict(zip(m3_prog_save.keys(), x)), zip(*m3_prog_save.values())))
        util.save_result(m3_prog_save, results_dir, 'm3_program', remove_duplicate='sample_id',)
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logging.info(f"End run\nElapsed time: {total_time_str}")
        
if __name__  == '__main__':
    main()