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
    contexts.append(datas_vid[video_id])
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

    all_s4_prog = []
    all_s3_prog = []
    all_s2_prog = []
    all_s1_prog = []
    all_timespan = []

    start_time = time.time()
    logging.info('Start run')
    metric_logger = util.MetricLogger(delimiter='  ')
    header = '[Retrieve baseline]'
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
                                    'anchor_frame_ids': [],
                                    'event_queue': ['none','none'],
                                    'conjunction': 'none',
                                    'require_ocr': False,
                                    'qa_type': '',
                                    'planning_error': None,
                                    'error': None,
                                    'VLM_answers': {'video': [], 'image': []},
                                    'planning': None,
                                    'S1_understanding': None,
                                    'S2_understanding': None,
                                    'S3_understanding': None,
                                    'S4_understanding': None,
                                    'process_planning': True,
                                    'process_stage1': True,
                                    'process_stage2': True,
                                    'process_stage3': True,
                                    'process_stage4': True})
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
        
        # # Global planning
        # logging.info('Start global planning')
        # planning_input = [{'question': memory['question'], 'is_process': memory['process_planning']} for memory in EXTERNAL_MEMORY]
        # EXTERNAL_MEMORY = Global_planning(config, EXTERNAL_MEMORY, device=device, data=planning_input, prompt_type='planning')
        
        # Stage1 understanding and program generation
        logging.info('Start stage1 understanding and program generation')
        S1_input = [{'question': memory['question'], 'is_process': memory['process_stage1']} for memory in EXTERNAL_MEMORY]
        S1_understanding, S1_programs = Understanding_generation(config, device=device, data=S1_input, prompt_type='stage1')
        
        # update External Memory (understanding)
        for idx, S1_und in enumerate(S1_understanding): EXTERNAL_MEMORY[idx]['S1_understanding'] = S1_und

        # Stage1 processing then update External Memory
        logging.info('Start stage1 processing')
        S1_input = [{'program': program} for program in S1_programs]
        EXTERNAL_MEMORY = Stage1(config, EXTERNAL_MEMORY, data=S1_input, stage='stage1', device=device)
        
        # Stage2 understanding and program generation
        logging.info('Start stage2 understanding and program generation')
        S2_input = [{'question': memory['event_queue'][0], 'image': image, 'is_process': memory['process_stage2']} for memory, image in zip(EXTERNAL_MEMORY, batch['image'])]
        S2_understanding, S2_programs = Understanding_generation(config, device=device, data=S2_input, prompt_type='stage2')

        # update External Memory (understanding)
        for idx, S2_und in enumerate(S2_understanding): EXTERNAL_MEMORY[idx]['S2_understanding'] = S2_und
        
        # Stage2 processing then update External Memory
        logging.info('Start stage2 processing')
        S2_input = [{'program': program, 'image': image, 'video_path': video_path} for program, image, video_path in zip(S2_programs, batch['image'], batch['video_path'])]
        EXTERNAL_MEMORY = Stage2(config, EXTERNAL_MEMORY, data=S2_input, stage='stage2', device=device)
        
        # Stage3 understanding and program generation
        logging.info('Start stage3 understanding and program generation')
        S3_input = [{'question': memory['event_queue'][1], 'image': image, 'is_process': memory['process_stage3']} for memory, image in zip(EXTERNAL_MEMORY, batch['image'])]
        S3_understanding, S3_programs = Understanding_generation(config, device=device, data=S3_input, prompt_type='stage3')
        
        # update External Memory (understanding)
        for idx, S3_und in enumerate(S3_understanding): EXTERNAL_MEMORY[idx]['S3_understanding'] = S3_und
        
        # Stage3 processing then update External Memory
        logging.info('Start stage3 processing')
        S3_input = [{'program': program, 'image': image, 'video_path': video_path} for program, image, video_path in zip(S3_programs, batch['image'], batch['video_path'])]
        EXTERNAL_MEMORY = Stage3(config, EXTERNAL_MEMORY, data=S3_input, stage='stage3', device=device)
        
        # Stage4 understanding and program generation
        logging.info('Start stage4 understanding and program generation')
        S4_input = [{'question': memory['question'], 'frame_ids': memory['frame_ids'], 'require_ocr': memory['require_ocr'], 'qa_type': memory['qa_type'], 'is_process': memory['process_stage4']} for memory in EXTERNAL_MEMORY]
        S4_understanding, S4_programs = Understanding_generation(config, device=device, data=S4_input, prompt_type='stage4')
        
        # update External Memory (understanding)
        for idx, S4_und in enumerate(S4_understanding): EXTERNAL_MEMORY[idx]['S4_understanding'] = S4_und
        
        # Stage4 processing then update External Memory (imageQA)
        logging.info('Start stage4[image] processing')
        S4_input = [{'program': program, 'image': image} for program, image in zip(S4_programs, batch['image'])]
        EXTERNAL_MEMORY = Stage4_image(config, EXTERNAL_MEMORY, data=S4_input, stage='stage4_image', device=device)

        # Stage4 processing then update External Memory (videoQA)
        logging.info('Start stage4[video] processing')
        S4_input = [{'program': program, 'image': image, 'video_path': video_path} for program, image, video_path in zip(S4_programs, batch['image'], batch['video_path'])]
        EXTERNAL_MEMORY = Stage4_video(config, EXTERNAL_MEMORY, data=S4_input, stage='stage4_video', device=device)

        # Final prediction
        logging.info('Start final prediction')
        Final_input = [{'video_context': load_video_context(config, video_id, memory['VLM_answers']), 'question': question, 'option': option, 'is_process': True} \
                                for video_id, question, option, memory in zip(batch['video_id'], batch['query'], batch['possible_answers'], EXTERNAL_MEMORY)]
        Final_predictions = Program_generation(config, device=device, data=Final_input, prompt_type='final')
        
        # update information
        all_results += Final_predictions
        all_memories += EXTERNAL_MEMORY

        # update s1, s2, m3 program (intermediate generated program)
        s4_prog_list = [i.split('\n') for i in S4_programs]
        s3_prog_list = [i.split('\n') for i in S3_programs]
        s2_prog_list = [i.split('\n') for i in S2_programs]
        s1_prog_list = [i.split('\n') for i in S1_programs]
        all_s4_prog += s4_prog_list
        all_s3_prog += s3_prog_list
        all_s2_prog += s2_prog_list
        all_s1_prog += s1_prog_list
        
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

            s1_prog_save = {'sample_id': all_sample_ids, 'video_id': all_ids, 'query': all_queries, 's1_prog': all_s1_prog}
            s1_prog_save = list(map(lambda x: dict(zip(s1_prog_save.keys(), x)), zip(*s1_prog_save.values())))
            util.save_result(s1_prog_save, results_dir, 's1_program', remove_duplicate='sample_id',)
            
            s2_prog_save = {'sample_id': all_sample_ids, 'video_id': all_ids, 'query': all_queries, 's2_prog': all_s2_prog}
            s2_prog_save = list(map(lambda x: dict(zip(s2_prog_save.keys(), x)), zip(*s2_prog_save.values())))
            util.save_result(s2_prog_save, results_dir, 's2_program', remove_duplicate='sample_id',)

            s3_prog_save = {'sample_id': all_sample_ids, 'video_id': all_ids, 'query': all_queries, 's3_prog': all_s3_prog}
            s3_prog_save = list(map(lambda x: dict(zip(s3_prog_save.keys(), x)), zip(*s3_prog_save.values())))
            util.save_result(s3_prog_save, results_dir, 's3_program', remove_duplicate='sample_id',)

            s4_prog_save = {'sample_id': all_sample_ids, 'video_id': all_ids, 'query': all_queries, 's4_prog': all_s4_prog}
            s4_prog_save = list(map(lambda x: dict(zip(s4_prog_save.keys(), x)), zip(*s4_prog_save.values())))
            util.save_result(s4_prog_save, results_dir, 's4_program', remove_duplicate='sample_id',)

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

        s1_prog_save = {'sample_id': all_sample_ids, 'video_id': all_ids, 'query': all_queries, 's1_prog': all_s1_prog}
        s1_prog_save = list(map(lambda x: dict(zip(s1_prog_save.keys(), x)), zip(*s1_prog_save.values())))
        util.save_result(s1_prog_save, results_dir, 's1_program', remove_duplicate='sample_id',)
        
        s2_prog_save = {'sample_id': all_sample_ids, 'video_id': all_ids, 'query': all_queries, 's2_prog': all_s2_prog}
        s2_prog_save = list(map(lambda x: dict(zip(s2_prog_save.keys(), x)), zip(*s2_prog_save.values())))
        util.save_result(s2_prog_save, results_dir, 's2_program', remove_duplicate='sample_id',)

        s3_prog_save = {'sample_id': all_sample_ids, 'video_id': all_ids, 'query': all_queries, 's3_prog': all_s3_prog}
        s3_prog_save = list(map(lambda x: dict(zip(s3_prog_save.keys(), x)), zip(*s3_prog_save.values())))
        util.save_result(s3_prog_save, results_dir, 's3_program', remove_duplicate='sample_id',)

        s4_prog_save = {'sample_id': all_sample_ids, 'video_id': all_ids, 'query': all_queries, 's4_prog': all_s4_prog}
        s4_prog_save = list(map(lambda x: dict(zip(s4_prog_save.keys(), x)), zip(*s4_prog_save.values())))
        util.save_result(s4_prog_save, results_dir, 's4_program', remove_duplicate='sample_id',)
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logging.info(f"End run\nElapsed time: {total_time_str}")
        
if __name__  == '__main__':
    main()