import math
import torch
import torch.distributed as dist
from .utils import ProgramInterpreter
from .step_interpreters import unload_model
import util

def Stage1(config, EXTERNAL_MEMORY, **kwargs):
    # initial interpreter
    interpreter = ProgramInterpreter(config, mode=config.mode, stage=kwargs['stage'], device=kwargs['device'])
    # make data iterable (bsz: 1)
    dataset = util.CustomDataset(kwargs['data'])
    dataloader = util.make_loader(dataset, 1, config)

    metric_logger = util.MetricLogger(delimiter='  ')
    header = '[Stage1 processing]'

    # External Memory update using program execution output (iterate over only batch 1)
    for i, data in enumerate(metric_logger.log_every(dataloader, config.log_freq, header)):
        # only continue stage2 if process_stage1==True
        if EXTERNAL_MEMORY[i]['process_stage1']:
            try:
                # get output by program execution
                final_output, output_state = interpreter.execute(data['program'][0], init_state=None) # assume only batch 1
                # update 'question' and 'frame_ids' field of External Memory when trim() != 'none'
                if output_state['TRIM0']['cue'] != 'none':
                    # update 'frame_ids' field
                    num_frames = int(len(EXTERNAL_MEMORY[i]['frame_ids'])*0.4) # like MoReVQA paper, mentioned 'truncating 40%'
                    if output_state['TRIM0']['cue'] == 'beginning':
                        start_idx = 0
                    elif output_state['TRIM0']['cue'] == 'middle':
                        start_idx = round(len(EXTERNAL_MEMORY[i]['frame_ids'])*0.3)
                    elif output_state['TRIM0']['cue'] == 'end':
                        start_idx = round(len(EXTERNAL_MEMORY[i]['frame_ids'])*0.6)
                    else:
                        raise Exception('wrong cue option')
                    EXTERNAL_MEMORY[i]['frame_ids'] = [i for i in range(start_idx, start_idx + num_frames)]
                
                # update 'conjunction' field of External Memory
                EXTERNAL_MEMORY[i]['conjunction'] = output_state['PARSE_EVENT0']['conj']
                
                # update 'question' field of External Memory
                EXTERNAL_MEMORY[i]['question'] = output_state['PARSE_EVENT0']['independent_clause']
                    
                # update 'event_queue' field of External Memory
                EXTERNAL_MEMORY[i]['event_queue'] = [output_state['PARSE_EVENT0']['dependent_clause'], output_state['PARSE_EVENT0']['independent_clause']]
                
                # global planning을 작성하기 전에 임시로 작성
                if output_state['PARSE_EVENT0']['dependent_clause'] == 'none':
                    EXTERNAL_MEMORY[i]['process_stage2'] = False
                
                # update 'require_ocr' field of External Memory
                if output_state['REQUIRE_OCR0'] != 'no':
                    EXTERNAL_MEMORY[i]['require_ocr'] = True

                # update 'qa_type' field of External Memory
                EXTERNAL_MEMORY[i]['qa_type'] = output_state['CLASSIFY0']
            except:
                EXTERNAL_MEMORY[i]['frame_ids'] = []
                EXTERNAL_MEMORY[i]['event_queue'] = ['none', EXTERNAL_MEMORY[i]['question']]
                EXTERNAL_MEMORY[i]['qa_type'] = EXTERNAL_MEMORY[i]['question'].split(' ')[0]
                EXTERNAL_MEMORY[i]['error'] = 'stage1'

    metric_logger.synchronize_between_processes()
    if util.is_dist_avail_and_initialized():
        dist.barrier()
    
    return EXTERNAL_MEMORY

def Stage2(config, EXTERNAL_MEMORY, **kwargs):
    # initial interpreter
    interpreter = ProgramInterpreter(config, mode=config.mode, stage=kwargs['stage'], device=kwargs['device'])
    # make data iterable (bsz: 1)
    dataset = util.CustomDataset(kwargs['data'])
    dataloader = util.make_loader(dataset, 1, config)

    metric_logger = util.MetricLogger(delimiter='  ')
    header = '[Stage2 processing]'
    # External Memory update using program execution output
    for i, data in enumerate(metric_logger.log_every(dataloader, config.log_freq, header)):
        # only continue stage2 if process_stage2==True
        if EXTERNAL_MEMORY[i]['process_stage2']:
            # initialize frame_id (indicator)
            frames = data['image'][0]
            indicator = torch.zeros(frames.size(0))
            indicator[EXTERNAL_MEMORY[i]['frame_ids']] = 1
            try:
                # get output by program execution
                final_output, output_state = interpreter.execute(data['program'][0], init_state={'video_path': data['video_path'][0],
                                                                                                'image': frames,
                                                                                                'indicator': indicator.bool(),
                                                                                                'conj': 'none',
                                                                                                'qa_type': None},) # assume only batch 1
                # update 'anchor_frame_ids' field
                EXTERNAL_MEMORY[i]['anchor_frame_ids'] = output_state['RETRIEVE0']
            except:
                EXTERNAL_MEMORY[i]['anchor_frame_ids'] = []
                if EXTERNAL_MEMORY[i]['error'] == None:
                    EXTERNAL_MEMORY[i]['error'] = 'stage2'
                    
            # continue truncate function if stage2 program exists
            try:
                # program is same. only 'conjunction; change
                program = 'TRUNCATE0=truncate(conj="{conj}", anchor=RETRIEVE0)'.format(conj=EXTERNAL_MEMORY[i]['conjunction']) 
                # get output by program execution
                final_output, output_state = interpreter.execute(program, init_state={'indicator': indicator.bool(),
                                                                                        'RETRIEVE0': EXTERNAL_MEMORY[i]['anchor_frame_ids']},) # assume only batch 1
                # update 'frame_ids' field
                EXTERNAL_MEMORY[i]['frame_ids'] = output_state['TRUNCATE0']
            except:
                if EXTERNAL_MEMORY[i]['error'] == None:
                    EXTERNAL_MEMORY[i]['error'] = 'truncate'

    metric_logger.synchronize_between_processes()
    if util.is_dist_avail_and_initialized():
        dist.barrier()
    # unload model
    loaded_model = interpreter.loaded_model
    for step_name, model in loaded_model.items():
        unload_model(model)

    return EXTERNAL_MEMORY

def Stage3(config, EXTERNAL_MEMORY, **kwargs):
    # initial interpreter
    interpreter = ProgramInterpreter(config, mode=config.mode, stage=kwargs['stage'], device=kwargs['device'])
    # make data iterable (bsz: 1)
    dataset = util.CustomDataset(kwargs['data'])
    dataloader = util.make_loader(dataset, 1, config)

    metric_logger = util.MetricLogger(delimiter='  ')
    header = '[Stage3 processing]'
    # External Memory update using program execution output
    for i, data in enumerate(metric_logger.log_every(dataloader, config.log_freq, header)):
        # only continue stage2 if process_stage3==True
        if EXTERNAL_MEMORY[i]['process_stage3']:
            # initialize frame_id (indicator)
            frames = data['image'][0]
            indicator = torch.zeros(frames.size(0))
            indicator[EXTERNAL_MEMORY[i]['frame_ids']] = 1
            try:
                # get output by program execution
                final_output, output_state = interpreter.execute(data['program'][0], init_state={'video_path': data['video_path'][0],
                                                                                                'image': frames,
                                                                                                'indicator': indicator.bool(),
                                                                                                'conj': EXTERNAL_MEMORY[i]['conjunction'],
                                                                                                'qa_type': EXTERNAL_MEMORY[i]['qa_type']},) # assume only batch 1
                # update 'frame_ids' field
                EXTERNAL_MEMORY[i]['frame_ids'] = output_state['RETRIEVE0']
            except:
                EXTERNAL_MEMORY[i]['frame_ids'] = []
                if EXTERNAL_MEMORY[i]['error'] == None:
                    EXTERNAL_MEMORY[i]['error'] = 'stage3'
                
    metric_logger.synchronize_between_processes()
    if util.is_dist_avail_and_initialized():
        dist.barrier()
    # unload model
    loaded_model = interpreter.loaded_model
    for step_name, model in loaded_model.items():
        unload_model(model)

    return EXTERNAL_MEMORY

def Stage4_image(config, EXTERNAL_MEMORY, **kwargs):
    # initial interpreter
    interpreter = ProgramInterpreter(config, mode=config.mode, stage=kwargs['stage'], device=kwargs['device'])
    
    dataset = util.CustomDataset(kwargs['data'])
    dataloader = util.make_loader(dataset, 1, config)
    
    metric_logger = util.MetricLogger(delimiter='  ')
    header = '[Stage4 image processing]'
    # External Memory update using program execution output
    for i, data in enumerate(metric_logger.log_every(dataloader, config.log_freq, header)):
        # only continue stage2 if process_stage4==True
        if EXTERNAL_MEMORY[i]['process_stage4']:
            try:
                # get output by program execution
                final_output, output_state = interpreter.execute(data['program'][0], init_state={'image': data['image'][0],
                                                                                                 'frame_ids': EXTERNAL_MEMORY[i]['frame_ids'],}) # assume only batch 
                # determine whether to proeed [video] processing
                if output_state['REQUIRE_VIDEO0'] == 'yes':
                    EXTERNAL_MEMORY[i]['process_video'] = True
                elif output_state['REQUIRE_VIDEO0'] == 'no':
                    EXTERNAL_MEMORY[i]['process_video'] = False
                else:
                    raise Exception('Invalid process_video type')
                
                # image VQA output
                QA_pools = []
                QA_pools += output_state['VQA']
                QA_pools += output_state['IMAGE_VQA']
                
                # sort in ascending order based on frame_id
                sorted_QA_pools = sorted(QA_pools, key=lambda x:x['frame_id'])
                
                # answer formatting
                answers = []
                for qa in sorted_QA_pools:
                    answers.append(f"[frame{qa['frame_id']:>3}]{qa['question']}: {qa['answer']}")
                EXTERNAL_MEMORY[i]['VLM_answers']['image'] = '\n'.join(answers)
            except:
                EXTERNAL_MEMORY[i]['frame_ids'] = []
                if EXTERNAL_MEMORY[i]['error'] == None:
                    EXTERNAL_MEMORY[i]['error'] = 'stage4_image'

    metric_logger.synchronize_between_processes()
    if util.is_dist_avail_and_initialized():
        dist.barrier()
    # unload model
    loaded_model = interpreter.loaded_model
    for step_name, model in loaded_model.items():
        unload_model(model)

    return EXTERNAL_MEMORY

def Stage4_video(config, EXTERNAL_MEMORY, **kwargs):
    # initial interpreter
    interpreter = ProgramInterpreter(config, mode=config.mode, stage=kwargs['stage'], device=kwargs['device'])
    
    dataset = util.CustomDataset(kwargs['data'])
    dataloader = util.make_loader(dataset, 1, config)
    
    metric_logger = util.MetricLogger(delimiter='  ')
    header = '[Stage4 video processing]'
    # External Memory update using program execution output
    for i, data in enumerate(metric_logger.log_every(dataloader, config.log_freq, header)):
        # only continue stage2 if process_stage4==True
        if EXTERNAL_MEMORY[i]['process_stage4'] and EXTERNAL_MEMORY[i]['process_video']:
            try:
                # get output by program execution
                final_output, output_state = interpreter.execute(data['program'][0], init_state={'video_path': data['video_path'][0],
                                                                                                 'frame_ids': EXTERNAL_MEMORY[i]['frame_ids'],}) # assume only batch 1
                # video VQA output
                QA_pools = []
                QA_pools += output_state['VQA']
                QA_pools += output_state['VIDEO_VQA']
                
                # answer formatting
                answers = []
                for qa in QA_pools:
                    answers.append(f"[video]{qa['question']}: {qa['answer']}")
                EXTERNAL_MEMORY[i]['VLM_answers']['video'] = '\n'.join(answers)
            except:
                EXTERNAL_MEMORY[i]['frame_ids'] = []
                if EXTERNAL_MEMORY[i]['error'] == None:
                    EXTERNAL_MEMORY[i]['error'] = 'stage4_video'

    metric_logger.synchronize_between_processes()
    if util.is_dist_avail_and_initialized():
        dist.barrier()
    # unload model
    loaded_model = interpreter.loaded_model
    for step_name, model in loaded_model.items():
        unload_model(model)

    return EXTERNAL_MEMORY