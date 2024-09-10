import os
import json
import torch
import math
from tqdm import tqdm
import time
import datetime
import pathlib
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.multiprocessing as mp
from functools import partial
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

from configs import config
from engine.utils import ProgramInterpreter
from util import save_json, FileLoggingConsole, set_seed

def my_collate(batch):
    # Avoid stacking images (different size). Return everything as a list
    to_return = {k: [d[k] for d in batch] for k in batch[0].keys()}
    return to_return

def load_video_context(config, video_id):
    with open(config.video_context, 'r') as f:
        datas = json.load(f)
    contexts = []
    # context formatting
    for frame_idx, caption in zip(datas[video_id]['frame_idx'], datas[video_id]['captions']):
        contexts.append(f"[frame{frame_idx:>4}]caption: {caption}.")
    return '\n'.join(contexts)

def main():
    mp.set_start_method('spawn')
    from datasets import get_dataset
    now = datetime.datetime.now()
    time_str = now.strftime("%Y-%m-%d_%H-%M")

    results_dir = pathlib.Path(config.results_dir)
    results_dir = results_dir / config.dataset.dataset_name / config.dataset.split / config.mode / f"{config.exp_name}_{time_str}"
    results_dir.mkdir(parents=True, exist_ok=True)
    filename_json = os.path.join(results_dir, "results.json")
    
    console = FileLoggingConsole(path=os.path.join(results_dir,"results.log"), highlight=False, record=True)
    console.log(OmegaConf.to_container(config))

    set_seed(config.seed)

    batch_size = config.dataset.batch_size
    num_processes = min(batch_size, 50)
    
    interpreter = ProgramInterpreter(config=config, mode=config.mode)
    
    dataset = get_dataset(config.dataset)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True,
                            collate_fn=my_collate)

    all_results = []
    all_answers = []
    all_possible_answers = []
    all_queries = []
    all_query_types = []
    all_ids = []

    start_time = time.time()
    console.log("Start run")
    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        # create Final prediction input
        Final_input = [{'video_context': load_video_context(config, video_id), 'VLM_answer': '','question': question, 'option': option } \
                            for video_id, question, option in zip(batch['video_id'], batch['query'], batch['possible_answers'])]
        # execute Final prediction then return final_answer
        final_answers = interpreter.step_interpreters['llama'].generate(Final_input, prompt_type='final')

        # update list
        all_results += final_answers
        all_answers += batch['answer']
        all_possible_answers += batch['possible_answers']
        all_queries += batch['query']
        all_query_types += batch['query_type']
        all_ids += batch['video_id']
        
        if (i + 1) % config.log_every == 0:
            try:
                accuracy, all_corrections = dataset.accuracy(all_results, all_answers, all_possible_answers, all_query_types)
                console.log(f'Accuracy at Batch {i}/{len(dataloader)}: {accuracy}')
            except Exception as e:
                console.log(f'Error computing accuracy: {e}')
                
            if config.save:
                final_datas = {"video_id": all_ids, "query": all_queries, "query_type": all_query_types, "answer": all_answers,
                            "possible_answer": all_possible_answers, "result": all_results, "correction": all_corrections}
                final_datas = list(map(lambda x: dict(zip(final_datas.keys(), x)), zip(*final_datas.values())))
                save_json(final_datas, filename_json)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    console.log(f"End run\nElapsed time: {total_time_str}")

    try:
        accuracy, all_corrections = dataset.accuracy(all_results, all_answers, all_possible_answers, all_query_types)
        console.log(f'Final accuracy: {accuracy}')
    except Exception as e:
        print(f'Error computing accuracy: {e}')

    if config.save:
        final_datas = {"video_id": all_ids, "query": all_queries, "query_type": all_query_types, "answer": all_answers,
                       "possible_answer": all_possible_answers, "result": all_results, "correction": all_corrections}
        final_datas = list(map(lambda x: dict(zip(final_datas.keys(), x)), zip(*final_datas.values())))
        save_json(final_datas, filename_json)
        
if __name__  == '__main__':
    main()