import json
import os
import random
import pandas as pd
from torch.utils.data import Dataset
import decord
from decord import cpu, gpu
import numpy as np
import spacy

import numpy as np

from collections import Counter

def load_file(file_name):
    annos = None
    if os.path.splitext(file_name)[-1] == '.csv':
        return pd.read_csv(file_name)
    with open(file_name, 'r') as fp:
        if os.path.splitext(file_name)[1]== '.txt':
            annos = fp.readlines()
            annos = [line.rstrip() for line in annos]
        if os.path.splitext(file_name)[1] == '.json':
            annos = json.load(fp)

    return annos

class IntentQADataset(Dataset):
    def __init__(self, split, data_path="", tokenize=None, max_samples=None, version='multiplechoice', fps=30,
                 max_num_frames=30, start_sample=0, **kwargs):
        
        self.split = split
        self.split = split
        self.data_path = data_path
        self.tokenize = tokenize
        self.version = version
        self.fps = fps
        self.input_type = 'video'
        self.max_num_frames = max_num_frames
        self.num_options = 5
        
        sample_list_path = os.path.join(self.data_path, f'{split}.csv')
        self.sample_list = load_file(sample_list_path)
        
        if max_samples is not None:
            # self.sample_list = self.sample_list.sample(n=max_samples)
            self.sample_list = self.sample_list[start_sample:start_sample+max_samples]
        
        self.sample_ids = self.sample_list.index
        self.sample_id_to_index = {sample_id: idx for idx, sample_id in enumerate(self.sample_ids)}

        print("Loading spacy..")
        self.nlp = spacy.load('en_core_web_lg')
        print("Finished loading spacy..")

    def get_video(self, video_path):
        # If fixed width and height are required, VideoReader takes width and height as arguments.
        video_reader = decord.VideoReader(video_path, num_threads=1, ctx=cpu(0))
        decord.bridge.set_bridge('torch')
        vlen = len(video_reader)
        original_fps = video_reader.get_avg_fps()
        num_frames = int(vlen * self.fps / original_fps)
        # num_frames = min(self.max_num_frames, num_frames) # do not set maximum length
        frame_idxs = np.linspace(0, vlen, num_frames, endpoint=False).astype(np.int)
        video = video_reader.get_batch(frame_idxs).byte()
        video = video.permute(0, 3, 1, 2)
        return video

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        cur_sample = self.sample_list.loc[sample_id]
        
        question = str(cur_sample['question'])
        if self.tokenize:
            question = self.tokenize(question)
        question = question + '?'
        video_name = str(cur_sample['video_id'])
        video_path = os.path.join(self.data_path, 'videos', video_name + '.mp4')
        video = self.get_video(video_path)
        
        assert self.version == 'multiplechoice'
        answer_idx = int(cur_sample['answer'])
        possible_answers = [str(cur_sample[f'a{i}']) for i in range(self.num_options)]
        answer = possible_answers[answer_idx]
        
        query_type = str(cur_sample['type'])
        qid = str(cur_sample['qid'])
        
        out_dict = {"sample_id": sample_id, "answer": answer, "image": video, "query": question, 'pil_img': -1,
                    "query_type": query_type, 'index': idx, 'possible_answers': possible_answers,
                    'extra_context': possible_answers, 'video_id': video_name, 'video_path': video_path}
        return out_dict
    
    def __len__(self):
        return self.sample_list.shape[0]
    
    def accuracy(self, prediction, ground_truth, possible_answers, query_type):
        """
        Args:
            prediction (list): List of predicted answers.
            ground_truth (list): List of ground truth answers.
            possible_answers (list): List of possible answers.
            query_type (list): List of query types
        Returns:
            score (float): Score of the prediction.
        """
        
        assert len(prediction) == len(ground_truth)
        score = 0
        corrections = []

        num = {'CW': 0, 'CH': 0, 'TN': 0, 'TC': 0}
        num_over = {'C': 0, 'T': 0}
        num_all = 0
        # accuracy init
        acc = {'CW': 0, 'CH': 0, 'TN': 0, 'TC': 0}
        acc_over = {'C': 0, 'T': 0}
        if self.version == 'openended':
            for p, g, qt in zip(prediction, ground_truth, query_type):
                if qt == 'TP':
                    qt = 'TN'
                num_all += 1
                num_over[qt[0]] += 1
                num[qt] += 1
                
                # accuracy
                if p.lower() == g.lower():
                    score += 1
                    acc_over[qt[0]] += 1
                    acc[qt] += 1
                    corrections.append(True)
                else:
                    corrections.append(False)
            for qt in acc.keys():
                acc[qt] = round(acc[qt] / num[qt],4) if num[qt] > 0 else None
            for qt in acc_over.keys():
                acc_over[qt] = round(acc_over[qt] / num_over[qt],4) if num_over[qt] > 0 else None
        else: # multiplechoice
            for p, g, a, qt in zip(prediction, ground_truth, possible_answers, query_type):
                if qt == 'TP':
                    qt = 'TN'
                num_all += 1
                num_over[qt[0]] += 1
                num[qt] += 1
                if isinstance(p, list) or isinstance(p, tuple):
                    if len(p) == 2:
                        p = p[0]  # p[1] is the info dict
                    else:  # Multiple predictions
                        all_answers = []
                        for pp in p:
                            if pp not in a:
                                pred_tokens = self.nlp(pp)
                                a.sort(key=lambda x: pred_tokens.similarity(self.nlp(x)), reverse=True)
                                pp = a[0]
                            all_answers.append(pp)
                        # Majority vote
                        c = Counter(all_answers).most_common(1)[0]
                        if c[1] == 1:
                            # If no majority, select the middle one
                            p = all_answers[1]
                        else:
                            p = c[0]
                if p.isdigit():
                    if int(p) > self.num_options or int(p) < 0:
                        p = random.randint(1,self.num_options)
                    p = a[int(p)-1]
                if '(A)' in p:
                    p = a[0]
                elif '(B)' in p:
                    p = a[1]
                elif '(C)' in p:
                    p = a[2]
                elif '(D)' in p:
                    p = a[3]
                elif '(E)' in p:
                    p = a[4]
                if len(p) == 1:
                    if 'A' == p:
                        p = a[0]
                    elif 'B' == p:
                        p = a[1]
                    elif 'C' == p:
                        p = a[2]
                    elif 'D' == p:
                        p = a[3]
                    elif 'E' == p:
                        p = a[4]
                if p not in a:
                    if p is None:
                        print('None case')  # Should not happen
                    else:
                        pred_tokens = self.nlp(p)
                        a.sort(key=lambda x: pred_tokens.similarity(self.nlp(x)), reverse=True)
                    p = a[0]
                if p == g:
                    score += 1
                    acc_over[qt[0]] += 1
                    acc[qt] += 1
                    corrections.append(True)
                else:
                    corrections.append(False)
            for qt in acc.keys():
                acc[qt] = round(acc[qt] / num[qt],4) if num[qt] > 0 else None
            for qt in acc_over.keys():
                acc_over[qt] = round(acc_over[qt] / num_over[qt],4) if num_over[qt] > 0 else None
        # return score / len(prediction), corrections, acc
        return score / len(prediction), corrections, acc_over | acc