import json
import os
import random
import torch
import pickle
import pandas as pd
from torch.utils.data import Dataset
import decord
from decord import cpu, gpu
import numpy as np
import spacy
import logging
import time
import datetime

from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
import numpy as np

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

class ActivityNetQADataset(Dataset):
    def __init__(self, split, data_path="", max_samples=None, fps=30, max_num_frames=30, start_sample=0, **kwargs):
        """
        Args:
            split (str): Data split.
            data_path (str): Path to the data folder
            input_type (str): Type of input. One of ["image", "video"]
            image_transforms (callable, optional): Optional transform to be applied on an image. Only used if input_type
                is "image".
            fps (int): Frames per second. Only used if input_type is "video".
            max_num_frames (int): Maximum number of frames to use. Only used if input_type is "video".
            max_samples (int, optional): Maximum number of samples to load. If None, load all samples.
            start_sample (int, optional): Index of the first sample to load. If None, start from the beginning.
        """

        self.split = split
        self.data_path = data_path
        self.fps = fps
        self.input_type = 'video'
        self.max_num_frames = max_num_frames
        
        question_list_path = os.path.join(self.data_path, f'{split}_q.json')
        answer_list_path = os.path.join(self.data_path, f'{split}_a.json')

        question_list = load_file(question_list_path)
        answer_list = load_file(answer_list_path)
        
        if max_samples is not None:
            # self.sample_list = self.sample_list.sample(n=max_samples)
            question_list = question_list[start_sample: start_sample + max_samples]
            answer_list = answer_list[start_sample: start_sample + max_samples]
        
        self.question_list = question_list
        self.answer_list = answer_list
        
        self.video2id = load_file(os.path.join(self.data_path, 'test_video2id.json'))
        
    def get_video(self, video_path):
        # If fixed width and height are required, VideoReader takes width and height as arguments.
        video_reader = decord.VideoReader(str(video_path), num_threads=1, ctx=cpu(0))
        decord.bridge.set_bridge('torch')
        vlen = len(video_reader)
        original_fps = video_reader.get_avg_fps()
        num_frames = int(vlen * self.fps / original_fps)
        # num_frames = min(self.max_num_frames, num_frames)
        frame_idxs = np.linspace(0, vlen, num_frames, endpoint=False).astype(np.int)
        video = video_reader.get_batch(frame_idxs).byte()
        video = video.permute(0, 3, 1, 2)
        return video

    def __getitem__(self, idx):
        sample_id = self.question_list[idx]['question_id']
        assert self.question_list[idx]['question_id'] == self.answer_list[idx]['question_id'], "question answer pair's must be equal"
        
        question =  self.question_list[idx]['question']
        question = question + '?'

        video_name = 'v_' + self.question_list[idx]['video_name']
        video_path = os.path.join(self.data_path, 'videos', self.video2id[video_name])
        # video = self.get_video(video_path)
        video = torch.load(os.path.join(self.data_path, 'video_encoded', video_name + '.pkl'))
        
        answer = self.answer_list[idx]['answer']
        
        query_type = str(self.answer_list[idx]['type'])
        
        out_dict = {'sample_id': sample_id, 'answer': answer, 'image': video, 'query': question, 'pil_img': -1,
                    "query_type": query_type, 'index': idx, 'possible_answers': [],
                    'extra_context': [], 'video_id': video_name, 'video_path': video_path}

        return out_dict

    def __len__(self):
        return len(self.question_list)

    
    def accuracy(self, prediction, ground_truth, possible_answers, query_type):
        assert len(prediction) == len(ground_truth)
        corrections= []
        num = {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': 0, '8': 0}
        num_all = 0
        # accuracy init
        acc = {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': 0, '8': 0}
        acc_all = 0
        for p, g, qt in zip(prediction, ground_truth, query_type):
            num[qt] += 1
            num_all += 1
            
            # accuaracy
            if p.lower() == g.lower():
                acc_all += 1
                acc[qt] += 1
                corrections.append(True)
            else:
                corrections.append(False)
        for qt in acc.keys():
            acc[qt] = round(acc[qt] / num[qt],4) if num[qt] > 0 else None
        return acc_all / num_all, corrections, acc

    def report_wups(self, prediction, ground_truth, possible_answers, query_type):
        assert len(prediction) == len(ground_truth)
        wups = []
        num = {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': 0, '8': 0}
        num_all = 0
        # wups init
        wups0 = {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': 0, '8': 0}
        wups9 = {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': 0, '8': 0}
        wups0_all, wups9_all = 0, 0
        
        for p, g, qt in zip(prediction, ground_truth, query_type):
            num[qt] += 1
            num_all += 1
            
            # wups
            s0 = get_wups(p, g, 0)
            s9 = get_wups(p, g, 0.9)
            wups0[qt] += s0
            wups9[qt] += s9
            wups0_all += s0
            wups9_all += s9
            wups.append({'wups0': s0, 'wups9': s9})
        
        for qt in wups0.keys():
            wups0[qt] = round(wups0[qt] / num[qt],4) if num[qt] > 0 else None
            wups9[qt] = round(wups9[qt] / num[qt],4) if num[qt] > 0 else None
        return wups0_all / num_all, wups9_all / num_all, wups, wups0, wups9

def wup(word1, word2, alpha):
    """
    calculate the wup similarity
    :param word1:
    :param word2:
    :param alpha:
    :return:
    """
    # print(word1, word2)
    if word1 == word2:
        return 1.0

    w1 = wordnet.synsets(word1)
    w1_len = len(w1)
    if w1_len == 0: return 0.0
    w2 = wordnet.synsets(word2)
    w2_len = len(w2)
    if w2_len == 0: return 0.0

    #match the first
    word_sim = w1[0].wup_similarity(w2[0])
    if word_sim is None:
        word_sim = 0.0

    if word_sim < alpha:
        word_sim = 0.1*word_sim
    return word_sim


def wups(words1, words2, alpha):
    """
    :param pred:
    :param truth:
    :param alpha:
    :return:
    """
    sim = 1.0
    flag = False
    for w1 in words1:
        max_sim = 0
        for w2 in words2:
            word_sim = wup(w1, w2, alpha)
            if word_sim > max_sim:
                max_sim = word_sim
        if max_sim == 0: continue
        sim *= max_sim
        flag = True
    if not flag:
        sim = 0.0
    return sim


def get_wups(pred, truth, alpha):
    """
    calculate the wups score
    :param pred:
    :param truth:
    :return:
    """
    pred = word_tokenize(pred)
    truth = word_tokenize(truth)
    item1 = wups(pred, truth, alpha)
    item2 = wups(truth, pred, alpha)
    value = min(item1, item2)
    return value
