import json
import os
import random
import pandas as pd
from torch.utils.data import Dataset
import decord
from decord import cpu, gpu
import numpy as np
import spacy

from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
import numpy as np

from pywsd.utils import lemmatize_sentence
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


def save_file(obj, filename):
    """
    save obj to filename
    :param obj:
    :param filename:
    :return:
    """
    filepath = os.path.dirname(filename)
    if filepath != '' and not os.path.exists(filepath):
        os.makedirs(filepath)
    else:
        with open(filename, 'w') as fp:
            json.dump(obj, fp, indent=4)


class NExTGQADataset(Dataset):
    def __init__(self, split, data_path="", tokenize=None, max_samples=None, version='openended', fps=30,
                 max_num_frames=30, start_sample=0, **kwargs):

        assert version in ['openended', 'multiplechoice']
        
        # directory = 'nextqa' if version == 'multiplechoice' else 'nextoe'
        directory = '' if version == 'multiplechoice' else 'nextoe' # unlike NExT-QA, the files are within the folder
        
        if directory == 'nextoe':
            raise ValueError(f"NExT-GQA open-ended not built")
        
        print("Loading spacy..")
        self.nlp = spacy.load('en_core_web_lg')
        print("Finished loading spacy..")

        self.split = split
        self.data_path = data_path
        
        # NExT-GQA shares videos with NExT-QA. If video_path not specified then read from its own dataset directory
        video_path = kwargs.get('video_path', None)
        self.video_path = data_path if video_path == None else video_path
        
        self.tokenize = tokenize
        self.version = version
        self.fps = fps
        self.input_type = 'video'
        self.max_num_frames = max_num_frames

        sample_list_path = os.path.join(self.data_path, directory, f'{split}.csv')
        self.sample_list = load_file(sample_list_path)

        if max_samples is not None:
            # self.sample_list = self.sample_list.sample(n=max_samples)
            self.sample_list = self.sample_list[start_sample:start_sample+max_samples]

        self.sample_ids = self.sample_list.index
        self.sample_id_to_index = {sample_id: idx for idx, sample_id in enumerate(self.sample_ids)}

        self.video_to_dir = {}
        for directory in os.listdir(os.path.join(self.video_path, 'videos')):
            for video in os.listdir(os.path.join(self.video_path, 'videos', directory)):
                self.video_to_dir[video.split('.')[0]] = directory

        timespan_anno_path = os.path.join(self.data_path, f'gsub_{split}.json')
        self.timespan_anno = load_file(timespan_anno_path)

    
    def get_sample_path(self, index):
        sample_id = self.sample_ids[index]
        cur_sample = self.sample_list.loc[sample_id]
        video_name = str(cur_sample['video'])
        video_path = os.path.join(self.data_path, 'videos', self.video_to_dir[video_name], video_name + '.mp4')
        return video_path

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
        video_path = os.path.join(self.video_path, 'videos', self.video_to_dir[video_name], video_name + '.mp4')
        video = self.get_video(video_path)

        if self.version == 'openended':
            answer = str(cur_sample['answer'])
            if self.tokenize:
                answer = self.tokenize(answer)
            possible_answers = ''
        else:  # multiple choice
            possible_answers = [str(cur_sample[f'a{i}']) for i in range(5)]
            answer = cur_sample['answer']

        query_type = str(cur_sample['type'])
        qid = str(cur_sample['qid'])
        timespan = self.timespan_anno[video_name]['location'][qid]

        out_dict = {"sample_id": sample_id, "answer": answer, "image": video, "query": question, 'pil_img': -1,
                    "query_type": query_type, 'index': idx, 'possible_answers': possible_answers,
                    'extra_context': possible_answers, 'video_id': video_name, 'video_path': video_path, 'timespan' : timespan}

        return out_dict

    def __len__(self):
        return self.sample_list.shape[0]

    def get_index_from_sample_id(self, sample_id):
        return self.sample_id_to_index[sample_id]

    def get_img_path(self, index):
        sample_id = self.sample_ids[index]
        cur_sample = self.sample_list.loc[sample_id]
        video_name = str(cur_sample['video'])
        video_path = os.path.join(self.data_path, 'videos', self.video_to_dir[video_name], video_name + '.mp4')
        return video_path

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
        if self.version == 'openended':
            for p, g, qt in zip(prediction, ground_truth, query_type):
                if isinstance(p, list) or isinstance(p, tuple):
                    p = p[0]  # p[1] is the info dict
                if p is None:
                    print('None case')
                    p = 'object'  # To select some word
                if qt == 'DC' or qt == 'DB':
                    s = 1 if remove_stop(p) == remove_stop(g) else 0
                else:
                    s = get_wups(remove_stop(p), remove_stop(g), 0)
                score += 100 * s
        else:
            nlp = self.nlp
            for p, g, a in zip(prediction, ground_truth, possible_answers):
                if isinstance(p, list) or isinstance(p, tuple):
                    if len(p) == 2:
                        p = p[0]  # p[1] is the info dict
                    else:  # Multiple predictions
                        all_answers = []
                        for pp in p:
                            if pp not in a:
                                pred_tokens = nlp(pp)
                                a.sort(key=lambda x: pred_tokens.similarity(nlp(x)), reverse=True)
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
                    if int(p) > 5 or int(p) < 0:
                        p = random.randint(1,5)
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
                if p not in a:
                    if p is None:
                        print('None case')  # Should not happen
                    else:
                        pred_tokens = nlp(p)
                        a.sort(key=lambda x: pred_tokens.similarity(nlp(x)), reverse=True)
                    p = a[0]
                if p == g:
                    score += 1
                    corrections.append(True)
                else:
                    corrections.append(False)
        return score / len(prediction), corrections
    
    
    def grounding(self, prediction, ground_truth, possible_answers, query_type, loc_prediction, loc_ground_truth, num_frames):
        """
        Args:
            prediction (list): List of predicted answers.
            ground_truth (list): List of ground truth answers.
            possible_answers (list): List of possible answers.
            query_type (list): List of query types
            loc_prediction (list): List of predicted answers.
            loc_ground_truth (list): List of ground truth answers.
            all_num_frames (list): List of num frames
        Returns:
            grounding result (dict): Result of IoU and IoP for different threshold.
        """

        assert len(loc_prediction) == len(loc_ground_truth)
        mIoU, mIoP = 0, 0
        
        cnt, cnt_empty = 0, 0
        thresholds = [0.3, 0.5]
        cnt_IoU = {i:0 for i in thresholds}
        cnt_IoP = {i:0 for i in thresholds}
        acc = {i:0 for i in thresholds}
        
        
        for idx, (pred, gt_locs, n_frames) in enumerate(zip(loc_prediction, loc_ground_truth, num_frames)):
            frame_ids = pred['frame_ids']
            
            # For empty frame_ids, consider the entire video
            if frame_ids == []:
                frame_ids = [0, n_frames - 1]
                cnt_empty += 1
            
            # use final frame_ids as the grounded region 
            pred_loc = [frame_ids[0], frame_ids[-1]]
            
            # add zero for initialization
            all_tIoU, all_tIoP = [0], [0]
            
            for loc in gt_locs:
                tIoU, tIoP = get_tIoU(loc, pred_loc)
                all_tIoU.append(tIoU)
                all_tIoP.append(tIoP)
            
            # select one with maximum IoU, IoP 
            max_tIoU = max(all_tIoU)
            max_tIoP = max(all_tIoP)
            
            for thd in thresholds:
                cnt_IoU[thd] += (max_tIoU >= thd)
                cnt_IoP[thd] += (max_tIoP >= thd)
                    
                # Compute Accuracy of the sample for QA
                qa_score, _ = self.accuracy([prediction[idx]], [ground_truth[idx]], [possible_answers[idx]], [query_type[idx]])
                acc[thd] += (qa_score and (max_tIoP >= thd))
                
            cnt += 1
            mIoU += max_tIoU
            mIoP += max_tIoP
                    
        mIoU = mIoU / cnt
        mIoP = mIoP / cnt
        IoU = {k: v / cnt for k,v in cnt_IoU.items()}
        IoP = {k: v / cnt for k,v in cnt_IoP.items()}
        acc = {k: v / cnt for k,v in acc.items()}
        
        return {'mIoU':mIoU, 'mIoP': mIoP, 'IoU': IoU, 'IoP':IoP, 'cnt_empty': cnt_empty, 'acc': acc}
                
        
        

# Below is code from https://github.com/doc-doc/NExT-OE/blob/main/eval_oe.py

def get_tIoU(loc, span):
    """
    Args:
        loc (list): Ground-truth timepan 
        span (list): Predicted timespan 

    Returns:
        IoU (float): Intersection over Union
        IoP (float): Intersection over Prediction 
    """
    if span[0] == span[-1]:
        if loc[0] <= span[0] and span[0] <= loc[1]:
            return 0, 1
        else:
            return 0, 0
    
    span_u =  (min(loc[0], span[0]), max(loc[-1], span[-1]))
    span_i = (max(loc[0], span[0]), min(loc[-1], span[-1]))
    dis_i = (span_i[1] - span_i[0])
    
    if span_u[1] > span_u[0]:
        IoU = dis_i / (span_u[1] - span_u[0]) 
    else: 
        IoU = 0.0
    if span[-1] > span[0]:
        IoP = dis_i / (span[-1] - span[0]) 
    else:
        IoP = 0.0

    return IoU, IoP


stopwords = "i, me, my, myself, we, our, ours, ourselves, you, you're, you've, you'll, you'd, your, yours, yourself, " \
            "yourselves, he, him, his, himself, she, she's, her, hers, herself, it, it's, its, itself, they, them, " \
            "their, theirs, themselves, what, which, who, whom, this, that, that'll, these, those, am, is, are, was, " \
            "were, be, been, being, have, has, had, having, do, does, did, doing, a, an, the, and, but, if, or, " \
            "because, as, until, while, to, from, of, at, for, with, about, into, through, during, again, further, " \
            "then, here, there, when, where, why, how, all, any, each, most, other, some, such, only, own, so, than, " \
            "too, very, s, t, can, will, just, don, don't, should, should've, now, d, ll, m, o, re, ve, y, ain, " \
            "aren, aren't, couldn, couldn't, didn, didn't, doesn, doesn't, hadn, hadn't, hasn, hasn't, haven, " \
            "haven't, isn, isn't, ma, mightn, mightn't, mustn, mustn't, needn, needn't, shan, shan't, shouldn, " \
            "shouldn't, wasn, wasn't, weren, weren't, won, won't, wouldn, wouldn't"


def remove_stop(sentence):

    words = lemmatize_sentence(sentence)
    words = [w for w in words if not w in stopwords]
    return ' '.join(words)


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