import gc
import os
import torch
import clip
import math
from torchvision import transforms
import numpy as np
import copy
import io, tokenize
import decord
from decord import cpu
from torch.nn import functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, OwlViTProcessor, OwlViTForObjectDetection
from util import load_json

def parse_step(step_str,partial=False):
    tokens = list(tokenize.generate_tokens(io.StringIO(step_str).readline))
    try:
        output_var = tokens[0].string
        step_name = tokens[2].string
        # print(output_var, step_name)
    except:
        raise Exception('Invalid program token parsing!')
    parsed_result = dict(
        output_var=output_var,
        step_name=step_name)
    if partial:
        return parsed_result

    arguments = dict()
    processing_list = False # whether argument type is list
    current_arg = None # current argument name to process
    current_list = [] # list to save value if processing_list=True
    arg_tokens = [token for token in tokens[4:-3] if token.string not in [',','=']]
    token_string = [token.string for token in arg_tokens]
    for i, token in enumerate(token_string):
        if current_arg is None:
            current_arg = token
        elif token == '[':
            processing_list = True
            current_list = []
        elif token == ']':
            processing_list = False
            arguments[current_arg] = current_list
            current_arg = None
        elif processing_list:
            current_list.append(token.strip('"'))
        else:
            arguments[current_arg] = token.strip('"')
            current_arg = None
    parsed_result['args'] = arguments
    return parsed_result

class TRIMInterpreter():
    step_name = 'trim'
    def __init__(self):
        # print(f'Registering {self.step_name} step')
        pass
    
    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        output_var = parse_result['output_var']
        args = parse_result['args']
        trim_option = args['trim']
        truncated_question = None
        if trim_option != 'none':
            truncated_question = args['truncated_question']
        assert(step_name==self.step_name)
        return trim_option, truncated_question, output_var

    def execute(self,prog_step,inspect=False):
        trim_option, truncated_question, output_var = self.parse(prog_step)
        out_value = {'trim': trim_option, 'truncated_question': truncated_question}
        prog_step.state[output_var] = out_value
        return out_value

class PARSEEVENTInterpreter():
    step_name = 'parse_event'
    def __init__(self):
        # print(f'Registering {self.step_name} step')
        pass

    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        output_var = parse_result['output_var']
        args = parse_result['args']
        conj_option = args['conj']
        event_to_localize, truncated_question = None, None
        if conj_option != 'none':
            event_to_localize = args['event_to_localize']
            truncated_question = args['truncated_question']
        assert(step_name==self.step_name)
        return conj_option, event_to_localize, truncated_question, output_var

    def execute(self,prog_step,inspect=False):
        conj_option, event_to_localize, truncated_question, output_var = self.parse(prog_step)
        out_value = {'conj': conj_option, 'event_to_localize': event_to_localize, 'truncated_question': truncated_question}
        prog_step.state[output_var] = out_value
        return out_value

class CLASSIFYInterpreter():
    step_name = 'classify'
    def __init__(self):
        # print(f'Registering {self.step_name} step')
        pass

    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        output_var = parse_result['output_var']
        args = parse_result['args']
        qatype_option = args['type']
        assert(step_name==self.step_name)
        return qatype_option, output_var

    def execute(self,prog_step,inspect=False):
        qatype_option, output_var = self.parse(prog_step)
        out_value = qatype_option
        prog_step.state[output_var] = out_value
        return out_value

class REQUIREOCRInterpreter():
    step_name = 'require_ocr'
    def __init__(self):
        # print(f'Registering {self.step_name} step')
        pass

    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        output_var = parse_result['output_var']
        args = parse_result['args']
        ocr_option = args['bool']
        assert(step_name==self.step_name)
        return ocr_option, output_var

    def execute(self,prog_step,inspect=False):
        ocr_option, output_var = self.parse(prog_step)
        out_value = ocr_option
        prog_step.state[output_var] = out_value
        return out_value

class LOCALIZEInterpreter(torch.nn.Module):
    step_name = 'localize'
    def __init__(self, config, device=None):
        super().__init__()
        # print(f'Registering {self.step_name} step')
        self.dev = device
        
        self.config = config
        # localize model
        localize_model_id = config.owlvit.model_path
        self.localize_processor = OwlViTProcessor.from_pretrained(localize_model_id)
        self.localize_model = OwlViTForObjectDetection.from_pretrained(localize_model_id)
        self.localize_model.eval()
        
        # verify model
        verify_model_id = config.clip.model_path
        assert verify_model_id in ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']
        self.verify_model, self.verify_processor = clip.load(verify_model_id, device=self.dev)
        self.verify_model.eval()
        self.verify_model.requires_grad_ = False
        self.verify_transform = self.get_clip_transforms_from_tensor(336 if "336" in verify_model_id else 224)
        
        self.possible_options = load_json('datas/possible_options.json')
        self.to_PIL = transforms.ToPILImage()
        self.prompt = "a photo of "

    def get_clip_transforms_from_tensor(self, n_px=336):
        def _convert_image_to_rgb(image):
            return image.convert("RGB")
        return transforms.Compose([
            # transforms.ToPILImage(),
            transforms.Resize(n_px, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(n_px),
            # _convert_image_to_rgb,
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
    
    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        output_var = parse_result['output_var']
        args = parse_result['args']
        noun_var = args['noun']
        modifier_var = args['noun_with_modifier']
        assert(step_name==self.step_name)
        return noun_var, modifier_var, output_var

    @torch.no_grad()
    def predict(self, img, noun_obj_name, modifier_obj_name):
        inputs = self.localize_processor(text=[[f'{self.prompt}{noun_obj_name}']],
                                images=img,
                                return_tensors='pt').to(self.dev)
        outputs = self.localize_model(**inputs)
        target_sizes = torch.Tensor([img.size[::-1]])
        results = self.localize_processor.post_process_object_detection(outputs=outputs, threshold=self.config.owlvit.threshold, target_sizes=target_sizes)
        boxes, scores = results[0]["boxes"], results[0]["scores"]
        if boxes.size(0) > 0:
            boxes = boxes.type(torch.int64).tolist()
            if modifier_obj_name != '' and noun_obj_name != modifier_obj_name:
                return self.classify(img, noun_obj_name, modifier_obj_name, boxes)
            else:
                return True
        else:
            return False
    
    @torch.no_grad()
    def classify(self, img, noun_obj_name, modifier_obj_name, boxes):
        negative_categories = [f"{att} {noun_obj_name}" for att in self.possible_options['attributes']]
        for box in boxes:
            crop_img = img.crop(box)
            img_clip = self.verify_transform(crop_img).unsqueeze(0).to(self.dev)
            
            img_features = self.verify_model.encode_image(img_clip)
            img_features = F.normalize(img_features, dim=-1)

            obj_name = self.prompt + modifier_obj_name
            obj_name = clip.tokenize([obj_name]).to(self.dev)

            positive_text_features = self.verify_model.encode_text(obj_name)
            positive_text_features = F.normalize(positive_text_features, dim=-1)
            
            negative_text_features = self.text_negatives(negative_categories)
            
            text_features = torch.concat([positive_text_features, negative_text_features], axis=0)
            
            sim = (100.0 * img_features @ text_features.T).squeeze(dim=0)
            res = F.softmax(torch.cat((sim[0].broadcast_to(1, sim.shape[0] - 1),
                                       sim[1:].unsqueeze(0)), dim=0), dim=0)[0].mean()
            if res.item() > self.config.clip.threshold:
                return True
        return False
        
    @torch.no_grad()
    def text_negatives(self, negative_categories=None):
        if negative_categories is None:
            with open('datas/random_negatives.txt') as f:
                negative_categories = [x.strip() for x in f.read().split()]

        negative_categories = [self.prompt + x for x in negative_categories]
        negative_tokens = clip.tokenize(negative_categories).to(self.dev)

        negative_text_features = self.verify_model.encode_text(negative_tokens)
        negative_text_features = F.normalize(negative_text_features, dim=-1)

        return negative_text_features

    def execute(self,prog_step,inspect=False):
        noun_var, modifier_var, output_var = self.parse(prog_step)
        
        # initialize indicator
        indicator = copy.deepcopy(prog_step.state['indicator'].detach())
        candidate_frame_ids = torch.where(indicator==True)[0].tolist()
        # do not update frame_id when noun=="" and noun_with_modifier==""
        if noun_var == '' and modifier_var == '':
            prog_step.state[output_var] = indicator
            return indicator
        # iterate over images
        for i in candidate_frame_ids:
            img = prog_step.state['image'][i]
            img = self.to_PIL(img)
            noun_obj_name = noun_var
            modifier_obj_name = modifier_var
            # update to False if object is not detected
            indicator[i] = self.predict(img, noun_obj_name, modifier_obj_name)                
        prog_step.state[output_var] = indicator
        return indicator

class TRUNCATEInterpreter():
    step_name = 'truncate'
    def __init__(self):
        # print(f'Registering {self.step_name} step')
        pass
    
    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        output_var = parse_result['output_var']
        args = parse_result['args']
        truncate_option = args['truncate']
        anchor_option = args['anchor']
        assert(step_name==self.step_name)
        return truncate_option, anchor_option, output_var
    
    def execute(self,prog_step,inspect=False):
        truncate_option, anchor_option, output_var = self.parse(prog_step)
        prev_frame_ids = prog_step.state[anchor_option]
        if truncate_option == 'when':
            prog_step.state['indicator'] = torch.zeros(prog_step.state['indicator'].size(0)).bool()
            prog_step.state['indicator'][prev_frame_ids] = True
        elif truncate_option == 'before':
            if len(prev_frame_ids) == 0: # nothing is detected in the previous step
                prog_step.state['indicator'] = torch.zeros(prog_step.state['indicator'].size(0)).bool()
            else:
                anchor_index = min(prev_frame_ids)
                prog_step.state['indicator'][anchor_index:] = False
        elif truncate_option == 'after':
            if len(prev_frame_ids) == 0: # nothing is detected in the previous step
                prog_step.state['indicator'] = torch.zeros(prog_step.state['indicator'].size(0)).bool()
            else:
                anchor_index = max(prev_frame_ids)
                prog_step.state['indicator'][:anchor_index+1] = False
        # if temporal does not exist, escape
        if torch.all(prog_step.state['indicator']==False):
            prog_step.state[output_var] = []
            return []
        frame_id = torch.where(prog_step.state['indicator']==True)[0].tolist()
        prog_step.state[output_var] = frame_id
        return frame_id

class InternVLInterpreter(torch.nn.Module):
    step_name = 'vqa'
    tokenizer = None
    model = None
    def __init__(self, config, device=None):
        super().__init__()
        # print(f'Registering {self.step_name} step')
        self.dev = device
        self.config = config
        model_id = config.internvl.model_path
        if InternVLInterpreter.tokenizer is None or InternVLInterpreter.model is None:
            InternVLInterpreter.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=False)
            InternVLInterpreter.model = AutoModel.from_pretrained(model_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, use_flash_attn=True, trust_remote_code=True)
            InternVLInterpreter.model.eval()
        self.tokenizer = InternVLInterpreter.tokenizer
        self.model = InternVLInterpreter.model
        
        self.prompt = {'image_vqa': '<image>\n{} Answer the question shortly.',
                       'video_vqa': '{}{} Answer the question shortly.',
                       'video_baseline': '{}{}',
                       'verify': '<image>\n{} Answer the question either yes or no.'}
        self.generation_config = dict(max_new_tokens=1024, do_sample=False)
        self.to_PIL = transforms.ToPILImage()
        self.max_batch_size = self.config.internvl.max_batch_size
        
    def build_transform(self, input_size):
        transform = transforms.Compose([
            transforms.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            transforms.Resize((input_size, input_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        return transform

    def find_closest_aspect_ratio(self, aspect_ratio, target_ratios, width, height, image_size):
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    def dynamic_preprocess(self, image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        # calculate the existing image aspect ratio
        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
            i * j <= max_num and i * j >= min_num)
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        # find the closest aspect ratio to the target
        target_aspect_ratio = self.find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size)

        # calculate the target width and height
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        # resize the image
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size
            )
            # split the image
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        assert len(processed_images) == blocks
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        return processed_images

    def load_image(self, image, input_size=448, max_num=12):
        # image = Image.open(image_file).convert('RGB')
        transform = self.build_transform(input_size=input_size)
        images = self.dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        return pixel_values

    @torch.no_grad()
    def predict(self, img, question, prompt_type='image_vqa'):
        if isinstance(question, str):
            question = self.prompt[prompt_type].format(question)
        else:
            raise Exception('invalide question type')
        pixel_values = self.load_image(img, max_num=12).to(torch.bfloat16).to(self.dev)
        output_text = self.model.chat(self.tokenizer, pixel_values, question, self.generation_config, device=self.dev)
        return output_text

    @torch.no_grad()
    def batch_predict(self, imgs, questions, prompt_type='image_vqa'):
        if isinstance(questions, list):
            questions = [self.prompt[prompt_type].format(question) for question in questions]
        else:
            raise Exception('invalide question type')
        imgs = [self.load_image(img, max_num=12).to(torch.bfloat16).to(self.dev) for img in imgs]
        num_patches_list = [img.size(0) for img in imgs]
        output_text = []
        for i in range(0, len(imgs), self.max_batch_size):
            batch_imgs = imgs[i:i+self.max_batch_size]
            pixel_values = torch.cat(batch_imgs, dim=0)
            output_text += self.model.batch_chat(self.tokenizer, pixel_values,
                                                 num_patches_list=num_patches_list[i:i+self.max_batch_size],
                                                 questions=questions[i:i+self.max_batch_size],
                                                 generation_config=self.generation_config,
                                                 device=self.dev)
        return output_text

    @torch.no_grad()
    def image_predict(self, prog_step, questions):
        candidate_frame_ids = prog_step.state['frame_ids']
        # initialize QA pool. save {frame_id, Q, A} pair
        QA_pool = []
        # initialize index for selecting question
        if isinstance(questions, str):
            questions = [questions]
            q_idxs = np.zeros(len(candidate_frame_ids), dtype=int).tolist()
        elif isinstance(questions, list):
            if len(questions) == 0:
                return QA_pool
            q_idxs = np.random.randint(0, len(questions), size=len(candidate_frame_ids), dtype=int).tolist()
        # iterate over images, make QA pair (batch)
        imgs = [self.to_PIL(prog_step.state['image'][i]) for i in candidate_frame_ids]
        questions = [questions[q_idx] for q_idx in q_idxs]
        answers = self.batch_predict(imgs, questions, prompt_type='image_vqa')
        for i, question, answer in zip(candidate_frame_ids, questions, answers):
            QA_pool.append({'frame_id': i, 'question': question, 'answer': answer})
        return QA_pool

    def get_index(self, bound, fps, max_frame, first_idx=0, num_segments=32):
        if bound:
            start, end = bound[0], bound[1]
        else:
            start, end = -100000, 100000
        start_idx = max(first_idx, round(start * fps))
        end_idx = min(round(end * fps), max_frame)
        seg_size = float(end_idx - start_idx) / num_segments
        frame_indices = np.array([
            int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
            for idx in range(num_segments)
        ])
        return frame_indices
        
    def load_video(self, video_path, bound=None, input_size=448, max_num=1, num_segments=32):
        vr = decord.VideoReader(video_path, num_threads=1, ctx=cpu(0))
        decord.bridge.set_bridge('torch')
        max_frame = len(vr) - 1
        fps = float(vr.get_avg_fps())
        pixel_values_list, num_patches_list = [], []
        transform = self.build_transform(input_size=input_size)
        frame_indices = self.get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
        for frame_index in frame_indices:
            img = vr[frame_index].byte().permute(2, 0, 1)
            img = self.to_PIL(img).convert('RGB')
            img = self.dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
            pixel_values = [transform(tile) for tile in img]
            pixel_values = torch.stack(pixel_values)
            num_patches_list.append(pixel_values.shape[0])
            pixel_values_list.append(pixel_values)
        pixel_values = torch.cat(pixel_values_list)
        return pixel_values, num_patches_list
    
    def load_prompt(self, data):
        if self.config.question_type == 'mc':
            prompt_path = 'datas/prompt/llm_only_mc.prompt'
        elif self.config.question_type == 'oe':
            prompt_path = 'datas/prompt/llm_only_oe.prompt'
        else:
            raise Exception('Invalid question type!')
        with open(prompt_path) as f:
            base_prompt = f.read().strip()

        if isinstance(data, list):
            if self.config.question_type == 'mc':
                prompt = [base_prompt.replace('INSERT_QUESTION_HERE', d['question']).format(len_options=len(d['option']), options=d['option']) for d in data]
            elif self.config.question_type == 'oe':
                prompt = [base_prompt.replace('INSERT_QUESTION_HERE', d['question']) for d in data]
            else:
                raise Exception('Invalid question type!')
        else:
            raise TypeError('data must be list of strings')
        return prompt
    
    # @torch.no_grad()
    # def video_predict(self, data):
    #     # convert data
    #     data = [dict(zip(data.keys(), values)) for values in zip(*data.values())]
    #     # prepare data
    #     prompt = self.load_prompt(data)
    #     bound = [[min(d['frame_ids']), max(d['frame_ids']) + 1] if len(d['frame_ids']) > 0 else None for d in data]
    #     vids = [self.load_video(d['video_path'], bound=b, num_segments=8, max_num=1)[0].to(torch.bfloat16).to(self.dev) for d, b in zip(data, bound)]
    #     num_patches_list = [self.load_video(d['video_path'], bound=b, num_segments=8, max_num=1)[1] for d, b in zip(data, bound)]
    #     video_prefixs = [''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches))]) for num_patches in num_patches_list]
    #     questions = [self.prompt['video_baseline'].format(prefix, p) for prefix, p in zip(video_prefixs, prompt)]
    #     answer = []
    #     for i in range(0, len(prompt), self.max_batch_size):
    #         batch_videos = vids[i:i+self.max_batch_size]
    #         pixel_values = torch.cat(batch_videos, dim=0)
    #         answer += self.model.batch_chat(self.tokenizer, pixel_values,
    #                                         questions[i:i+self.max_batch_size],
    #                                         self.generation_config,
    #                                         num_patches_list=num_patches_list[i:i+self.max_batch_size],
    #                                         device=self.dev)
    #     return answer

    @torch.no_grad()
    def video_predict(self, prog_step, questions):
        candidate_frame_ids = prog_step.state['frame_ids']
        start_idx, end_idx = min(candidate_frame_ids), max(candidate_frame_ids)
        # initialize QA pool. save {Q, A} pair
        QA_pool = []
        # initialize question
        if isinstance(questions, str):
            questions = [questions]
        elif isinstance(questions, list):
            if len(questions) == 0:
                return QA_pool
        pixel_values, num_patches_list = self.load_video(prog_step.state['video_path'], num_segments=8, max_num=1)
        pixel_values = pixel_values.to(torch.bfloat16).to(self.dev)
        video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
        # iterate over question
        for question in questions:
            question = self.prompt['video_vqa'].format(video_prefix, question)
            answer = self.model.chat(self.tokenizer, pixel_values, question, self.generation_config, num_patches_list=num_patches_list, device=self.dev)
            QA_pool.append({'question': question, 'answer': answer})
        return QA_pool

    def parse(self, prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        output_var = parse_result['output_var']
        args = parse_result['args']
        questions = args['question']
        require_ocr = args['require_ocr']
        assert(step_name==self.step_name)
        return questions, require_ocr ,output_var

    def execute(self,prog_step,inspect=False):
        questions, require_ocr, output_var = self.parse(prog_step)
        # initialize QA_pool. save video and image result {video: {Q, A} pair, image: {frame_id, Q, A} pair}
        QA_pool = {'video': [], 'image': []}
        # reasoning over video
        # TODO: video reasoning code는 구체적으로 작성되지 않은 상태
        if prog_step.state['is_video']:
            QA_pool['video'] += self.video_predict(prog_step, questions)
        # reasoning over image
        if prog_step.state['is_image']:
            QA_pool['image'] += self.image_predict(prog_step, questions)
        prog_step.state[output_var] = QA_pool
        return QA_pool

class InternVLInterpreterverify(InternVLInterpreter):
    step_name = 'verify_action'
    def __init__(self, config, device=None):
        super().__init__(config, device)
        # print(f'Registering {self.step_name} step')
        
    def parse(self, prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        output_var = parse_result['output_var']
        args = parse_result['args']
        action = args['action']
        nouns = args['nouns']
        return action, nouns, output_var
    
    def execute(self,prog_step,inspect=False):
        action, nouns, output_var = self.parse(prog_step)
    
        # initialize indicator
        indicator = copy.deepcopy(prog_step.state['indicator'].detach())
        # update frame_id based on condition (anchor)
        for noun in nouns:
            # assert noun in prog_step.state.keys()
            if noun in prog_step.state.keys():
                indicator = indicator * prog_step.state[noun].clone().detach()
        candidate_frame_ids = torch.where(indicator==True)[0].tolist()
        # do not update frame_id when action=='no_action'
        if action == 'no_action':
            prog_step.state[output_var] = candidate_frame_ids
            return candidate_frame_ids
        # iterate over images, update frame_id (batch)
        imgs = [self.to_PIL(prog_step.state['image'][i]) for i in candidate_frame_ids]
        questions = [action] * len(imgs)
        answers = self.batch_predict(imgs, questions, prompt_type='verify')
        for i, answer in enumerate(answers):
            if 'yes' in answer.lower():
                indicator[i] = True
            elif 'no' in answer.lower(): # 일단 'yes'가 없으면 'no'라고 가정
                indicator[i] = False
            else:
                raise Exception("Invalid answer type. Should be either 'yes' or 'no'")
        frame_id = torch.where(indicator==True)[0].tolist()
        prog_step.state[output_var] = frame_id
        return frame_id

class InternLMXComposerInterpreter(torch.nn.Module):
    step_name = 'vqa'
    tokenizer = None
    model = None
    def __init__(self, config, device=None):
        super().__init__()
        # print(f'Registering {self.step_name} step')
        self.dev = device
        self.config = config
        model_id = config.internlmxcomposer.model_path
        self.model_type = model_id.split('/')[-1]
        # TODO: internLMXComposer-4khd-7b, internLMXComposer2d5 경우에 대해서도 정상작동하는지 double-check
        assert self.model_type in ['internlm-xcomposer2-vl-7b', 'internlm-xcomposer2-4khd-7b', 'internlm-xcomposer2d5-7b']
        if InternLMXComposerInterpreter.tokenizer is None or InternLMXComposerInterpreter.model is None:
            InternLMXComposerInterpreter.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            InternLMXComposerInterpreter.model = AutoModelForCausalLM.from_pretrained(model_id, device_map='cuda', trust_remote_code=True)
            InternLMXComposerInterpreter.model.half().eval()
        self.tokenizer = InternLMXComposerInterpreter.tokenizer
        self.model = InternLMXComposerInterpreter.model
        if self.model_type in ['internlm-xcomposer2-4khd-7b', 'internlm-xcomposer2d5-7b']:
            self.model = self.model.to(torch.bfloat16)
        if self.model_type == 'internlm-xcomposer2d5-7b':
            self.model.tokenizer = self.tokenizer

        # self.prompt = {'image_vqa': '<ImageHere>{} Please answer the question in a few words or phrases.',
        #                'video_vqa': '<ImageHere>{} Please answer the question in a few words or phrases.',
        #                'verify': '<ImageHere>{} Please answer the question only yes or no.'}
        self.prompt = {'image_vqa': '<ImageHere>{} Please answer the question in a single word or phrase.',
                       'video_vqa': '<ImageHere>{} Please answer the question in a single word or phrase.',
                       'verify': '<ImageHere>{} Please answer the question only yes or no.'}
        self.to_PIL = transforms.ToPILImage()
        self.max_batch_size = self.config.internlmxcomposer.max_batch_size

    @ torch.no_grad()
    def image_predict(self, prog_step, questions):
        candidate_frame_ids = prog_step.state['frame_ids']
        # initialize QA pool. save {frame_id, Q, A} pair
        QA_pool = []
        # initialize index for selecting question
        if isinstance(questions, str):
            questions = [questions]
            q_idxs = np.zeros(len(candidate_frame_ids), dtype=int).tolist()
        elif isinstance(questions, list):
            if len(questions) == 0:
                return QA_pool
            q_idxs = np.random.randint(0, len(questions), size=len(candidate_frame_ids), dtype=int).tolist()
        # iterate over images, make QA pair (batch)
        images = [self.to_PIL(prog_step.state['image'][i]) for i in candidate_frame_ids]
        questions = [questions[q_idx] for q_idx in q_idxs]
        answers = self.caption_generate(images, questions, prompt_type='image_vqa')
        for idx, question, answer in zip(candidate_frame_ids, questions, answers):
            QA_pool.append({'frame_id': idx, 'question': question, 'answer': answer.strip('.')})
        return QA_pool

    @ torch.no_grad()
    def caption_generate(self, images, questions, prompt_type='image_vqa'):
        if isinstance(questions, list):
            questions = [self.prompt[prompt_type].format(question) for question in questions]
        else:
            raise Exception('invalide question type')
        torch.set_grad_enabled(False)
        answer = []
        # TODO: internLMXComposer2-vl 경우에 대해서만 batch inference 구현된 상태. 나머지 두 경우에 대해서도 추가
        if self.model_type == 'internlm-xcomposer2-vl-7b':
            # use batch inference
            with torch.cuda.amp.autocast():
                for i in range(0, len(images), self.max_batch_size):
                    batch_imgs = images[i:i+self.max_batch_size]
                    batch_questions = questions[i:i+self.max_batch_size]
                    answer += self.model.batch_chat(self.tokenizer, query=batch_questions, image=batch_imgs, do_sample=False)[0]
        elif self.model_type == 'internlm-xcomposer2-4khd-7b':
            with torch.cuda.amp.autocast():
                for image, question in zip(images, questions):
                    answer.append(self.model.chat(self.tokenizer, query=question, image=image, hd_num=55, do_sample=False)[0])
        elif self.model_type == 'internlm-xcomposer2d5-7b':
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                for image, question in zip(images, questions):
                    answer.append(self.model.chat(self.tokenizer, question, [image], do_sample=False, use_meta=True, is_video=False)[0])
        return answer

    @torch.no_grad()
    def video_predict(self, data):
        assert self.model_type == 'internlm-xcomposer2d5-7b'
        torch.set_grad_enabled(False)
        # convert data
        data = [dict(zip(data.keys(), values)) for values in zip(*data.values())]
        # prepare data
        prompt = self.load_prompt(data)
        bounds = [[min(d['frame_ids']), max(d['frame_ids']) + 1] if len(d['frame_ids']) > 0 else None for d in data]
        images = [d['video_path'] for d in data]
        questions = [p for p in prompt]
        answer = []
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            for image, question, bound in zip(images, questions, bounds):
                answer.append(self.model.chat(self.tokenizer, question, [image], do_sample=False, use_meta=True, is_video=True, bound=bound)[0])
        return answer
    
    def parse(self, prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        output_var = parse_result['output_var']
        args = parse_result['args']
        questions = args['question']
        require_ocr = args['require_ocr']
        assert(step_name==self.step_name)
        return questions, require_ocr ,output_var

    def execute(self,prog_step,inspect=False):
        questions, require_ocr, output_var = self.parse(prog_step)
        # initialize QA_pool. save video and image result {video: {Q, A} pair, image: {frame_id, Q, A} pair}
        QA_pool = {'video': [], 'image': []}
        # reasoning over video
        # TODO: video reasoning code는 구체적으로 작성되지 않은 상태
        if prog_step.state['is_video']:
            QA_pool['video'] += self.video_predict(prog_step, questions)
        # reasoning over image
        if prog_step.state['is_image']:
            QA_pool['image'] += self.image_predict(prog_step, questions)
        prog_step.state[output_var] = QA_pool
        return QA_pool

class InternLMXComposerInterpreterverify(InternLMXComposerInterpreter):
    step_name = 'verify_action'
    def __init__(self, config, device=None):
        super().__init__(config, device)
        # print(f'Registering {self.step_name} step')``

    def parse(self, prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        output_var = parse_result['output_var']
        args = parse_result['args']
        action = args['action']
        nouns = args['nouns']
        return action, nouns, output_var
    
    def execute(self,prog_step,inspect=False):
        action, nouns, output_var = self.parse(prog_step)
    
        # initialize indicator
        indicator = copy.deepcopy(prog_step.state['indicator'].detach())
        # update frame_id based on condition (anchor)
        for noun in nouns:
            # assert noun in prog_step.state.keys()
            if noun in prog_step.state.keys():
                indicator = indicator * prog_step.state[noun].clone().detach()
        candidate_frame_ids = torch.where(indicator==True)[0].tolist()
        # do not update frame_id when action=='no_action'
        if action == 'no_action':
            prog_step.state[output_var] = candidate_frame_ids
            return candidate_frame_ids
        # iterate over images, update frame_id (batch)
        imgs = [self.to_PIL(prog_step.state['image'][i]) for i in candidate_frame_ids]
        questions = [action] * len(imgs)
        answers = self.caption_generate(imgs, questions, prompt_type='verify')
        for idx, answer in zip(candidate_frame_ids, answers):
            if 'yes' in answer.strip('.').lower():
                indicator[idx] = True
            elif 'no' in answer.strip('.').lower(): # 일단 'yes'가 없으면 'no'라고 가정
                indicator[idx] = False
            else:
                raise Exception("Invalid answer type. Should be either 'yes' or 'no'")
        frame_id = torch.where(indicator==True)[0].tolist()
        prog_step.state[output_var] = frame_id
        return frame_id

class InternLM(torch.nn.Module):
    step_name = 'internlm'
    def __init__(self, config, device=None):
        super().__init__()
        # print(f'Registering {self.step_name} step')
        self.dev = device
        
        self.config = config
        model_id = config.internlm.model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.tokenizer.padding_side = 'left'
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, trust_remote_code=True)
        self.model.eval()
        
        self.max_batch_size = self.config.internlm.max_batch_size
    
    def load_prompt(self, data, prompt_type):
        if prompt_type == 'module1':
            additional_system_prompt = 'Only answer with the final answer similar to given examples.'
            prompt_path = 'datas/prompt/module1.prompt'
        elif prompt_type == 'module2':
            additional_system_prompt = 'Only answer with the final answer similar to given examples.'
            prompt_path = 'datas/prompt/module2.prompt'
        elif prompt_type == 'module2_retrieve':
            additional_system_prompt = 'Only answer with the final answer similar to given examples.'
            prompt_path = 'datas/prompt/module2_spatioretrieve.prompt'
        elif prompt_type == 'module3':
            additional_system_prompt = 'Only answer with the final answer similar to given examples.'
            prompt_path = 'datas/prompt/module3.prompt'
        elif prompt_type == 'final':
            additional_system_prompt = 'Only answer with the final answer.'
            if self.config.question_type == 'mc':
                prompt_path = 'datas/prompt/final_prediction_mc.prompt'
            elif self.config.question_type == 'oe':
                prompt_path = 'datas/prompt/final_prediction_oe.prompt'
            else:
                raise Exception('Invalid question type!')
        elif prompt_type == 'llm_only':
            additional_system_prompt = 'Only answer with the final answer.'
            if self.config.question_type == 'mc':
                prompt_path = 'datas/prompt/llm_only_mc.prompt'
            elif self.config.question_type == 'oe':
                prompt_path = 'datas/prompt/llm_only_oe.prompt'
            else:
                raise Exception('Invalid question type!')
        else:
            raise Exception('wrong prompt type')
        with open(prompt_path) as f:
            base_prompt = f.read().strip()

        if prompt_type == 'module1':
            # convert data
            data = [dict(zip(data.keys(), values)) for values in zip(*data.values())]
            if isinstance(data, list):
                prompt = [base_prompt.replace('INSERT_QUESTION_HERE', d['question']) for d in data]
            else:
                raise TypeError('data must be list of strings')
        elif prompt_type == 'module2':
            # convert data
            data = [dict(zip(data.keys(), values)) for values in zip(*data.values())]
            if isinstance(data, list):
                prompt = [base_prompt.replace('INSERT_CONJUNCTION_HERE', d['conjunction'])
                                    .replace('INSERT_PHRASE1_HERE', d['event_queue'][0])
                                    .replace('INSERT_PHRASE2_HERE', d['event_queue'][1]) for d in data]
            else:
                raise TypeError('data must be list of strings')
        elif prompt_type == 'module2_retrieve':
            # convert data
            data = [dict(zip(data.keys(), values)) for values in zip(*data.values())]
            if isinstance(data, list):
                prompt = [base_prompt.replace('INSERT_CONJUNCTION_HERE', d['conjunction'])
                                    .replace('INSERT_EVENT1_HERE', d['event_queue'][0])
                                    .replace('INSERT_EVENT2_HERE', d['event_queue'][1]) for d in data]
            else:
                raise TypeError('data must be list of strings')
        elif prompt_type == 'module3':
            # convert data
            data = [dict(zip(data.keys(), values)) for values in zip(*data.values())]
            if isinstance(data, list):
                prompt = [base_prompt.replace('INSERT_QATYPE_HERE', d['qa_type']).replace('INSERT_QUESTION_HERE', d['question']) for d in data]
            else:
                raise TypeError('data must be list of strings')
        elif prompt_type == 'final':
            # convert data
            data = [dict(zip(data.keys(), values)) for values in zip(*data.values())]
            if isinstance(data, list):
                if self.config.question_type == 'mc':
                    prompt = [base_prompt.replace('INSERT_SUMMARY_HERE', d['video_context']).replace('INSERT_QUESTION_HERE', d['question']).format(len_options=len(d['option']), options=d['option']) for d in data]
                elif self.config.question_type == 'oe':
                    prompt = [base_prompt.replace('INSERT_SUMMARY_HERE', d['video_context']).replace('INSERT_QUESTION_HERE', d['question']) for d in data]
                else:
                    raise Exception('Invalid question type!')
            else:
                raise TypeError('data must be list of strings')
        elif prompt_type == 'llm_only':
            # convert data
            data = [dict(zip(data.keys(), values)) for values in zip(*data.values())]
            if isinstance(data, list):
                if self.config.question_type == 'mc':
                    prompt = [base_prompt.replace('INSERT_QUESTION_HERE', d['question']).format(len_options=len(d['option']), options=d['option']) for d in data]
                elif self.config.question_type == 'oe':
                    prompt = [base_prompt.replace('INSERT_QUESTION_HERE', d['question']) for d in data]
                else:
                    raise Exception('Invalid question type!')
            else:
                raise TypeError('data must be list of strings')
        else:   
            raise Exception('wrong prompt type')
        
        return prompt, additional_system_prompt
    
    @torch.no_grad()
    def generate(self, data, prompt_type='module1'):
        prompt, additional_system_prompt = self.load_prompt(data, prompt_type)
        response = []
        llm_batch_size = 1 if prompt_type == 'final' else self.max_batch_size
        for i in range(0, len(prompt), llm_batch_size):
            response += self.model.batch_chat(self.tokenizer, prompt[i: i + llm_batch_size],
                                              max_new_tokens=self.config.internlm.max_new_tokens,
                                              do_sample=self.config.internlm.do_sample,
                                              temperature=self.config.internlm.temperature,
                                              top_p=self.config.internlm.top_p,
                                              additional_system_prompt=additional_system_prompt)
        return response

class RETRIEVEInterpreterQDDETR(torch.nn.Module):
    step_name = 'retrieve'
    def __init__(self, config, device=None):
        super().__init__()
        # print(f'Registering {self.step_name} step')
        self.dev = device
        
        from pretrained_model.QD_DETR.run_on_video.run import QDDETRPredictor
        ckpt_path = config.qd_detr.model_checkpoint_path
        clip_model_name_or_path = config.qd_detr.clip_model
        self.predictor = QDDETRPredictor(ckpt_path=ckpt_path, clip_model_name_or_path=clip_model_name_or_path, device=self.dev)
        self.config = config

    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        output_var = parse_result['output_var']
        args = parse_result['args']
        query = args['query']
        assert(step_name==self.step_name)
        return query, output_var
    
    def execute(self,prog_step,inspect=False):
        query, output_var = self.parse(prog_step)
        # initialize indicator
        indicator = copy.deepcopy(prog_step.state['indicator'].detach())
        # if temporal does not exist, escape
        if torch.all(indicator==False):
            prog_step.state[output_var] = []
            return []
        # initialize start and end frame id of temporal window
        candidate_frame_ids = torch.where(indicator==True)[0].tolist()
        start_idx, end_idx = min(candidate_frame_ids), max(candidate_frame_ids)
        if query == '':
            frame_id = [i for i in range(start_idx, end_idx + 1)]
            prog_step.state[output_var] = frame_id
            return frame_id
        # select top 1 prediction
        predictions = self.predictor.localize_moment(video_path=prog_step.state['video_path'], query_list=[query], start_idx=start_idx, end_idx=end_idx)
        temporal_windows = predictions[0]['pred_relevant_windows']
        start_idx_new = max(start_idx, start_idx + round(temporal_windows[0][0]))
        end_idx_new = min(end_idx, start_idx + round(temporal_windows[0][1]))
        # update frame_id
        frame_id = [i for i in range(start_idx_new, end_idx_new + 1)] if start_idx_new <= end_idx_new else []
        prog_step.state[output_var] = frame_id
        return frame_id
    
    @torch.no_grad()
    def predict_window(self, data):
        frame_ids = []
        # iterate over data (video_path, question, frame_ids)
        for d in data:
            start_idx, end_idx = min(d['frame_ids']), max(d['frame_ids'])
            # select top 1 prediction
            predictions = self.predictor.localize_moment(video_path=d['video_path'], query_list=[d['question']], start_idx=start_idx, end_idx=end_idx)
            temporal_windows = predictions[0]['pred_relevant_windows']
            start_idx_new = max(start_idx, start_idx + round(temporal_windows[0][0]))
            end_idx_new = min(end_idx, start_idx + round(temporal_windows[0][1]))
            # update frame_id
            frame_id = [i for i in range(start_idx_new, end_idx_new + 1)] if start_idx_new <= end_idx_new else []
            frame_ids.append(frame_id)
        return frame_ids

class RETRIEVEInterpreterUniVTG(torch.nn.Module):
    step_name = 'retrieve'
    predictor = None
    def __init__(self, config, device=None):
        super().__init__()
        # print(f'Registering {self.step_name} step')
        self.dev = device
        # set video feature path
        if config.dataset.dataset_name == 'NExTQA':
            directory = 'nextqa' if config.dataset.version == 'multiplechoice' else 'nextoe'
        else:
            directory = ''
        self.clip_vid_feat_path = os.path.join(config.dataset.data_path, directory, config.dataset.split + '_' + config.univtg.clip_vid_feat_path)
        self.slowfast_vid_feat_path = os.path.join(config.dataset.data_path, directory, config.dataset.split + '_' + config.univtg.slowfast_vid_feat_path)

        if RETRIEVEInterpreterUniVTG.predictor is None:
            from pretrained_model.UniVTG.run_on_video.run import UniVTGPredictor
            ckpt_path = config.univtg.model_checkpoint_path
            clip_model_name_or_path = config.univtg.clip_model
            RETRIEVEInterpreterUniVTG.predictor = UniVTGPredictor(config, ckpt_path=ckpt_path, clip_model_name_or_path=clip_model_name_or_path, device=self.dev,
                                            clip_vid_feat_path=self.clip_vid_feat_path, slowfast_vid_feat_path=self.slowfast_vid_feat_path)
        self.predictor = RETRIEVEInterpreterUniVTG.predictor
        self.config = config
        
    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        output_var = parse_result['output_var']
        args = parse_result['args']
        query = args['query']
        assert(step_name==self.step_name)
        return query, output_var
    
    def execute(self,prog_step,inspect=False):
        query, output_var = self.parse(prog_step)
        # initialize indicator
        indicator = copy.deepcopy(prog_step.state['indicator'].detach())
        # if temporal does not exist, escape
        if torch.all(indicator==False):
            prog_step.state[output_var] = []
            return []
        # initialize start and end frame id of temporal window
        candidate_frame_ids = torch.where(indicator==True)[0].tolist()
        start_idx, end_idx = min(candidate_frame_ids), max(candidate_frame_ids)
        if query == 'none':
            frame_id = [i for i in range(start_idx, end_idx + 1)]
            prog_step.state[output_var] = frame_id
            return frame_id
        # convert video_path into vid
        vid = str(prog_step.state['video_path'].split('/')[-1].split('.')[0])
        # add punctuation at the end od the phrase, except for question
        query = query + '.' if query[-1] != '?' else query
        # select top 1 prediction
        predictions = self.predictor.localize_moment(vid=vid, query_list=[query], start_idx=start_idx, end_idx=end_idx)
        temporal_windows = predictions[0]['pred_relevant_windows']
        # saliencys = predictions[0]['pred_saliency_scores'] # for saliency/hightlight detection
        start_idx_new = max(start_idx, start_idx + round(temporal_windows[0][0]))
        end_idx_new = min(end_idx, start_idx + round(temporal_windows[0][1]))
        # update frame_id
        frame_id = [i for i in range(start_idx_new, end_idx_new + 1)] if start_idx_new <= end_idx_new else []
        prog_step.state[output_var] = frame_id
        return frame_id
    
    @torch.no_grad()
    def predict_window(self, data):
        frame_ids = []
        # iterate over data (video_path, question, frame_ids)
        for d in data:
            start_idx, end_idx = min(d['frame_ids']), max(d['frame_ids'])
            # convert video_path into vid
            vid = str(d['video_path'].split('/')[-1].split('.')[0])
            # select top 1 prediction
            predictions = self.predictor.localize_moment(vid=vid, query_list=[d['question']], start_idx=start_idx, end_idx=end_idx)
            temporal_windows = predictions[0]['pred_relevant_windows']
            # saliencys = predictions[0]['pred_saliency_scores'] # for saliency/hightlight detection
            start_idx_new = max(start_idx, start_idx + round(temporal_windows[0][0]))
            end_idx_new = min(end_idx, start_idx + round(temporal_windows[0][1]))
            # update frame_id
            frame_id = [i for i in range(start_idx_new, end_idx_new + 1)] if start_idx_new <= end_idx_new else []
            frame_ids.append(frame_id)
        return frame_ids

class InternVideo(torch.nn.Module):
    step_name = 'vqa'
    def __init__(self, config, device=None):
        super().__init__()
        # print(f'Registering {self.step_name} step')
        self.dev = device
        self.config = config
        model_id = config.internvideo.model_path
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=False,)
        self.model = AutoModel.from_pretrained(model_id, torch_dtype=torch.bfloat16, trust_remote_code=True)
        self.generation_config = dict(max_new_tokens=512, do_sample=False, pad_token_id=self.tokenizer.eos_token_id)
        self.instruction_prompt = 'Carefully watch the video and pay attention to the cause and sequence of events, the detail and movement of objects, and the action and pose of persons. Based on your observations, select the best option that accurately addresses the question.\n'

    def get_index(self, bound, fps, max_frame, first_idx=0, num_segments=32):
        if bound:
            start, end = bound[0], bound[1]
        else:
            start, end = -100000, 100000
        start_idx = max(first_idx, round(start * fps))
        end_idx = min(round(end * fps), max_frame)
        seg_size = float(end_idx - start_idx) / num_segments
        frame_indices = np.array([
            int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
            for idx in range(num_segments)
        ])
        return frame_indices
    
    def load_video(self, video_path, bound=None, num_segments=8, return_msg=False, resolution=224, hd_num=4, padding=False):
        vr = decord.VideoReader(video_path, ctx=cpu(0), num_threads=1)
        decord.bridge.set_bridge('torch')
        max_frame = len(vr) - 1
        fps = float(vr.get_avg_fps())
        frame_indices = self.get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

        transform = transforms.Compose([
            transforms.Lambda(lambda x: x.float().div(255.0)),
            transforms.Normalize(mean, std)
        ])

        frames = vr.get_batch(frame_indices).byte()
        frames = frames.permute(0, 3, 1, 2)

        if padding:
            frames = self.HD_transform_padding(frames.float(), image_size=resolution, hd_num=hd_num)
        else:
            frames = self.HD_transform_no_padding(frames.float(), image_size=resolution, hd_num=hd_num)

        frames = transform(frames)
        # print(frames.shape)
        T_, C, H, W = frames.shape

        sub_img = frames.reshape(
            1, T_, 3, H//resolution, resolution, W//resolution, resolution
        ).permute(0, 3, 5, 1, 2, 4, 6).reshape(-1, T_, 3, resolution, resolution).contiguous()

        glb_img = F.interpolate(
            frames.float(), size=(resolution, resolution), mode='bicubic', align_corners=False
        ).to(sub_img.dtype).unsqueeze(0)

        frames = torch.cat([sub_img, glb_img]).unsqueeze(0)

        if return_msg:
            fps = float(vr.get_avg_fps())
            sec = ", ".join([str(round(f / fps, 1)) for f in frame_indices])
            # " " should be added in the start and end
            msg = f"The video contains {len(frame_indices)} frames sampled at {sec} seconds."
            return frames, msg
        else:
            return frames
    
    def load_video_no_HD(self, video_path, bound=None, num_segments=8, return_msg=False, resolution=224, hd_num=4, padding=False):
        vr = decord.VideoReader(video_path, ctx=cpu(0), num_threads=1)
        decord.bridge.set_bridge('torch')
        max_frame = len(vr) - 1
        fps = float(vr.get_avg_fps())
        frame_indices = self.get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

        transform = transforms.Compose([
            transforms.Lambda(lambda x: x.float().div(255.0)),
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.Normalize(mean, std)
        ])
        
        frames = vr.get_batch(frame_indices).byte()
        frames = frames.permute(0, 3, 1, 2)
        frames = transform(frames)
        
        T_, C, H, W = frames.shape

        if return_msg:
            fps = float(vr.get_avg_fps())
            sec = ", ".join([str(round(f / fps, 1)) for f in frame_indices])
            # " " should be added in the start and end
            msg = f"The video contains {len(frame_indices)} frames sampled at {sec} seconds."
            return frames, msg
        else:
            return frames
        
    def HD_transform_padding(self, frames, image_size=224, hd_num=6):
        def _padding_224(frames):
            _, _, H, W = frames.shape
            tar = int(np.ceil(H / 224) * 224)
            top_padding = (tar - H) // 2
            bottom_padding = tar - H - top_padding
            left_padding = 0
            right_padding = 0

            padded_frames = F.pad(
                frames,
                pad=[left_padding, right_padding, top_padding, bottom_padding],
                mode='constant', value=255
            )
            return padded_frames

        _, _, H, W = frames.shape
        trans = False
        if W < H:
            frames = frames.flip(-2, -1)
            trans = True
            width, height = H, W
        else:
            width, height = W, H

        ratio = width / height
        scale = 1
        while scale * np.ceil(scale / ratio) <= hd_num:
            scale += 1
        scale -= 1
        new_w = int(scale * image_size)
        new_h = int(new_w / ratio)

        resized_frames = F.interpolate(
            frames, size=(new_h, new_w),
            mode='bicubic',
            align_corners=False
        )
        padded_frames = _padding_224(resized_frames)

        if trans:
            padded_frames = padded_frames.flip(-2, -1)

        return padded_frames

    def find_closest_aspect_ratio(self, aspect_ratio, target_ratios, width, height, image_size):
            best_ratio_diff = float('inf')
            best_ratio = (1, 1)
            area = width * height
            for ratio in target_ratios:
                target_aspect_ratio = ratio[0] / ratio[1]
                ratio_diff = abs(aspect_ratio - target_aspect_ratio)
                if ratio_diff < best_ratio_diff:
                    best_ratio_diff = ratio_diff
                    best_ratio = ratio
                elif ratio_diff == best_ratio_diff:
                    if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                        best_ratio = ratio
            return best_ratio

    def HD_transform_no_padding(self, frames, image_size=224, hd_num=6, fix_ratio=(2,1)):
        min_num = 1
        max_num = hd_num
        _, _, orig_height, orig_width = frames.shape
        aspect_ratio = orig_width / orig_height

        # calculate the existing video aspect ratio
        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
            i * j <= max_num and i * j >= min_num)
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        # find the closest aspect ratio to the target
        if fix_ratio:
            target_aspect_ratio = fix_ratio
        else:
            target_aspect_ratio = self.find_closest_aspect_ratio(
                aspect_ratio, target_ratios, orig_width, orig_height, image_size)

        # calculate the target width and height
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        # resize the frames
        resized_frame = F.interpolate(
            frames, size=(target_height, target_width),
            mode='bicubic', align_corners=False
        )
        return resized_frame

    def load_prompt(self, data):
        if self.config.question_type == 'mc':
            prompt_path = 'datas/prompt/llm_only_mc.prompt'
        elif self.config.question_type == 'oe':
            prompt_path = 'datas/prompt/llm_only_oe.prompt'
        else:
            raise Exception('Invalid question type!')
        with open(prompt_path) as f:
            base_prompt = f.read().strip()

        if isinstance(data, list):
            if self.config.question_type == 'mc':
                prompt = [base_prompt.replace('INSERT_QUESTION_HERE', d['question']).format(len_options=len(d['option']), options=d['option']) for d in data]
            elif self.config.question_type == 'oe':
                prompt = [base_prompt.replace('INSERT_QUESTION_HERE', d['question']) for d in data]
            else:
                raise Exception('Invalid question type!')
        else:
            raise TypeError('data must be list of strings')

        return prompt

    @torch.no_grad()
    def video_predict(self, data):
        # convert data
        data = [dict(zip(data.keys(), values)) for values in zip(*data.values())]
        # prepare data
        prompt = self.load_prompt(data)
        bound = [[min(d['frame_ids']), max(d['frame_ids']) + 1] if len(d['frame_ids']) > 0 else None for d in data]
        video_tensors = [self.load_video_no_HD(d['video_path'], bound=b, num_segments=8, return_msg=False).to(self.model.device) for d, b in zip(data, bound)]
        answer = []
        for p, video_tensor in zip(prompt, video_tensors):
            answer.append(self.model.chat(self.tokenizer, '', p, instruction=self.instruction_prompt, media_type='video', media_tensor=video_tensor, generation_config=self.generation_config))
        return answer
    
    def parse(self, prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        output_var = parse_result['output_var']
        args = parse_result['args']
        questions = args['question']
        require_ocr = args['require_ocr']
        assert(step_name==self.step_name)
        return questions, require_ocr ,output_var
    
    def execute(self,prog_step,inspect=False):
        questions, require_ocr, output_var = self.parse(prog_step)
        # initialize QA_pool. save video and image result {video: {Q, A} pair, image: {frame_id, Q, A} pair}
        QA_pool = {'video': [], 'image': []}
        # reasoning over video
        if prog_step.state['is_video']:
            QA_pool['video'] += self.video_predict(prog_step, questions)
        if prog_step.state['is_image']:
            QA_pool['image'] += self.image_predict(prog_step, questions)
        prog_step.state[output_var] = QA_pool
        return QA_pool

### further defined interpreter ###
class TRIMInterpreter2():
    step_name = 'trim'
    def __init__(self):
        # print(f'Registering {self.step_name} step')
        pass
    
    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        output_var = parse_result['output_var']
        args = parse_result['args']
        trim_option = args['trim']
        assert(step_name==self.step_name)
        return trim_option, output_var

    def execute(self,prog_step,inspect=False):
        trim_option, output_var = self.parse(prog_step)
        out_value = {'trim': trim_option}
        prog_step.state[output_var] = out_value
        return out_value

class PARSEEVENTInterpreter2():
    step_name = 'parse_event'
    def __init__(self):
        # print(f'Registering {self.step_name} step')
        pass

    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        output_var = parse_result['output_var']
        args = parse_result['args']
        conj_option = args['conj']
        event_to_localize = args['event_to_localize']
        main_question = args['main_question']
        assert(step_name==self.step_name)
        return conj_option, event_to_localize, main_question, output_var

    def execute(self,prog_step,inspect=False):
        conj_option, event_to_localize, main_question, output_var = self.parse(prog_step)
        out_value = {'conj': conj_option, 'event_to_localize': event_to_localize, 'main_question': main_question}
        prog_step.state[output_var] = out_value
        return out_value

class TRUNCATEInterpreter2():
    step_name = 'truncate'
    def __init__(self):
        # print(f'Registering {self.step_name} step')
        pass
    
    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        output_var = parse_result['output_var']
        args = parse_result['args']
        conj_option = args['conj']
        anchor_option = args['anchor']
        assert(step_name==self.step_name)
        return conj_option, anchor_option, output_var
    
    def execute(self,prog_step,inspect=False):
        conj_option, anchor_option, output_var = self.parse(prog_step)
        prev_frame_ids = prog_step.state[anchor_option]
        if conj_option == 'when':
            prog_step.state['indicator'] = torch.zeros(prog_step.state['indicator'].size(0)).bool()
            prog_step.state['indicator'][prev_frame_ids] = True
        elif conj_option == 'before':
            if len(prev_frame_ids) == 0: # nothing is detected in the previous step
                prog_step.state['indicator'] = torch.zeros(prog_step.state['indicator'].size(0)).bool()
            else:
                anchor_index = min(prev_frame_ids)
                prog_step.state['indicator'][anchor_index:] = False
        elif conj_option == 'after':
            if len(prev_frame_ids) == 0: # nothing is detected in the previous step
                prog_step.state['indicator'] = torch.zeros(prog_step.state['indicator'].size(0)).bool()
            else:
                anchor_index = max(prev_frame_ids)
                prog_step.state['indicator'][:anchor_index+1] = False
        # if temporal does not exist, escape
        if torch.all(prog_step.state['indicator']==False):
            prog_step.state[output_var] = []
            return []
        frame_id = torch.where(prog_step.state['indicator']==True)[0].tolist()
        prog_step.state[output_var] = frame_id
        return frame_id

class RETRIEVEInterpreterUniVTG2(RETRIEVEInterpreterUniVTG):
    step = 'retrieve'
    def __init__(self, config, device=None):
        super().__init__(config, device)
        # print(f'Registering {self.step_name} step')
    
    def parse(self, prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        output_var = parse_result['output_var']
        args = parse_result['args']
        query = args['query']
        nouns = args['nouns']
        return query, nouns, output_var
    
    def execute(self,prog_step,inspect=False):
        query, nouns, output_var = self.parse(prog_step)
        # initialize indicator
        indicator = copy.deepcopy(prog_step.state['indicator'].detach())
        # update frame_id based on condition (anchor)
        for noun in nouns:
            # assert noun in prog_step.state.keys()
            if noun in prog_step.state.keys():
                indicator = indicator * prog_step.state[noun].clone().detach()
        # if temporal does not exist, escape
        if torch.all(indicator==False):
            prog_step.state[output_var] = []
            return []
        # initialize start and end frame id of temporal window
        candidate_frame_ids = torch.where(indicator==True)[0].tolist()
        # since candidate_frame_ids may noy consecutive, using min and max value to set temporal windows
        start_idx, end_idx = min(candidate_frame_ids), max(candidate_frame_ids)
        if query == 'none':
            frame_id = [i for i in range(start_idx, end_idx + 1)]
            prog_step.state[output_var] = frame_id
            return frame_id
        # convert video_path into vid
        vid = str(prog_step.state['video_path'].split('/')[-1].split('.')[0])
        # add punctuation at the end od the phrase, except for question
        query = query + '.' if query[-1] != '?' else query
        # select top 1 prediction
        predictions = self.predictor.localize_moment(vid=vid, query_list=[query], start_idx=start_idx, end_idx=end_idx)
        temporal_windows = predictions[0]['pred_relevant_windows']
        # saliencys = predictions[0]['pred_saliency_scores'] # for saliency/hightlight detection
        start_idx_new = max(start_idx, start_idx + round(temporal_windows[0][0]))
        end_idx_new = min(end_idx, start_idx + round(temporal_windows[0][1]))
        # update frame_id
        frame_id = [i for i in range(start_idx_new, end_idx_new + 1)] if start_idx_new <= end_idx_new else []
        prog_step.state[output_var] = frame_id
        return frame_id
    
class InternLM2(InternLM):
    step_name = 'internlm'
    def __init__(self, config, device=None):
        super().__init__(config, device)
    
    def load_prompt(self, data, prompt_type):
        if prompt_type == 'stage1_understanding':
            additional_system_prompt = 'Only answer with the final answer similar to given examples.'
            prompt_path = 'datas/prompt/stage1_understanding.prompt'
        elif prompt_type == 'stage1':
            additional_system_prompt = 'Only answer with the final answer similar to given examples.'
            prompt_path = 'datas/prompt/stage1.prompt'
        elif prompt_type == 'stage2_understanding':
            additional_system_prompt = 'Only answer with the final answer similar to given examples.'
            prompt_path = 'datas/prompt/stage2_understanding.prompt'
        elif prompt_type == 'stage2':
            additional_system_prompt = 'Only answer with the final answer similar to given examples.'
            prompt_path = 'datas/prompt/stage2.prompt'
        elif prompt_type == 'stage3_understanding':
            additional_system_prompt = 'Only answer with the final answer similar to given examples.'
            prompt_path = 'datas/prompt/stage3_understanding.prompt'
        elif prompt_type == 'stage3':
            additional_system_prompt = 'Only answer with the final answer similar to given examples.'
            prompt_path = 'datas/prompt/stage3.prompt'
        elif prompt_type == 'stage4_understanding':
            additional_system_prompt = 'Only answer with the final answer similar to given examples.'
            prompt_path = 'datas/prompt/stage4_understanding.prompt'
        elif prompt_type == 'stage4':
            additional_system_prompt = 'Only answer with the final answer similar to given examples.'
            prompt_path = 'datas/prompt/stage4.prompt'
        ### 아래는 작업 중 임시로 사용하는 코드 ###
        elif prompt_type == 'module2':
            additional_system_prompt = 'Only answer with the final answer similar to given examples.'
            prompt_path = 'datas/prompt/module2_spatioretrieve.prompt' # 그냥 question으로 넣는 것 better. default: use spatioretrieve in module2
        elif prompt_type == 'module3':
            additional_system_prompt = 'Only answer with the final answer similar to given examples.'
            prompt_path = 'datas/prompt/module3.prompt'
        #############################
        elif prompt_type == 'final':
            additional_system_prompt = 'Only answer with the final answer.'
            if self.config.question_type == 'mc':
                prompt_path = 'datas/prompt/final_prediction_mc.prompt'
            elif self.config.question_type == 'oe':
                prompt_path = 'datas/prompt/final_prediction_oe.prompt'
            else:
                raise Exception('Invalid question type!')
        elif prompt_type == 'llm_only':
            additional_system_prompt = 'Only answer with the final answer.'
            if self.config.question_type == 'mc':
                prompt_path = 'datas/prompt/llm_only_mc.prompt'
            elif self.config.question_type == 'oe':
                prompt_path = 'datas/prompt/llm_only_oe.prompt'
            else:
                raise Exception('Invalid question type!')
        else:
            raise Exception('wrong prompt type')
        with open(prompt_path) as f:
            base_prompt = f.read().strip()
        
        if prompt_type in ['stage1_understanding', 'stage2_understanding', 'stage3_understanding', 'stage4_understanding']:
            # convert data
            data = [dict(zip(data.keys(), values)) for values in zip(*data.values())]
            if isinstance(data, list):
                prompt = [base_prompt.replace('INSERT_QUESTION_HERE', d['question']) for d in data]
            else:
                raise TypeError('data must be list of strings')
        elif prompt_type in ['stage1', 'stage2', 'srage3', 'stage4']:
            # convert data
            data = [dict(zip(data.keys(), values)) for values in zip(*data.values())]
            if isinstance(data, list):
                prompt = [base_prompt.replace('INSERT_QUESTION_HERE', d['question']).replace('INSERT_UNDERSTANDING_HERE', d['understanding']) for d in data]
            else:
                raise TypeError('data must be list of strings')
        elif prompt_type == 'module2':
            # convert data
            data = [dict(zip(data.keys(), values)) for values in zip(*data.values())]
            if isinstance(data, list):
                prompt = [base_prompt.replace('INSERT_CONJUNCTION_HERE', d['conjunction'])
                                    .replace('INSERT_EVENT1_HERE', d['event_queue'][0])
                                    .replace('INSERT_EVENT2_HERE', d['event_queue'][1]) for d in data]
            else:
                raise TypeError('data must be list of strings')
        elif prompt_type == 'module3':
            # convert data
            data = [dict(zip(data.keys(), values)) for values in zip(*data.values())]
            if isinstance(data, list):
                prompt = [base_prompt.replace('INSERT_QATYPE_HERE', d['qa_type']).replace('INSERT_QUESTION_HERE', d['question']) for d in data]
            else:
                raise TypeError('data must be list of strings')
        elif prompt_type == 'final':
            # convert data
            data = [dict(zip(data.keys(), values)) for values in zip(*data.values())]
            if isinstance(data, list):
                if self.config.question_type == 'mc':
                    prompt = [base_prompt.replace('INSERT_SUMMARY_HERE', d['video_context'])
                                        .replace('INSERT_QUESTION_HERE', d['question'])
                                        .format(len_options=len(d['option']), options=d['option']) for d in data]
                elif self.config.question_type == 'oe':
                    prompt = [base_prompt.replace('INSERT_SUMMARY_HERE', d['video_context'])
                                        .replace('INSERT_QUESTION_HERE', d['question']) for d in data]
                else:
                    raise Exception('Invalid question type!')
            else:
                raise TypeError('data must be list of strings')
        elif prompt_type == 'llm_only':
            # convert data
            data = [dict(zip(data.keys(), values)) for values in zip(*data.values())]
            if isinstance(data, list):
                if self.config.question_type == 'mc':
                    prompt = [base_prompt.replace('INSERT_QUESTION_HERE', d['question'])
                                        .format(len_options=len(d['option']), options=d['option']) for d in data]
                elif self.config.question_type == 'oe':
                    prompt = [base_prompt.replace('INSERT_QUESTION_HERE', d['question']) for d in data]
                else:
                    raise Exception("Invalid question type!")
            else:
                raise TypeError('data must be list of strings')
        else:   
            raise Exception('wrong prompt type')
        
        return prompt, additional_system_prompt
    
def register_step_interpreters(config, **kwargs):
    vqa_mapping = {'internvl': InternVLInterpreter, 'internlmxcomposer': InternLMXComposerInterpreter}
    verify_mapping = {'internvl': InternVLInterpreterverify, 'internlmxcomposer':InternLMXComposerInterpreterverify}
    retrieve_mapping = {'univtg': RETRIEVEInterpreterUniVTG2, 'qddetr': RETRIEVEInterpreterQDDETR} # default UniVTG is spatiotemporal (localize->retrieve)
    if kwargs['mode'] == 'morevqa':
        # load model
        vqa_model = vqa_mapping[config.vlm_type](config, kwargs['device'])
        verify_model = verify_mapping[config.vlm_type](config, kwargs['device'])
        localize_model = LOCALIZEInterpreter(config, kwargs['device'])
        vqa_model = load_model(vqa_model, kwargs['device'], config)
        verify_model = load_model(verify_model, kwargs['device'], config)
        localize_model = load_model(localize_model, kwargs['device'], config)
        vqa_model.eval()
        verify_model.eval()
        localize_model.eval()
        
        loaded_model = dict(vqa=vqa_model, verify_action=verify_model, localize=localize_model)
        step_interpreters = dict(
            trim=TRIMInterpreter(),
            parse_event=PARSEEVENTInterpreter(),
            classify=CLASSIFYInterpreter(),
            require_ocr=REQUIREOCRInterpreter(),
            localize=localize_model,
            truncate=TRUNCATEInterpreter(),
            vqa=vqa_model,
            verify_action=verify_model,
        )
        return step_interpreters, loaded_model
    elif kwargs['mode'] == 'morevqa_retrieve':
        # load model
        vqa_model = vqa_mapping[config.vlm_type](config, kwargs['device'])
        retrieve_model = retrieve_mapping[config.retrieve_type](config, kwargs['device'])
        localize_model = LOCALIZEInterpreter(config, kwargs['device'])
        vqa_model = load_model(vqa_model, kwargs['device'], config)
        retrieve_model.to(kwargs['device'])
        localize_model = load_model(localize_model, kwargs['device'], config)
        vqa_model.eval()
        retrieve_model.eval()
        localize_model.eval()
        
        loaded_model = dict(vqa=vqa_model, retrieve=retrieve_model, localize=localize_model)
        step_interpreters = dict(
            trim=TRIMInterpreter(),
            parse_event=PARSEEVENTInterpreter(),
            classify=CLASSIFYInterpreter(),
            require_ocr=REQUIREOCRInterpreter(),
            localize=localize_model,
            truncate=TRUNCATEInterpreter2(),
            retrieve=retrieve_model,
            vqa=vqa_model,
        )
        return step_interpreters, loaded_model
    elif kwargs['mode'] == 'morevqa_understanding':
        # load mode
        vqa_model = vqa_mapping[config.vlm_type](config, kwargs['device'])
        retrieve_model = retrieve_mapping[config.retrieve_type](config, kwargs['device'])
        localize_model = LOCALIZEInterpreter(config, kwargs['device'])
        vqa_model = load_model(vqa_model, kwargs['device'], config)
        retrieve_model.to(kwargs['device'])
        localize_model = load_model(localize_model, kwargs['device'], config)
        vqa_model.eval()
        retrieve_model.eval()
        localize_model.eval()
        
        loaded_model = dict(vqa=vqa_model, retrieve=retrieve_model, localize=localize_model)
        step_interpreters = dict(
            trim=TRIMInterpreter2(),
            parse_event=PARSEEVENTInterpreter2(),
            classify=CLASSIFYInterpreter(),
            require_ocr=REQUIREOCRInterpreter(),
            localize=localize_model,
            truncate=TRUNCATEInterpreter2(),
            retrieve=retrieve_model,
            vqa=vqa_model,
        )
        return step_interpreters, loaded_model
    else:
        raise Exception('Invalid mode type!')

def load_model(model, device, config):
    model.to(device)
    if config.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.gpu])
        model_without_ddp = model.module
        return model_without_ddp
    return model

def unload_model(model):
    model.to('cpu')
    del model
    gc.collect()
    torch.cuda.empty_cache()