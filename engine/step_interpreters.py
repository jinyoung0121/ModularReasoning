import cv2
import os
import sys
import re
import torch
import clip
import random
from torchvision import transforms
import numpy as np
import copy
import io, tokenize
from torch.nn import functional as F
from PIL import Image,ImageDraw,ImageFont,ImageFilter
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from transformers import Blip2Processor, Blip2ForConditionalGeneration, OwlViTProcessor, OwlViTForObjectDetection
from util import load_json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)))

device = "cuda" if torch.cuda.is_available() else "cpu"

def parse_step(step_str,partial=False, module_idx=2):
    tokens = list(tokenize.generate_tokens(io.StringIO(step_str).readline))
    try:
        output_var = tokens[0].string
        step_name = tokens[2].string
        print(output_var, step_name)
    except:
        raise Exception('Invalid program token parsing!')
    parsed_result = dict(
        output_var=output_var,
        step_name=step_name)
    if partial:
        return parsed_result

    arg_tokens = [token for token in tokens[4:-3] if token.string not in [',','=']]
    token_string = [token.string for token in arg_tokens]
    contain_list = True if ('[' in token_string) and (']' in token_string) else False
    num_tokens = len(arg_tokens) // 2
    arguments = dict()
    if contain_list:
        # 일단 heuristic하게 작성
        if module_idx == 2:
            end_idx = token_string.index(']')
            arguments[arg_tokens[0].string] = arg_tokens[1].string
            arguments[arg_tokens[2].string] = [arg_tokens[i].string for i in range(4,end_idx)]
            parsed_result['args'] = arguments
        elif module_idx == 3:
            end_idx = token_string.index(']')
            arguments[arg_tokens[0].string] = [arg_tokens[i].string for i in range(2,end_idx)]
            arguments[arg_tokens[end_idx+1].string] = arg_tokens[end_idx+2].string
            parsed_result['args'] = arguments
        
        else:
            raise Exception('Invalid module index! (when handling list)')
    else:
        for i in range(num_tokens):
            arguments[arg_tokens[2*i].string] = arg_tokens[2*i+1].string
        parsed_result['args'] = arguments
    return parsed_result

class TRIMInterpreter():
    step_name = 'trim'
    def __init__(self):
        print(f'Registering {self.step_name} step')
    
    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        output_var = parse_result['output_var']
        args = parse_result['args']
        trim_option = args['trim']
        truncated_question = None
        if trim_option != '"none"':
            truncated_question = args['truncated_question']
        assert(step_name==self.step_name)
        return trim_option, truncated_question, output_var

    def execute(self,prog_step,inspect=False):
        trim_option, truncated_question, output_var = self.parse(prog_step)
        out_value = {'trim': trim_option, 'truncated_question': truncated_question}
        prog_step.state[output_var] = out_value
        if inspect:
            html_str = self.html(trim_option,output_var)
            return out_value, html_str

        return out_value

class PARSEEVENTInterpreter():
    step_name = 'parse_event'
    def __init__(self):
        print(f'Registering {self.step_name} step')

    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        output_var = parse_result['output_var']
        args = parse_result['args']
        conj_option = args['conj']
        event_to_localize, truncated_question = None, None
        if conj_option != '"none"':
            event_to_localize = args['event_to_localize']
            truncated_question = args['truncated_question']
        assert(step_name==self.step_name)
        return conj_option, event_to_localize, truncated_question, output_var

    def execute(self,prog_step,inspect=False):
        conj_option, event_to_localize, truncated_question, output_var = self.parse(prog_step)
        out_value = {'conj': conj_option, 'event_to_localize': event_to_localize, 'truncated_question': truncated_question}
        prog_step.state[output_var] = out_value
        if inspect:
            html_str = self.html(conj_option, event_to_localize, truncated_question, output_var)
            return out_value, html_str

        return out_value

class CLASSIFYInterpreter():
    step_name = 'classify'
    def __init__(self):
        print(f'Registering {self.step_name} step')

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
        if inspect:
            html_str = self.html(qatype_option,output_var)
            return out_value, html_str

        return out_value

class REQUIREOCRInterpreter():
    step_name = 'require_ocr'
    def __init__(self):
        print(f'Registering {self.step_name} step')

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
        if inspect:
            html_str = self.html(ocr_option,output_var)
            return out_value, html_str

        return out_value

class LOCALIZEInterpreter():
    step_name = 'localize'
    def __init__(self, config, gpu_number=0):
        print(f'Registering {self.step_name} step')
        self.dev = f'cuda:{gpu_number}' if device == 'cuda' else device
        
        self.config = config
        # localize model
        localize_model_id = config.owlvit.model_path
        self.localize_processor = OwlViTProcessor.from_pretrained(localize_model_id)
        self.localize_model = OwlViTForObjectDetection.from_pretrained(localize_model_id).to(self.dev)
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
        if noun_var == '""' and modifier_var == '""':
            prog_step.state[output_var] = indicator
            if inspect:
                html_str = self.html(noun_var, modifier_var, output_var)
                return indicator, html_str
            return indicator
        # iterate over images
        for i in candidate_frame_ids:
            img = prog_step.state['image'][i]
            img = self.to_PIL(img)
            noun_obj_name = noun_var[1:-1]
            modifier_obj_name = modifier_var[1:-1]
            # update to False if object is not detected
            indicator[i] = self.predict(img, noun_obj_name, modifier_obj_name)                
            
        prog_step.state[output_var] = indicator
        if inspect:
            html_str = self.html(noun_var, modifier_var, output_var)
            return indicator, html_str
        
        return indicator

class TRUNCATEInterpreter():
    step_name = 'truncate'
    def __init__(self):
        print(f'Registering {self.step_name} step')
    
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
        if truncate_option == '"when"':
            prog_step.state['indicator'] = torch.zeros(prog_step.state['image'].size(0)).bool()
            prog_step.state['indicator'][prev_frame_ids] = True
        elif truncate_option == '"before"':
            if len(prev_frame_ids) == 0: # nothing is detected in the previous step
                prog_step.state['indicator'] = torch.zeros(prog_step.state['image'].size(0)).bool()
            else:
                anchor_index = min(prev_frame_ids)
                prog_step.state['indicator'][anchor_index:] = False
        elif truncate_option == '"after"':
            if len(prev_frame_ids) == 0: # nothing is detected in the previous step
                prog_step.state['indicator'] = torch.zeros(prog_step.state['image'].size(0)).bool()
            else:
                anchor_index = max(prev_frame_ids)
                prog_step.state['indicator'][:anchor_index+1] = False
        frame_id = torch.where(prog_step.state['indicator']==True)[0].tolist()
        prog_step.state[output_var] = frame_id
        if inspect:
            html_str = self.html(truncate_option,anchor_option,output_var)
            return frame_id, html_str
        
        return frame_id

class VQAInterpreter():
    step_name = 'vqa'
    def __init__(self, config, gpu_number=1):
        print(f'Registering {self.step_name} step')
        self.dev = f'cuda:{gpu_number}' if device == 'cuda' else device
        
        self.config = config
        model_id = config.blip.model_path
        self.processor = Blip2Processor.from_pretrained(model_id)
        self.model = Blip2ForConditionalGeneration.from_pretrained(model_id).to(self.dev)
        self.model.eval()
        self.prompt = {'vqa': "Question: {} Short answer:",
                       'verify': "Question: {} Please answer yes or no. Answer:"}
        self.max_words = 100
        self.to_PIL = transforms.ToPILImage()
    
    def parse(self, prog_step):
        parse_result = parse_step(prog_step.prog_str, module_idx=3)
        step_name = parse_result['step_name']
        output_var = parse_result['output_var']
        args = parse_result['args']
        questions = args['question']
        require_ocr = args['require_ocr']
        assert(step_name==self.step_name)
        return questions, require_ocr ,output_var

    @torch.no_grad()
    def predict(self, img, question, prompt_type='vqa'):
        def pre_question(question):
            # from LAVIS blip_processors
            question = re.sub(
                r"([.!\"()*#:;~])",
                "",
                question.lower(),
            )
            question = question.rstrip(" ")

            # truncate question
            question_words = question.split(" ")
            if len(question_words) > self.max_words:
                question = " ".join(question_words[: self.max_words])

            return question
        if isinstance(question, str):
            prompt = [self.prompt[prompt_type].format(pre_question(question))]
        else:
            raise Exception('invalide question type')
        inputs = self.processor(images=img, text=prompt, return_tensors='pt', padding='longest').to(self.dev)
        outputs = self.model.generate(**inputs)
        output_text = self.processor.batch_decode(outputs, skip_special_tokens=True)
        return output_text

    def execute(self,prog_step,inspect=False):
        questions, require_ocr, output_var = self.parse(prog_step)
        
        candidate_frame_ids = prog_step.state['frame_ids']
        # initialize QA pool. save {frame_id, Q, A} pair
        QA_pool = []
        # initialize index for selecting question
        if isinstance(questions, str):
            questions = [questions]
            q_idxs = np.zeros(len(candidate_frame_ids), dtype=int).tolist()
        elif isinstance(questions, list):
            if len(questions) == 0:
                prog_step.state[output_var] = QA_pool
                return QA_pool
            q_idxs = np.random.randint(0, len(questions), size=len(candidate_frame_ids), dtype=int).tolist()
        # iterate over images, make QA pair
        for i, q_idx in zip(candidate_frame_ids, q_idxs):
            img = prog_step.state['image'][i]
            img = self.to_PIL(img)
            question = questions[q_idx][1:-1]
            answer = self.predict(img, question, prompt_type='vqa')[0]
            QA_pool.append({'frame_id': i, 'question': question, 'answer': answer})
        
        prog_step.state[output_var] = QA_pool
        if inspect:
            html_str = self.html(img, answer, output_var)
            return QA_pool, html_str

        return QA_pool

class VERIFYACTIONInterpreter(VQAInterpreter):
    step_name = 'verify_action'
    def parse(self, prog_step):
        parse_result = parse_step(prog_step.prog_str, module_idx=2)
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
                indicator = indicator * torch.tensor(prog_step.state[noun])
        candidate_frame_ids = torch.where(indicator==True)[0].tolist()
        # do not update frame_id when action=='no_action'
        if action == '"no_action"':
            prog_step.state[output_var] = candidate_frame_ids
            if inspect:
                html_str = self.html(action, nouns, output_var)
                return candidate_frame_ids, html_str
            return candidate_frame_ids
        # iterate over images, update frame_id
        for i in candidate_frame_ids:
            img = prog_step.state['image'][i]
            img = self.to_PIL(img)
            question = action[1:-1]
            answer = self.predict(img, question, prompt_type='verify')[0]
            if 'yes' == answer.lower():
                indicator[i] = True
            elif 'no'== answer.lower(): # 일단 'yes'가 없으면 'no'라고 가정
                indicator[i] = False
            else:
                raise Exception("Invalid answer type. Should be either 'yes' or 'no'")
        frame_id = torch.where(indicator==True)[0].tolist()
        prog_step.state[output_var] = frame_id
        if inspect:
            html_str = self.html(action, nouns, output_var)
            return frame_id, html_str
        
        return frame_id

class VQAInterpreterInternVL():
    step_name = 'vqa'
    def __init__(self, config, gpu_number=1):
        print(f'Registering {self.step_name} step')
        self.dev = f'cuda:{gpu_number}' if device == 'cuda' else device

        self.config = config
        model_id = config.internvl.model_path
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=False)
        self.model = AutoModel.from_pretrained(model_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, use_flash_attn=True, trust_remote_code=True).to(self.dev)
        self.model.eval()
        
        self.prompt = {'vqa': '<image>\n{} Answer the question shortly.',
                       'verify': '<image>\n{} Answer the question either yes or no.'}
        self.generation_config = dict(max_new_tokens=1024, do_sample=False)
        self.to_PIL = transforms.ToPILImage()
        self.max_batch_size = 6
        
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
    
    def parse(self, prog_step):
        parse_result = parse_step(prog_step.prog_str, module_idx=3)
        step_name = parse_result['step_name']
        output_var = parse_result['output_var']
        args = parse_result['args']
        questions = args['question']
        require_ocr = args['require_ocr']
        assert(step_name==self.step_name)
        return questions, require_ocr ,output_var
    
    @torch.no_grad()
    def predict(self, img, question, prompt_type='vqa'):
        if isinstance(question, str):
            question = self.prompt[prompt_type].format(question)
        else:
            raise Exception('invalide question type')
        pixel_values = self.load_image(img, max_num=12).to(torch.bfloat16).to(self.dev)
        output_text = self.model.chat(self.tokenizer, pixel_values, question, self.generation_config, device=self.dev)
        return output_text

    @torch.no_grad()
    def batch_predict(self, imgs, questions, prompt_type='vqa'):
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

    def execute(self, prog_step, inspect=False):
        questions, require_ocr, output_var = self.parse(prog_step)
        
        candidate_frame_ids = prog_step.state['frame_ids']
        # initialize QA pool. save {frame_id, Q, A} pair
        QA_pool = []
        # initialize index for selecting question
        if isinstance(questions, str):
            questions = [questions]
            q_idxs = np.zeros(len(candidate_frame_ids), dtype=int).tolist()
        elif isinstance(questions, list):
            if len(questions) == 0:
                prog_step.state[output_var] = QA_pool
                return QA_pool
            q_idxs = np.random.randint(0, len(questions), size=len(candidate_frame_ids), dtype=int).tolist()
        # # iterate over images, make QA pair
        # for i, q_idx in zip(candidate_frame_ids, q_idxs):
        #     img = prog_step.state['image'][i]
        #     img = self.to_PIL(img)
        #     question = questions[q_idx][1:-1]
        #     answer = self.predict(img, question, prompt_type='vqa')
        #     QA_pool.append({'frame_id': i, 'question': question, 'answer': answer})
        # iterate over images, make QA pair (batch)
        imgs = [self.to_PIL(prog_step.state['image'][i]) for i in candidate_frame_ids]
        questions = [questions[q_idx][1:-1] for q_idx in q_idxs]
        answers = self.batch_predict(imgs, questions, prompt_type='vqa')
        for i, question, answer in zip(candidate_frame_ids, questions, answers):
            QA_pool.append({'frame_id': i, 'question': question, 'answer': answer})
        prog_step.state[output_var] = QA_pool
        if inspect:
            html_str = self.html(QA_pool, answer, output_var)
            return QA_pool, html_str

        return QA_pool

class VERIFYACTIONInterpreterInternVL(VQAInterpreterInternVL):
    step_name = 'verify_action'
    def parse(self, prog_step):
        parse_result = parse_step(prog_step.prog_str, module_idx=2)
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
                indicator = indicator * torch.tensor(prog_step.state[noun])
        candidate_frame_ids = torch.where(indicator==True)[0].tolist()
        # do not update frame_id when action=='no_action'
        if action == '"no_action"':
            prog_step.state[output_var] = candidate_frame_ids
            if inspect:
                html_str = self.html(action, nouns, output_var)
                return candidate_frame_ids, html_str
            return candidate_frame_ids
        # # iterate over images, update frame_id
        # for i in candidate_frame_ids:
        #     img = prog_step.state['image'][i]
        #     img = self.to_PIL(img)
        #     question = action[1:-1]
        #     answer = self.predict(img, question, prompt_type='verify')
        #     if 'yes' in answer.lower():
        #         indicator[i] = True
        #     elif 'no' in answer.lower(): # 일단 'yes'가 없으면 'no'라고 가정
        #         indicator[i] = False
        #     else:
        #         raise Exception("Invalid answer type. Should be either 'yes' or 'no'")
        # iterate over images, update frame_id (batch)
        imgs = [self.to_PIL(prog_step.state['image'][i]) for i in candidate_frame_ids]
        questions = [action[1:-1]] * len(imgs)
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
        if inspect:
            html_str = self.html(action, nouns, output_var)
            return frame_id, html_str
        
        return frame_id
    
class Llama():
    step_name = 'llama'
    def __init__(self, config, gpu_number=2):
        print(f'Registering {self.step_name} step')
        self.dev = f'cuda:{gpu_number}' if device == 'cuda' else device
        
        self.config = config
        model_id = config.llama.model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left'
        self.model = AutoModelForCausalLM.from_pretrained(model_id).to(self.dev)
        self.model.eval()
        
        self.max_batch_size = self.config.llama.max_batch_size
    
    def apply_chat(self, prompt, prompt_type):
        if prompt_type in ['final', 'llm_only']:
            message = [{"role": "system", "content": "Only answer with the final answer."},
                    {"role": "user", "content": prompt}]
        elif prompt_type in ['module1', 'module2', 'module3']:
            message = [{"role": "system", "content": "Only answer with the final answer similar to given examples."},
                    {"role": "user", "content": prompt}]
        else:
            raise Exception('Invalid prompt type.')
        return message
    
    def load_prompt(self, data, prompt_type):
        if prompt_type == 'module1':
            prompt_path = 'datas/prompt/module1.prompt'
        elif prompt_type == 'module2':
            prompt_path = 'datas/prompt/module2.prompt'
        elif prompt_type == 'module3':
            prompt_path = 'datas/prompt/module3.prompt'
        elif prompt_type == 'final':
            if self.config.question_type == 'mc':
                prompt_path = 'datas/prompt/final_prediction_mc.prompt'
            elif self.config.question_type == 'oe':
                prompt_path = 'datas/prompt/final_prediction_oe.prompt'
            else:
                raise Exception('Invalid question type!')
        elif prompt_type == 'llm_only':
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
            if isinstance(data, list):
                prompt = [base_prompt.replace('INSERT_QUESTION_HERE', d) for d in data]
            elif isinstance(data, str):
                prompt = [base_prompt.replace('INSERT_QUESTION_HERE', data)]
            else:
                raise TypeError("question must be a string or a list of strings")
        elif prompt_type == 'module2':
            if isinstance(data, list):
                prompt = []
                for d in data:
                    if d['conjunction'] == 'none':
                        if len(d['event_queue']) == 1: # conj=="none"인 경우 이전 question 혹은 truncated_question을 넘겨받기 때문에 len(event_queue)==1
                            temp = base_prompt.replace('INSERT_PHRASE_HERE', d['event_queue'][0])
                        else:
                            raise Exception("'event_queue' length should be equal to 1 if 'conj'=='none'")
                    else: # len(event_queue)==2
                        if len(d['event_queue']) == 2: # for sanity check
                            temp1 = base_prompt.replace('INSERT_PHRASE_HERE', d['event_queue'][0])
                            temp2 = base_prompt.replace('INSERT_PHRASE_HERE', d['event_queue'][1])
                            temp = [temp1, temp2, d["conjunction"]]
                        else:
                            raise Exception("'event_queue' length should be equal to 2 if 'conj'!='none'")
                    prompt.append(temp)
            else:
                if data['conjunction'] == 'none':
                    if len(data['event_queue']) == 1: # conj=="none"인 경우 이전 question 혹은 truncated_question을 넘겨받기 때문에 len(event_queue)==1
                        temp = base_prompt.replace('INSERT_PHRASE_HERE', data['event_queue'][0])
                    else:
                        raise Exception("'event_queue' length should be equal to 1 if 'conj'=='none'")
                else: # len(event_queue)==2
                    if len(data['event_queue']) == 2: # for sanity check
                        temp1 = base_prompt.replace('INSERT_PHRASE_HERE', data['event_queue'][0])
                        temp2 = base_prompt.replace('INSERT_PHRASE_HERE', data['event_queue'][1])
                        temp = [temp1, temp2, data["conjunction"]]
                    else:
                        raise Exception("'event_queue' length is not equal to 2")
                prompt = [temp]
        elif prompt_type == 'module3':
            if isinstance(data, list):
                prompt = [base_prompt.replace('INSERT_QATYPE_HERE', d['qa_type']).replace('INSERT_QUESTION_HERE', d['question']) for d in data]
            elif isinstance(data, str):
                prompt = base_prompt.replace('INSERT_QATYPE_HERE', data['qa_type']).replace('INSERT_QUESTION_HERE', data['question'])
            else:
                raise TypeError("Invalid data type")
        elif prompt_type == 'final':
            if isinstance(data, list):
                if self.config.question_type == 'mc':
                    prompt = [base_prompt.replace('INSERT_SUMMARY_HERE', d['video_context']+'\n'+d['VLM_answer'] if d['VLM_answer'] != '' else d['video_context']).replace('INSERT_QUESTION_HERE', d['question']).format(len_options=len(d['option']), options=d['option']) for d in data]
                elif self.config.question_type == 'oe':
                    prompt = [base_prompt.replace('INSERT_SUMMARY_HERE', d['video_context']+'\n'+d['VLM_answer'] if d['VLM_answer'] != '' else d['video_context']).replace('INSERT_QUESTION_HERE', d['question']) for d in data]
                else:
                    raise Exception("Invalid question type!")
            elif isinstance(data, str):
                if self.config.question_type == 'mc':
                    prompt = [base_prompt.replace('INSERT_SUMMARY_HERE', d['video_context']+'\n'+d['VLM_answer'] if d['VLM_answer'] != '' else d['video_context']).replace('INSERT_QUESTION_HERE', data['question']).format(len_options=len(data['option']), options=data['option'])]
                elif self.config.question_type == 'oe':
                    prompt = [base_prompt.replace('INSERT_SUMMARY_HERE', d['video_context']+'\n'+d['VLM_answer'] if d['VLM_answer'] != '' else d['video_context']).replace('INSERT_QUESTION_HERE', data['question'])]
                else:
                    raise Exception("Invalid question type!")
        elif prompt_type == 'llm_only':
            if isinstance(data, list):
                if self.config.question_type == 'mc':
                    prompt = [base_prompt.replace('INSERT_QUESTION_HERE', d['question']).format(len_options=len(d['option']), options=d['option']) for d in data]
                elif self.config.question_type == 'oe':
                    prompt = [base_prompt.replace('INSERT_QUESTION_HERE', d['question']) for d in data]
                else:
                    raise Exception("Invalid question type!")
            elif isinstance(data, str):
                if self.config.question_type == 'mc':
                    prompt = [base_prompt.replace('INSERT_QUESTION_HERE', data['question']).format(len_options=len(data['option']), options=data['option'])]
                elif self.config.question_type == 'oe':
                    prompt = [base_prompt.replace('INSERT_QUESTION_HERE', data['question'])]
                else:
                    raise Exception("Invalid question type!")
        else:   
            raise Exception('wrong prompt type')
        
        return prompt
    
    @torch.no_grad()
    def generate_(self, prompt, prompt_type):
        if isinstance(prompt, str):
            message = [self.apply_chat(prompt, prompt_type)]
        else:
            raise Exception('Invalid prompt type!')
        
        input_ids = self.tokenizer.apply_chat_template(
            message,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.dev)

        outputs = self.model.generate(
            input_ids,
            max_new_tokens=self.config.llama.max_tokens,
            do_sample=self.config.llama.do_sample,
            temperature=self.config.llama.temperature,
            top_p=self.config.llama.top_p,
            pad_token_id=self.tokenizer.eos_token_id
        )
        outputs = outputs[0][input_ids.shape[-1]:]
        output_text = self.tokenizer.decode(outputs, skip_special_tokens=True)
        output_text = [output_text.strip().replace('Answer:\n','')]
        return output_text
    
    @torch.no_grad()
    def generate(self, data, prompt_type='module1'):
        prompt = self.load_prompt(data, prompt_type)
        if prompt_type == 'module2': # generation per sample
            response = []
            for p in prompt:
                if isinstance(p, list): # len(event_queue)==2 case
                    final_p = self.generate_(p[0],prompt_type)[0] + '\n'
                    final_p += f'TRUNCATE0=truncate(truncate="{p[2]}", anchor=VERIFY_ACTION0)' + '\n'
                    final_p += self.generate_(p[1],prompt_type)[0]
                    response += [final_p]
                else:
                    response += self.generate_(p, prompt_type)
            return response
        else: # for module1, module3
            response = []
            for p in prompt:
                response += self.generate_(p, prompt_type)
            return response

def register_step_interpreters(config, mode='modular'):
    if mode == 'modular':
        return dict(
            trim=TRIMInterpreter(),
            parse_event=PARSEEVENTInterpreter(),
            classify=CLASSIFYInterpreter(),
            require_ocr=REQUIREOCRInterpreter(),
            localize=LOCALIZEInterpreter(config, gpu_number=0),
            truncate=TRUNCATEInterpreter(),
            verify_action=VERIFYACTIONInterpreterInternVL(config, gpu_number=1),
            vqa=VQAInterpreterInternVL(config, gpu_number=2),
            llama=Llama(config, gpu_number=3),
        )
    elif mode == 'jcef' or mode == 'llm_only':
        return dict(
            llama=Llama(config, gpu_number=0)
        )
    else:
        raise Exception('Invalid mode type!')