#asset_[step_name].py
 
# -*- coding: utf-8 -*-

import os
import logging
import warnings

from tqdm import tqdm
from PIL import Image

from alolib.asset import Asset
from groundingdino import GroundingDino

from visualize import annotate, annotate_mask
import torch
import numpy as np
warnings.filterwarnings('ignore')

#--------------------------------------------------------------------------------------------------------------------------
#    CLASS
#--------------------------------------------------------------------------------------------------------------------------
class NoLog:
    def __enter__(self):
        logging.disable()

    def __exit__(self, type, val, trace):
        logging.disable(logging.NOTSET)


class UserAsset(Asset):
    def __init__(self, asset_structure):
        super().__init__(asset_structure)
        self.args       = self.asset.load_args()
        self.config     = self.asset.load_config()
        self.data       = self.asset.load_data()    

        self.device = self.args['device']

        if self.device == 'cpu':
            self.asset.save_warning('CUDA not Working!')
            self.asset.save_warning('It is strongly recommended to use CUDA for fast inference')
        else:
            if torch.cuda.is_available():
                self.asset.save_info(f"PyTorch version: {torch.__version__}")
                self.asset.save_info(f"CUDA available: {torch.cuda.is_available()}")
                self.asset.save_info(f"CUDA version: {torch.version.cuda}")
            else:
                self.device = 'cpu'
                self.asset.save_warning('CUDA not Working!')
                self.asset.save_warning('It is strongly recommended to use CUDA for fast inference')
 
    @Asset.decorator_run
    def run(self):
        results = []
        image_files = self.data['image_files']
        text_prompts = self.data['text_prompts']

        if text_prompts is None:
            text_prompts = [self.args['text_prompt'] for _ in image_files]

        bert_pretraind = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bert-base-uncased')
        self.asset.save_info(f"pretraind_path: {bert_pretraind}")

        with NoLog():
            model = GroundingDino(device=self.device, tokenizer_path=bert_pretraind)

        output_dir = self.asset.get_output_path()
        extra_output_dir = self.asset.get_extra_output_path()

        self.asset.save_info('GroundingDino Inference Start')

        pbar = tqdm(zip(image_files, text_prompts), total=len(image_files), desc='GroundingDino Inference')

        try:
            for idx,(path, text) in enumerate(pbar):
                image = Image.open(path).convert('RGB')
                image_name = path.split('/')[-1]
                
                boxes, logits, phrases = model(image, text, 
                                                box_threshold=self.args['box_threshold'], 
                                                text_threshold=self.args['text_threshold'],
                                                nms_threshold=self.args['nms_threshold'])
                
                if self.args['postprocess'] == True:
                    from postprocess import remove_overlap
                    keep = remove_overlap(boxes)
                    boxes, logits, phrases = boxes[keep], logits[keep], [phrases[i] for i in keep]
                
                results.append([path, boxes, logits, phrases])
                result_image = annotate(image, boxes, logits, phrases)
                if idx ==0:
                    Image.fromarray(result_image).save(os.path.join(output_dir, image_name))
                else:
                    Image.fromarray(result_image).save(os.path.join(extra_output_dir, image_name))
        except Exception as e:
            self.asset.save_warning(f"{e}")
            image_array = np.array(image)
            Image.fromarray(image_array).save(os.path.join(output_dir, image_name))

        del model


        self.data['inference_results'] = results
        self.asset.save_data(self.data)
        self.asset.save_config(self.config)
 
 
#--------------------------------------------------------------------------------------------------------------------------
#    MAIN
#--------------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    envs, argv, data, config = {}, {}, {}, {}
    ua = UserAsset(envs, argv, data, config)
    ua.run()
