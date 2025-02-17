#asset_[step_name].py
 
# -*- coding: utf-8 -*-
import os
import sys
from alolib.asset import Asset
import numpy as np 
import pandas as pd 
 
#--------------------------------------------------------------------------------------------------------------------------
#    CLASS
#--------------------------------------------------------------------------------------------------------------------------
class UserAsset(Asset):
    def __init__(self, asset_structure):
        super().__init__(asset_structure)
        self.args       = self.asset.load_args()
        self.config     = self.asset.load_config()
        self.data       = self.asset.load_data()
 
    @Asset.decorator_run
    def run(self):
        # data from inference asset 
        results = self.data['inference_results']
        result_df = []
        average_logits = []

        for path, boxes, logits, phrases in results:
            for box, logit, phrase in zip(boxes, logits, phrases):
                result_df.append({
                    'file_path': path,
                    'box': box.tolist(),
                    'logit': logit,
                    'phrase': phrase
                })

                average_logits.append(float(logit))

        result_df = pd.DataFrame(result_df)
        
        output_path = self.asset.get_output_path()
        result_df.to_csv(os.path.join(output_path, 'output.csv'))

        num_object = len(average_logits)
        average_logits = sum(average_logits) / num_object if num_object != 0 else 0.0        
        self.asset.save_summary(result=f"#detected objects: {num_object}", \
                score=average_logits, \
                note="Score means average logits")
        
        self.asset.save_data(self.data)
        self.asset.save_config(self.config)
 
 
#--------------------------------------------------------------------------------------------------------------------------
#    MAIN
#--------------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    envs, argv, data, config = {}, {}, {}, {}
    ua = UserAsset(envs, argv, data, config)
    ua.run()
