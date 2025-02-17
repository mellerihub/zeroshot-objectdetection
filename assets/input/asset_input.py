#asset_[step_name].py
 
# -*- coding: utf-8 -*-
import os
import sys
from alolib.asset import Asset
import pandas as pd  
from glob import glob

#--------------------------------------------------------------------------------------------------------------------------
#    CLASS
#--------------------------------------------------------------------------------------------------------------------------
class UserAsset(Asset):
    def __init__(self, asset_structure):
        super().__init__(asset_structure)
        self.args       = self.asset.load_args() # experimental_plan.yaml args
        self.config     = self.asset.load_config() # asset - asset interface config
        self.data       = {} # asset - asset interface data

    def get_csv(self):
        path = os.path.join(self.asset.get_input_path(), 'inference/*.csv')
        return glob(path)
    
    def df_from_csvs(self, csvs):
        csvs = [pd.read_csv(f) for f in csvs]
        df = pd.concat(csvs)

        df['image_path'] = df['image_path'].apply(
            lambda x: os.path.join('inference', x.replace('./', '')) if x.startswith('./') else x)
        
        return df

    def get_image_files(self):
        allowed_formats = ['jpg', 'png', 'jpeg']

        file_path_list = sum([glob(os.path.join(self.asset.get_input_path(), f'**/*{ext}'),recursive=True) for ext in allowed_formats], [])

        return file_path_list
 
    @Asset.decorator_run
    def run(self):
        if self.args['input_mode'] == 'csv':
            csv_files = self.get_csv()
            df = self.df_from_csvs(csv_files)

            image_files = df['image_path'].tolist()
            text_prompts = df['text_prompts'].tolist()
        
        else:
            image_files = self.get_image_files()
            text_prompts = None
            
        self.data['image_files'] = image_files
        self.data['text_prompts'] = text_prompts
            
        self.asset.save_data(self.data) # to next asset
        self.asset.save_config(self.config) # to next asset
#--------------------------------------------------------------------------------------------------------------------------
#    MAIN
#--------------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    envs, argv, data, config = {}, {}, {}, {}
    ua = UserAsset(envs, argv, data, config)
    ua.run()
