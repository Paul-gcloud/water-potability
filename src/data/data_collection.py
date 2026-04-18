import pandas as pd
import numpy as np
import os
import yaml
from sklearn.model_selection import train_test_split

#load data
#data = pd.read_csv(r"C:\Users\USER\Documents\Data Science\Types of Analysis\Data Analyst Package\Prac\Model Deployment\mlops\water_potability.csv")
def load_data(filepath:str)->pd.DataFrame:
    try: 
        return pd.read_csv(filepath)
    except Exception as e:
        raise Exception(f'Error loading data from {filepath}:{e}')
    
#split data
#test_size=yaml.safe_load(open('params.yaml'))['data_collection']['test_size']
def load_params(filepath:str)->float:
    try:
        with open(filepath, 'r')as file:
            params=yaml.safe_load(file)
        return params.get('data_collection', {}).get('test_size', 0.20)
    except ValueError as e:
        raise ValueError(f'Error loading parameters from {filepath}:{e}')
#train_data, test_data = train_test_split(data, test_size=test_size, random_state=42)
def split_data(data:pd.DataFrame, test_size:float):
    try:
        return train_test_split(data, test_size=test_size, random_state=42)
    except Exception as e:
        raise Exception(f'error spliting data :{e}')

def save_data(df:pd.DataFrame, filepath:str)->None:
    try:
        df.to_csv(filepath, index=False)
    except Exception as e:
        raise Exception(f'Error saving data to {filepath}:{e}')

def main():
    try:
        data_path = r'c:\Users\USER\Documents\Data Science\Types of Analysis\Data Analyst Package\Prac\Model Deployment\mlops\water_potability.csv'
        params_path = 'params.yaml'
        raw_data_path = os.path.join('data','raw')

        data= load_data(data_path)
        test_size= load_params(params_path)
        train_data, test_data = split_data(data, test_size)

        os.makedirs(raw_data_path)

        train_path = os.path.join(raw_data_path, 'train.csv')
        test_path = os.path.join(raw_data_path, 'test.csv')

        save_data(train_data, train_path)
        save_data(test_data, test_path)
    except Exception as e:
        raise Exception(f'Error occured :{e}')

if __name__=="__main__":
    main()



#create folders
#data_path = os.path.join('data','raw')
#os.makedirs(data_path)

#train_data.to_csv(os.path.join(data_path, 'train.csv'), index=False)
#test_data.to_csv(os.path.join(data_path, 'test.csv'), index=False)