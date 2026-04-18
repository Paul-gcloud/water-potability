import numpy as pd
import pandas as pd
import yaml
import os
import pickle
from sklearn.ensemble import RandomForestClassifier

#load_data
#train_data = pd.read_csv('./data/processed/train_processed.csv')
def load_data(filepath:str)->pd.DataFrame:
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        raise Exception(f'Error loading data from {filepath}:{e}')

#prepare data
#X_train = train_data.drop(columns=['Potability'])
#y_train = train_data['Potability']
def prepare_data(data:pd.DataFrame)-> tuple[pd.DataFrame, pd.Series]:
    try:
        X=data.drop(columns=['Potability'])
        y=data['Potability']
        return X,y
    except Exception as e:
        raise Exception(f'Error preparing data :{e}')

#load_model
#n_estimators=yaml.safe_load(open('params.yaml'))['model_building']['n_estimators']
def load_params(filepath:str)->int:
    try:
        with open(filepath, 'r')as file:
            params=yaml.safe_load(open('params.yaml'))
        return params.get('model_building', {}).get('n_estimators', 100)
    except Exception as e:
        raise Exception(f'Error loading parameters from {filepath}:{e}')
#clf=RandomForestClassifier(n_estimators=n_estimators)
#clf.fit(X_train, y_train)

def train_model(X:pd.DataFrame, y:pd.Series, n_estimators:int)-> RandomForestClassifier:
    try:
        clf=RandomForestClassifier(n_estimators=n_estimators)
        clf.fit(X,y)
        return clf
    except Exception as e:
        raise Exception(f'Error training model :{e}')

def save_model(model:RandomForestClassifier, filepath:str)-> None:
    try:
        with open(filepath, 'wb')as file:
            pickle.dump(model, file)
    except Exception as e:
        raise Exception(f'Error loading model to {filepath}')
#pickle.dump(clf,open('model.pkl', 'wb'))

def main():
    try:
        data_path = './data/processed/train_processed_mean.csv'
        params_path='params.yaml'
        model_name='models/model.pkl'

        train_data = load_data(data_path)
        n_estimators = load_params(params_path)
        X_train,y_train = prepare_data(train_data)

        model = train_model(X_train, y_train, n_estimators)
        save_model(model, model_name)
    except Exception as e:
        raise Exception(f'Error occured :{e}')

if __name__=='__main__':
    main()