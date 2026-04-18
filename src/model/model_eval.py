import numpy as np
import pandas as pd
import os
import pickle
import json
from sklearn.metrics import accuracy_score, classification_report

#load data
#test_data = pd.read_csv('./data/processed/test_processed.csv')
def load_data(filepath:str)->pd.DataFrame:
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        raise Exception(f'Error loading data from {filepath}')

#prepare data
#X_test = test_data.drop(columns=['Potability'])
#y_test = test_data['Potability']
def prepare_data(data:pd.DataFrame)->tuple[pd.DataFrame, pd.Series]:
    try:
        X=data.drop(columns=['Potability'])
        y=data['Potability']
        return X,y
    except Exception as e:
        raise Exception(f'Error preparing data : {e}')

#load model
#model = pickle.load(open('model.pkl', 'rb'))
def load_model(filepath:str):
    try:
        with open(filepath, 'rb') as file:
            model =pickle.load(file)
        return model
    except Exception as e:
        raise Exception(f'')

#prediction
#y_pred = model.predict(X_test)

#Evaluation
#acc=accuracy_score(y_test, y_pred)
#class_rpt = classification_report(y_test, y_pred)

#metrics_dict ={
#    'accuracy_score': acc,
 #   'classification_report':class_rpt 
#}


def model_evaluation(model, X_test:pd.DataFrame, y_test:pd.Series)-> dict:
    try:
        y_pred = model.predict(X_test)

        #Evaluation
        acc=accuracy_score(y_test, y_pred)
        class_rpt = classification_report(y_test, y_pred)

        metrics_dict ={
                'accuracy_score': acc,
                'classification_report':class_rpt 
        }
        return metrics_dict
    except Exception as e:
        raise Exception(f'error evaluating data :{e}')

#os.makedirs('metrics', exist_ok=True)

def save_metrics(metrics_dict:dict, filepath:str)->None:
    try:
        with open('reports/metrics.json','w')as file:
            json.dump(metrics_dict, file, indent=4)
    except Exception as e:
        raise Exception(f'error saving metrics to {filepath}:{e}')

#with open('metrics.json', 'w')as file:
#    json.dump(metrics_dict, file, indent=4)

def main():
    try:
        data_path='./data/processed/test_processed.csv'
        model_path='models/model.pkl'
        metrics_path ='reports/metrics.json'

        test_data=load_data(data_path)
        X_test, y_test = prepare_data(test_data)
        model = load_model(model_path)
        metrics=model_evaluation(model, X_test, y_test)
        
        save_metrics(metrics, metrics_path)
    except Exception as e:
        raise Exception(f'error occured : {e}')
    
if __name__=='__main__':
    main()




