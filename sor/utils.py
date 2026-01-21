import os
import sys
import pickle
import pandas as pd
import numpy as np
import dill

from sklearn.metrics import r2_score
from sor.exception import CustomException

def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        
        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)
    
    except Exception as e:
        raise CustomException(e,sys)

def evaluate_models(x_train,y_train,x_test,y_test,models):
    try:
        report={}

        for i in range(len(list(models))):
            processes=list(models.values())[i]
            processes.fit(x_train,y_train)
            y_pred=processes.predict(x_test)
            
            r_score=r2_score(y_test,y_pred)
            
            report[list(models.keys())[i]]=r_score
            
        return report
    except Exception as e:
        raise CustomException(e,sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)