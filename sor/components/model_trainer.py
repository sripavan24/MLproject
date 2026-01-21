import os
import sys
import numpy as np
import pandas as pd


from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score,mean_squared_error,root_mean_squared_error


from sor.exception import CustomException
from sor.logger import logging

from dataclasses import dataclass

from sor.utils import save_object,evaluate_models


@dataclass

class Modeltrainercongif:
    trained_model_file_path=os.path.join("artifacts","model.pkl")
    
class ModelTrainer:
    def __init__(self):
        self.model_train_config=Modeltrainercongif()
        
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("spliting training and test inpute data")
            x_train,y_train,x_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1],
                
            )
            
            models={
                "LogisticRegression":LinearRegression(),
        
                "AdaBoostRegressor":AdaBoostRegressor(),
                "GradientBoostingRegressor":GradientBoostingRegressor(),
                "RandomForestRegressor":RandomForestRegressor(),
                "KNeighborsRegressor":KNeighborsRegressor(),
                "DecisionTreeRegressor":DecisionTreeRegressor()
            }
            
            
            model_report:dict=evaluate_models(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,models=models)
            best_model_score=max(sorted(model_report.values()))
            
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model=models[best_model_name]
            
            if best_model_score<0.6:
                raise CustomException("no best model founded")
            
            logging.info("best founded model on the trainning data set")
            
            
            save_object(
                file_path=self.model_train_config.trained_model_file_path,obj=best_model
            )
            
            
            predicted=best_model.predict(x_test)
            r_score=r2_score(y_test,predicted)
            
            
            return r_score
            
            
        except Exception as e:
            raise CustomException(e,sys)
    

    
    


