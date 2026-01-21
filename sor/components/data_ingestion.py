import os
import sys
from sor.exception import CustomException
from sor.logger import logging


import pandas as pd


from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from sor.components.data_tranasformatin import DataTransformation
from sor.components.data_tranasformatin import Datatramsformationcogig

from sor.components.model_trainer import ModelTrainer
from sor.components.model_trainer import Modeltrainercongif


@dataclass
class dataingestioncofig:
    tarin_path: str=os.path.join("artifacts","train_data.csv")
    test_path: str=os.path.join("artifacts","test_data.csv")
    raw_data_path: str=os.path.join("artifacts","raw_data.csv")
    
class Dataingestion:
    def __init__(self):
        self.ingestion=dataingestioncofig()
        
    def initiate_data_ingestion(self):
        logging.info("Enterd data ingestion or compount")
        try:
            df = pd.read_csv(r"notebook\data\std.csv")

            logging.info("read dataset as dataframe")
            
            df = df.rename(columns={
            "race/ethnicity": "race_ethnicity",
            "parental level of education": "parental_level_of_education",
            "test preparation course": "test_preparation_course",
            "math score":"math_score",
            "reading score": "reading_score",
            "writing score": "writing_score"
            
            })
            
            os.makedirs(os.path.dirname(self.ingestion.tarin_path),exist_ok=True)

            df.to_csv(self.ingestion.raw_data_path,index=False,header=True)
            
            logging.info("Train test split initited")
            
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=41)
            
            train_set.to_csv(self.ingestion.tarin_path,index=False,header=True)
            
            test_set.to_csv(self.ingestion.test_path,index=False,header=True)
            
            logging.info("inmgerstion of the data id completed")
            
            return (self.ingestion.tarin_path,
                    self.ingestion.test_path)
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=="__main__":
    ooj=Dataingestion()
    train_data,test_data=ooj.initiate_data_ingestion() 

    data_transformation=DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)

    modeltrainer=ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr))