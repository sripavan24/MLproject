import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sor.exception import CustomException
from sor.logger import logging
from dataclasses import dataclass
from sor.utils import save_object
from sor.exception import CustomException


@dataclass
class Datatramsformationcogig:
    preprocessor_obj_file_path=os.path.join("artifacts","proprocessor.pkl")
    
    
class DataTransformation:
    def __init__(self):
        
        self.data_transformation_config=Datatramsformationcogig()
        
    def get_data_transformer_object(self):
        """
        this function is responsible for data transformation
        """
        
        try:
           numeric_features=['writing score','reading score']
           categorical_features= ['gender',
                                    'race/ethnicity',
                                    'parental level of education',
                                    'lunch',
                                    'test preparation course',
                                ] 
           num_pipeline= Pipeline(
                            steps=[
                                ("imputer",SimpleImputer(strategy="median")),
                                ("scaler",StandardScaler(with_mean=False))
                                 
                                ]
                            )
           cat_pipeline=Pipeline(steps=[
                                    ("impute",SimpleImputer(strategy="most_frequent")),
                                    ("One_hot_encoder",OneHotEncoder()),
                                    ("scaler",StandardScaler(with_mean=False)),
                                    ]
                                 )
           logging.info("numerical columns encoder completed")
           
           logging.info("categorical columns encoder completed")
           
           preprocessor=ColumnTransformer(
               [
                   ("num_pipeline",num_pipeline,numeric_features),
                   ("cat_pipline",cat_pipeline,categorical_features)
               ]
               
           )
           
           return preprocessor

           
        except Exception as e:
            
            raise CustomException(e,sys)
    def initiate_data_transformation(self,train_set,test_set):
        try:
            train_df=pd.read_csv(train_set)
            test_df=pd.read_csv(test_set)
            
            logging.info("read train and test data completed")
            logging.info("Obtaining preprocessing object")
            
            preprocessing_obj=self.get_data_transformer_object()
            
            target_column_name="math score"
            numeric_features=['writing score','reading score']
            
            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]
            
            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]
            
            logging.info(
                f"Applying apreprocessing abject on training dataframe and testing dataframe"
                )
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)
            
            train_arr=np.c_[input_feature_train_arr,np.array(input_feature_train_df)]
            test_arr=np.c_[input_feature_test_arr,np.array(input_feature_test_df)]
            
            logging.info(f"saved preprocessing object.")
            
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            
            raise CustomException(e,sys)
            
