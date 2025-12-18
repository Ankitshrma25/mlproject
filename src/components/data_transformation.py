import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer    # To handle missing values
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):   # This function is responsible for data transformation
        '''
        This function is responsible for data transformation

        '''
        try:
            numerical_columns = ['writing_score', 'reading_score']
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            ]

            # Create a pipeline for numerical columns
            num_pipeline = Pipeline(
                steps=[
                    ("Imputer", SimpleImputer(strategy='median')),  # Hnalde missing Values replacing them with median of the column
                    ("scaler",StandardScaler())
                ]
            
            )

            # Create a pipeline for categorical columns
            cat_pipeline = Pipeline(
                steps=[
                    ("Imputer",SimpleImputer(strategy='most_frequent')),    # Handle missing values by replacing them with most frequent category(Mode)
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            logging.info("Numerical Column standard scalering completed ")
            logging.info("Categorical columns encoding completed ")

            # Combine both pipelines using ColumnTransformer
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline,numerical_columns),
                    ("cat_pipeline", cat_pipeline,categorical_columns)
                ]
            )
            logging.info("Column Transformer Completed")
            return preprocessor


        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df = pd.read_csv(train_path)     # Read training dataset from given path
            test_df = pd.read_csv(test_path)       # Read testing dataset from given path

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessor_obj = self.get_data_transformer_object()

            target_column_name = "math_score"      # Target column name
            numerical_column = ["writing_score","reading_score"]

            input_feature_train_df = train_df.drop(columns=[target_column_name],axis=1)   # Input features for training dataset
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name],axis=1)     # Input features for testing dataset
            target_feature_test_df = test_df[target_column_name]

            logging.info(f"Applying preprocessing object on training dataframe and testing dataframe.")

            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr,np.array(target_feature_train_df)]

            test_arr = np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
                
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)