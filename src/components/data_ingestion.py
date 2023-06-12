import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts',"train.csv")
    test_data_path: str=os.path.join('artifacts',"test.csv")
    raw_data_path: str=os.path.join('artifacts',"data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):

        logging.info("Entered the data ingestion method or component")
        
        try:

            df=pd.read_csv('notebook/Higgs Boson Machine Learning Challenge train.csv')

            logging.info('Read the dataset as dataframe')

            le = LabelEncoder()

            df.iloc[:,32] = le.fit_transform (df.iloc[:,32])

            logging.info("Label encoder applied on label column of the raw data set")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info("Raw Data file saved")

            y1 = df.iloc[:, 32]

            logging.info("y1 created for stratification")

            train_set,test_set=train_test_split(df,test_size=0.2,random_state=0, stratify =y1, shuffle=True)

            logging.info("Train Test Split Completed")

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)

            logging.info("Train Set Saved")

            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Test Set Saved")

            logging.info("Ingestion of the data has completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path)
        
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=="__main__":
    obj=DataIngestion()
    # obj.initiate_data_ingestion()

    # these pieces of codes are added later, after data creating data transformation
    train_data,test_data=obj.initiate_data_ingestion()

    data_transformation=DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)

    modeltrainer=ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr))