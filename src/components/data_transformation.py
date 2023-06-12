import sys
import os
from dataclasses import dataclass
import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation
        
        '''
        try:
            numerical_columns = ['DER_mass_MMC', 'DER_mass_transverse_met_lep', 'DER_mass_vis',
                                   'DER_pt_h', 'DER_deltaeta_jet_jet', 'DER_mass_jet_jet',
                                   'DER_prodeta_jet_jet', 'DER_deltar_tau_lep', 'DER_pt_tot', 'DER_sum_pt',
                                   'DER_pt_ratio_lep_tau', 'DER_met_phi_centrality',
                                   'DER_lep_eta_centrality', 'PRI_tau_pt', 'PRI_tau_eta', 'PRI_tau_phi',
                                   'PRI_lep_pt', 'PRI_lep_eta', 'PRI_lep_phi', 'PRI_met', 'PRI_met_phi',
                                   'PRI_met_sumet', 'PRI_jet_num', 'PRI_jet_leading_pt',
                                   'PRI_jet_leading_eta', 'PRI_jet_leading_phi', 'PRI_jet_subleading_pt',
                                   'PRI_jet_subleading_eta', 'PRI_jet_subleading_phi', 'PRI_jet_all_pt',
                                   'Weight']
            
            num_pipeline= Pipeline(steps=[("scaler",StandardScaler())])

            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor=ColumnTransformer([("num_pipeline",num_pipeline,numerical_columns)])

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)

            logging.info("Read train data")

            train_df = train_df.drop ('EventId', axis =1)

            logging.info("Event Id dropped from train set")
            
            test_df=pd.read_csv(test_path)

            logging.info("Read test data")

            test_df = test_df.drop ('EventId', axis =1)
            
            logging.info("Event Id dropped from test set")

            preprocessing_obj=self.get_data_transformer_object()

            logging.info("Preprocessing object obtained")

            input_feature_train_df=train_df.drop('Label',axis=1)

            logging.info("Dropped label column from the train set to make the input data frame for model training")

            target_feature_train_df=train_df['Label']

            logging.info("Target feature obtained for model training")

            input_feature_test_df=test_df.drop(columns=['Label'],axis=1)

            logging.info("Dropped label column from the test set to make the input data frame for model testing")
            
            target_feature_test_df=test_df['Label']

            logging.info("Target feature obtained for model testing")

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            
            logging.info("Preprocessing object applied on training dataframe.")

            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)
            
            logging.info("Preprocessing object applied on testing dataframe.")

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]

            logging.info("Combined the input features and target feature of train set as an array.")

            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Combined the input features and target feature of test set as an array.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            logging.info("Saved preprocessing object.")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)
