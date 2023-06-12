import os
import sys
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve, precision_recall_curve


@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array):
        try:
            x_train,y_train,x_test,y_test=(train_array[:,:-1], train_array[:,-1], test_array[:,:-1],
                                           test_array[:,-1])
            logging.info("Made x_train, y_train, x_test, y_test")

            lr = LogisticRegression()

            logging.info("Logistic Regression Initiated")
            
            lr.fit (x_train, y_train)

            logging.info("Training Data Fitted On The Model")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=lr)

            y_pred_train=lr.predict(x_train)

            logging.info("Made prediction on training data")

            y_pred_test=lr.predict(x_test)

            logging.info("Made prediction on test data")

            training_data_accuracy = accuracy_score(y_pred_train, y_train)

            print ("Accuracy On Training Data:", training_data_accuracy*100)

            logging.info("Train Accuracy Printed")
            test_data_accuracy = accuracy_score(y_pred_test, y_test)
            print ("Accuracy On Test Data:", test_data_accuracy*100)
            print ("Correct predictions Train:", sum (y_train == y_pred_train))
            print ("Correct predictions Test:", sum (y_test == y_pred_test))
            print ("Incorrect predictions Train:", sum (y_train != y_pred_train))
            print ("Incorrect predictions Test:", sum (y_test != y_pred_test))
            print ("F1 Score Train:", f1_score(y_train, y_pred_train))
            print ("F1 Score Test:", f1_score(y_test, y_pred_test))
            print('Precision Train: %.3f' % precision_score(y_train, y_pred_train))
            print('Precision Test: %.3f' % precision_score(y_test, y_pred_test))
            print('Recall Train: %.3f' % recall_score(y_train, y_pred_train))
            print('Recall Test: %.3f' % recall_score(y_test, y_pred_test))
            FPR, TPR, threshold = roc_curve(y_train, y_pred_train)
            print('roc_auc_score train: ', roc_auc_score(y_train, y_pred_train))
            FPR, TPR, threshold = roc_curve(y_test, y_pred_test)
            print('roc_auc_score Test: ', roc_auc_score(y_test, y_pred_test))
            
            print ("Confusion Matrix Train:\n", confusion_matrix(y_train , y_pred_train))

            print ("Confusion Matrix Test:\n", confusion_matrix(y_test , y_pred_test))

            print ("Classification Report Train:\n", classification_report (y_train, y_pred_train, digits = 4))

            print ("Classification Report Test:\n", classification_report (y_test, y_pred_test, digits = 4))

        except Exception as e:
            raise CustomException(e,sys)