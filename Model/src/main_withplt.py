import os

from model import DNN, Random_Forest, SVM, Logistic_Regression, XGBoost
import config
from keras.models import load_model
import matplotlib.pyplot as plt

data_paths = [config.DATA_PATH_WITHOUT_ITEMS, config.DATA_PATH_5MIN, config.DATA_PATH_10MIN, config.DATA_PATH_15MIN, config.DATA_PATH_20MIN,
              config.DATA_PATH_25MIN, config.DATA_PATH_30MIN, config.DATA_PATH_35MIN, config.DATA_PATH_FULL]
save_result = 1

if __name__ == '__main__':
    plt.figure(figsize=(10, 8))
    for path in data_paths:
        print(f'Training on data: {path}')
        fpr, tpr, auc_score = DNN(path, save_result)
        # DNN2() #test
        '''Random_Forest(path, save_result)
        Logistic_Regression(path, save_result)
        XGBoost(path, save_result)'''
        plt.plot(fpr, tpr, label=f'{os.path.basename(path)} (AUC = {auc_score:.2f})')

    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='best')
    plt.savefig('combined_roc_curve.png')
    plt.show()
