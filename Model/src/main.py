from model import DNN, Random_Forest, SVM, Logistic_Regression, XGBoost
import config
from keras.models import load_model

data_paths = [config.DATA_PATH_WITHOUT_ITEMS, config.DATA_PATH_5MIN, config.DATA_PATH_10MIN, config.DATA_PATH_15MIN, config.DATA_PATH_20MIN,
              config.DATA_PATH_25MIN, config.DATA_PATH_30MIN, config.DATA_PATH_35MIN, config.DATA_PATH_FULL]
save_result = 1

if __name__ == '__main__':
    for path in data_paths:
        print(f'Training on data: {path}')
        DNN(path, save_result)
        # DNN2() #test
        Random_Forest(path, save_result)
        '''SVM()'''
        Logistic_Regression(path, save_result)
        XGBoost(path, save_result)
