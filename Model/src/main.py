from model import DNN, Random_Forest, SVM, Logistic_Regression, XGBoost
import config

data_paths = [config.DATA_PATH_20MIN, config.DATA_PATH_25MIN, config.DATA_PATH_30MIN, config.DATA_PATH]

if __name__ == '__main__':
    for path in data_paths:
        print(f'Training on data: {path}')
        DNN(path)
        # DNN2() #test
        Random_Forest(path)
        '''SVM()'''
        Logistic_Regression(path)
        XGBoost(path)
