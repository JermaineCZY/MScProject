import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from kerastuner.tuners import RandomSearch
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import TensorBoard
from keras.regularizers import l2
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import os


# Adjust parameter weights!!!
def DNN(path, save_result):
    # load data
    df = pd.read_csv(path)

    # divide the dataset
    X = df.drop(['radiant_win', 'match_id', 'start_time'], axis=1)
    y = df['radiant_win']

    # divide training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # define model
    model = Sequential()
    model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dropout(0.45))  # Dropout
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.45))  # Dropout
    model.add(Dense(1, activation='sigmoid'))

    # compile model
    model.compile(loss='binary_crossentropy', optimizer='adamax', metrics=['accuracy'])

    # define tensorboard callbacks
    filename = os.path.basename(path)

    # define tensorboard callbacks
    log_dir = "./logs/" + filename
    tensorboard = TensorBoard(log_dir=log_dir)

    # training model
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=64, callbacks=[tensorboard])

    # evaluation model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'DNN: Loss: {loss}, Accuracy: {accuracy}')

    # After training model
    y_pred = model.predict(X_test)
    auc_score = roc_auc_score(y_test, y_pred)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)

    # save the AUC score to CSV
    if save_result:
        dataset_name = os.path.basename(path).split('processed_data_')[1].split('.csv')[0]
        df = pd.DataFrame(
            data={'dataset': [dataset_name], 'model': ['DNN'], 'accuracy': [accuracy], 'AUC': [auc_score]})
        df.to_csv('results.csv', mode='a', index=False)  # append to existing file

    # save model
    model.save('my_model.h5')
    return fpr, tpr, auc_score


def Random_Forest(path, save_result):
    # load data
    df = pd.read_csv(path)

    # splitting the dataset
    X = df.drop(['radiant_win', 'match_id', 'start_time'], axis=1)
    y = df['radiant_win']

    # partitioning the training set and the test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # define
    model = RandomForestClassifier(n_estimators=100, max_features=30, max_depth=None,
                                   min_samples_leaf=1, random_state=42)

    # training
    model.fit(X_train, y_train)

    # predict
    y_pred = model.predict(X_test)

    # evaluation
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Random_Forest: Accuracy: {accuracy}')

    # save results to CSV
    if save_result:
        dataset_name = os.path.basename(path).split('processed_data_')[1].split('.csv')[0]
        df = pd.DataFrame(
            data={'dataset': [dataset_name], 'model': ['Random_Forest'], 'accuracy': [accuracy], 'AUC': "None"})
        df.to_csv('results.csv', mode='a', index=False)  # append to existing file


def Logistic_Regression(path, save_result):
    # load data
    df = pd.read_csv(path)

    # splitting the dataset
    X = df.drop(['radiant_win', 'match_id', 'start_time'], axis=1)
    y = df['radiant_win']

    # partitioning the training set and the test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # define
    model = LogisticRegression(max_iter=10000, solver='saga', random_state=42)

    # training
    model.fit(X_train, y_train)

    # predict
    y_pred = model.predict(X_test)

    # evaluation
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Logistic Regression: Accuracy: {accuracy}')

    # save results to CSV
    if save_result:
        dataset_name = os.path.basename(path).split('processed_data_')[1].split('.csv')[0]
        df = pd.DataFrame(
            data={'dataset': [dataset_name], 'model': ['Logistic_Regression'], 'accuracy': [accuracy], 'AUC': "None"})
        df.to_csv('results.csv', mode='a', index=False)  # append to existing file


def XGBoost(path, save_result):
    # load data
    df = pd.read_csv(path)

    # splitting the dataset
    X = df.drop(['radiant_win', 'match_id', 'start_time'], axis=1)
    y = df['radiant_win']

    # partitioning the training set and the test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # define model
    model = XGBClassifier(random_state=42)

    # training
    model.fit(X_train, y_train)

    # predict
    y_pred = model.predict(X_test)

    # evaluation
    accuracy = accuracy_score(y_test, y_pred)
    print(f'xgboost: Accuracy: {accuracy}')

    # save results to CSV
    if save_result:
        dataset_name = os.path.basename(path).split('processed_data_')[1].split('.csv')[0]
        df = pd.DataFrame(
            data={'dataset': [dataset_name], 'model': ['XGBoost'], 'accuracy': [accuracy], 'AUC': "None"})
        df.to_csv('results.csv', mode='a', index=False)  # append to existing file
