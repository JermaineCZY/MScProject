import config
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import TensorBoard
from keras.layers import Dropout
from keras.regularizers import l2
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score


def DNN():
    # load data
    df = pd.read_csv(config.DATA_PATH)

    # divide the dataset
    X = df.drop(['radiant_win', 'match_id'], axis=1)
    y = df['radiant_win']

    # divide training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    # define model
    model = Sequential()
    model.add(Dense(128, input_dim=X_train.shape[1], activation='relu', kernel_regularizer=l2(0.01)))  # L2正则化
    model.add(Dropout(0.5))  # Dropout
    model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.01)))  # L2正则化
    model.add(Dropout(0.5))  # Dropout
    model.add(Dense(1, activation='sigmoid'))  # 输出层

    # compile model
    model.compile(loss='binary_crossentropy', optimizer='adamax', metrics=['accuracy'])

    # define tensorboard callbacks
    tensorboard = TensorBoard(log_dir='./logs')

    # training model
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32, callbacks=[tensorboard])

    # evaluation model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'DNN: Loss: {loss}, Accuracy: {accuracy}')

    # save model
    model.save('my_model.h5')


def Random_Forest():
    # 加载数据
    df = pd.read_csv(config.DATA_PATH)

    # 划分数据集
    X = df.drop(['radiant_win', 'match_id'], axis=1)
    y = df['radiant_win']

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 建立模型
    model = RandomForestClassifier(n_estimators=100, random_state=42)

    # 训练模型
    model.fit(X_train, y_train)

    # 预测测试集结果
    y_pred = model.predict(X_test)

    # 计算预测准确率
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Random_Forest: Accuracy: {accuracy}')


def SVM():
    # 加载数据
    df = pd.read_csv(config.DATA_PATH)

    # 划分数据集
    X = df.drop(['radiant_win', 'match_id'], axis=1)
    y = df['radiant_win']

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 建立模型
    model = svm.SVC(kernel='linear', C=1.0, random_state=42)

    # 训练模型
    model.fit(X_train, y_train)

    # 预测测试集结果
    y_pred = model.predict(X_test)

    # 计算预测准确率
    accuracy = accuracy_score(y_test, y_pred)
    print(f'SVM: Accuracy: {accuracy}')


def Logistic_Regression():
    # 加载数据
    df = pd.read_csv(config.DATA_PATH)

    # 划分数据集
    X = df.drop(['radiant_win', 'match_id'], axis=1)
    y = df['radiant_win']

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 创建模型
    model = LogisticRegression(random_state=42)

    # 训练模型
    model.fit(X_train, y_train)

    # 预测
    y_pred = model.predict(X_test)

    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Logistic Regression: Accuracy: {accuracy}')


def XGBoost():
    # 加载数据
    df = pd.read_csv(config.DATA_PATH)

    # 划分数据集
    X = df.drop(['radiant_win', 'match_id'], axis=1)
    y = df['radiant_win']

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 创建模型
    model = XGBClassifier(random_state=42)

    # 训练模型
    model.fit(X_train, y_train)

    # 预测
    y_pred = model.predict(X_test)

    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    print(f'xgboost: Accuracy: {accuracy}')


if __name__ == '__main__':
    DNN()
    Random_Forest()
    SVM()
    Logistic_Regression()
    XGBoost()
