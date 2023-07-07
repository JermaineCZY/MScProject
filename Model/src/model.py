import config
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


def DNN():
    # load data
    df = pd.read_csv(config.TRAIN_DATA_PATH)

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


def DNN2():
    # 加载数据
    df = pd.read_csv(config.TRAIN_DATA_PATH)

    # 划分数据集
    X = df.drop(['radiant_win', 'match_id'], axis=1)
    y = df['radiant_win']

    # 特征缩放
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 将类别标签转换为one-hot encoding（如果需要）
    # y = to_categorical(y)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    def build_model(hp):
        model = Sequential()
        model.add(Dense(units=hp.Int('units_input',
                                     min_value=32,
                                     max_value=512,
                                     step=32),
                        input_dim=X_train.shape[1],
                        activation='relu'))
        model.add(Dense(units=hp.Int('units_hidden',
                                     min_value=32,
                                     max_value=512,
                                     step=32),
                        activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy',
                      optimizer=hp.Choice('optimizer', ['adam', 'sgd', 'rmsprop']),
                      metrics=['accuracy'])

        return model

    tuner = RandomSearch(
        build_model,
        objective='val_accuracy',
        max_trials=5,
        executions_per_trial=3,
        directory='project',
        project_name='Dota 2 Win Prediction')

    tuner.search_space_summary()

    tuner.search(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

    tuner.results_summary()

    best_model = tuner.get_best_models()[0]

    # 评估模型
    loss, accuracy = best_model.evaluate(X_test, y_test)
    print(f'Loss: {loss}, Accuracy: {accuracy}')

    # 交叉验证
    kfold = KFold(n_splits=10, shuffle=True, random_state=42)
    results = cross_val_score(best_model, X, y, cv=kfold)
    print(f'Cross-validation accuracy: {results.mean()}')


def Random_Forest():
    # load data
    df = pd.read_csv(config.TRAIN_DATA_PATH)

    # splitting the dataset
    X = df.drop(['radiant_win', 'match_id'], axis=1)
    y = df['radiant_win']

    # partitioning the training set and the test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # define
    model = RandomForestClassifier(n_estimators=100, random_state=42)

    # training
    model.fit(X_train, y_train)

    # predict
    y_pred = model.predict(X_test)

    # evaluation
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Random_Forest: Accuracy: {accuracy}')


def SVM():
    # load data
    df = pd.read_csv(config.TRAIN_DATA_PATH)

    # splitting the dataset
    X = df.drop(['radiant_win', 'match_id'], axis=1)
    y = df['radiant_win']

    # partitioning the training set and the test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # define
    model = svm.SVC(kernel='linear', C=1.0, random_state=42)

    # training
    model.fit(X_train, y_train)

    # predict
    y_pred = model.predict(X_test)

    # evaluation
    accuracy = accuracy_score(y_test, y_pred)
    print(f'SVM: Accuracy: {accuracy}')


def Logistic_Regression():
    # load data
    df = pd.read_csv(config.TRAIN_DATA_PATH)

    # splitting the dataset
    X = df.drop(['radiant_win', 'match_id'], axis=1)
    y = df['radiant_win']

    # partitioning the training set and the test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # define
    model = LogisticRegression(random_state=42)

    # training
    model.fit(X_train, y_train)

    # predict
    y_pred = model.predict(X_test)

    # evaluation
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Logistic Regression: Accuracy: {accuracy}')


def XGBoost():
    # load data
    df = pd.read_csv(config.TRAIN_DATA_PATH)

    # splitting the dataset
    X = df.drop(['radiant_win', 'match_id'], axis=1)
    y = df['radiant_win']

    # partitioning the training set and the test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # define
    model = XGBClassifier(random_state=42)

    # training
    model.fit(X_train, y_train)

    # predict
    y_pred = model.predict(X_test)

    # evaluation
    accuracy = accuracy_score(y_test, y_pred)
    print(f'xgboost: Accuracy: {accuracy}')


if __name__ == '__main__':
    # DNN()
    DNN2()
    # Random_Forest()
    # SVM()
    # Logistic_Regression()
    # XGBoost()
