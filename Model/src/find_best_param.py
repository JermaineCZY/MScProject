import pandas as pd
from keras import Sequential, optimizers
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from matplotlib import pyplot as plt
from sklearn.model_selection import RandomizedSearchCV, train_test_split
import optuna


def random_model(X_train, hidden_units1=128, hidden_units2=64, dropout_rate=0.5):
    model = Sequential()
    model.add(Dense(hidden_units1, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(hidden_units2, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adamax', metrics=['accuracy'])
    return model


def use_random():
    # load data
    df = pd.read_csv('../../Data/src/processed_data_25_min.csv')

    # divide the dataset
    X = df.drop(['radiant_win', 'match_id', 'start_time'], axis=1)
    y = df['radiant_win']

    # divide training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Keras model
    model = KerasClassifier(build_fn=random_model(X_train), epochs=20, batch_size=64)

    # define the hyperparameter space to search
    param_dist = {
        'hidden_units1': [64, 128, 256],
        'hidden_units2': [32, 64, 128],
        'dropout_rate': [0.2, 0.5, 0.7]
    }

    # use RandomizedSearchCV
    random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=10, cv=3)
    random_search.fit(X_train, y_train)

    # output the best parameters
    print("best parameters: ", random_search.best_params_)

    # Obtain the best model
    best_model = random_search.best_estimator_.model

    # evaluate the best model using the test set
    loss, accuracy = best_model.evaluate(X_test, y_test)
    print("the loss on the test set:", loss)
    print("the accuracy on the test set:", accuracy)


def create_model(X_train, hidden_units1, hidden_units2, dropout_rate, optimizer):
    model = Sequential()
    model.add(Dense(hidden_units1, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(hidden_units2, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


def use_optuna():
    # load data and divide as before
    df = pd.read_csv('../../Data/src/processed_data_25_min.csv')
    X = df.drop(['radiant_win', 'match_id', 'start_time'], axis=1)
    y = df['radiant_win']
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)

    def objective(trial):
        hidden_units1 = trial.suggest_categorical('hidden_units1', [16, 32, 64, 128, 256])
        hidden_units2 = trial.suggest_categorical('hidden_units2', [16, 32, 64, 128])
        dropout_rate = trial.suggest_float('dropout_rate', 0.2, 0.7)
        epochs = trial.suggest_int('epochs', 10, 25)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True)
        optimizer = optimizers.Adamax(learning_rate=learning_rate)

        model = create_model(X_train, hidden_units1, hidden_units2, dropout_rate, optimizer)
        model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, verbose=0)
        loss, accuracy = model.evaluate(X_val, y_val, verbose=0)
        return -accuracy

    study = optuna.create_study()
    study.optimize(objective, n_trials=3)

    # output the best parameters
    print("best parameters: ", study.best_params)

    # retrain the model with the best parameters
    best_params = study.best_params
    best_optimizer = optimizers.Adamax(learning_rate=best_params['learning_rate'])
    best_model = create_model(X_train_val, best_params['hidden_units1'], best_params['hidden_units2'],
                              best_params['dropout_rate'], best_optimizer)
    best_model.fit(X_train_val, y_train_val, epochs=best_params['epochs'], batch_size=best_params['batch_size'])

    # evaluate the best model using the test set
    loss, accuracy = best_model.evaluate(X_test, y_test)
    print("the loss on the test set:", loss)
    print("the accuracy on the test set:", accuracy)


if __name__ == '__main__':
    use_optuna()
    # use_random()
