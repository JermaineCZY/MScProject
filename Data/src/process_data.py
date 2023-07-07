import pandas as pd
import numpy as np

def select_heroes(csv_path):
    # load raw data
    df = pd.read_csv(csv_path)

    # select the columns
    heroes_columns = [f"radiant_hero_{i + 1}" for i in range(5)] + [f"dire_hero_{i + 1}" for i in range(5)]
    selected_columns = ["match_id"] + heroes_columns + ["radiant_win"]
    selected_df = df[selected_columns]

    # write selected columns to a new csv file
    selected_df.to_csv('selected_match_details.csv', index=False)


def one_hot_encoding(csv_path):

    df = pd.read_csv(csv_path)

    # hero list
    heroes_columns_radiant = [f"radiant_hero_{i + 1}" for i in range(5)]
    heroes_columns_dire = [f"dire_hero_{i + 1}" for i in range(5)]

    # one hot encoding radiant heroes
    one_hot_radiant = pd.get_dummies(df[heroes_columns_radiant].apply(lambda x: tuple(x), axis=1).explode(),
                                     prefix='radiant_hero').groupby(level=0).sum()

    # one hot encoding dire heroes
    one_hot_dire = pd.get_dummies(df[heroes_columns_dire].apply(lambda x: tuple(x), axis=1).explode(),
                                  prefix='dire_hero').groupby(level=0).sum()

    # add one hot encoded columns to df
    df = pd.concat([df[['match_id', 'radiant_win']], one_hot_radiant, one_hot_dire], axis=1)

    df.to_csv('encoded_match_details.csv', index=False)


if __name__ == '__main__':
    # select_heroes('match_details.csv')
    one_hot_encoding('selected_match_details.csv')
