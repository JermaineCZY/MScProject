import pandas as pd

def save_data(data, path):
    df = pd.DataFrame(data)
    df.to_csv(path, index=False)
