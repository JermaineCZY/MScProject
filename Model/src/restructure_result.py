import pandas as pd

# read the csv file
df = pd.read_csv('results.csv', header=None)

# delete empty rows
df = df.dropna()

# get the new header
new_header = df.iloc[0]
df = df[1:]
df.columns = new_header

# remove the extra dataset model accuracy line
df = df[df["dataset"] != "dataset"]

# save as a new csv file
df.to_csv('new_results.csv', index=False)
