import pandas as pd
def load_dataset() :
    df = pd.read_csv('dataset/gigaword/test.csv')
    print(df.head())
    return df