import pandas as pd
import numpy as np
import re


def cleanComment(comment):
    return re.sub(r'[^a-zA-Z0-9\s]', '', comment)

def cleanData(path):
    df = pd.read_csv(path)
    df = df.drop(columns=['Name', 'Review ID'])
    # Change date to day number format
    df['Date'] = pd.to_datetime(df['Date'], format="%Y-%m-%d %H:%M:%S")
    df['Day Number'] = (df['Date'] - df['Date'].min()).dt.days + 1
    # Clean emojis from comments
    df['Comment'] = df['Comment'].apply(cleanComment)
    return df


def main():
    return

if __name__=='__main__':
    main()