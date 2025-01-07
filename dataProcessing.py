import pandas as pd
import numpy as np
import re


def cleanComment(comment):
    return re.sub(r'[^a-zA-Z0-9\s]', '', comment)

def cleanData(path):
    df = pd.read_csv(path)
    df = df.drop(columns=['Name', 'Review ID'])
    # Change date to monthYear format
    df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%m%Y')
    # Clean emojis from comments
    df['Comment'] = df['Comment'].apply(cleanComment)
    # print(df.head())
    # print(df.shape)  # 100000, 6
    return df


def main():
    data = cleanData('data/GPT_reviews.csv')


if __name__=='__main__':
    main()