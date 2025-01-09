import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
from tabulate import tabulate


def main():
    sns.set_theme()
    # Graphics in SVG format are more sharp and legible
    rcParams['figure.dpi'] = 100
    plt.rcParams['svg.fonttype'] = 'none'
    df = pd.read_csv('data/processedReviews.csv')

    # Comparing Sentiment Analysis to Ratings, bar plot
    _, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5), sharey=True)
    sns.countplot(x="Sentiment Class", data=df, ax=axes[0], order=["Very Negative", "Negative", "Neutral", "Positive", "Very Positive"])
    sns.countplot(x="Rating", data=df, ax=axes[1])
    plt.savefig("docs/sentimentVsRating.svg", format="svg")
    plt.show()

    # Comparing Sentiment to app version, contingency table
    crosstab = pd.crosstab(df["Sentiment Class"], df["App Version"])
    # Filter columns (App Versions) where total reviews are greater than 100
    filtered_crosstab = crosstab.loc[:, crosstab.sum() > 100]
    print(tabulate(filtered_crosstab.T, headers = 'keys', tablefmt = 'pretty'))

    # Mean number of thumbs up for each sentiment class of reviews
    meanThumbs = df.groupby(['Sentiment Class'])["Thumbs Up"].agg(["mean"]).sort_values(by="mean", ascending=False)
    print(tabulate(meanThumbs, headers = 'keys', tablefmt = 'pretty'))

    # Time-series plot of very neg + neg percentage, very pos + pos percentage over day intervals
    negativeReviews = df[df["Sentiment Class"].isin(["Negative", "Very Negative"])]
    positiveReviews = df[df["Sentiment Class"].isin(["Positive", "Very Positive"])]
    dailyNegativeReviews = negativeReviews.groupby("Day Number").size()
    dailyPositiveReviews = positiveReviews.groupby("Day Number").size()
    dailyTotalReviews = df.groupby("Day Number").size()
    negativePercentage = (dailyNegativeReviews / dailyTotalReviews) * 100
    positivePercentage = (dailyPositiveReviews / dailyTotalReviews) * 100
    # Filling missing months with 0% if no reviews of that sentiment
    negativePercentage = negativePercentage.reindex(dailyTotalReviews.index, fill_value=0)
    positivePercentage = positivePercentage.reindex(dailyTotalReviews.index, fill_value=0)
    plt.figure(figsize=(10, 6))
    plt.plot([i for i in range(1, 242)], negativePercentage.to_numpy(), color='red', label='Negative + Very Negative (%)')
    plt.plot([i for i in range(1, 242)], positivePercentage.to_numpy(), color='green', label='Positive + Very Positive (%)')
    plt.title('Review percentage trends over time')
    plt.xlabel('Days from 2023-11-21 to 2024-07-18')
    plt.ylabel('Percentage of Reviews (%)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig("docs/timeSeriesSentiment.svg", format="svg")
    plt.show()

    return

if __name__=='__main__':
    main()
