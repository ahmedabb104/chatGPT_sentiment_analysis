from dataProcessing import cleanData
from transformers import pipeline
from collections import Counter

def getSentimentAnalysisOutput():
    sentiment_pipeline = pipeline("text-classification", model="tabularisai/multilingual-sentiment-analysis")
    df = cleanData('data/GPT_reviews.csv')
    reviews = [str(i) for i in df['Comment'].values]
    scores = sentiment_pipeline(reviews)
    labels = [output['label'] for output in scores]
    classCounts = Counter(labels)
    classes = ['Very Negative', 'Negative', 'Neutral', 'Positive', 'Very Positive']
    percentages = [
        {
            "label": reviewClass, 
            "percentage": (classCounts[reviewClass] / 100000) * 100
        }
        for reviewClass in classes
    ]
    for item in percentages:
        print(f"{item["label"]}: {item["percentage"]}%")
    return percentages

###### Final Percentages:
######### Very Negative: 6.104%
######### Negative:      5.693%
######### Neutral:       13.331%
######### Positive:      24.171%
######### Very Positive: 50.701%
getSentimentAnalysisOutput()