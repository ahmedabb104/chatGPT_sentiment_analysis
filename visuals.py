from dataProcessing import cleanData
import matplotlib.pyplot as plt

df = cleanData('data/GPT_reviews.csv')
print(df.head())

# def corrMartrix(df):
    # Drop 
