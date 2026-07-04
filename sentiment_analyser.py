from transformers import pipeline
from datasets import load_datasets
import pandas as pd
from sklearn.metrics import accuracy_score, clasification_report, confusion_matrix

#Load IMDB dataset
print("Loading Dataset...")
dataset = load_dataset("imdb", split="test[:1000]" )

#Load a pre-trained sentiment analysis model
print("loading...")
classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

#Function to predict sentiment for evaluation
def predict_sentiment(texts, batch_size=32): 
    predictions=[]
    for i in range(0, len(texts),batch_size );
        batch=texts[i:i+batch_size]
        results=classifier(batch, truncation=true, max_length=512)
        predictions.extend(results)
    return predictions

#Get predictions
print("Make predictions on 1000 reviews")
texts = dataset['text']
predictions = predict_sentiment(texts)

#Convert predictions to binary (POSITIVE=1, NEGATIVE=0)
pred_labels = [1 if p['label'] ==  'POSITIVE' else 0 for p in predictions]
true_labels = dataset['label']


#Test it with a simple example
text= "An incredible feat Enthrals anew every time Cannot believe its two decades ago Love it so much"

result= classifier(text)

print(f"Text: {text}")
print(f"result: {result}")

