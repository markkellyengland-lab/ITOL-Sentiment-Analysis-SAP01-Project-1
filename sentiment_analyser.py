from transformers import pipeline
from datasets import load_dataset
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#Load IMDB dataset
print("Loading Dataset...")
dataset = load_dataset("stanfordnlp/imdb", split="test", download_mode="force_redownload").shuffle(seed=42).select(range(500)) #shuffles the entire test deck first, and then takes 500 random reviews from the top.

#Load a pre-trained sentiment analysis model
print("loading...")
classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

#Function to predict sentiment for evaluation
def predict_sentiment(texts, batch_size=32): 
    predictions=[]
    for i in range(0, len(texts), batch_size):
        batch=texts[i:i+batch_size]
        results=classifier(batch, truncation=True, max_length=512)
        predictions.extend(results)
    return predictions

#Get predictions
print("Make predictions on 500 reviews")
texts = dataset['text']
predictions = predict_sentiment(texts)

#Convert predictions to binary (POSITIVE=1, NEGATIVE=0)
pred_labels = [1 if p['label'].upper() == 'POSITIVE' or p['label'] == 'LABEL_1'else 0 for p in predictions]
true_labels = dataset['label']


#Evaluate our model
accuracy = accuracy_score(true_labels, pred_labels)
print(f"Accuracy Score:{accuracy:.4f}")
print(classification_report(true_labels, pred_labels, target_names=['Negative','Positive']))

#Show some examples
print ("--- Example Predictions ---")
for i in range(5):
    print(f"Review: {texts[i][:200]}...")
    print(f"True: {'Positive' if true_labels[i] ==1 else 'Negative'}")
    print(f"Predicted: {predictions[i]['label']} (Confidence: {predictions[i]['score']:.3f})")
   
