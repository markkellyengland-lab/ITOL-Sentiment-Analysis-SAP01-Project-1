from transformers import pipeline

#Load the model
print("loading model...")
classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

 