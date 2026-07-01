from transformers import pipeline

# Load a pre-trained sentiment analysis model
print("loading...")
classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

#Test it with a simple example
text= "An incredible feat Enthrals anew every time Cannot believe its two decades ago Love it so much"

result= classifier(text)

print(f"Text: {text}")
print(f"result: {result}")

