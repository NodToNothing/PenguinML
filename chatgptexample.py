import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load a pre-trained transformer model and tokenizer from Hugging Face
model_name = "distilbert-base-uncased-finetuned-sst-2-english"  # Sentiment analysis model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Define an input sentence
text = "Hugging Face transformers are amazing!"

# Tokenize the input sentence
inputs = tokenizer(text, return_tensors="pt")

# Run the model and get the outputs
with torch.no_grad():
    outputs = model(**inputs)

# Extract predicted probabilities and labels
probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
pred_label = torch.argmax(probs, dim=1).item()

# Output results
labels = ["Negative", "Positive"]
print(f"Text: {text}\nPredicted Sentiment: {labels[pred_label]} (Confidence: {probs[0][pred_label]:.4f})")
