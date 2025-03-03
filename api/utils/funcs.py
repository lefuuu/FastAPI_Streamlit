import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_checkpoint = 'cointegrated/rubert-tiny-sentiment-balanced'
def load_model():
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint)
    model.eval()
    return model

def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    return tokenizer
model = load_model()
tokenizer = load_tokenizer()

def get_sentiment(text, return_type='label'):
    """ Calculate sentiment of a text. `return_type` can be 'label', 'score' or 'proba' """
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
        proba = torch.sigmoid(model(**inputs).logits).cpu().numpy()[0]
        cls = model.config.id2label[proba.argmax()]
    return cls, proba

# text = 'Какая гадость эта ваша заливная рыба!'
# # classify the text
# print(get_sentiment(text, 'label'))  # negative
# # score the text on the scale from -1 (very negative) to +1 (very positive)
# print(get_sentiment(text, 'score'))  # -0.5894946306943893
# # calculate probabilities of all labels
# print(get_sentiment(text, 'proba'))  # [0.7870447  0.4947824  0.19755007]


