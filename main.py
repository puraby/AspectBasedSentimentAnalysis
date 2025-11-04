import os
import re
import pandas as pd
from transformers import pipeline, DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import TextClassificationPipeline

# Load IMDB dataset
def load_imdb_dataset(base_dir):
    data = {'text': [], 'label': []}
    for label in ['pos', 'neg']:
        dir_path = os.path.join(base_dir, label)
        for file_name in os.listdir(dir_path):
            if file_name.endswith('.txt'):
                with open(os.path.join(dir_path, file_name), 'r', encoding='utf-8') as f:
                    data['text'].append(f.read())
                    data['label'].append(1 if label == 'pos' else 0)
    return pd.DataFrame(data)

train_df = load_imdb_dataset('../aclImdb/train')
test_df = load_imdb_dataset('../aclImdb/test')
full_df = pd.concat([train_df, test_df]).reset_index(drop=True)

# Define aspect keywords
aspect_keywords = {
    'Cast': ['cast', 'actor', 'actress', 'performance'],
    'Cinematography': ['cinematography', 'visual', 'shot', 'scene'],
    'Music': ['music', 'soundtrack', 'score'],
    'Editing': ['editing', 'cut'],
    'Direction': ['direction', 'director']
}

# Extract aspect sentences
def extract_aspect_sentences(review, keywords):
    sentences = re.split(r'(?<=[.!?]) +', review)
    aspect_sentences = [sentence for sentence in sentences if any(keyword in sentence.lower() for keyword in keywords)]
    return ' '.join(aspect_sentences) if aspect_sentences else None

for aspect, keywords in aspect_keywords.items():
    full_df[aspect] = full_df['text'].apply(lambda review: extract_aspect_sentences(review, keywords))

# Initialize sentiment analysis pipeline
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertForSequenceClassification.from_pretrained(model_name)
sentiment_analyzer = TextClassificationPipeline(model=model, tokenizer=tokenizer)

# Truncate text to a maximum of 512 tokens
def truncate_text(text, tokenizer, max_length=512):
    encoded_input = tokenizer(text, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
    return tokenizer.decode(encoded_input['input_ids'][0], skip_special_tokens=True)

# Apply truncation to reviews and aspect sentences
full_df['text'] = full_df['text'].apply(lambda x: truncate_text(x, tokenizer))
for aspect in aspect_keywords.keys():
    full_df[aspect] = full_df[aspect].apply(lambda x: truncate_text(x, tokenizer) if x else x)

# Label sentiment
def label_sentiment(aspect_text):
    if aspect_text:
        truncated_text = truncate_text(aspect_text, tokenizer)
        results = sentiment_analyzer(truncated_text)
        sentiments = [result['label'] for result in results]
        positive_count = sentiments.count('POSITIVE')
        negative_count = sentiments.count('NEGATIVE')
        return 1 if positive_count > negative_count else 0
    else:
        return -1  # Not defined

for aspect in aspect_keywords.keys():
    full_df[aspect] = full_df[aspect].apply(label_sentiment)

# Save the DataFrame to a CSV file
full_df.to_csv('aspect_labeled_imdb_data.csv', index=False)

print("Aspect-based sentiment analysis completed and saved to 'aspect_labeled_imdb_data.csv'.")

# Evaluate the model
train_data = full_df.sample(frac=0.8, random_state=42)
test_data = full_df.drop(train_data.index)

def evaluate_aspect_sentiment(data, aspect):
    correct = data[data[aspect] != -1]
    accuracy = (correct[aspect] == correct['label']).mean()
    return accuracy

for aspect in aspect_keywords.keys():
    accuracy = evaluate_aspect_sentiment(test_data, aspect)
    print(f'Accuracy for {aspect}: {accuracy:.2f}')

