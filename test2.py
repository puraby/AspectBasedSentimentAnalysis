import re
import pandas as pd
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
import os
from sklearn.model_selection import train_test_split

# Download and extract the IMDB dataset


# Read the IMDB dataset
def load_imdb_data(data_dir):
    data = {'Review': [], 'Sentiment': []}
    for label in ['pos', 'neg']:
        labeled_dir = f'{data_dir}/{label}'
        for file in os.listdir(labeled_dir):
            if file.endswith('.txt'):
                with open(os.path.join(labeled_dir, file), 'r', encoding='utf-8') as f:
                    review = f.read()
                    data['Review'].append(review)
                    data['Sentiment'].append(1 if label == 'pos' else 0)
    return pd.DataFrame(data)

train_df = load_imdb_data('../aclImdb/train')
test_df = load_imdb_data('../aclImdb/test')

# Add dummy aspect columns for now
aspects = ['Cast', 'Cinematography', 'Music', 'Editing', 'Direction']
for aspect in aspects:
    train_df[aspect] = train_df['Sentiment']
    test_df[aspect] = test_df['Sentiment']

# Preprocess the data
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'<br />', ' ', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    return text

train_df['Review'] = train_df['Review'].apply(preprocess_text)
test_df['Review'] = test_df['Review'].apply(preprocess_text)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_text(texts):
    return tokenizer(
        texts.tolist(),
        max_length=512,
        truncation=True,
        padding=True,
        return_tensors='tf'
    )

train_encodings = tokenize_text(train_df['Review'])
test_encodings = tokenize_text(test_df['Review'])

aspect_columns = aspects
train_labels = train_df[aspect_columns].values
test_labels = test_df[aspect_columns].values

# Define the Model
num_aspects = len(aspects)
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_aspects)
model.trainable = True

model.summary()

optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
model.compile(optimizer=optimizer,
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

def prepare_dataset(encodings, labels):
    dataset = tf.data.Dataset.from_tensor_slices((dict(encodings), labels))
    return dataset.shuffle(1000).batch(8)

train_dataset = prepare_dataset(train_encodings, train_labels)
test_dataset = prepare_dataset(test_encodings, test_labels)

# Train the Model
history = model.fit(train_dataset, validation_data=test_dataset, epochs=3, verbose=2)

# Evaluate the Model
results = model.evaluate(test_dataset)
print(f"Test results - Loss: {results[0]} - Accuracy: {results[1]}")
