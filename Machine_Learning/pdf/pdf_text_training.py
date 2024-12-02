import os
import fitz  # PyMuPDF
import pandas as pd
import re
import nltk
import numpy as np
from nltk.corpus import stopwords
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

nltk.download('stopwords')

def extract_text_from_pdf(pdf_path):
    with fitz.open(pdf_path) as doc:
        extracted_text = ""
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            extracted_text += page.get_text()
    return extracted_text

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove numbers
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])  # Remove stop words
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace
    text = re.sub(r'[^\x00-\x7F]+', '', text)  # Remove special characters
    return text

def process_pdfs_in_directory(directory_path):
    all_cleaned_texts = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(directory_path, filename)
            extracted_text = extract_text_from_pdf(pdf_path)
            cleaned_text = clean_text(extracted_text)
            all_cleaned_texts.append(cleaned_text)
    return all_cleaned_texts

# Example usage
directory_path = './pdf/PDFs'  # Replace with the path to your directory containing PDFs
cleaned_texts = process_pdfs_in_directory(directory_path)

# Combine all cleaned texts into one
corpus = ' '.join(cleaned_texts)

# Tokenize the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts([corpus])
total_words = len(tokenizer.word_index) + 1

# Create sequences
input_sequences = []
for line in corpus.split('.'):
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

# Pad sequences
max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre')

# Create predictors and label
input_sequences = np.array(input_sequences)
X, y = input_sequences[:,:-1],input_sequences[:,-1]

# One-hot encode the labels
y = tf.keras.utils.to_categorical(y, num_classes=total_words)

# Define the model
model = Sequential()
model.add(Embedding(total_words, 100, input_length=max_sequence_len-1))
model.add(LSTM(150))
model.add(Dense(total_words, activation='softmax'))

# Compile and train the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X, y, epochs=50, verbose=1)