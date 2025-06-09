# Data pre-processing
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('punkt')

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub(r"http\S+|@\w+|#\w+", "", str(text))  # remove URLs, mentions, hashtags
    text = re.sub(r"[^a-zA-Z\s]", "", text)             # remove non-alphabetic characters
    text = text.lower()
    return " ".join(w for w in word_tokenize(text) if w not in stop_words)

def preprocess_dataset(input_path, output_path):
    df = pd.read_csv(input_path)
    df['clean_text'] = df['text'].apply(clean_text)
    df.to_csv(output_path, index=False)
    print(f"Saved cleaned data to {output_path}")