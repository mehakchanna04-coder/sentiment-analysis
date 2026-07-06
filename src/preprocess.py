import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# Negation-safe stop words
stop_words = set(stopwords.words('english'))
negation_words = {'no', 'not', 'nor', 'neither', 'never',
                  'nothing', 'nobody', 'nowhere', 'hardly',
                  'barely', 'scarcely', "don't", "won't",
                  "can't", "isn't", "aren't", "wasn't",
                  "weren't", "hasn't", "haven't", "hadn't"}
stop_words = stop_words - negation_words
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # Step 1: Remove HTML tags
    text = re.sub(r'<.*?>', ' ', text)
    # Step 2: Remove special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Step 3: Lowercase
    text = text.lower()
    # Step 4: Tokenize
    tokens = word_tokenize(text)
    # Step 5: Remove stop words (negation-safe)
    tokens = [t for t in tokens if t not in stop_words]
    # Step 6: Lemmatize
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return ' '.join(tokens)

def preprocess_dataframe(df):
    print("Preprocessing reviews...")
    df['clean_review'] = df['review'].apply(preprocess_text)
    df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})
    print(f"Done! Shape: {df.shape}")
    return df

if __name__ == "__main__":
    print("Preprocess module ready!")
    print("Functions: preprocess_text, preprocess_dataframe")