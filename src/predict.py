import pickle
from preprocess import preprocess_text
import warnings
warnings.filterwarnings('ignore')

def load_model():
    with open('outputs/nb_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('outputs/tfidf_vectorizer.pkl', 'rb') as f:
        tfidf = pickle.load(f)
    print("Model and vectorizer loaded!")
    return model, tfidf

def predict_sentiment(review, model, tfidf):
    clean   = preprocess_text(review)
    vector  = tfidf.transform([clean])
    pred    = model.predict(vector)[0]
    prob    = model.predict_proba(vector)[0]

    sentiment  = "POSITIVE" if pred == 1 else "NEGATIVE"
    confidence = max(prob) * 100

    print(f"\nReview:     {review[:80]}...")
    print(f"Sentiment:  {sentiment}")
    print(f"Confidence: {confidence:.2f}%")
    return sentiment, confidence

if __name__ == "__main__":
    model, tfidf = load_model()

    test_reviews = [
        "This movie was absolutely amazing! Best film I've seen this year.",
        "Terrible movie, complete waste of time. Not worth watching at all.",
        "Not bad but could have been better. Average film overall."
    ]

    print("\n" + "="*50)
    print("SENTIMENT PREDICTION DEMO")
    print("="*50)
    for review in test_reviews:
        predict_sentiment(review, model, tfidf)