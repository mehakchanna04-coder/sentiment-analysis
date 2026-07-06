import pandas as pd
from sklearn.model_selection import train_test_split
from preprocess import preprocess_dataframe
from train import vectorize, train_model, evaluate, save_model
import warnings
warnings.filterwarnings('ignore')

def run_pipeline(input_path):
    print("="*50)
    print("SENTIMENT ANALYSIS PIPELINE")
    print("="*50)

    # Step 1: Load data
    print("\nStep 1: Loading data...")
    df = pd.read_csv(input_path)
    print(f"Loaded: {df.shape}")

    # Step 2: Preprocess
    print("\nStep 2: Preprocessing...")
    df = preprocess_dataframe(df)

    # Step 3: Split
    print("\nStep 3: Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        df['clean_review'], df['label'],
        test_size=0.2, random_state=42,
        stratify=df['label']
    )
    print(f"Train: {X_train.shape} | Test: {X_test.shape}")

    # Step 4: Vectorize
    print("\nStep 4: TF-IDF Vectorization...")
    X_train_tfidf, X_test_tfidf, tfidf = vectorize(X_train, X_test)

    # Step 5: Train
    print("\nStep 5: Training model...")
    model = train_model(X_train_tfidf, y_train)

    # Step 6: Evaluate
    print("\nStep 6: Evaluating...")
    results = evaluate(model, X_test_tfidf, y_test)

    # Step 7: Save
    print("\nStep 7: Saving model...")
    save_model(model, tfidf)

    # Save results
    pd.DataFrame([results]).to_csv(
        'outputs/sentiment_results.csv', index=False)
    print("\nPipeline complete!")
    return model, tfidf, results

if __name__ == "__main__":
    run_pipeline('data/IMDB Dataset.csv')