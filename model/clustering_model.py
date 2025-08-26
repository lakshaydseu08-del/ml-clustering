import os
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib

DATA_PATH = 'data/Mall_Customers.csv'
MODEL_DIR = 'model'
MODEL_PATH = os.path.join(MODEL_DIR, 'model.pkl')

def choose_best_k(X, k_min=2, k_max=10, random_state=42):
    best_k = None
    best_score = -1
    for k in range(k_min, k_max + 1):
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('kmeans', KMeans(n_clusters=k, n_init=10, random_state=random_state))
        ])
        labels = pipe.fit_predict(X)
        score = silhouette_score(X, labels)
        if score > best_score:
            best_score = score
            best_k = k
    return best_k, best_score

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    X = df[['Annual Income (k$)', 'Spending Score (1-100)']].copy()

    best_k, best_score = choose_best_k(X)
    print(f"[INFO] Best k by silhouette = {best_k} (score={best_score:.3f})")

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('kmeans', KMeans(n_clusters=best_k, n_init=10, random_state=42))
    ])
    pipeline.fit(X)

    # Save model
    joblib.dump(pipeline, MODEL_PATH)
    print(f"[INFO] Saved model at: {MODEL_PATH}")

    # Save clustered data for reference
    labels = pipeline.named_steps['kmeans'].labels_
    df_out = df.copy()
    df_out['cluster'] = labels
    df_out.to_csv('data/clustered_customers.csv', index=False)
    print("[INFO] Wrote: data/clustered_customers.csv")

if __name__ == "__main__":
    main()
