import os
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt

DATA_PATH = 'data/Mall_Customers.csv'
MODEL_PATH = 'model/model.pkl'

def main():
    df = pd.read_csv(DATA_PATH)
    X = df[['Annual Income (k$)', 'Spending Score (1-100)']].copy()

    pipeline = joblib.load(MODEL_PATH)
    scaler = pipeline.named_steps['scaler']
    kmeans = pipeline.named_steps['kmeans']

    labels = pipeline.predict(X)
    sil = silhouette_score(X, labels)
    dbi = davies_bouldin_score(X, labels)
    print(f"Silhouette Score: {sil:.3f}")
    print(f"Davies-Bouldin Index (lower is better): {dbi:.3f}")
    print(f"n_clusters: {kmeans.n_clusters}")

    # Plot clusters in original scale
    X_vals = X.values
    centers_orig = scaler.inverse_transform(kmeans.cluster_centers_)

    plt.figure()
    plt.scatter(X_vals[:, 0], X_vals[:, 1], c=labels, s=35)
    plt.scatter(centers_orig[:, 0], centers_orig[:, 1], marker='X', s=200)
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    plt.title('Customer Clusters (Income vs Spending)')
    plt.tight_layout()
    plt.savefig('cluster_plot.png', dpi=160)
    print("Saved plot: cluster_plot.png")

if __name__ == "__main__":
    main()
