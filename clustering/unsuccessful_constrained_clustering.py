import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Extract sentences and convert to list
df = pd.read_csv('full-dataset.csv', na_values=[''], keep_default_na=False)
sentences = (df['sentence']).tolist()

# Record number of total sentences and number of labeled sentences
num_sentences = len(sentences)
num_labeled = df['category'].count()
print(num_sentences, num_labeled)

# Load model
model = SentenceTransformer('all-mpnet-base-v2')

# Separate the labeled & unlabeled sentences
labeled_df = df[df['category'].notna()].copy()
unlabeled_df = df[df['category'].isna()].copy()

label_to_embeddings = {}

# Compute centroid embedding for each label
for label in labeled_df['category'].unique():
    label_sentences = labeled_df[labeled_df['category'] == label]['sentence'].tolist()

    embeddings = model.encode(label_sentences, batch_size=64, show_progress_bar=True)

    if len(embeddings.shape) == 1:
        embeddings = embeddings[np.newaxis, :]

    centroid = np.mean(embeddings, axis=0)
    label_to_embeddings[label] = centroid

# Encode unlabeled sentences
unlabeled_sentences = unlabeled_df['sentence'].tolist()
unlabeled_embeddings = model.encode(unlabeled_sentences, batch_size=64, show_progress_bar=True)

# Assign each unlabeled sentence to the closest centroid
labels = list(label_to_embeddings.keys())

centroids = np.stack([label_to_embeddings[l] for l in labels])  # ← this recreates centroids

# Cosine similarity
threshold = 0.6
sims = cosine_similarity(unlabeled_embeddings, centroids)
max_sims = np.max(sims, axis=1)
assigned_labels = [labels[i] if max_sims[i] > threshold else "None" for i in np.argmax(sims, axis=1)]

# Update the original DataFrame
df.loc[unlabeled_df.index, 'predicted_label'] = assigned_labels