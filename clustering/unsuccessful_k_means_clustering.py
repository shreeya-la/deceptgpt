import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from collections import Counter

# Extract sentences and convert to list
df = pd.read_csv('full-dataset.csv', na_values=[''], keep_default_na=False)
sentences = (df['sentence']).tolist()

# Record number of total sentences and number of labeled sentences
num_sentences = len(sentences)
num_labeled = df['category'].count()
print(num_sentences, num_labeled)

# Set up model
model = SentenceTransformer('all-mpnet-base-v2')

# Generate embeddings from the sentences
embeddings = model.encode(sentences, batch_size=64, show_progress_bar=True, device='cuda')

# Perform clustering (7 deception categories + "None" = 8 Clusters)
k = 8
kmeans = KMeans(n_clusters=k, random_state=42)
clusters = kmeans.fit_predict(embeddings)

# Identify the labeled sentences
true_labels = df['category'][:num_labeled]
labeled_clusters = clusters[:num_labeled]

# Assign cluster labels based on the known labels
cluster_to_label = {}
for cluster_id in range(k):
    labels = [true_labels.iloc[i] for i in range(num_labeled) if labeled_clusters[i] == cluster_id]
    if labels:
        cluster_to_label[cluster_id] = Counter(labels).most_common(1)[0][0]

# Apply to unlabeled sentences
predicted_labels = [
    cluster_to_label.get(int(c), "Test") for c in clusters[num_labeled:]
]

# Assign labels in the DataFrame
df.loc[num_labeled:, 'category'] = predicted_labels