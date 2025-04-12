# DeceptGPT

Online services can use deceptive framing to manipulate users into making decisions that may compromise their privacy. Therefore, we use Natural Language Processing (NLP) techniques to analyze and detect manipulative language in online privacy policies of various websites and applications. We used the OPP-115 Corpus, a set of 115 online privacy policies.

An explanation of all uploaded files is below.

Dataset-Preparation:
- **rename.py**: The original file names have non-consecutive numbers (ranging from 20 to 1713) followed by the respective company name. Therefore, to better organize the policies, we wrote this python script to renumber the policies from 1 to 115. For example, the first policy was renamed from `20_www.theatlantic.com.html` to `1_www.theatlantic.com.html`.
- **extract-sentences.py**: The policies are html documents. Therefore, we wrote this python script to extract the sentences from each privacy policy and clean/process the text. The result is a csv file with three rows: document number, sentence number, and sentence. 

Clustering:
- **unsuccessful_k_means_clustering.py**: This is our approach to cluster the manually labeled sentences using a simple k-means approach. Unfortunately, this approach was unsuccessful because clusters were being made based on similarity and not the labels. Therefore, we moved to an apporach that seeds the clustering with known labels. 
- **unsuccessful_constrained_clustering.py**: This is our approach to cluster the manually labeled sentences using nearest centroid classification. Unfortunately, this approach was unsuccessful because our labeled datset is highly imbalaced (92.39% of sentences are labeled "None"). We proceeded using data augmentation.

Fine-Tuning:
- **finetuning.py**: This is where we fine-tune GPT-2. Because GPT-2 is originally a language model and not a classifier, we use Hugging Face’s GPT2ForSequenceClassification. We then train GPT-2 on our synthetic dataset (60/20/20 split).
- **evaluation.py**: Next, we load our fine-tuned GPT-2 model (i.e., DeceptGPT) and evaluate it on real-world data. We use the manually-labeled set of ~1,000 sentences.
- **comparison.py**: Additionally, we evaluate the vanilla GPT-2 model on the manually-labeled set of ~1,000 sentences. This allows us to compare DeceptGPT's performance to a baseline.
