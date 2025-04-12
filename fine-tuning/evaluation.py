import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import Dataset
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification
from transformers import Trainer
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

model = GPT2ForSequenceClassification.from_pretrained("./deceptgpt-model")
tokenizer = GPT2Tokenizer.from_pretrained("./deceptgpt-model")

label2id = {
    "None": 0,
    "Fear Appeal": 1,
    "Fine Print Disclaimers": 2,
    "Glittering Generalities": 3,
    "Loaded Language": 4,
    "Obfuscation": 5,
    "Plain Folks": 6,
    "Weasel Words": 7,
}

# Load labeled dataset
df = pd.read_csv("labeled-subset.csv", na_values=[''], keep_default_na=False)
df["category"] = df["category"].map(label2id)

# Convert to Hugging Face dataset
opp115 = Dataset.from_pandas(df)

def tokenize(example):
    tokens = tokenizer(
        example["sentence"],
        truncation=True,
        padding="max_length",
        max_length=128,
    )
    tokens["token_type_ids"] = [0] * len(tokens["input_ids"])
    return tokens

tokenized_new = opp115.map(tokenize, batched=True)
tokenized_new = tokenized_new.rename_column("category", "labels")
tokenized_new.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

trainer = Trainer(model=model, tokenizer=tokenizer)

predictions = trainer.predict(tokenized_new)

logits = predictions.predictions
labels = predictions.label_ids
y_pred = np.argmax(logits, axis=-1)

print("Accuracy:", accuracy_score(labels, y_pred))
print("Macro F1:", f1_score(labels, y_pred, average="macro"))
print("Classification Report:\n", classification_report(labels, y_pred, target_names=label2id.keys()))

binary_labels = (labels != 0).astype(int)
binary_preds = (y_pred != 0).astype(int)

# View Binary Accuracy (No Deception vs Any Deception Category)
print("Binary Accuracy (None vs Deceptive):", accuracy_score(binary_labels, binary_preds))
print("Binary F1:", f1_score(binary_labels, binary_preds, average="binary"))
print("Binary Report:\n", classification_report(binary_labels, binary_preds, target_names=["None", "Deceptive"]))

labels_list = list(label2id.keys())
cm = confusion_matrix(labels, y_pred, labels=list(range(8)))

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels_list, yticklabels=labels_list)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()
