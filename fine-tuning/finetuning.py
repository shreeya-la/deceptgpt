import pandas as pd
import numpy as np
import evaluate
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification
from transformers import DataCollatorWithPadding
from transformers import Trainer, TrainingArguments
from transformers import EarlyStoppingCallback
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load synthetic dataset
df = pd.read_csv('synthetic-dataset.csv', na_values=[''], keep_default_na=False)

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
id2label = {v: k for k, v in label2id.items()}

df["category"] = df["category"].map(label2id)

# Train-test split
train_df, temp_df = train_test_split(df, test_size=0.4, stratify=df['category'], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['category'], random_state=42)

# Convert to Hugging Face datasets
dataset = DatasetDict({
    "train": Dataset.from_pandas(train_df),
    "validation": Dataset.from_pandas(val_df),
    "test": Dataset.from_pandas(test_df)
})

# 7 Deception Categories + "None"
num_labels = 8
model_name = "gpt2"

# Load tokenizer and resize to match padding token
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Load GPT-2 for classification
model = GPT2ForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
model.config.pad_token_id = tokenizer.pad_token_id

# Max sentence is 169 char < 128 tokens ~ 170 char
# Tokenization function
def tokenize_function(example):
    return tokenizer(
        example["sentence"],
        truncation=True,
        padding='max_length',
        max_length=128
    )

# Apply tokenizer to dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Rename label column for Hugging Face
tokenized_datasets = tokenized_datasets.rename_column("category", "labels")
tokenized_datasets.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

training_args = TrainingArguments(
    output_dir="./deceptgpt-model",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=4,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    push_to_hub=False,
)

accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return accuracy.compute(predictions=predictions, references=labels)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    compute_metrics=compute_metrics,
)

trainer.train()

trainer.evaluate(tokenized_datasets["test"])

preds = trainer.predict(tokenized_datasets["test"])

logits = preds.predictions
labels = preds.label_ids
y_pred = np.argmax(logits, axis=-1)

print("Accuracy:", accuracy_score(labels, y_pred))
print("Macro F1:", f1_score(labels, y_pred, average="macro"))
print("Classification Report:\n", classification_report(labels, y_pred, target_names=label2id.keys()))



# View Binary Accuracy (No Deception vs Any Deception Category)
binary_labels = (labels != 0).astype(int)
binary_preds = (y_pred != 0).astype(int)

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

# Saves model + tokenizer
trainer.save_model("./deceptgpt-model")
tokenizer.save_pretrained("./deceptgpt-model")