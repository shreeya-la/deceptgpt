import os
import glob
import pandas as pd
from bs4 import BeautifulSoup
import nltk
import re

# Ensure the sentence tokenizer is available
nltk.download('punkt_tab')

# Folder where your HTML files are stored
html_folder = "sanitized_policies"

# Get a list of all HTML files
html_files = sorted(glob.glob(os.path.join(html_folder, "*.html")))

# Store extracted data
data = []

# Function to extract document number from filename
def extract_document_number(filename):
    match = re.match(r'(\d+)_', os.path.basename(filename))
    return int(match.group(1)) if match else None

# Function to clean and process text
def clean_text(text):
    lines = text.split("\n")  # Split text into lines
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        if line:  # Ignore empty lines
            if not re.search(r'[.!?]$', line):
                line += "."
            
            if "P.O." in line:
                line = line.replace("P.O.", "PO")
            
            # Regular expression to remove numbering at the start of a sentence
            pattern = r'^[A-Za-z0-9]+\.\s*'
            # Remove numbering from each sentence in the list
            line = re.sub(pattern, '', line)
            
            cleaned_lines.append(line)
    
    return "\n".join(cleaned_lines)  # Rejoin lines into text

# Process each file
for _, file_path in enumerate(html_files, start=1):
    doc_num = extract_document_number(file_path)
    if doc_num is None:
        continue

    with open(file_path, "r", encoding="utf-8") as file:
        soup = BeautifulSoup(file, "html.parser")

        # Extract visible text from HTML
        text = soup.get_text(separator="\n")

        # Remove `|||` or `||| ` from the text
        text = re.sub(r'\|\|\| ?', '', text)

        # Clean and process text
        text = clean_text(text)
        
        # Split text into sentences
        sentences = nltk.tokenize.sent_tokenize(text)

        # Remove added periods after extracting sentences
        sentences = [re.sub(r'\.$', '', sentence) for sentence in sentences]

        # Further split sentences that contain new lines
        final_sentences = []
        for sentence in sentences:
            split_sentences = sentence.split("\n")  # Split by new lines
            final_sentences.extend([s.strip() for s in split_sentences if s.strip()])
        
        # Remove elements that contain only a single "&"
        final_sentences = [sentence for sentence in final_sentences if not re.match(r'^[&A-Za-z0-9]$', sentence)]

        # Store each sentence with its document and sentence number
        for sent_num, sentence in enumerate(final_sentences, start=1):
            data.append({
                "document_number": doc_num,
                "sentence_number": sent_num,
                "sentence": sentence
            })

# Convert to a DataFrame for easy manipulation
df = pd.DataFrame(data)

# Sort by "document_number" and "sentence_number"
df = df.sort_values(by=["document_number", "sentence_number"]).reset_index(drop=True)

# Save to CSV
df.to_csv("extracted_sentences.csv", index=False)

print("Extraction complete. Data saved to extracted_sentences.csv")
