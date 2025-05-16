import pandas as pd
import ast
import time
from transformers import pipeline
from tqdm.auto import tqdm

#  Force GPU 
device = 0
print("Running on GPU (optimized mode)")


ml_insurance = pd.read_csv('ml_insurance_challenge.csv')
insurance_taxonomy = pd.read_csv('insurance_taxonomy.csv')

# Preprocess text
tqdm.pandas()
def combine_text(row):
    try:
        tags = ', '.join(ast.literal_eval(row['business_tags'])) if pd.notna(row['business_tags']) else ''
    except:
        tags = ''
    text = f"{row['description']} | Tags: {tags} | Sector: {row['sector']} | Category: {row['category']} | Niche: {row['niche']}"
    return text[0:600]  # optimize by truncating long texts

ml_insurance['full_text'] = ml_insurance.progress_apply(combine_text, axis=1)

# Clear data
ml_insurance = ml_insurance[ml_insurance['full_text'].notna()]
ml_insurance = ml_insurance[ml_insurance['full_text'].str.strip() != '']
ml_insurance = ml_insurance.reset_index(drop=True)
ml_insurance.drop_duplicates()
insurance_taxonomy.drop_duplicates()

#  Load classifier (small model for GPU)
classifier = pipeline(
    "zero-shot-classification",
    model="MoritzLaurer/deberta-v3-base-zeroshot-v1",
    device=device
)

# Candidate labels
labels = insurance_taxonomy['label'].dropna().unique().tolist()[0:160]

# Stable for loop inference
start_time = time.time()
predicted_labels = []

for text in tqdm(ml_insurance['full_text'], desc="Classifying companies"):
    try:
        output = classifier(text, candidate_labels=labels, multi_label=True)
        preds = [label for label, score in zip(output["labels"], output["scores"]) if score >= 0.5]
        predicted_labels.append(preds if preds else [output["labels"][0]])
    except Exception as e:
        print(f"Error: {e}")
        predicted_labels.append(["No label"])

ml_insurance['insurance_label'] = predicted_labels

end_time = time.time()
minutes = (end_time - start_time) // 60
seconds = (end_time - start_time) % 60
print(f"Classification completed in {int(minutes)} min {int(seconds)} sec.")

# Save results
ml_insurance.drop(columns='full_text', inplace=True)
ml_insurance.to_csv("data_labeled_output.csv", index=False)
print("Results saved as data_labeled_output.csv")
