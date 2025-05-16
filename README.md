# Insurance Company Classifier – Veridion Task

This project addresses Task 2 from the [Veridion Engineering Challenge](https://veridion.com/engineering-challenges/): building a classifier that assigns one or more insurance-related labels to a given company description using the provided taxonomy.

---

## Problem Statement

Given:

- A dataset (`ml_insurance_challenge.csv`) containing company descriptions, tags, sector, category, and niche.
- A taxonomy file (`insurance_taxonomy.csv`) containing insurance-related classification labels.

Goal:

- Automatically assign one or more relevant labels to each company using zero-shot classification.

---

## Solution Overview

### 1. **Data Preprocessing**

- Combined the fields (`description`, `tags`, `sector`, `category`, `niche`) into a single `full_text` input for classification.
- Removed rows with empty or null `full_text`.
- Deduplicated rows from both datasets.
- Truncate long texts

### 2. **Model**

- Used a zero-shot classification pipeline from Hugging Face:
  - Model: `MoritzLaurer/deberta-v3-base-zeroshot-v1`
  - Enabled `multi_label=True` to allow multiple labels per company.
  - Fallback: if no label scores above threshold (0.5), select the top-scoring label instead.

### 3. **Execution**

- Inference is performed in a `for` loop using `tqdm` to track progress.
- Processing is done on GPU if available (`device=0`), otherwise CPU.
- Runtime is measured and reported at the end.

### 4. **Output**

- Labels are added to a new column called `insurance_label`.
- Results are saved in a file named `data_labeled_output.csv`.

---

## Challenges Encountered

### Slow performance on GPU

- Zero-shot classification is resource-intensive.
- Processing on GPU can take several hours (900+ minutes for full dataset).

### Crashes with `datasets.map()`

- PyArrow errors when handling null values or empty lists in batch mapping.
- Switching to a `for` loop solved stability issues completely.

### Some companies received no labels

- Default threshold of 0.5 filtered out all results in some cases.
- Solution: fallback to top label (`labels[0]`) when no score exceeds threshold.

---

## Final Results

- All companies were successfully classified.
- No runtime crashes or data loss due to safe fallback mechanisms.
- Output is consistent  `data_labeled_output.csv`.

---

## Future Improvements

1. **Label Clustering**:
   - Group similar labels in taxonomy to reduce overlap and increase clarity.

2. **Threshold Tuning per Label**:
   - Instead of global threshold (0.5), apply dynamic thresholds based on label frequency or semantic similarity.

3. **Model Upgrade**:
   - Use a fine-tuned classification model trained specifically on insurance company data for better precision.

4. **Parallel Inference**:
   - Add multiprocessing or batch GPU inference to accelerate runtime.

---

## Lessons Learned

- Zero-shot classification is extremely powerful for tasks without labeled training data.
- Batch pipelines are fast but can be fragile – simple `for` loops are more stable.
- Performance monitoring and thresholds are key for trustable multi-label results.

---

## File Structure

| File                      | Description                                       |
|---------------------------|---------------------------------------------------|
| `ml_insurance_challenge.csv` | Input data with company metadata                |
| `insurance_taxonomy.csv`     | List of all possible classification labels      |
| `sort_companies.py`          | Main Python script for classification pipeline  |
| `data_labeled_output.csv`    | Final labeled output with predicted labels      |

---

## Requirements

- Python 3.8+
- `transformers`, `tqdm`, `pandas`
- Optional: PyTorch with CUDA for GPU acceleration

---

## How to Run

```bash
python -m venv venv
pip install transformers pandas tqdm
python sort_companies.py
