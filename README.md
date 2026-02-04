# Mental Health Crisis Detector

A machine learning system for detecting suicidal ideation and mental health crises from social media text using natural language processing.

## Overview

This project implements and compares three approaches for crisis detection:
- Baseline: TF-IDF + Logistic Regression (93.7% accuracy)
- Deep Learning: LSTM with word embeddings (93.8% accuracy)
- Transformers: Fine-tuned BERT (results pending)

## Dataset

- Source: Kaggle "Suicide and Depression Detection" dataset
- Size: 231,064 Reddit posts after preprocessing
- Distribution: Balanced (50% suicide / 50% non-suicide)
- Link: https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch

## Preprocessing Pipeline

1. Remove posts with fewer than 5 words
2. Filter spam using unique word ratio for long posts
3. Text cleaning: lowercase, remove URLs, remove special characters
4. Lemmatization using NLTK WordNetLemmatizer
5. Tokenization for neural models

## Models

### 1. TF-IDF + Logistic Regression
- Vocabulary size: 115,458 features
- Test accuracy: 93.68%
- Recall (suicide class): 93%

### 2. LSTM
- Architecture: Embedding (128d) -> LSTM (64 units) -> Dense
- Sequence length: 350 tokens
- Test accuracy: 93.83%
- Recall (suicide class): 95%

### 3. BERT
- Model: bert-base-uncased (109M parameters)
- Max sequence length: 512 tokens
- Training: 3 epochs with early stopping
- Results: [to be updated]

## Requirements
```
pandas
numpy
scikit-learn
nltk
tensorflow
transformers
torch
datasets
```

## Usage

Run the notebook sequentially to:
1. Load and explore the data
2. Preprocess text
3. Train baseline model
4. Train LSTM model
5. Fine-tune BERT model
6. Compare results

## Ethical Considerations

This system is designed for research and educational purposes. In production:
- Include crisis resources (988 Suicide & Crisis Lifeline)
- Prioritize recall over precision (better to flag false positives)
- Human review required for any automated interventions
- Regular model auditing for bias and fairness

## Results Comparison

| Model | Accuracy | Suicide Recall | Non-Suicide Recall | Precision | F1 Score |
|-------|----------|----------------|-------------------|-----------|----------|
| TF-IDF + LR | 93.68% | 93% | 94% | 94% | 0.94 |
| LSTM | 93.83% | 95% | 93% | 93% | 0.94 |
| BERT | 94.62% | 94.92% | 94.32% | 94.06% | 0.9449 |

## License

Educational and research use only.
