# src/data/ Data Flow Diagram

## Complete Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         EXTERNAL DATA SOURCES                               │
│  (Twitter API, CSV files, JSON dumps, HuggingFace datasets)                  │
└─────────────────────────────┬───────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           RAW DATA STORAGE                                  │
│                    dataset/raw/ (CSV/JSON/Parquet)                         │
│  Columns: ['text', 'label', 'timestamp', 'user_id', ...]                   │
└─────────────────────────────┬───────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 1: TEXT CLEANING                                                      │
│  ─────────────────────────────────────────────────────────────────────────  │
│  File: cleaner.py                                                           │
│  Class: TweetCleaner                                                        │
│                                                                             │
│  Input:  Raw text string                                                    │
│  Output: Cleaned text string                                                │
│                                                                             │
│  Pipeline:                                                                  │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐     │
│  │ remove_urls │ → │remove_mentns│ → │remove_hashtg│ → │remove_emojis│     │
│  │  https://  │   │   @user     │   │    #AI      │   │     😀       │     │
│  └─────────────┘   └─────────────┘   └─────────────┘   └─────────────┘     │
│         │                │                  │                │                │
│         ▼                ▼                  ▼                ▼                │
│  ┌─────────────┐   ┌─────────────┐   ┌──────────────────────────────────┐  │
│  │remove_special│ → │ lowercase  │ → │    normalize_whitespace()        │  │
│  │   chars!@#  │   │            │   │  "  hello   world  " → "hello world" │  │
│  └─────────────┘   └─────────────┘   └──────────────────────────────────┘  │
└─────────────────────────────┬───────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 2: DATAFRAME PROCESSING                                               │
│  ─────────────────────────────────────────────────────────────────────────  │
│  File: preprocessor.py                                                        │
│  Class: TextPreprocessor                                                      │
│                                                                             │
│  Input:  pd.DataFrame with ['text', 'label'] columns                          │
│  Output: pd.DataFrame with added ['cleaned_text'] column                      │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  TextPreprocessor().preprocess(df)                                  │    │
│  │                                                                     │    │
│  │  Raw:    "Love this! 😍 @user https://t.co/x #amazing"              │    │
│  │           ↓                                                         │    │
│  │  Cleaned: "love this amazing"                                       │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  TextPreprocessor().get_class_weights(labels)                         │    │
│  │                                                                     │    │
│  │  Input:  [0, 0, 0, 1, 1, 2]  (imbalanced classes)                   │    │
│  │  Output: tensor([0.5, 0.75, 1.5])  (for CrossEntropyLoss)           │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────┬───────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PROCESSED DATA STORAGE                              │
│                  dataset/processed/ (cleaned CSV)                          │
│  Columns: ['text', 'label', 'cleaned_text', 'tokenized', ...]              │
└─────────────────────────────┬───────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 3: PYTORCH DATASET                                                    │
│  ─────────────────────────────────────────────────────────────────────────  │
│  File: dataset.py                                                             │
│  Class: SentimentDataset                                                      │
│                                                                             │
│  Input:  DataFrame or CSV path + Tokenizer + max_length                      │
│  Output: Dict of tensors for BERT                                            │
│                                                                             │
│  SentimentDataset("train.csv", tokenizer, max_length=128)                   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  __getitem__(idx=0)                                                  │    │
│  │                                                                     │    │
│  │  Text: "love this amazing"                                           │    │
│  │       ↓                                                             │    │
│  │  Tokenizer.encode_plus()                                            │    │
│  │       ↓                                                             │    │
│  │  {                                                                  │    │
│  │    'input_ids':      tensor([101, 5310, 42, 6428, 102, 0, 0, ...])   │    │
│  │    'attention_mask': tensor([1, 1, 1, 1, 1, 0, 0, ...])             │    │
│  │    'labels':         tensor(1)                                       │    │
│  │  }                                                                  │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────┬───────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 4: DATA LOADING                                                       │
│  ─────────────────────────────────────────────────────────────────────────  │
│  File: loaders.py                                                             │
│  Class: DataLoaderFactory                                                     │
│                                                                             │
│  Input:  SentimentDataset                                                     │
│  Output: torch.utils.data.DataLoader                                          │
│                                                                             │
│  DataLoader(dataset, batch_size=32, shuffle=True)                            │
│                                                                             │
│  Batch Output:                                                              │
│  {                                                                          │
│    'input_ids':      tensor(shape=[32, 128])  # (batch, seq_len)             │
│    'attention_mask': tensor(shape=[32, 128])                                 │
│    'labels':         tensor(shape=[32])                                      │
│  }                                                                          │
└─────────────────────────────┬───────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MODEL TRAINING                                      │
│  src/models/bert_classifier.py → src/pipelines/model_training.py           │
└─────────────────────────────────────────────────────────────────────────────┘
```

## File Dependencies

```
cleaner.py
    └── No internal dependencies (base cleaning)

preprocessor.py
    └── imports TweetCleaner from cleaner.py
    └── uses sklearn for class weights
    └── returns pd.DataFrame

dataset.py
    └── imports pandas, torch, transformers
    └── reads from dataset/processed/
    └── returns torch tensors

loaders.py
    └── imports dataset.py
    └── creates torch.utils.data.DataLoader
```

## Import Chain

```python
# User-facing imports (from src.data import ...)
src/data/__init__.py
    ├── TweetCleaner       (from cleaner.py)
    ├── TextPreprocessor   (from preprocessor.py)
    ├── DataAugmenter      (from preprocessor.py)
    ├── SentimentDataset   (from dataset.py)
    └── DataLoaderFactory  (from loaders.py)
```

## Execution Flow Example

```python
from src.data import TextPreprocessor, SentimentDataset
from transformers import BertTokenizer
from torch.utils.data import DataLoader

# 1. Preprocess raw data
preprocessor = TextPreprocessor()
df = pd.read_csv("dataset/raw/tweets.csv")
cleaned_df = preprocessor.preprocess(df)  # cleaner.py methods chained

# 2. Save to processed
cleaned_df.to_csv("dataset/processed/train.csv", index=False)

# 3. Create PyTorch Dataset
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
dataset = SentimentDataset("train.csv", tokenizer, max_length=128)

# 4. Create DataLoader
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# 5. Training loop
for batch in loader:
    input_ids = batch['input_ids']      # (32, 128)
    attention_mask = batch['attention_mask']  # (32, 128)
    labels = batch['labels']            # (32,)
    # forward pass...
```
