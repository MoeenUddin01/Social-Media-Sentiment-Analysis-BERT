# BERT-based Social Media Sentiment Analysis

Fine-tune BERT models for sentiment classification on social media text.

## Project Structure

```
src/
  data/         - Data loading, cleaning, and preprocessing
  models/       - BERT model implementation and tokenizers
  pipelines/    - Training and evaluation pipelines
  utils/        - Logging, metrics, seeds, visualization
config.ymal     - YAML configuration file
pyproject.toml  - Dependencies and project metadata
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate  # Windows
pip install -e .
```

## Usage

### Data Preprocessing

**Using TweetCleaner (individual methods):**

```python
from src.data import TweetCleaner

cleaner = TweetCleaner()
text = cleaner.remove_urls("Check out https://example.com #cool")
text = cleaner.remove_hashtags(text)
text = cleaner.normalize_whitespace(text)
```

**Using TextPreprocessor (full pipeline with class weights):**

```python
from src.data import TextPreprocessor
import pandas as pd

preprocessor = TextPreprocessor(lowercase=True, remove_emojis=True)

df = pd.DataFrame({
    'text': ['Love this product! 😍 @user https://t.co/x #amazing'],
    'label': [1]
})

# Returns DataFrame with 'cleaned_text' column
cleaned_df = preprocessor.preprocess(df)

# Compute class weights for imbalanced data
weights = preprocessor.get_class_weights(cleaned_df['label'])
# Use with: torch.nn.CrossEntropyLoss(weight=weights)
```

**Using SentimentDataset (PyTorch Dataset for training):**

```python
from src.data import SentimentDataset
from transformers import BertTokenizer
from torch.utils.data import DataLoader

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Create dataset from processed CSV in dataset/processed/
dataset = SentimentDataset(
    "train_cleaned.csv",  # relative path resolved to dataset/processed/
    tokenizer,
    max_length=128,
    text_column='cleaned_text',
    label_column='label'
)

# Or create from DataFrame directly
dataset = SentimentDataset(cleaned_df, tokenizer, max_length=128)

# Use with DataLoader
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Each batch contains:
# - 'input_ids': tensor of token IDs (batch_size, max_length)
# - 'attention_mask': tensor of masks (batch_size, max_length)
# - 'labels': tensor of class labels (batch_size,)
```

### Training

```bash
python -m src.pipelines.model_training
```

### Evaluation

```bash
python -m src.pipelines.model_evaluation
```

## Configuration

Edit `config.ymal` to adjust model architecture, training hyperparameters, and inference settings.

## License

MIT License
