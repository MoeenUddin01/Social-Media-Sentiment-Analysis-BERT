# BERT-based Social Media Sentiment Analysis

Fine-tune BERT models for sentiment classification on social media text.

## Project Structure

```text
src/
  data/         - Data loading, cleaning, and preprocessing
  models/       - BERT model implementation and tokenizers
  pipelines/    - Training and evaluation pipelines
  utils/        - Logging, metrics, seeds, visualization
artifacts/      - Saved artifacts (checkpoints, tokenizers, logs, outputs)
  checkpoints/  - Model checkpoints (.pt files)
  tokenizers/   - Saved tokenizer files
  logs/         - Training and evaluation logs
  outputs/      - Generated outputs and predictions
config.yaml     - YAML configuration file
pyproject.toml  - Dependencies and project metadata
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate  # Windows
pip install -e .
```

### DagsHub Setup (Experiment Tracking)

Install DagsHub and MLflow dependencies:

```bash
pip install dagshub mlflow
```

Set up your DagsHub credentials:

```bash
export DAGSHUB_TOKEN=your_token_here
export MLFLOW_TRACKING_USERNAME=your_dagshub_username
export MLFLOW_TRACKING_PASSWORD=your_dagshub_token
dagshub login
```

Update `config.yaml` with your DagsHub repository:

```yaml
dagshub:
  enabled: true
  repo_owner: "your_dagshub_username"
  repo_name: "bert-sentiment"
  experiment_name: "bert-sentiment-analysis"
```

Then visit:
[https://dagshub.com/{repo_owner}/bert-sentiment/experiments](https://dagshub.com/{repo_owner}/bert-sentiment/experiments)

to see live charts updating every batch during training.

### Kaggle Training

Run this project on Kaggle with GPU (T4/P100):

**Setup Notebook Cells:**

```python
# Cell 1: Clone & Install
!git clone https://github.com/MoeenUddin01/Social-Media-Sentiment-Analysis-BERT.git
%cd Social-Media-Sentiment-Analysis-BERT
!pip install -q -r requirements.txt dagshub mlflow
!nvidia-smi
```

```python
# Cell 2: DagsHub Authentication
import os
from getpass import getpass
DAGSHUB_TOKEN = getpass("Enter DagsHub token: ")
os.environ['DAGSHUB_TOKEN'] = DAGSHUB_TOKEN
os.environ['MLFLOW_TRACKING_USERNAME'] = 'MoeenUddin01'
os.environ['MLFLOW_TRACKING_PASSWORD'] = DAGSHUB_TOKEN
```

```python
# Cell 3: Prepare Data
import os, shutil
os.makedirs('dataset/raw', exist_ok=True)
shutil.copy('/kaggle/input/YOUR-DATASET/data.csv', 'dataset/raw/data.csv')
```

```python
# Cell 4: Run Full Pipeline
from src.pipelines.data_preprocessin import DataPipeline
from src.pipelines.model_training import TrainingPipeline
from src.models.tokenizer import SentimentTokenizer
from src.models.bert_classifier import BertSentimentClassifier
from src.utils.seed import get_device, set_seed
from src.utils.logger import DagsHubLogger
import yaml, dagshub

# Setup
dagshub.init(repo_owner='MoeenUddin01', 
             repo_name='Social-Media-Sentiment-Analysis-BERT', mlflow=True)
with open('config.yaml') as f:
    config = yaml.safe_load(f)
device = get_device()
set_seed(42)

# Data pipeline
tokenizer = SentimentTokenizer()
data_pipeline = DataPipeline(config, tokenizer)
train_loader, val_loader, test_loader = data_pipeline.run('data.csv')

# Training
model = BertSentimentClassifier(num_labels=3).to(device)
dagshub_logger = DagsHubLogger(config)
training_pipeline = TrainingPipeline(
    model=model, train_loader=train_loader, val_loader=val_loader,
    device=device, config=config, dagshub_logger=dagshub_logger
)
training_pipeline.run(epochs=config['training']['epochs'])
```

**Important Artifacts to Save:**
- `artifacts/checkpoints/` - Model weights
- `artifacts/checkpoints/tokenizer/` - Tokenizer config
- `artifacts/checkpoints/label_map.json` - Class mappings

**Charts that will appear on DagsHub dashboard:**
- `train/batch_loss` → updates every batch (live)
- `train/batch_accuracy` → updates every batch (live)
- `val/f1_macro` → updates every epoch
- `val/loss` → updates every epoch
- `learning_rate` → updates every epoch (shows warmup curve)
- `test/accuracy` → appears once at end
- Confusion matrix → logged as image artifact every epoch

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

### Data Pipeline (End-to-End)

**Using DataPipeline for full preprocessing workflow:**

```python
from src.pipelines.data_preprocessin import DataPipeline
from src.models.tokenizer import SentimentTokenizer

# Initialize with config and tokenizer
tokenizer = SentimentTokenizer()
pipeline = DataPipeline(config_path="config.yaml", tokenizer=tokenizer)

# Run full pipeline: load -> clean -> split -> create DataLoaders
train_loader, val_loader, test_loader = pipeline.run("raw_data.csv")

# Access individual steps
raw_df = pipeline.load_raw("raw_data.csv")
cleaned_df = pipeline.clean(raw_df)
train_ds, val_ds, test_ds = pipeline.build_datasets(cleaned_df)
```

**Load checkpoint for inference:**

```python
# Restore pipeline with saved artifacts
pipeline = DataPipeline.load_checkpoint("artifacts/checkpoints/run_20240101/")
test_loader = pipeline.build_dataloaders(None, None, test_dataset)
```

### Model

**Using BertSentimentClassifier:**

```python
from src.models.bert_classifier import BertSentimentClassifier

# Initialize model (uses bert-base-uncased backbone)
model = BertSentimentClassifier(num_labels=3)  # positive, negative, neutral

# Forward pass with tokenized inputs
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
inputs = tokenizer("This product is amazing!", return_tensors="pt", padding=True)
logits = model(inputs)
```

**Save model checkpoint:**

```python
from src.models.bert_classifier import BertSentimentClassifier

model = BertSentimentClassifier(num_labels=3)

# Save to default artifacts/checkpoints/ directory
checkpoint_path = model.save("model.pt")

# Or specify custom checkpoint directory
checkpoint_path = model.save(
    checkpoint_name="best_model.pt",
    checkpoints_dir="/path/to/checkpoints"
)
```

**Loading from checkpoint:**

```python
from src.models.bert_classifier import BertSentimentClassifier

# Load from default artifacts/checkpoints/ directory
model = BertSentimentClassifier.from_pretrained("model.pt")

# Or specify custom checkpoint directory
model = BertSentimentClassifier.from_pretrained(
    checkpoint_name="best_model.pt",
    checkpoints_dir="/path/to/checkpoints"
)
```

**Labels mapping:**

- `0`: negative
- `1`: neutral
- `2`: positive

**Load for inference:**

```python
from src.models.bert_classifier import BertSentimentClassifier

# Load model and automatically set to eval mode
model = BertSentimentClassifier.load_for_inference("artifacts/checkpoints/")

# Or specify checkpoint file explicitly
model = BertSentimentClassifier.load_for_inference(
    checkpoint_dir="artifacts/checkpoints/",
    checkpoint_name="best_model.pt"
)
```

### Tokenizer

**Using SentimentTokenizer:**

```python
from src.models.tokenizer import SentimentTokenizer

# Initialize tokenizer (defaults to bert-base-uncased)
tokenizer = SentimentTokenizer()

# Tokenize a batch of texts
encoded = tokenizer.tokenize_batch(
    ["Love this product!", "Terrible experience", "It's okay"],
    max_length=128
)

# Access tensors
input_ids = encoded["input_ids"]        # (batch_size, max_length)
attention_mask = encoded["attention_mask"]  # (batch_size, max_length)
```

**Save and load tokenizer:**

```python
# Save to default artifacts/tokenizers/{model_name}/ directory
save_path = tokenizer.save()

# Or specify custom path
save_path = tokenizer.save("path/to/tokenizer_dir")

# Load from default artifacts/tokenizers/{model_name}/ directory
tokenizer = SentimentTokenizer.load()

# Or load from custom path
tokenizer = SentimentTokenizer.load("path/to/tokenizer_dir")
```

### Fine-Tuning

**Using FineTuner (layer freezing and LR decay):**

```python
from src.models.fine_tuner import FineTuner
from src.models.bert_classifier import BertSentimentClassifier

model = BertSentimentClassifier(num_labels=3)
fine_tuner = FineTuner(model)

# Freeze all BERT base layers (only classifier trainable)
fine_tuner.freeze_base_layers()

# Gradual unfreezing: unfreeze top N layers
fine_tuner.gradual_unfreeze(layer_num=4)  # Unfreeze top 4 layers

# Unfreeze all parameters
fine_tuner.unfreeze_all()
```

**Layer-wise learning rate decay for AdamW:**

```python
import torch

# Get parameter groups with decayed learning rates
# Lower layers get lower LR; classifier gets 2x base_lr
param_groups = fine_tuner.get_parameter_groups(base_lr=2e-5, decay_rate=0.95)

# Use with AdamW optimizer
optimizer = torch.optim.AdamW(param_groups)
```

### Training

**Quick start (with default config):**

```bash
python src/models/train.py
```

**With custom config:**

```python
from src.models.train import train, get_default_config

# Use default configuration
train()

# Or provide custom config
config = get_default_config()
config["training"]["epochs"] = 5
config["training"]["batch_size"] = 64
train(config)
```

**Configuration structure (config.yaml):**

```yaml
model:
  name: "bert-base-uncased"
  num_labels: 3
  dropout: 0.3

training:
  batch_size: 32
  max_length: 128
  epochs: 3
  learning_rate: 2.0e-5
  weight_decay: 0.01
  gradual_unfreeze: true
  freeze_epochs: 1

data:
  train_file: "train_cleaned.csv"
  val_file: "val_cleaned.csv"
  test_file: "test_cleaned.csv"
  text_column: "cleaned_text"
  label_column: "label"

checkpoints:
  save_dir: "artifacts/checkpoints"
  save_best: true
  save_every_epoch: false

evaluation:
  save_plots: true
  report_dir: "artifacts/checkpoints"
  top_k_checkpoints: 3

logging:
  level: "INFO"
  log_dir: "artifacts/logs"
```

**Training features:**

- Automatic gradual unfreezing (freeze base layers initially, unfreeze progressively)
- Layer-wise learning rate decay (lower layers → lower LR)
- Saves best checkpoint based on validation accuracy
- Saves final checkpoint and tokenizer to `artifacts/`

### Training Pipeline (End-to-End)

**Using TrainingPipeline with callbacks and checkpointing:**

```python
from src.pipelines.model_training import TrainingPipeline
from src.pipelines.callbacks import EarlyStopping, ModelCheckpoint
from src.pipelines.scheduler import get_optimizer, get_scheduler
from src.models.evaluator import ModelEvaluator
import torch

# Setup
model = BertSentimentClassifier(num_labels=3)
optimizer = get_optimizer(model, config)
scheduler = get_scheduler(optimizer, num_training_steps, config)
evaluator = ModelEvaluator(model, device)

# Callbacks
callbacks = [
    EarlyStopping(monitor="val_loss", patience=3, mode="min"),
    ModelCheckpoint(monitor="val_accuracy", save_best_only=True),
]

# Initialize pipeline
pipeline = TrainingPipeline(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    scheduler=scheduler,
    evaluator=evaluator,
    callbacks=callbacks,
    device=device,
    config=config,
)

# Train with automatic validation and checkpointing
history = pipeline.fit(num_epochs=10)

# Save all artifacts (model, tokenizer, config, metrics)
pipeline.save_all_artifacts("artifacts/checkpoints/run_20240101/")
```

### Evaluation Pipeline (End-to-End)

**Using EvaluationPipeline for model evaluation:**

```python
from src.pipelines.model_evaluation import EvaluationPipeline

# Load from checkpoint
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pipeline = EvaluationPipeline(
    checkpoint_dir="artifacts/checkpoints/run_20240101/",
    device=device,
)

# Run full evaluation
metrics = pipeline.run(test_loader)
print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"Macro F1: {metrics['macro_f1']:.4f}")

# Evaluate single text
result = pipeline.run_single("This product is amazing!")
print(f"Predicted: {result['predicted_label']}")
print(f"Confidence: {result['confidence']:.4f}")

# Compare multiple runs
import pandas as pd
df = pipeline.compare_runs([
    "artifacts/checkpoints/run_20240101/",
    "artifacts/checkpoints/run_20240102/",
])
print(df)  # DataFrame with metrics per run

# Find best checkpoint
best_path = EvaluationPipeline.load_best_checkpoint()
print(f"Best model at: {best_path}")
```

### Evaluation

**Using ModelEvaluator (programmatic):**

```python
from src.models.evaluator import ModelEvaluator
from src.models.bert_classifier import BertSentimentClassifier
import torch
from torch.utils.data import DataLoader

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertSentimentClassifier.load_for_inference("artifacts/checkpoints/")
model.to(device)

# Create evaluator
label_map = {0: "negative", 1: "neutral", 2: "positive"}
evaluator = ModelEvaluator(model, device, label_map)

# Evaluate on test data
metrics = evaluator.evaluate(test_loader)
print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"Macro F1: {metrics['macro_f1']:.4f}")

# Save comprehensive report
evaluator.save_report("artifacts/outputs/")
# Saves: eval_report.txt, eval_results.json, confusion_matrix.png,
#        confidence_histogram.png, per_class_f1_bar.png
```

**Evaluate single text:**

```python
from src.models.tokenizer import SentimentTokenizer

result = evaluator.evaluate_single("This product is amazing!", tokenizer)
print(f"Predicted: {result['predicted_label']}")
print(f"Confidence: {result['confidence']:.4f}")
print(f"All probs: {result['class_probabilities']}")
```

**Using CLI:**

```bash
# Train with default config
python main.py train --config config.yaml --data-file dataset/raw/train.csv

# Evaluate with default test data
python main.py evaluate --checkpoint artifacts/checkpoints/ --test-data dataset/processed/test_cleaned.csv

# Specify custom output directory
python main.py evaluate \
    --checkpoint artifacts/checkpoints/best_model.pt \
    --test-data dataset/processed/test_cleaned.csv \
    --output-dir artifacts/outputs/
```

## Configuration

Edit `config.yaml` to adjust model architecture, training hyperparameters, and inference settings.

## License

MIT License
