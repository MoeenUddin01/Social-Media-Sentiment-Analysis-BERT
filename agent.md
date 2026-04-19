# AI Coding Agent Operating Manual

## Agent Role

AI coding assistant for a BERT-based social media sentiment analysis project. Operates within the Python package structure, respects import hierarchies, and maintains data flow through designated pipeline modules.

## Tools

- DagsHub: experiment tracking dashboard with live MLflow integration
- MLflow: metric and artifact logging backend
- URL pattern: https://dagshub.com/{owner}/{repo}/experiments
- Transformers: HuggingFace library for BERT models
- PyTorch: deep learning framework
- ruff: Python linting and formatting
- uv: fast Python package manager

## File Map

### Root

- `main.py` - Default entry stub (replace with actual orchestration logic)
- `pyproject.toml` - Project dependencies and metadata
- `config.ymal` - Configuration file (rename to config.yaml per project conventions)
- `README.md` - Project documentation

### src/ (Package Root)

- `src/__init__.py` - Package exports and version

### src/data/

- `src/data/__init__.py` - Data module exports
- `src/data/loaders.py` - Dataset loading utilities
- `src/data/cleaner.py` - Text cleaning and normalization
- `src/data/dataset.py` - PyTorch Dataset definitions
- `src/data/preprocessor.py` - Preprocessing transforms

### src/models/

- `src/models/__init__.py` - Model module exports
- `src/models/bert_classifier.py` - BERT classification model
- `src/models/tokenizer.py` - Tokenization utilities
- `src/models/fine_tuner.py` - Fine-tuning wrapper
- `src/models/train.py` - Training step implementation
- `src/models/checkpoints/` - Saved model checkpoints directory

### src/pipelines/

- `src/pipelines/__init__.py` - Pipeline module exports
- `src/pipelines/data_preprocessin.py` - Data preprocessing orchestration (note: typo in filename)
- `src/pipelines/model_training.py` - Training pipeline
- `src/pipelines/model_evaluation.py` - Evaluation pipeline
- `src/pipelines/callbacks.py` - Training callbacks
- `src/pipelines/scheduler.py` - Learning rate scheduling
- `src/pipelines/evaluator.py` - Evaluation logic

### src/utils/

- `src/utils/__init__.py` - Utils module exports
- `src/utils/logger.py` - Logging configuration
- `src/utils/metrics.py` - Metric computation
- `src/utils/seed.py` - Random seed management
- `src/utils/visualizer.py` - Visualization utilities

### Other Directories

- `dataset/` - Raw data storage (not tracked by git)
- `.venv/` - Virtual environment

## Task Patterns

| Task                      | Files to Edit                                                                 |
| ------------------------- | ----------------------------------------------------------------------------- |
| Add new data source       | `src/data/loaders.py` + `src/data/cleaner.py` + `src/pipelines/data_preprocessin.py` |
| Change model architecture | `src/models/bert_classifier.py` + update `config.ymal`                       |
| Add preprocessing step    | `src/data/preprocessor.py` + `src/pipelines/data_preprocessin.py`             |
| Modify training loop      | `src/pipelines/model_training.py` + `src/models/train.py`                     |
| Add evaluation metric     | `src/utils/metrics.py` + `src/pipelines/evaluator.py`                       |

## Decision Rules

1. **Import Tracing**
   - Always trace imports back to `src/__init__.py`
   - Never import from module internals; use package-level exports only

2. **Configuration**
   - Config values come from `config.ymal` only
   - No hardcoded hyperparameters in model or pipeline files
   - Cast values to expected types after loading from YAML

3. **Data Flow**
   - Data always flows through `DataPipeline` in `src/pipelines/data_preprocessin.py`
   - Never load data directly in model files
   - All dataset interactions go through `src/data/loaders.py`

4. **Model Changes**
   - Architecture changes go in `src/models/bert_classifier.py`
   - Training logic changes go in `src/models/train.py`
   - Update `config.ymal` when adding new model parameters

5. **Code Style**
   - Follow PEP 8 (enforced by ruff)
   - Google-style docstrings required
   - Type annotations on all function signatures
   - Maximum line length 88 characters

## Output Format

Agent replies with a file-by-file diff summary:

```markdown
**src/models/bert_classifier.py**
- Added `dropout_rate` parameter to `__init__`
- Updated `forward()` to apply dropout before classification layer

**config.ymal**
- Added `model.dropout_rate: 0.3`

**src/pipelines/model_training.py**
- Updated training loop to pass dropout config to model
```
