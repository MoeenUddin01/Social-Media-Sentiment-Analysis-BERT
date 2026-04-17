# BERT-based Social Media Sentiment Analysis

Fine-tune BERT models for sentiment classification on social media text.

## Project Structure

```
src/
  data/         - Data loading and preprocessing
  models/       - BERT model implementation
  training/     - Training utilities
  inference/    - Inference and API serving
  utils/        - Shared utilities
tests/          - Test suite
scripts/        - Training and evaluation scripts
config/         - YAML configuration files
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

## Usage

### Training

```bash
python scripts/train.py --config config/train.yaml
```

### Evaluation

```bash
python scripts/evaluate.py --model path/to/checkpoint --data path/to/test.csv
```

### Inference

```bash
python scripts/predict_batch.py --input texts.txt --output predictions.json
```

### API Server

```bash
uvicorn src.inference.api_server:app --host 0.0.0.0 --port 8000
```

## Configuration

Edit YAML files in `config/` to adjust:
- Model architecture (`model.yaml`)
- Training hyperparameters (`train.yaml`)
- Inference settings (`infer.yaml`)

## License

MIT License
