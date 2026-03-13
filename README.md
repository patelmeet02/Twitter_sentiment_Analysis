# Twitter Sentiment Analysis — Bidirectional LSTM

A deep learning project for classifying tweets into **positive**, **negative**, or **neutral** sentiment using a Bidirectional LSTM model with pre-trained Word2Vec embeddings.

---

## Project Structure

```
├── sentiment_analysis_dl.ipynb   # Model training pipeline
├── test.ipynb                    # Inference / prediction notebook
├── requirements.txt              # Python dependencies
└── sentiment_analysis_lstm_model/
    ├── sentiment_lstm_model.keras # Saved Keras model
    ├── tokenizer.pickle           # Fitted Keras tokenizer
    ├── label_encoder.pickle       # Fitted label encoder
    └── model_config.json          # Hyperparameter config
```

---

## Dataset

The model is trained on a Twitter sentiment dataset (`train.csv`) with 27,481 samples. Each row includes:

| Column | Description |
|---|---|
| `text` | Full tweet text |
| `selected_text` | Sentiment-bearing phrase |
| `sentiment` | Label: `positive`, `negative`, or `neutral` |
| `Time of Tweet` | morning / noon / night |
| `Age of User` | Age bracket (e.g. 0–20, 21–30) |
| `Country` | User's country |

---

## Model Architecture

The model is a stacked **Bidirectional LSTM** with pre-trained embeddings:

```
Embedding (Word2Vec Google News 300d, frozen)
    ↓
Bidirectional LSTM (128 units, return_sequences=True)
    ↓
Dropout (0.5)
    ↓
Bidirectional LSTM (64 units)
    ↓
Dropout (0.5)
    ↓
Dense (3 units, softmax)
```

**Key hyperparameters:**
- `max_words`: 20,000
- `max_len`: 200 tokens
- `embedding_dim`: 300
- `optimizer`: Adam
- `loss`: Sparse categorical cross-entropy
- `epochs`: 15, `batch_size`: 64

**Training results (epoch 15):** ~89% train accuracy, ~74% validation accuracy.

---

## Text Preprocessing Pipeline

1. Fix encoding issues with `ftfy`
2. Lowercase
3. Strip HTML tags and URLs
4. Remove punctuation
5. Tokenize with NLTK `word_tokenize`
6. Lemmatize with NLTK `WordNetLemmatizer`

---

## Installation

```bash
pip install tensorflow numpy ftfy nltk gensim scikit-learn
```

Or using the provided requirements file:

```bash
pip install -r requirements.txt
```

> **Note:** The training notebook also downloads the `word2vec-google-news-300` model via `gensim.downloader` (~1.6 GB) and NLTK corpora (`punkt_tab`, `wordnet`).

---

## Usage

### Training (`sentiment_analysis_dl.ipynb`)

Run all cells in order. The notebook will:
1. Load and clean the dataset
2. Preprocess and tokenize text
3. Build a Word2Vec embedding matrix
4. Train the Bidirectional LSTM
5. Save the model, tokenizer, label encoder, and config to disk

### Inference (`test.ipynb`)

Load the saved model and run predictions on new text:

```python
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle, json, numpy as np, os

model_dir = 'path/to/sentiment_analysis_lstm_model'

model     = load_model(os.path.join(model_dir, 'sentiment_lstm_model.keras'))
tokenizer = pickle.load(open(os.path.join(model_dir, 'tokenizer.pickle'), 'rb'))
le        = pickle.load(open(os.path.join(model_dir, 'label_encoder.pickle'), 'rb'))
config    = json.load(open(os.path.join(model_dir, 'model_config.json')))

max_len = config['max_len']

text = "You are looking great today!"
seq  = tokenizer.texts_to_sequences([text])
pad  = pad_sequences(seq, maxlen=max_len, padding='post')
pred = model.predict(pad, verbose=0)
label = le.inverse_transform([np.argmax(pred)])[0]

print(f"Sentiment: {label}")   # → positive / negative / neutral
```

**Label mapping:**
| Encoded | Sentiment |
|---|---|
| 0 | negative |
| 1 | neutral |
| 2 | positive |

---

## Requirements

| Library | Purpose |
|---|---|
| `tensorflow` | Model building and training |
| `numpy` | Numerical operations |
| `scikit-learn` | Label encoding, train/test split |
| `gensim` | Word2Vec embeddings |
| `nltk` | Tokenization and lemmatization |
| `ftfy` | Unicode/encoding fixes |
| `pandas` | Data loading and manipulation |
