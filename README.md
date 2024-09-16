# Project: Sentence Reconstruction using Transformers

## Overview

The objective of this project is to reconstruct an original sentence from a random permutation of words. The project imposes several constraints to encourage the use of efficient models. The dataset comprises sentences from the **generics_kb dataset** provided by Hugging Face, with a vocabulary limited to the 10,000 most frequent words.

## Key Features
- **Task**: Reordering shuffled words in a sentence.
- **Constraints**:
  - No pretrained models are allowed.
  - The model must contain fewer than **20M parameters**.
  - No post-processing techniques (e.g., beam search) can be applied.
  - Only the provided dataset can be used for training.

## Dataset

The dataset consists of generic sentences filtered for length (over 8 words) and with a restricted vocabulary of 10,000 words. Sentences are shuffled and fed into the model to reconstruct the original order.

```python
from datasets import load_dataset

ds = load_dataset('generics_kb')['train']
ds = ds.filter(lambda row: len(row["generic_sentence"].split(" ")) > 8)
```

## Model Architecture

### Transformer Model
The model is based on the Transformer architecture, utilizing both an **Encoder** and **Decoder** to capture dependencies in scrambled sentences.

### Encoder
- **Embedding Layers**: Token and position embeddings are used to encode word semantics and positional information.
- **Multi-Head Attention**: Two attention layers allow the model to focus on various parts of the input sequence.
- **Feed-Forward Network**: A two-layer network with ReLU activation and dropout to reduce overfitting.

### Decoder
- **Masked Multi-Head Attention**: Prevents the model from looking at future tokens during training.
- **Attention Layers**: Attends to both the encoder's output and the decoder's previous steps.
- **Feed-Forward Network**: Similar to the encoder, processes the data further with a two-layer network.

### Hyperparameters
- Vocabulary size: **10,000**
- Maximum sequence length: **28**
- Embedding dimension: **64**
- Attention heads: **16**
- Feed-forward size: **4096**
- Optimizer: **AdamW**
- Epochs: **30**

## Training

The model is trained using **sparse categorical crossentropy**, and the dataset is split into training (220,000 samples) and test sets. Batch size is set to 32, and the model trains for 30 epochs.

```python
results = transformer.fit(train_generator, batch_size=32, epochs=EPOCHS)
```

### Results
- **Training Accuracy**: Up to **80%** after 30 epochs.
- **Test Score**: Average score of **0.39** on the test set, based on the given evaluation metric.

## Evaluation Metric

The quality of sentence reconstruction is measured by comparing the longest common substring between the original and predicted sentences, normalized by the length of the sentences.

```python
from difflib import SequenceMatcher

def score(s, p):
    match = SequenceMatcher(None, s, p).find_longest_match()
    return match.size / max(len(s), len(p))
```

## Conclusions

- The transformer model successfully reconstructs shuffled sentences with a moderate average score of **0.39**, which is an acceptable result given the constraints on the number of parameters.
- A balance between model size and performance was achieved, as the model contains only **4.3M trainable parameters**, well below the 20M limit.
