
# Adversarial Attacks on Sentence-T5 Using EOS Token Space

This repository investigates how certain tokens in the Sentence-T5 embedding space behave similarly to the `</s>` (end-of-sequence) token. The focus is on the Romanian word **lucrarea**, which was discovered to have a high embedding similarity to `</s>` in the TensorFlow implementation. This similarity can be exploited to artificially boost **Sharpened Cosine Similarity (SCS)** scores — a phenomenon observed during the [Kaggle Prompt Recovery Competition](https://www.kaggle.com/competitions/llm-prompt-recovery).

Importantly, [top-ranking solutions on the leaderboard](https://www.kaggle.com/competitions/llm-prompt-recovery/leaderboard) used this insight as part of their adversarial attack strategy. For example, [this 1st place solution](https://www.kaggle.com/competitions/llm-prompt-recovery/discussion/494343) heavily relies on the similarity between `lucrarea` and `</s>` to construct high-scoring prompts.


## Notebooks

- `01_tfhub_eos_analysis.ipynb`:  
  Analyze all tokens from TensorFlow Hub’s Sentence-T5 embedding space and rank them by similarity to `</s>`.  
  `lucrarea` is shown to have a sharpened cosine similarity ~0.7965 and a dot product score ~0.9270 to `</s>` — indicating it acts as a “soft-EOS”.

- `02_hf_lucrarea_inspection.ipynb`:  
  Run the same inspection using HuggingFace’s PyTorch version. This reveals that `lucrarea` is tokenized as `['▁', 'lucrarea']`, and thus is not part of the embedding matrix as a single unit — leading to different behavior.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download the TensorFlow Sentence-T5 model and tokenizer files:
- Put `spiece.model` and the SavedModel folder (from [TFHub link or Kaggle TF version]) into a local folder: `t5/`
  
Directory should look like:
```
t5/
├── spiece.model
├── saved_model.pb
└── variables/
```

3. Run the notebooks.

## Notes

- `lucrarea` appears as a strong similarity artifact due to SentencePiece tokenization and embedding structure.
- TensorFlow version preserves `▁lucrarea` as a valid token, while HuggingFace splits it.
- Run TensorFlow jobs on CPU (or limit GPU) if memory errors occur during batch encoding.
