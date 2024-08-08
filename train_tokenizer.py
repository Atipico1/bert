import tokenizers
import re
from datasets import load_dataset, load_from_disk

dataset = load_from_disk("dataset/corpus_cleaned")
dataset = dataset["train"]
dataset = dataset.shuffle(seed=42).select(range(int(len(dataset)*0.1)))
bwpt = tokenizers.BertWordPieceTokenizer(clean_text=True)

bwpt.train_from_iterator(
    dataset["text"],
    vocab_size=30528,
    min_frequency=10,
    limit_alphabet=3000,
    show_progress=True,
    )

bwpt.save_model('./', 'final-clean')