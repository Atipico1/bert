from transformers import (
    AutoTokenizer,
    BertConfig,
    BertTokenizer,
    BertForMaskedLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer
)
from datasets import load_dataset, load_from_disk
import torch, os

dataset = load_from_disk("dataset/corpus_cleaned")
#train_set = dataset["train"].select(range(int(len(dataset["train"])*0.01)))
train_set = dataset["train"]
eval_set = dataset["test"]
del dataset
train_set = train_set.to_iterable_dataset(num_shards=32)
eval_set = eval_set.to_iterable_dataset(num_shards=32)
tokenizer = BertTokenizer.from_pretrained('final-vocab.txt')
tokenizer.model_max_length = 512
tokenizer.eos_token = tokenizer.pad_token
config = BertConfig.from_pretrained('bert-base-uncased')
config.vocab_size = tokenizer.vocab_size

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", max_length=512, truncation=True)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.3
)
model = BertForMaskedLM(config)
#train_set = train_set.shuffle(seed=42).select(range(int(len(train_set)*1)))
train_set = train_set.map(tokenize_function, batched=True, batch_size=1024, remove_columns=["subset"])
eval_set = eval_set.map(tokenize_function, batched=True, batch_size=1024, remove_columns=["subset"])

training_args = TrainingArguments(
    do_eval=True,
    learning_rate=5e-4,
    logging_steps=1,
    gradient_accumulation_steps=4,
    logging_strategy="steps",
    lr_scheduler_type="linear",
    adam_epsilon=1e-5,
    weight_decay=1e-5,
    max_steps=500000,
    bf16=True,
    dataloader_num_workers=32,
    dataloader_pin_memory=True,
    output_dir='result2/',
    overwrite_output_dir=True,
    per_device_eval_batch_size=128,
    save_steps=5000,
    per_device_train_batch_size=64, # originally set to 8
    push_to_hub=False,
    save_strategy="steps",
    seed=42,
)

trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    train_dataset=train_set,
    eval_dataset=eval_set,
)

# Perfrom pre-training and save the model
trainer.train()
trainer.save_model('./test_train')

# trainer = SFTTrainer(
#         model=model,
#         args=training_args,
#         train_dataset=train_set,
#         eval_dataset=eval_set,
#         tokenizer=tokenizer,
#         dataset_batch_size=10000,
#         max_seq_length=512,
#         dataset_num_proc=os.cpu_count(),
#         dataset_text_field="text",
#         data_collator=data_collator,
#     )
# To clear out cache for unsuccessful run