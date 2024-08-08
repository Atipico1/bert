from transformers import BertConfig, BertForMaskedLM
import torch
config = BertConfig.from_pretrained('bert-base-uncased',
                                    attn_implementation="flash_attention_2",
                                    dtype=torch.bfloat16,
                                    )
model = BertForMaskedLM(config)