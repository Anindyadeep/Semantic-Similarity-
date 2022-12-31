import torch 
import transformers

DEVICE="cuda" if torch.cuda.is_available() else "cpu"
EPOCHS=3
TRAIN_BATCH_SIZE=64
VALID_BATCH_SIZE=8
ACCUMULATED_STEPS=4
MAX_LEN=256

TOKENIZER=transformers.BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)