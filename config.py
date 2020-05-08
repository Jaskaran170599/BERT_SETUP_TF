import tokenizers
import os

MAX_LEN = 128
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 16
EPOCHS = 3
BERT_PATH = "bert-base-cased"
MODEL_PATH = "model.bin"
TRAINING_FILE = "/kaggle/input/tweet-sentiment-extraction/train.csv"
TESTING_FILE="/kaggle/input/tweet-sentiment-extraction/train.csv"
TOKENIZER = tokenizers.BertWordPieceTokenizer("bert-base-cased-vocab.txt",
             lowercase=False)