from transformers import BertModel, BertTokenizer
from tokenizers import BertWordPieceTokenizer
import torch
import torch.nn as nn
import string

class config:
    MAX_LEN = 128
    TOKENIZER = BertWordPieceTokenizer('/home/koushik/Documents/Pretrained Models/bert-base-uncased/vocab.txt')
    BERT_PATH = '/home/koushik/Documents/Pretrained Models/bert-base-uncased'

class BERTBaseUncased(nn.Module):
    def __init__(self):
        super(BERTBaseUncased, self).__init__()
        self.bert = BertModel.from_pretrained(config.BERT_PATH)
        self.l0 = nn.Linear(768, 2)

    def forward(self, ids, mask, token_type_ids):
        sequence_output, pooled_output = self.bert(
            ids,
            attention_mask = mask,
            token_type_ids = token_type_ids
        )
        logits = self.l0(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        return start_logits, end_logits

model = BERTBaseUncased()
model = nn.DataParallel(model)
model.load_state_dict(torch.load('sentiment_extraction/bert model/model.bin'))
model.eval()

def predict(tweet, sentiment, max_len=config.MAX_LEN, tokenizer=config.TOKENIZER):
    tweet = (" ".join(str(tweet).split(" "))).strip()
    
    if sentiment == "neutral" or len(tweet.split()) < 4:
        return tweet

    encoded_tweet = tokenizer.encode(tweet)
    tweet_tokens = encoded_tweet.tokens
    tweet_ids = encoded_tweet.ids
    mask = [1] * len(tweet_ids)
    token_type_ids = [0] * len(tweet_ids)

    padding_len = max_len - len(tweet_ids)
    ids = tweet_ids + [0] * padding_len
    mask = mask + [0] * padding_len
    token_type_ids = token_type_ids + [0] * padding_len

    ids =  torch.tensor([ids], dtype=torch.long)
    mask =  torch.tensor([mask], dtype=torch.long)
    token_type_ids = torch.tensor([token_type_ids], dtype=torch.long)

    start, end = model(ids, mask, token_type_ids)
    start = start.cpu().view(-1)[1:-(padding_len+1)]
    end = end.cpu().view(-1)[1:-(padding_len+1)]

    _, idx_start = start.max(0)
    _, idx_end = end.max(0)

    output = ""
    tweet_tokens = tweet_tokens[1: -1]
    for i in range(idx_start, idx_end+1):
        if tweet_tokens[i] in ('CLS', 'SEP'):
            continue
        elif tweet_tokens[i].startswith("##"):
            output += tweet_tokens[i][2:]
        elif len(tweet_tokens[i])==1 and tweet_tokens[i] in string.punctuation:
            output += tweet_tokens[i]
        else:
            output += (" "+tweet_tokens[i])
    if len(output) == 0:
        output = tweet
    return output.strip()