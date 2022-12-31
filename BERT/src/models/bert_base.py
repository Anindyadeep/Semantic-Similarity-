import torch 
import transformers 
import torch.nn as nn 
from typing import List 

# Model for Dataset Approach 1 

class BertBaseUncasedSingleSentence(nn.Module):
    def __init__(self, dropout_prob=0.2):
        super(BertBaseUncasedSingleSentence, self).__init__() 
        self.bert=transformers.BertModel.from_pretrained('bert-base-uncased')
        self.bert_dop=nn.Dropout(dropout_prob)
        self.classifier=nn.Linear(768, 1)
    
    def forward(self, ids : torch.Tensor, token_type_ids : torch.Tensor, mask : torch.Tensor) -> torch.Tensor:
        """Computes the similarity between two sentences provided using the dataset format 

        Args:
            ids (torch.Tensor): Token ids to be used
            token_type_ids (torch.Tensor): Token type ids to be used
            mask (torch.Tensor): Attention mask

        Returns:
            torch.Tensor: Returns logits between 0 to 1 for computing the probability of similarity
        """
        sequence_encodings, pooled_encodings = self.bert(
            ids=ids, 
            token_type_ids=token_type_ids,
            attention_mask=mask
        )
        embeddings = self.bert_drop(pooled_encodings)
        return torch.sigmoid(self.classifier(embeddings)) # Check if it is doing better with simple logits or not 
    

# Model for Dataset Approach 2 

class BertBaseUncasedSentencePair(nn.Module):
    def __init__(self, dropout_prob=0.2):
        super(BertBaseUncasedSentencePair, self).__init__() 
        self.bert=transformers.BertModel.from_pretrained('bert-base-uncased')
        self.bert_dop=nn.Dropout(dropout_prob)
        self.classifier=nn.Linear(768, 1)
    
    def forward(self):
        ...