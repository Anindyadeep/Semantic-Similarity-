import config 
import torch 
from typing import List, Dict

# Dataset Approach 1 

class BERTDatasetMerged:
    def __init__(self, sentence_list1 : List[str], sentence_list2 : List[str], targets : List[int]) -> None:
        """Creates a BERT TOKENIZER instance that takes two sentences and merges them
        in the format like this: [CLS] text1 [SEP] text2 [SEP]

        Args:
            sentence_list1 (List[str]): A list of sentences
            sentence_list2 (List[str]): A list of sentences
            targets (List[int]): Whether text1 and text2 are similar schemantically or not 
        
        Returns:
            None
        """ 
        self.sentence_list1, self.sentence_list2 = sentence_list1, sentence_list2
        self.targets = targets 

    def __len__(self):
        return len(self.sentence_list1)
    
    def __getitem__(self, item_num : int) -> Dict[str, torch.Tensor]:
        """Fetches the requested token of a pair of sentences 

        Args:
            item_num (int): The idx of the senetence in the sentence list

        Returns:
            Dict[str, torch.Tensor]: A dict that contains [ids, mask, token_type_ids, targets] as the required keys
        """

        text1, text2 = self.sentence_list1[item_num], self.sentence_list2[item_num]

        # TODO: Use a text cleaning function from utils 
        text1 = " ".join(text1.split())
        text2 = " ".join(text2.split())


        inputs=config.TOKENIZER.encode_plus(
                    text1, text2, 
                    add_special_tokens=True, 
                    max_length=config.MAX_LEN,
                    padding='max_length'
                )

        ids=inputs['input_ids']
        token_types=inputs['token_type_ids']
        mask=inputs['attention_mask']

        return {
            'ids' : torch.tensor(ids, dtype=torch.long), 
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids' : torch.tensor(token_types, dtype=torch.long),
            'targets' : torch.tensor(int(self.targets[item_num]), dtype=torch.long)
        }


# Dataset Approach 1 

class SentencePairBERTDataset:
    def __init__(self, sentence_list1 : List[str], sentence_list2 : List[str], targets : List[int]) -> None:
        """Creates a BERT TOKENIZER instance that takes two sentences and stores them
        in the format like this: 
        input1: [CLS] sentence1 [SEP]
        input2: [CLS] sentence2 [SEP]

        Args:
            sentence_list1 (List[str]): A list of sentences
            sentence_list2 (List[str]): A list of sentences
            targets (List[int]): Whether text1 and text2 are similar schemantically or not 
        
        Returns:
            None
        """ 
        self.sentence_list1, self.sentence_list2 = sentence_list1, sentence_list2
        self.targets = targets 
    
    def __len__(self):
        return len(self.sentence_list1)
    
    def __getitem__(self, item_num : int) -> Dict[str, torch.Tensor]:
        """Fetches the requested token of a pair of sentences 

        Args:
            item_num (int): The idx of the senetence in the sentence list

        Returns:
            Dict[str, torch.Tensor]: ... TODO 
        """

        text1, text2 = self.sentence_list1[item_num], self.sentence_list2[item_num]

        # TODO: Use a text cleaning function from utils 
        text1 = " ".join(text1.split())
        text2 = " ".join(text2.split())


        inputs1=config.TOKENIZER.encode_plus(
                    text1, None, 
                    add_special_tokens=True, 
                    max_length=config.MAX_LEN,
                    padding='max_length'
                )

        inputs2=config.TOKENIZER.encode_plus(
                    text1, None, 
                    add_special_tokens=True, 
                    max_length=config.MAX_LEN,
                    padding='max_length'
                )

        ids1, ids2 = inputs1['input_ids'], inputs2['input_ids']
        token_types1, token_types2 = inputs1['token_type_ids'], inputs2['token_type_ids']
        mask1, mask2 = inputs1['attention_mask'], inputs2['attention_mask']

        return {
            'ids' : [torch.tensor(ids1, dtype=torch.long), torch.tensor(ids2, dtype=torch.long)],
            'mask': [torch.tensor(mask1, dtype=torch.long), torch.tensor(mask2, dtype=torch.long)],
            'token_type_ids' : [torch.tensor(token_types1, dtype=torch.long), torch.tensor(token_types2, dtype=torch.long)],
            'targets' : torch.tensor(int(self.targets[item_num]), dtype=torch.long)
        }



if __name__ == '__main__':
    import pandas as pd 

    train_dataset = pd.read_csv('../Data/train_dataset_random_negative_sample_from_iteself.csv')
    train_text=train_dataset['text'].tolist()
    train_reason=train_dataset['reason'].tolist()
    train_targets=train_dataset['label'].tolist()

    dataset = SentencePairBERTDataset(
        sentence_list1=train_text, sentence_list2=train_reason, targets=train_targets
    )

    from torch.utils.data import DataLoader 
    dataloader = DataLoader(dataset, batch_size=config.TRAIN_BATCH_SIZE)
    loader_dict = next(iter(dataloader))