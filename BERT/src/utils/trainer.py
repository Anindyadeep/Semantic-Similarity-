import torch 
import pytorch_lightning as pl 

class BertSemanticSimilarity(pl.LightningDataModule):
    def __init__(self, model):
        super(BertSemanticSimilarity, self).__init__()
        self.model = model 
    
    def training_step(self):
        pass 

    def configure_optimizers(self, lr):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        return optimizer
    
