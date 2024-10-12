from torch.utils.data import DataLoader,TensorDataset
from torch.utils.data import Dataset

class SentimentDataset(Dataset):
    def __init__(self,df):
        self.dataset = df
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        text = self.dataset.loc[idx, "text"]
        label = self.dataset.loc[idx, "label"]
        input_ids = self.dataset.loc[idx, "input_ids"]
        attention_mask = self.dataset.loc[idx, "attention_mask"]
        sample = {"text": text, "label": label,"input_ids":input_ids,"attention_mask":attention_mask}
        # print(sample)
        return sample