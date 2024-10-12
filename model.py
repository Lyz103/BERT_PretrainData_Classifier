from transformers import BertTokenizer, BertModel, AutoModel
import torch.nn as nn

class fn_cls(nn.Module):
    def __init__(self, args, tokenizer):
        super(fn_cls, self).__init__()
        self.args = args
        self.num_classes = args.num_classes
        self.model = AutoModel.from_pretrained(args.model)
        self.model.resize_token_embeddings(len(tokenizer))  # Adjust token embeddings
        self.l1 = nn.Linear(768, args.num_classes)  # Change output to 3 for multi-class classification

    def forward(self, x, attention_mask=None):
        outputs = self.model(x, attention_mask=attention_mask)
        x = outputs[1]  # Use the pooled output ([CLS] token representation)
        x = self.l1(x)  # Linear layer with 3 outputs for 3 classes
        return x  # Return raw logits; apply softmax during evaluation or with CrossEntropyLoss during training