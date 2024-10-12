import torch
import torch.nn as nn
from torch.optim import Adam
import time
from sklearn import metrics

class Finetune_Trainer:
    def __init__(self, model, train_dataloader,
                 eval_dataloader, args):

        self.args = args
        self.cuda_condition = True and not self.args.no_cuda
        self.device = torch.device("cuda" if self.cuda_condition else "cpu")

        self.model = model
        if self.cuda_condition:
            self.model.cuda()

        # Setting the train and test data loader
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader

        # self.data_name = self.args.data_name
        # self.optim = ScheduledOptim(Adam(model.parameters(), betas=(args.adam_beta1, args.adam_beta2), eps=1e-09, weight_decay=args.weight_decay), n_warmup_steps=args.n_warmup_steps,init_lr=args.lr)
        # self.optim = Adam(self.model.parameters(), lr=self.args.lr, betas=betas, weight_decay=self.args.weight_decay)
        self.optim  = Adam(self.model.parameters(), lr=self.args.lr)

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))
        self.criterion = nn.CrossEntropyLoss()

    def train(self, epoch):
        self.iteration(epoch, self.train_dataloader)

    def test(self, epoch):
        return self.iteration(epoch, self.eval_dataloader, train=False)

    def iteration(self, epoch, dataloader, train=True):
        if train:
            self.model.train()
            print("______________________________________________")
            print("______________________________________________")
            print("_______________", epoch, "start_______________")
            print("______________________________________________")
            print("______________________________________________")
            epoch_loss = 0
            total = 0
            correct = 0
            output_all = []
            label_all = []
            for batch_idx, batch in enumerate(dataloader):
                self.optim.zero_grad()
                label = batch['label'].to(self.device) # No need to convert to float for multi-class
                input_ids = torch.stack(batch['input_ids']).t().to(self.device) # batch size * 100
                attention_mask = torch.stack(batch['attention_mask']).t().to(self.device)  # batch size * 100

                # Forward pass to compute the output logits
                output = self.model(input_ids, attention_mask=attention_mask)  # batch size * 3

                # Calculate loss using CrossEntropyLoss (multi-class)
                loss = self.criterion(output, label)
                loss.backward()
                self.optim.step()

                with torch.no_grad():
                    # Get the predicted class from the logits
                    predicted = torch.argmax(output, dim=1)  # batch size * 1 (predicted class index)
                    output_all.append(predicted)
                    label_all.append(label)

                    total += len(label)

                    # Accumulate loss and calculate average loss
                    epoch_loss += loss.item()
                    ave_loss = epoch_loss / total

                    # Calculate accuracy
                    add_correct = (predicted == label).sum().item()
                    correct += add_correct
                    acc = correct / total * 100

                    if batch_idx % 5 == 0:
                        print('[{}/{} ({:.0f}%)]\tCorrect: {}, Total: {}, Accuracy: {:.2f}%, Avg Loss: {}'.format(
                            batch_idx, len(dataloader), 100. * batch_idx / len(dataloader),
                            correct, total, acc, ave_loss
                        ), end="\r")

            # End of epoch
            print('Correct: {}, Total: {}, Accuracy: {:.2f}%, Avg Loss: {}'.format(
                correct, total, acc, ave_loss))
            return epoch_loss
        else:
            self.model.eval()
            epoch_loss = 0
            total = 0
            correct = 0
            output_all = []
            T = 0
            label_all = []
            
            for batch_idx, batch in enumerate(dataloader):
                with torch.no_grad():
                    # Move labels and input data to the correct device
                    label = batch['label'].to(self.device)  # batch size * 1 (not float for multi-class)
                    label_all.append(label)
                    
                    input_ids = torch.stack(batch['input_ids']).t().to(self.device)  # batch size * 100
                    attention_mask = torch.stack(batch['attention_mask']).t().to(self.device) # batch size * 100
                    
                    # Calculate model output (logits for 3 classes)
                    t1 = time.time()
                    output = self.model(input_ids, attention_mask=attention_mask)  # batch size * 3 (logits for 3 classes)
                    t2 = time.time()
                    T += t2-t1
                    # Calculate loss using CrossEntropyLoss (no need for sigmoid or softmax)
                    loss = self.criterion(output, label)
                    epoch_loss += loss.item()
                    total += len(label)
                    ave_loss = epoch_loss / total
                    
                    # Get the predicted class (argmax over logits for 3 classes)
                    predicted = torch.argmax(output, dim=1)  # batch size * 1 (predicted class index)
                    output_all.append(predicted)
                    
                    # Calculate accuracy
                    add_correct = (predicted == label).sum().item()
                    correct += add_correct
                    acc = correct / total * 100
                    
                    if batch_idx % 5 == 0:
                        print('[{}/{} ({:.0f}%)]\tCorrect: {}, Total: {}, Accuracy: {:.2f}%, Avg Loss: {}'.format(
                            batch_idx, len(dataloader), 100. * batch_idx / len(dataloader), 
                            correct, total, acc, ave_loss), end="\r")
            
            # At the end of the test loop
            print('Correct: {}, Total: {}, Accuracy: {:.2f}%, Avg Loss: {}'.format(
                            correct, total, acc, ave_loss))
            print("Time:", T / len(dataloader) * 1000)
            
            # Concatenate all outputs and labels
            output_all = torch.cat(output_all, 0)
            label_all = torch.cat(label_all, 0)
            
            # Move tensors to CPU and convert to numpy arrays
            output_all = output_all.cpu().numpy()
            label_all = label_all.cpu().numpy()
            
            # Calculate accuracy and classification report using sklearn
            acc_score = metrics.accuracy_score(label_all, output_all)
            print(metrics.classification_report(label_all, output_all))
            print("Accuracy:", acc_score)
            
            return acc, epoch_loss
