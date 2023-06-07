import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

class trainer():
    def __init__(self, model, train_loader, test_loader, epochs, lr):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.criterion = CrossEntropyLoss()
        self.optimizer = Adam(model.parameters(), lr=lr)
        self.epochs = epochs
        
    def train(self):
        self.acc_history = []
        self.loss_history = []
        
        # Training loop
        for epoch in range(self.epochs):
            train_loss = 0.0
            correct = 0
            total = 0
            for i, data in enumerate(self.train_loader):
                inputs, label = data
                inputs = inputs.to(self.device)
                label = label.to(self.device)

                # Zero the parameter gradients
                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.model(inputs.to(self.device))
                loss = self.criterion(outputs, label)

                # Backward pass and optimization
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                _, corrected = torch.max(label.data, 1)
                total += label.size(0)
                correct += (predicted == corrected).sum().item()

            self.loss_history.append(train_loss)
            self.acc_history.append(100*correct/total)
            print(f"--------------------Epoch {epoch+1}--------------------")
            print(f"Train_accuracy: {(100*correct/total):.2f}% | Train_loss: {train_loss:.4f}")
            
        # print the curves of the training accuracy and validation accuracy
        self.plot_acc()
        self.plot_loss()
        
    def predict(self):
        correct = 0
        total = 0
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        timing = 0
        
        with torch.no_grad():
            for data in self.test_loader:
                inputs, label = data
                inputs = inputs.to(self.device)
                label = label.to(self.device)

                # inference
                starter.record()
                outputs = self.model(inputs)
                ender.record()

                # check correct
                _, predicted = torch.max(outputs.data, 1)
                _, corrected = torch.max(label.data, 1)
                total += label.size(0)
                correct += (predicted == corrected).sum().item()

                # record time
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender)
                timing = curr_time

        print(f'Accuracy on the test images: {(100*correct/total):.2f}%')
        print(f'Inference time: {timing:.4f} milliseconds')
        
    def plot_acc(self):
        plt.plot(self.acc_history)
        plt.title('Accuracy History')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['train'], loc='upper left')
        plt.show()
        
    def plot_loss(self):
        plt.plot(self.loss_history)
        plt.title('Loss History')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['train'], loc='upper left')
        plt.show()
        
class imageDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.labels[index]
        return x, y