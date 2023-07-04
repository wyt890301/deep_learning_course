import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, confusion_matrix
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from thop import profile
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import cv2

class trainer():
    def __init__(self, model, train_loader, test_loader, epochs, lr, unet):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.criterion = CrossEntropyLoss()
        self.optimizer = Adam(model.parameters(), lr=lr)
        self.epochs = epochs
        self.unet = unet
        
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
                
                if(self.unet):
                    # 將三維壓縮為一維
                    avg_pool = nn.AdaptiveAvgPool2d(1)
                    squeeze_outputs = avg_pool(outputs).squeeze()
                    # calculate loss
                    loss = self.criterion(squeeze_outputs, label)
                else:
                    loss = self.criterion(outputs, label)
                

                # Backward pass and optimization
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
                
                # calculate accuracy
                if(self.unet):
                    _, predicted = torch.max(squeeze_outputs, 1)
                else:
                    _, predicted = torch.max(outputs.data, 1)
                    
                _, corrected = torch.max(label.data, 1)
                total += label.size(0)
                correct += (predicted == corrected).sum().item()

            self.loss_history.append(train_loss)
            self.acc_history.append(correct/total)
            print(f"--------------------Epoch {epoch+1}--------------------")
            print(f"Train_accuracy: {(correct/total):.3f} | Train_loss: {train_loss:.3f}")
            
        # print the curves of the training accuracy and validation accuracy
        self.plot_acc()
        self.plot_loss()
        
    def predict(self):
        correct = 0
        total = 0
        sen_list = []
        spe_list = []
        label_list = []
        output_list = []
        time_list = []
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        
        with torch.no_grad():
            for data in self.test_loader:
                inputs, label = data
                inputs = inputs.to(self.device)
                label = label.to(self.device)

                # inference
                starter.record()
                outputs = self.model(inputs)
                ender.record()
                
                if(self.unet):
                    # 將三維壓縮為一維
                    avg_pool = nn.AdaptiveAvgPool2d(1)
                    squeeze_outputs = avg_pool(outputs).squeeze()
                    # check correct
                    _, predicted = torch.max(squeeze_outputs, 1)
                else:
                    # check correct
                    _, predicted = torch.max(outputs.data, 1)
                    
                _, corrected = torch.max(label.data, 1)
                # accuracy
                total += label.size(0)
                correct += (predicted == corrected).sum().item()

                # Specificity and Sensitivity
                corrected_np = corrected.cpu().numpy()
                predicted_np = predicted.cpu().numpy()

                for i in range(3):
                    true_labels = [1 if x == i else 0 for x in corrected_np]
                    predicted_labels = [1 if x == i else 0 for x in predicted_np]
                    tn, fp, fn, tp = confusion_matrix(true_labels, predicted_labels).ravel()
                    sen_list.append(tp / (tp + fn))
                    spe_list.append(tn / (tn + fp))

                # AUC 
                label_list = np.concatenate((label_list, label.data.cpu().numpy()), axis=None)
                if(self.unet):
                    output_list = np.concatenate((output_list, squeeze_outputs.cpu().numpy()), axis=None)
                else:
                    output_list = np.concatenate((output_list, outputs.data.cpu().numpy()), axis=None)

                # record time
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender)
                time_list.append(curr_time)
                
        acc = correct / total
        sen = sum(sen_list) / len(sen_list)
        spe = sum(spe_list) / len(spe_list)
        auc = roc_auc_score(label_list, output_list, multi_class='ovr')
        timing = sum(time_list) / len(time_list)
        print(f"----------------------------------------------------")
        print(f'Accuracy on the test images: {acc:.3f}')
        print(f'Specificity on the test images: {sen:.3f}')
        print(f'Sensitivity on the test images: {spe:.3f}')
        print(f'AUC Score on the test images: {auc:.3f}')
        print(f'Inference time: {timing:.3f} milliseconds')
        
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
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.labels[index]
        return x, y
    
class model_complexity():
    def __init__(self, model, batch_size):
        self.batch_size = batch_size
        self.model = model
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    def call(self):
        inputs = torch.randn(self.batch_size, 3, 224, 224).to(self.device)
        macs, params = profile(self.model, inputs=(inputs, ))
        print(f'Macs = {(macs/1000**3):.3f}G')
        print(f'Params = {(params/1000**2):.3f}M')
        

class visualizing_tool():
    def __init__(self, train_loader, models):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.input, self.output = next(iter(train_loader))
        self.input = self.input.to(self.device)
        self.output = self.output.to(self.device)
        self.models = models

    def call(self):
        # set plt size
        plt.figure(figsize=(20, 14))
        
        # 七個model
        for i, model in enumerate(self.models):
            feature_image = self.feature_map(i, model)
            for j in range(10):
                plt.subplot(7, 10, 10*i+j+1)
                plt.imshow(normalize(feature_image[0][j]), cmap='gray')
                plt.axis('off')
                
    def feature_process(self):
        # 視覺化連續的特徵圖(vgg19, ViT)
        for i, model in enumerate(self.models):
            if i == 1:
                print("VGG19 前5層架構的特徵視覺化圖")
                self.vgg19_feature_map(model)
            elif i == 4:
                print("Vision Transformer 12層架構的特徵視覺化圖")
                self.vit_feature_map(model)
            else:
                pass

    def feature_map(self, index, model):
        # resnet50
        if index == 2:
            feature_map = model.conv1.forward(self.input)
            
        # ViT, SwinTransformer
        elif index == 4 or index == 5:
            layer1 = model.patch_embed.proj
            feature_map = layer1.forward(self.input)
        
        # UNet
        elif index == 6:
            layer1 = model.encoder.conv1
            feature_map = layer1.forward(self.input)
            
        # alexnet, vgg19, inception_v4
        else:
            # 第一層Convolution layers
            layer1 = model.features[0]
            feature_map = layer1.forward(self.input)
            
        # Show the feature map
        feature_image = feature_map.squeeze().cpu().detach().numpy()
        return feature_image

    def vgg19_feature_map(self, model):
        # set the feature layer
        for i in range(5):
            if i == 0:
                layer = model.features[0:4]
            else:
                layer = model.features[0:i*9]
                
            # feature_map
            feature_map = layer.forward(self.input)
            feature_image = feature_map.squeeze().cpu().detach().numpy()

            # plt
            fig, axs = plt.subplots(1,6, constrained_layout=True, figsize=(12, 2))
            for j in range(6):
                axs[j].imshow(normalize(feature_image[0][j]), cmap='gray')
                # axs[j].imshow(normalize(feature_image[0][j]), cmap='jet')
                axs[j].axis('off')
                # plt.tight_layout()
            fig.suptitle(f'Block {i+1}', fontsize=10)
            plt.show()
            
    def vit_feature_map(self, model):
        att_mat = []
        feature_map = model.patch_embed.forward(self.input)
        feature_map = model.pos_drop.forward(feature_map)

        
        for i in range(12): # 12 blocks
            feature_map_tmp = model.blocks[i].forward(feature_map)
            # att_mat.append(model.blocks[i].attn.proj.weight.mean(dim=1).view(14, 14).detach())
            att_mat.append(feature_map_tmp.reshape(32, 14, 14, 768))

        att_mat = torch.stack(att_mat).squeeze().cpu().detach().numpy()
        im = self.input[0].cpu().detach().numpy().transpose(1,2,0)
        im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        
        for i, v in enumerate(att_mat):
            mask = v[0].mean(axis=2)
            mask = cv2.resize(mask / mask.max(), (im.shape[0], im.shape[1]))
            result = (mask * im)#.astype("uint8")
            new_im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
            result[:,:,1:2] = 0
            result = cv2.addWeighted(new_im, 0.5, result, 0.5, 0)
            
            fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(5, 3))
            ax1.set_title('Original')
            ax2.set_title('Attention Map')
            _ = ax1.imshow(im, cmap="gray")
            _ = ax2.imshow(result, cmap="gray")
            fig.suptitle(f"Attention Map for Block {i+1}", fontsize=16)
    