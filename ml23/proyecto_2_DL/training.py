from torchvision.datasets import FER2013
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import torch.optim as optim
import torch
import torch.nn as nn
from tqdm import tqdm
from dataset import get_loader
from network import Network
from plot_losses import PlotLosses
from torchvision import transforms

def validation_step(val_loader, net, cost_function):
    '''
        Realiza un epoch completo en el conjunto de validación
        args:
        - val_loader (torch.DataLoader): dataloader para los datos de validación
        - net: instancia de red neuronal de clase Network
        - cost_function (torch.nn): Función de costo a utilizar

        returns:
        - val_loss (float): el costo total (promedio por minibatch) de todos los datos de validación
    '''
    val_loss = 0.0
    total = 0
    all_predictions = []
    all_labels = []
    for batch in val_loader:
        batch_imgs, batch_labels = batch['transformed'], batch['label']
        batch_imgs, batch_labels = batch_imgs.to(net.device), batch_labels.to(net.device)
        outputs = net(batch_imgs)
        loss = cost_function(outputs, batch_labels)
        val_loss += loss.item()
        total += batch_labels.size(0)

        _, predicted = torch.max(outputs, 1)
        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(batch_labels.cpu().numpy())
    return val_loss / total, np.array(all_predictions), np.array(all_labels)

def train():
    # Hyperparametros
    learning_rate = 1e-4
    n_epochs=50
    batch_size = 128

    transformations = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5), # Flips the image horizontally with probability of 0.5
        transforms.RandomApply([transforms.ColorJitter(brightness=0.5, contrast=0.5)], p=0.3), # Randomly adjusts brightness and contrast
        transforms.RandomRotation(degrees=15), # Randomly rotates the image
        transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)), # Random cropping and resizing back to original size
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)), # Applies Gaussian Blur
    ])

    # Train, validation, test loaders
    train_dataset, train_loader = get_loader("train", batch_size, transformations=transformations, shuffle=True)
    val_dataset, val_loader = get_loader("val", batch_size, transformations=transformations, shuffle=False) 
    print(f"Cargando datasets --> entrenamiento: {len(train_dataset)}, validacion: {len(val_dataset)}")

    plotter = PlotLosses()
    # Instanciamos tu red
    net = Network(input_dim=1, n_classes=7)

    # TODO: Define la funcion de costo
    criterion = nn.CrossEntropyLoss()

    # Define el optimizador
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    best_epoch_loss = np.inf

    for epoch in range(n_epochs):
        net.train()
        running_loss = 0.0

        for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}/{n_epochs}")):
            optimizer.zero_grad()
            batch_imgs, batch_labels = batch['transformed'], batch['label']
            batch_imgs, batch_labels = batch_imgs.to(net.device), batch_labels.to(net.device)
            outputs = net(batch_imgs)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        train_loss = running_loss / len(train_loader)
        net.eval()
        val_loss, val_predictions, val_labels = validation_step(val_loader, net, criterion)
        val_accuracy = accuracy_score(val_labels, val_predictions)
        val_precision = precision_score(val_labels, val_predictions, average='macro', zero_division=0)
        val_recall = recall_score(val_labels, val_predictions, average='macro', zero_division=0)
        val_f1 = f1_score(val_labels, val_predictions, average='macro', zero_division=0)


        tqdm.write(f"Epoch: {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                   f"Val Accuracy: {val_accuracy:.4f}, Val Precision: {val_precision:.4f}, "
                   f"Val Recall: {val_recall:.4f}, Val F1: {val_f1:.4f}")

        if val_loss < best_epoch_loss:
            best_epoch_loss = val_loss
            net.save_model('best_model.pth')

        plotter.on_epoch_end(epoch, train_loss, val_loss)
    plotter.on_train_end()  

if __name__=="__main__":
    train()