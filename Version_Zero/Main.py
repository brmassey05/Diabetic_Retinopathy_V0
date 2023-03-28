import pandas as pd
import os
import cv2
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score
import random
import json
from typing import *
from tqdm import tqdm
from PIL import Image

# PyTorch related
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# Scallop related
import scallopy

def retrieve_im_data():
  data = [x for x in os.listdir('/data3/masseybr/train')]
  final_res = []
  trainLabels = pd.read_csv("/data3/masseybr/trainLabels.csv").values.tolist()
  for i in range(len(data)):
    final_res.append((data[i], trainLabels[i]))

  return final_res

class HemorrhageDataset(torch.utils.data.Dataset):
  def __init__(self, images, train: bool = True):
    self.images = images
    self.train = train

  def __len__(self):
    return len(self.images)

  def __getitem__(self, idx):
    img = self.images[idx]
    # Convert image to tensor and normalize
    img_tensor = torchvision.transforms.ToTensor()(img)
    img_tensor = torchvision.transforms.Normalize((0.5,), (1,))(img_tensor)
    # Pass image through neural network to identify contours
    # Replace this with your own neural network model
    contours = cv2.findContours(np.array(img), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]
    # Return identified contours
    return img_tensor, contours


def hemorrhage_loader(batch_size, images):
  dataset_train = HemorrhageDataset(images, train=True)
  dataset_test = HemorrhageDataset(images, train=False)
  train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
  test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=True)
  return train_loader, test_loader

def identify_contours(image):
    # Read the image
    img = cv2.imread(image, 0)

    # Apply Gaussian blur to remove noise
    img_blur = cv2.GaussianBlur(img, (5, 5), 0)

    # Apply Otsu thresholding to segment the image
    _, thresh = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on the original image
    img_contours = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 50:  # Filter out small contours (noise)
            cv2.drawContours(img_contours, contour, -1, (0, 0, 255), 2)

    return contours

class ConvolutionNeuralNet(nn.Module):
  def __init__(self):
    super(ConvolutionNeuralNet, self).__init__()
    self.conv1 = nn.Conv2d(1, 32, 3, stride = 1, padding = 1)
    self.conv2 = nn.Conv2d(32, 64, 3, stride = 1, padding = 1)
    self.fc1 = nn.Linear(7744, 128)
    self.fc2 = nn.Linear(128, 2)

  def forward(self, x):
    x = F.max_pool2d(self.conv1(x), 2)
    x = F.max_pool2d(self.conv2(x), 2)
    x = torch.flatten(x, 1)
    x = F.relu(self.fc1(x))
    x = F.dropout(x, p = 0.5, training=self.training)
    x = self.fc2(x)
    return F.softmax(x, dim=1)

class HemorrhageNet(nn.Module):
  def __init__(self):
    super(HemorrhageNet, self).__init__()

    # Symbol Recognition CNN(s)
    # ==== YOUR CODE START HERE ====
    #
    # TODO: Setup your Convolutional Neural Network(s) to process potential digit or symbol data
    #
    # ==== YOUR CODE END HERE ====

    # Scallop Context
    self.scl_ctx = scallopy.ScallopContext("difftopkproofs")
    self.scl_ctx.add_program("""
    type hemorrhage(contour_id: usize, is_hemorrhage: bool)
    type severity(g: i8)
    rel num_hemorrhage(x) = x := count(id: hemorrhage(id, true))
    rel severity(0) = num_hemorrhage(0)
    rel severity(1) = num_hemorrhage(n), n > 0, n <= 2
    rel severity(2) = num_hemorrhage(n), n > 2, n <= 4
    rel severity(3) = num_hemorrhage(n), n > 4
    """)

    # The Scallop module to evaluate the expression, taking neural network outputs
    # The output will be integers ranging from -9 (if you have `0 - 9`) to 81 (if you have `9 * 9`)
    # This suggests that our `forward` function will return tensors of dimension 91
    self.compute = self.scl_ctx.forward_function("severity", output_mapping=list(range(5)))

  def forward(self, x: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
    # ==== YOUR CODE START HERE ====
    #
    # TODO: Write your code to invoke the CNNs and our Scallop module to evaluate the hand-written expression
    hemorrhage_distrs = [contour for contour in x]

    hemorrhages = torch.cat(tuple(hemorrhage_distrs), dim=1)

    result = self.compute(hemorrhages)

    return result
    #
    # ==== YOUR CODE END HERE ====

class Trainer():
  def __init__(self, train_loader, test_loader, learning_rate):
    self.network = HemorrhageNet()
    self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
    self.train_loader = train_loader
    self.test_loader = test_loader

  def loss(self, output, ground_truth):
    output_mapping = [0,1,2,3,4]
    (_, dim) = output.shape
    gt = torch.stack([torch.tensor([1.0 if output_mapping[i] == t else 0.0 for i in range(dim)]) for t in ground_truth])
    return F.binary_cross_entropy(output, gt)

  def train_model(model, train_loader, test_loader, num_epochs, learning_rate):
    # Set up loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Test model after each epoch
        model.eval()
        with torch.no_grad():
            total_loss = 0
            total_correct = 0
            for images, labels in test_loader:
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total_loss += criterion(outputs, labels).item()
                total_correct += (predicted == labels).sum().item()
            avg_loss = total_loss / len(test_loader)
            avg_acc = total_correct / len(test_loader.dataset)
            print(f"Epoch {epoch+1} | Validation Loss: {avg_loss:.4f} | Validation Accuracy: {avg_acc:.4f}")

