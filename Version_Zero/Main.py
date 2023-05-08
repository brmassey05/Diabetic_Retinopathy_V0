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
  return final_res, data, trainLabels

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
    self.hemorrhageNet = ConvolutionNeuralNet()
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

    processed_hemorrhages = self.hemorrhageNet(hemorrhages)

    result = self.compute(processed_hemorrhages)

    return result
    #
    # ==== YOUR CODE END HERE ====

class HemorrhageTrainer():
  def __init__(self, train_loader, test_loader, learning_rate):
    self.network = HemorrhageNet()
    self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
    self.train_loader = train_loader
    self.test_loader = test_loader

  def loss(self, output, ground_truth):
    output_mapping = list(range(-9, 82))
    (_, dim) = output.shape
    gt = torch.stack([torch.tensor([1.0 if output_mapping[i] == t else 0.0 for i in range(dim)]) for t in ground_truth])
    return F.binary_cross_entropy(output, gt)

  def train_epoch(self, epoch):
    self.network.train()
    train_loss = 0.0
    iter = tqdm(self.train_loader, total=len(self.train_loader))
    for (batch_id, (data, target)) in enumerate(iter):
      self.optimizer.zero_grad()
      output = self.network(data)
      loss = self.loss(output, target)
      loss.backward()
      self.optimizer.step()
      train_loss += loss.item()
      avg_loss = train_loss / (batch_id + 1)
      iter.set_description(f"[Train Epoch {epoch}] Batch Loss: {loss.item():.4f}, Avg Loss: {avg_loss:.4f}")

  def test_epoch(self, epoch):
    self.network.eval()
    num_items = 0
    test_loss = 0
    correct = 0
    with torch.no_grad():
      iter = tqdm(self.test_loader, total=len(self.test_loader))
      print(iter)
      for (image, (data, target)) in enumerate(iter):
        image = np.array(image)
        batch_id = int(data[0:2])
        output = self.network(image)
        test_loss += self.loss(output, target).item()
        avg_loss = test_loss / (batch_id + 1)
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum()
        num_items += pred.shape[0]
        perc = 100. * correct / num_items
        iter.set_description(f"[Test Epoch {epoch}] Avg loss: {avg_loss:.4f}, Accuracy: {correct}/{num_items} ({perc:.2f}%)")

  def train(self, n_epochs):
    self.test_epoch(0)
    for epoch in range(1, n_epochs + 1):
      self.train_epoch(epoch)
      self.test_epoch(epoch)

# Parameters
n_epochs = 3
batch_size = 32
learning_rate = 0.001
seed = 1234

# Random seed
torch.manual_seed(seed)
random.seed(seed)

# Retrieve Images
myimageset, images, labels = retrieve_im_data()
print(myimageset[0:10])

# Dataloaders
train_loader, test_loader = hemorrhage_loader(batch_size, myimageset)
trainer = HemorrhageTrainer(train_loader, test_loader, learning_rate)
trainer.train(n_epochs)