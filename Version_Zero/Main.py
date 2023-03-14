import cv2
import pandas as pd
import os
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
def retrieve_data():
  trainLabels = pd.read_csv("/data3/masseybr/trainLabels.csv")
  data = [x for x in os.listdir('/data3/masseybr/train')]
  print(len(data))
  return trainLabels

"""TO DO: Modify this to match the current dataset being utilized.
class HemorrhageDataset(torch.utils.data.Dataset):
  def __init__(self, train: bool = True):
    split = "train" if train else "test"
    self.data = [x for x in os.path.join(f"/data3/masseybr/{split}")]
    #self.metadata = json.load(open(os.path.join(f"data3/masseybr/{split}.json")))
    self.img_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.5,), (1,))])

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    datapoint = self.metadata[idx]
    imgs = []
    for img_path in datapoint["img_paths"]:
      img_full_path = os.path.join("hwf/symbols", img_path)
      img = Image.open(img_full_path).convert("L")
      img = self.img_transform(img)
      imgs.append(img)
    res = datapoint["res"]
    return (tuple(imgs), res)


def hemorrhage_loader(batch_size, hemorrhageDatset):
  train_loader = torch.utils.data.DataLoader(HemorrhageDataset(train=True), batch_size=batch_size, shuffle=True)
  test_loader = torch.utils.data.DataLoader(HemorrhageDataset(train=False), batch_size=batch_size, shuffle=True)
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
  def __init__(self, num_classes):
    super(ConvolutionNeuralNet, self).__init__()
    self.conv1 = nn.Conv2d(1, 32, 3, stride = 1, padding = 1)
    self.conv2 = nn.Conv2d(32, 64, 3, stride = 1, padding = 1)
    self.fc1 = nn.Linear(7744, 128)
    self.fc2 = nn.Linear(128, num_classes)

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

    # ==== YOUR CODE START HERE ====
    #
    # TODO: Setup the Scallop Context so that it contains relations to hold digit and symbol distributions
    #
    # ==== YOUR CODE END HERE ====

    # The Scallop module to evaluate the expression, taking neural network outputs
    # The output will be integers ranging from -9 (if you have `0 - 9`) to 81 (if you have `9 * 9`)
    # This suggests that our `forward` function will return tensors of dimension 91
    self.compute = self.scl_ctx.forward_function("result", output_mapping=list(range(-9, 82)))

  def forward(self, x: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
    # ==== YOUR CODE START HERE ====
    #
    # TODO: Write your code to invoke the CNNs and our Scallop module to evaluate the hand-written expression
    return None
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
      for (batch_id, (data, target)) in enumerate(iter):
        output = self.network(data)
        test_loss += self.loss(output, target).item()
        avg_loss = test_loss / (batch_id + 1)
        pred = output.data.max(1, keepdim=True)[1] - 9
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
n_epochs = 10
batch_size = 32
learning_rate = 0.001
seed = 1234

# Random seed
torch.manual_seed(seed)
random.seed(seed)

# Dataloaders
train_loader, test_loader = hemorrhage_loader(batch_size)
trainer = Trainer(train_loader, test_loader, learning_rate)
trainer.train(n_epochs)"""
retrieve_data()