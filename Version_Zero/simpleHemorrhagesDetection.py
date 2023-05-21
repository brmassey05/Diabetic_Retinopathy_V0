import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import pandas as pd
from PIL import Image

# Open the image file
image = Image.open('image.jpg')

# Get the size of the image
width, height = image.size

# Print the size
print('Image size:', width, 'x', height)

# Define the model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

class HemorrhageNet(nn.Module):
  def __init__(self):
    super(HemorrhageNet, self).__init__()

    self.hemorrhageNet = ConvolutionNeuralNet()

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
    self.compute = self.scl_ctx.forward_function("severity", output_mapping=list(range(5)))

  def forward(self, x: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
    hemorrhage_distrs = [contour for contour in x]

    hemorrhages = torch.cat(tuple(hemorrhage_distrs), dim=1)

    processed_hemorrhages = self.hemorrhageNet(hemorrhages)

    result = self.compute(processed_hemorrhages)

    return result


# Train the model
# Assuming you have your dataset prepared with images and corresponding labels
train_images = "/data3/masseybr/train"
df = pd.read_csv('data3/masseybr/trainLabels.csv')
train_labels = df['level']
model.fit(train_images, train_labels, epochs=10, batch_size=32)

# Evaluate the model
# Assuming you have a separate test set for evaluation
test_images = "/data3/masseybr/train"
test_labels = df['level']
loss, accuracy = model.evaluate(test_images, test_labels)
print('Test accuracy:', accuracy)