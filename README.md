# Developing a Neural Network Classification Model

## AIM
To develop a neural network classification model for the given dataset.

## THEORY
An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model
<img width="720" height="637" alt="image" src="https://github.com/user-attachments/assets/d01441f7-b766-4808-8cd1-7dfb94882674" />


## DESIGN STEPS
## Step 1: Load and Preprocess Data
Load the dataset, remove irrelevant columns (ID), handle missing values, encode categorical features using Label Encoding, and encode the target class (Segmentation).

## Step 2: Feature Scaling and Data Split
Split the dataset into training and testing sets, then normalize the input features using StandardScaler for better neural network performance.

# Step 3: Convert Data to PyTorch Tensors
Convert the scaled training and testing data into PyTorch tensors and create DataLoader objects for batch-wise training and evaluation.

# Step 4: Define the Neural Network Model
Design a feedforward neural network with multiple fully connected layers and ReLU activation functions, ending with an output layer for multi-class classification.

# Step 5: Train the Model
Train the model using CrossEntropyLoss and Adam optimizer by performing forward propagation, loss calculation, backpropagation, and weight updates over multiple epochs.

# Step 6: Evaluate and Predict
Evaluate the trained model on test data using accuracy, confusion matrix, and classification report, and perform prediction on a sample input.


## PROGRAM

### Name: Rabin R

### Register Number: 212224230213

```python
class PeopleClassifier(nn.Module):
    def __init__(self, input_size):
        super(PeopleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size,32)
        self.fc2 = nn.Linear(32,16)
        self.fc3 = nn.Linear(16,8)
        self.fc4 = nn.Linear(8,4)

    def forward(self, x):
      x = F.relu(self.fc1(x))
      x = F.relu(self.fc2(x))
      x = F.relu(self.fc3(x))
      x= self.fc4(x)
      return x
        
model =PeopleClassifier(input_size=X_train.shape[1])
criterion =nn.CrossEntropyLoss()
optimizer =optim.Adam(model.parameters(),lr=0.001)

def train_model(model, train_loader, criterion, optimizer, epochs):
  model.train()
  for epoch in range(epoch):
    for inputs, labels in train_loader:
      optimizer.zero_grad()
      outputs = model(inputs)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

```

### Dataset Information
<img width="1274" height="250" alt="image" src="https://github.com/user-attachments/assets/c8adfdb0-5f3f-40d0-b453-421277679a6e" />


### OUTPUT
<img width="805" height="223" alt="image" src="https://github.com/user-attachments/assets/9f9fe1a5-fede-4539-9035-be0cc5c9674e" />


## Confusion Matrix

<img width="887" height="567" alt="image" src="https://github.com/user-attachments/assets/8f2f1454-bab5-4091-b1bb-06a90623d42b" />


## Classification Report

<img width="1089" height="439" alt="image" src="https://github.com/user-attachments/assets/701198af-3b0f-4bdc-be85-7abded4f4b99" />

### New Sample Data Prediction

<img width="671" height="95" alt="image" src="https://github.com/user-attachments/assets/d1594818-7ab2-473a-aa95-2bba4af97ce1" />


## RESULT
This program has been executed successfully.
