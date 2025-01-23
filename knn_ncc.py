import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.metrics import accuracy_score
import numpy as np
import time


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

# Load CIFAR-10 dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Create data loaders
train_loader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset), shuffle=True)
test_loader = torch.utils.data.DataLoader(testset, batch_size=len(testset), shuffle=False)

#Flatten the X data , and normilize them
X_train = trainset.data.reshape((trainset.data.shape[0], -1)) / 255.0  
y_train = np.array(trainset.targets)
X_test = testset.data.reshape((testset.data.shape[0], -1)) / 255.0   
y_test = np.array(testset.targets)

#normilize between [-1 , 1]
X_train = (X_train - 0.5) / 0.5
X_test = (X_test - 0.5) / 0.5


# Nearest Class Centroid Algorithm
ncc = NearestCentroid()
start_training_time = time.time()
ncc.fit(X_train, y_train)
finish_training_time = time.time()

start_time = time.time()
predictions = ncc.predict(X_test)
finish_time = time.time()

accuracy = accuracy_score(predictions, y_test)
print(f"NCC Classifier Accuracy on CIFAR-10 Test Data: {accuracy * 100:.2f}%")
print(f"Training took {finish_training_time-start_training_time} to finish")
print(f"Algorithm took {finish_time-start_time} to predict \n")


# K Nearest Kneighbor Algorithm, for k = 1
knn = KNeighborsClassifier(n_neighbors=1)
start_training_time = time.time()
knn.fit(X_train, y_train)
finish_training_time = time.time()

start_time = time.time()
predictions = knn.predict(X_test)
finish_time = time.time()

accuracy = accuracy_score(predictions, y_test)
print(f"KNN Classifier for k=1, Accuracy on CIFAR-10 Test Data: {accuracy * 100:.2f}%")
print(f"Training took {finish_training_time-start_training_time} to finish")
print(f"Algorithm took {finish_time-start_time} to predict \n")

# K Nearest Kneighbor Algorithm, for k = 3
knn = KNeighborsClassifier(n_neighbors=3)
start_training_time = time.time()
knn.fit(X_train, y_train)
finish_training_time = time.time()

start_time = time.time()
predictions = knn.predict(X_test)
finish_time = time.time()

accuracy = accuracy_score(predictions, y_test)
print(f"KNN Classifier for k=3, Accuracy on CIFAR-10 Test Data: {accuracy * 100:.2f}%")
print(f"Training took {finish_training_time-start_training_time} to finish")
print(f"Algorithm took {finish_time-start_time} to predict \n")

