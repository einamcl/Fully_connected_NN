import torch
import matplotlib.pyplot as plt
from torchvision import datasets as dsets
import torchvision.transforms as transforms
import numpy as np
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.1307, 0.3081)
    ])
num_epochs = 5
batch_size = 100
learning_rate = 0.1

# Image Preprocessing

train_dataset = dsets.MNIST(root='./data/',
                               train=True,
                               transform=transform,
                               download=True)

test_dataset = dsets.MNIST(root='./data/',
                              train=False,
                              transform=transform,
                              download=True)
# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

print(list(train_dataset.train_data.size()))
print(train_dataset.train_data.float().mean()/255)
print(train_dataset.train_data.float().std()/255)

# Neural Network
def tanh(t):
    # t: torhc tensor
    # return tensor after applying tanh function
    return torch.div(torch.exp(t) - torch.exp(-t), torch.exp(t) + torch.exp(-t))


def tanhPrime(t):
    # derivative of tanh
    # t: tanh output
    return 1 - t * t


def softmax(t):
    # wrapper fucntion for softmax method
    # return tensor where all rows undergone softmax
    x = torch.zeros(t.shape)
    for index, item in enumerate(t):
        x[index] = row_softmax(item)
    return x


def row_softmax(a):
    # returns softmax func on torch tensor
    max_a = torch.max(a).item()
    a = a - max_a
    return torch.exp(a) / sum(torch.exp(a))


def one_hot(Y, size):
    # changes torch tensor into one hot tensor
    # Y: torch tensor vector
    # size: num of classes
    one_hot_Y = torch.zeros((Y.size(0), size))
    for i, val in enumerate(Y):
        one_hot_Y[i][int(val)] = 1
    return one_hot_Y


def make_p(y):
    # creates prediction vector from softmax
    p = torch.argmax(y, 1)
    return p


class NN:
    def __init__(self, input_size=784, output_size=10, hidden_size=10):
        # parameters
        self.inputSize = input_size
        self.outputSize = output_size
        self.hiddenSize = hidden_size

        # weights
        self.W1 = torch.randn(self.inputSize, self.hiddenSize)
        self.W1 = self.W1 / torch.norm(self.W1)
        self.b1 = torch.zeros(self.hiddenSize)

        self.W2 = torch.randn(self.hiddenSize, self.outputSize)
        self.W2 = self.W2 / torch.norm(self.W2)
        self.b2 = torch.zeros(self.outputSize)

    def set_weight(self, weights):
        self.W1 = weights["w1"]
        self.W2 = weights["w2"]
        self.b1 = weights["b1"]
        self.b2 = weights["b2"]

    def save_weights(self):
        torch.save({"w1": self.W1, "w2": self.W2, "b1": self.b1, "b2": self.b2}, "w_q1.pkl")

    def forward(self, X):
        self.z1 = torch.matmul(X, self.W1) + self.b1

        self.h = tanh(self.z1)
        self.z2 = torch.matmul(self.h, self.W2) + self.b2
        return softmax(self.z2)

    def backward(self, X, y, y_hat, lr=.1):
        batch_size = y.size(0)

        y = one_hot(y, self.outputSize)
        dl_dz2 = (y_hat - y) / batch_size

        dl_dh = torch.matmul(dl_dz2, torch.t(self.W2))
        dl_dz1 = dl_dh * tanhPrime(self.h)

        self.W1 -= lr * torch.matmul(torch.t(X), dl_dz1)
        self.b1 -= lr * torch.matmul(torch.t(dl_dz1), torch.ones(batch_size))
        self.W2 -= lr * torch.matmul(torch.t(self.h), dl_dz2)
        self.b2 -= lr * torch.matmul(torch.t(dl_dz2), torch.ones(batch_size))

    def train(self, X, y, lr=0.1):
        # forward + backward pass for training
        o = self.forward(X)
        self.backward(X, y, o, lr)

    def predict(self, X):
        # reaturns prediction of batch
        x = self.forward(X)
        a = make_p(x)
        return a

    def cross_ent_loss(self, y, y_hat):
        # calculate cross entropy loss
        # return loss
        loss = 0
        for i in range(y.shape[0]):
            for j in range(y.shape[1]):
                if y[i][j].item() != 0:
                    inner = (y[i][j] * torch.log(y_hat[i][j] + 0.000000001))
                    loss += inner.item()
        return -loss


# Train NN:

# if __name__ == "__main__":
train_batch = []
train_accuracy = []
test_accuracy = 0
test_accuracy_list = []
total_size = 0
t_size = 0
accuracy = 0
loss = 0
num_epochs = 5
learning_rate = 0.1
model = NN(784, 10, 200)
for epoch in range(num_epochs):

    if epoch > 2:
        learning_rate = 0.01
    if epoch > 4:
        learning_rate = 0.001

    for i, (images, labels) in enumerate(train_loader):
        images = images.view(-1, 784)
        y_hat = model.forward(images)
        model.backward(images, labels, y_hat, learning_rate)

        prediction = make_p(y_hat)
        accuracy += (prediction == labels).sum().item()
        total_size += len(labels)

    train_accuracy.append(accuracy / total_size)

    for i, (images_t, labels_t) in enumerate(test_loader):
        t_size += len(labels_t)

        images_t = images_t.view(-1, 784)
        y_hatt = model.forward(images_t)
        prediction_t = make_p(y_hatt)

        test_accuracy += (prediction_t == labels_t).sum().item()

    test_accuracy_list.append(test_accuracy / t_size)

plt.plot([x for x in range(1, num_epochs + 1)], test_accuracy_list, c='gold', label="Test accuracy")

plt.plot([x for x in range(1, num_epochs + 1)], train_accuracy, c='b', label="Train accuracy")

plt.title("Train & Test accuracy")
plt.xlabel("Epoch")
plt.ylabel("Cumulative accuracy")
plt.legend()
plt.show()
