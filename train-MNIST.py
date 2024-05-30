import time
import torch
from torch import nn # Neuronenschicten
from torch.utils.data import DataLoader # Daten schneller als for-Schleife laden
from torchvision import datasets # Die MNIST-Dateien laden
from torchvision.transforms import ToTensor # Transformation der Bilddaten in mehrdimensionale Arrays ("Tensoren")

training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

batch_size = 64

train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for input_data, labels in test_dataloader:
    print(f"Shape of input_data [N, C, H, W]: {input_data.shape}")
    print(f"Shape of labels: {labels.shape} (type: {labels.dtype})")
    break

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


# Netzwerk definieren
class FeedForwardNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512), # Eingabeschicht, Bilder bestehen aus 28 x 28 Pixeln
            nn.ReLU(),             # Aktivierungsfunktion
            nn.Linear(512, 512),   # hidden layer
            nn.ReLU(),             # Aktivierungsfunktion
            nn.Linear(512, 10)     # Ausgabeschicht, 10 Neuronen entsprechen 10 Ziffern
                                   # Auagabeschicht benötigt keine Aktivierungsfunktion
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


feed_forward_model = FeedForwardNetwork().to(device)
print(feed_forward_model)

# Loss-Funktion definieren
# Ausgabe: kleiner Wert, wenn Netzwerk gut funktioniert, großer Wert sonst
cross_entropy_loss_fn = nn.CrossEntropyLoss()
sgd_optimizer = torch.optim.SGD(feed_forward_model.parameters(), lr=0.05) # die Lernrate lr ist typischerweise der wichtigste Hyperparamter zur Steurung des Lernprozesses
                                                                          # lr zu klein => Training dauert zu lang
                                                                          # lr zu groß => optimale Lösung kann verpasst werden


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss = loss.item()
            current = (batch + 1) * len(X)
            print(f"loss: {loss:>.3f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error:  Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8.3f} \n")


start = time.time()
epochs = 15
for t in range(epochs):
    print(f"Epoch {t+1} (lr: {sgd_optimizer.state_dict()['param_groups'][0]['lr']})\n-------------------------------")
    train(train_dataloader,
          feed_forward_model,
          cross_entropy_loss_fn,
          sgd_optimizer)
    test(test_dataloader,
         feed_forward_model,
         cross_entropy_loss_fn)
end = time.time()
print(f"Done, elapsed {(end - start):.1f} sec")
