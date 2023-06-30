# Import dependencies
import os
import torch 
from PIL import Image
from torch import nn, save, load
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# Get data 
train = datasets.MNIST(root="data", download=True, train=True, transform=ToTensor())
dataset = DataLoader(train, 32)
#1,28,28 - classes 0-9

# Image Classifier Neural Network
class ImageClassifier(nn.Module): 
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, (3,3)), 
            nn.ReLU(),
            nn.Conv2d(32, 64, (3,3)), 
            nn.ReLU(),
            nn.Conv2d(64, 64, (3,3)), 
            nn.ReLU(),
            nn.Flatten(), 
            nn.Linear(64*(28-6)*(28-6), 10)  
        )

    def forward(self, x): 
        return self.model(x)

# Instance of the neural network, loss, optimizer 
clf = ImageClassifier().to('cpu')
opt = Adam(clf.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss() 

# Training flow or loading weights
 
if __name__ == "__main__": 
    if( os.path.isfile('model_state.pt') ):                            # Loading weights or Training then saving weights
        print("Saved weights found and loaded.")
        with open('model_state.pt', 'rb') as f: 
            clf.load_state_dict(load(f))
    else:        
        for epoch in range(10): # train for 10 epochs
            for batch in dataset: 
                X,y = batch 
                X, y = X.to('cpu'), y.to('cpu') 
                yhat = clf(X) 
                loss = loss_fn(yhat, y) 

                # Apply backprop 
                opt.zero_grad()
                loss.backward() 
                opt.step() 

            print(f"Epoch:{epoch} loss is {loss.item()}")
        
        with open('model_state.pt', 'wb') as f: 
            save(clf.state_dict(), f)
  

    # User inputs a drawing
    drawingPathInit = os.path.dirname(os.path.abspath(__file__))+ "\\28by28.jpg"
    
    while(1):
        
        try:
            drawingPath = input("\nPlease enter the path to a 28 by 28 image:")
            if(drawingPath == ""):
                drawingPath=drawingPathInit
                print("Default used: ",drawingPathInit)
            img = Image.open(drawingPath) 
            img_tensor = ToTensor()(img).unsqueeze(0).to('cpu')

            print(torch.argmax(clf(img_tensor)))
        except:
            print("Error")