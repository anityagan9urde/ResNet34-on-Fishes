# ResNet34-on-Fishes
- In this notebook, I have trained a ResNet-34 model using PyTorch on GPUs for classifying 9 types of fishes through their images.

## Dataset:
- ### Downloading:
  - The dataset was taken from Kaggle(Link: https://www.kaggle.com/crowww/a-large-scale-fish-dataset).
  - It contains two folders for each category of fish - images folder and ground truth images folder.
  - There are a total of 9 categories of fish in the dataset.
- ### Transforms:
  - I used an image manipulation library called Albumentations to augment the images.
  - Augmentations are done so that the model receives more diverse data and it is less likely to overfit to the data.
  - I resized, flipped(horizontal and vertical), and normalized the training images using the `.compose()` method which collectively applies all the transformations on the images at once.
  - The validation images were simply resized and normalized as they do not require such transformations.
- ### Generating the Dataset:
    - I created a class called `FishDataset()` which imports from `torch.utils.data.Dataset`.  
    - It takes in images, labels and transforms and gives out transformed images and labels based on the type of transform applied when calling.
    - These datasets when created can then be passed to the DataLoader.
    
```python
class FishDataset(torch.utils.data.Dataset):

    def __init__(self, images: list, labels: list, transform=None):
        super().__init__()
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self, ):
        return len(self.labels)

    def __getitem__(self, index):
        input_image = self.images[index]
        label = self.labels[index]
        image = np.array(Image.open(input_image).convert("RGB"))

        if self.transform is not None:
            image = self.transform(image=image)["image"]

        return image, label
```      
- ### Iterating through the dataset using DataLoaders:
    - Data Loaders convert the data into a set of batches making it easier to pass to the model.
    - They also let us shuffle the data and load it parallely using multiprocessing workers.
    - We import `DataLoader` from `torch.utils.data` and pass in our dataset with the batch size and number of other parameters.
    - We get different data loaders which can now be passed to the model for training in batches.<br>
## Model:
- ### Selecting ResNet-34 as a pretrained model:
    - #### Residual Networks:
        - ResNets or Residual Networks are a type of neural networks where some layers are skipped while training and the performance of the network is improved drastically.
        - They are used as a method to learn the identity function for a network (which is when the output equals the input).
        - Basic Structure of a ResNet:<br>
        ![](https://github.com/AnityaGan9urde/ResNet34-on-Fishes/blob/main/images/Residual-Block.jpg)<br>
        - ResNets when used with CNNs have been shown to perform much better than simply using a CNN. The problem being that as the number of layers increase the problem of vanishing/exploding gradients occur. 
        - ResNets have been shown to negate that problem as they directly skip some layers hence leading to fewer layers being trained at the initial stages of training and then gradually adjusts the skip connections to train the entire network. Hence, removing the problem of vanishing gradients as there are fewer layers to train and leading to a more better model.
        - Architecture of a ResNet as compared to simple neural networks:<br>
        ![](https://github.com/AnityaGan9urde/ResNet34-on-Fishes/blob/main/images/ResNet.jpg)<br>
        - The skip connections within layers in the third diagram is the basic idea behind ResNets and it helps with improving the performance of any deep layered neural network model. 
    - #### Code:
      - To import a Pretrained ResNet-34 in your code use the below command and replace the last layer with the required number of classes.
      - ResNet-34 was selected as it is much lighter and can work more efficiently for this simple dataset with just nine classes.
        ```python
        class FishModel(nn.Module):
    
            def __init__(self, num_classes, pretrained=True):
                super().__init__()
                self.network = models.resnet34(pretrained=pretrained)
                self.network.fc = nn.Linear(self.network.fc.in_features, num_classes)

            def forward(self, xb):
                return self.network(xb)
                :
                :
          ```
- ### Creating Some Helper functions: 
```python
# A training step function which takes the predictions for a batch of images and calculates the loss
def training_step(self, batch):
    images, labels = batch
    out = self(images)                  # Generate predictions
    loss = F.cross_entropy(out, labels)  # Calculate loss
    return loss
    
# A function to evaluate on the validation data and get performance scores
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)
  ```
## Training:
- ### Deploying to GPU:
    - GPUs are used for deep learning training because of their faster parallel computation and matrix manipulation abilities. Hence, we will train our model on a GPU.
    - Using the following code to check the availability of a GPU:
   ```python
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    ```
          
- ### Fitting the data on the model:
    - A function to fit the data to the model, train it for any number of epochs, backpropagate errors and evaluating the model to return the scores.
```python 
def fit_one_cycle(epochs, max_lr, model, train_loader, val_loader,
                  weight_decay=0, grad_clip=None, opt_func=torch.optim.SGD):
    torch.cuda.empty_cache()
    history = []

    # Set up custom optimizer with weight decay
    optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
    # Set up one-cycle learning rate scheduler
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs, steps_per_epoch=len(train_loader))

    for epoch in range(epochs):
        model.train()                               #Training the model for a single epoch
        train_losses = []
        lrs = []
        for batch in tqdm(train_loader):            #Calculating the loss for each batch
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()                         #Calculating the gradients to update the weights

            if grad_clip:                           #Gradient clipping
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)

            optimizer.step()                        #Moving down the slope of loss function
            optimizer.zero_grad()                   #Zero out gradients to be updated for the next step

            lrs.append(get_lr(optimizer))           # Record & update learning rate
            sched.step()

        result = evaluate(model, val_loader)        #Evaluating the model on the validation data
        result['train_loss'] = torch.stack(train_losses).mean().item()
        result['lrs'] = lrs
        model.epoch_end(epoch, result)
        history.append(result)
    return history        
  ```
  - The model starts training when the above function is called as such:
  ```python
  history += fit_one_cycle(epochs, max_lr, model, train_dl, valid_dl, 
                         grad_clip=grad_clip, 
                         weight_decay=weight_decay, 
                         opt_func=opt_func)  
```
  - Training is finished when the model trains for the number of epochs which I have selected to be six:
  ```
  0%|          | 0/54 [00:00<?, ?it/s]

Epoch [0],last_lr: 0.00596, train_loss: 0.38011255860328674, val_loss: 578.192138671875, val_acc: 0.10717707872390747

  0%|          | 0/54 [00:00<?, ?it/s]

Epoch [1],last_lr: 0.00994, train_loss: 0.3608550727367401, val_loss: 25.896209716796875, val_acc: 0.2486458271741867

  0%|          | 0/54 [00:00<?, ?it/s]

Epoch [2],last_lr: 0.00812, train_loss: 0.24963808059692383, val_loss: 1.502174735069275, val_acc: 0.6577916741371155

  0%|          | 0/54 [00:00<?, ?it/s]

Epoch [3],last_lr: 0.00463, train_loss: 0.0408831387758255, val_loss: 0.1541007161140442, val_acc: 0.9476354122161865

  0%|          | 0/54 [00:00<?, ?it/s]

Epoch [4],last_lr: 0.00133, train_loss: 0.008193316869437695, val_loss: 0.14649251103401184, val_acc: 0.9397916793823242

  0%|          | 0/54 [00:00<?, ?it/s]

Epoch [5],last_lr: 0.00000, train_loss: 0.002655074931681156, val_loss: 0.011105048470199108, val_acc: 0.9973958134651184
CPU times: user 58.3 s, sys: 22.5 s, total: 1min 20s
Wall time: 7min 59s
  ```

## Testing:
- The model can now be tested on the Test dataset using the Test Dataloader.
- After testing the model, we get an accuracy of around **99.41%**.
- Function to test random images:
```python
def predict_image(image):
    xb = to_device(image.unsqueeze(0), device) #Sending the images to the device i.e. GPU
    yb = model(xb)                             #Generating predictions
    _, preds  = torch.max(yb, dim=1)
    return classes[preds[0].item()]
```
- Selecting random images and testing the model:
```python
img, label = test_ds[100] #or test_df[500]
plt.imshow(img.permute(1, 2, 0))
print('Label:', classes[label], ', Predicted:', predict_image(img))
```
- Output:
> Label: Red Sea Bream, Predicted: Red Sea Bream<br> | Label: Gilt-Head Bream , Predicted: Gilt-Head Bream
> ------------|-------------
>![](https://github.com/AnityaGan9urde/ResNet34-on-Fishes/blob/main/images/test_fish.png) | ![](https://github.com/AnityaGan9urde/ResNet34-on-Fishes/blob/main/images/test_fish1.png)

## Results:
Loss vs. No. of Epochs| Accuracy vs. No. of Epochs
---------|---------
![](https://github.com/AnityaGan9urde/ResNet34-on-Fishes/blob/main/images/loss.jpg)|![](https://github.com/AnityaGan9urde/ResNet34-on-Fishes/blob/main/images/accuracy.jpg)

- We can clearly see that the loss decreases as the number of epochs increases. Even though it reaches the highest point on the first epoch, as it learns more the loss goes to almost zero.
- Similarly, for the accuracy, the model gives out a lower accuracy at the beginning of the training but then goes higher and reaches to almost 1.0.
## Conclusion:
- ResNets have been shown to perform better when deep neural networks are involved which can be seen by the accuracy generated by this model.
- We can use a more complicated model such as ResNet-50 but it would be excessive in this case as I have achieved an accuracy more than 99% on test set.
- Also, PyTorch can be a very powerful deep learning framework to build such models as they give a lot of flexibility and much better control on what actually happens behind the curtains of deep learning.
