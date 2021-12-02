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
- ### Creating DataLoaders:
    - Data Loaders convert the data into a set of batches making it easier to pass to the model
    - We import `DataLoader` from `torch.utils.data` and pass in our dataset with the batch size and number of other parameters.
    - We get different data loaders which we can now pass to the model for training, batch by batch.<br>
## Model:
- ### Selecting ResNet-34 as a pretrained model:
    - #### Residual Networks:
        - Basic Structure:<br>
        ![](https://github.com/AnityaGan9urde/ResNet34-on-Fishes/blob/main/images/Residual-Block.jpg)<br><br>
        - Architecture:<br>
    ![](https://github.com/AnityaGan9urde/ResNet34-on-Fishes/blob/main/images/ResNet.jpg)<br><br>
    - #### Code:
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
- ### Creating Helper functions: 
```python
def training_step(self, batch):
    images, labels = batch
    out = self(images)                  # Generate predictions
    loss = F.cross_entropy(out, labels)  # Calculate loss
    return loss
    
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)
  ```
## Training:
- ### Deploying to GPU:
    - Before starting the training we should always check for what device is available to train on.
    - Using the following code to check:
    ```python
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    ```
          
- ### Fitting the data on the model:
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
        # Training Phase
        model.train()
        train_losses = []
        lrs = []
        for batch in tqdm(train_loader):
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()

            # Gradient clipping
            if grad_clip:
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)

            optimizer.step()
            optimizer.zero_grad()

            # Record & update learning rate
            lrs.append(get_lr(optimizer))
            sched.step()

        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        result['lrs'] = lrs
        model.epoch_end(epoch, result)
        history.append(result)
    return history
  ```

## Testing:
- After testing the model on the test dataset, we get an accuracy of around **99.41%**.
## Results:
- ### Loss vs. No. of Epochs:<br>
![](https://github.com/AnityaGan9urde/ResNet34-on-Fishes/blob/main/images/loss.jpg)<br>
- ### Accuracy vs. No. of Epochs:<br>
![](https://github.com/AnityaGan9urde/ResNet34-on-Fishes/blob/main/images/accuracy.jpg)<br>
