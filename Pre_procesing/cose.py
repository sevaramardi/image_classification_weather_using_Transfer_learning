import torch
import numpy as np
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:',device)

file_path = '.weather/dataset/'
classes = ['dew', 'fogsmog', 'frost', 'glaze', 'hail', 'lightning' , 'rain', 'rainbow', 'rime', 'sandstorm', 'snow']

def getting_data(directory, size, label):
    images = []
    labels = []

    for file in os.listdir(directory):
        if file.endswith('.jpg'):
            img = Image.open(os.path.join(directory,file)) #(675, 950, 3)
            #plt.imshow(img)
            #plt.show()
            img = np.array(img.resize(size))
            if img.shape == (224,224,3):
                images.append(img)
                labels.append(label)
                
    return np.array(images),np.array(labels)   
#train 
train_images = []
train_labels = []
for lab,pclass in enumerate(classes):
    train_path = os.path.join(file_path,pclass)
    images, labels = getting_data(train_path, size=(224,224),label = lab)
    train_images.append(images) #(691, 224, 224, 3) for one
    train_labels.append(labels)

train_data_images = []
train_data_labels = []

test_data_labels = []
test_data_images = []



for i in range(11):
    #print(f'{classes[i]}: {len(train_images[i])}')
    #dividing for test 20% data for train 80%
    per = int(len(train_images[i])*0.8) #we are taking 80% for train

    train_data_images.append(train_images[i][:per])
 
    train_data_labels.append(train_labels[i][:per])

    test_data_images.append(train_images[i][per:])
    test_data_labels.append(train_labels[i][per:])

train_data_images = np.concatenate(train_data_images, axis=0)
train_data_labels = np.concatenate(train_data_labels, axis=0)
test_data_images = np.concatenate(test_data_images, axis=0)
test_data_labels = np.concatenate(test_data_labels, axis=0)

class CustomDataset():
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# Define transformations
transform = transforms.Compose([
    transforms.ToPILImage(),  # Convert numpy array to PIL Image
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.229, 0.224, 0.225])
])

# Create datasets
train_dataset = CustomDataset(train_data_images, train_data_labels, transform=transform)
test_dataset = CustomDataset(test_data_images, test_data_labels, transform=transform)

#data loader
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


#vizualization
# data_iter = iter(train_loader)
# images,labels = next(data_iter)

# def imshow(img):
#     mean=[0.5, 0.5, 0.5]
#     std=[0.229, 0.224, 0.225]
#     img = img.numpy().transpose((1,2,0))
#     img = img*std + mean
#     img = np.clip(img,0,1)
#     plt.imshow(img)

# plt.figure(figsize=(10,10))
# for i in range(6):
#     plt.subplot(2,3,i+1)
#     imshow(images[i])
#     plt.title(classes[labels[i]])
# plt.show()
