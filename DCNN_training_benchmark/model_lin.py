
from DCNN_training_benchmark.init import *

# VGG-16 datasets initialisation
data_transforms = {
    
    'train': transforms.Compose([
        transforms.RandomResizedCrop(int(image_size-32)), # random crop of the image to 224x224
        transforms.RandomHorizontalFlip(), # randomly horizontal flip
        #transforms.Grayscale(3),      # convert the image in grayscale
        transforms.ToTensor(),      # Convert the image to PyTorch Tensor data type.
        transforms.Normalize(       # Normalize the image by adjusting
        mean=[0.485, 0.456, 0.406],  #  its average and
        std=[0.229, 0.224, 0.225]    #  its standard deviation at the specified values.
        )]),
    
    'val': transforms.Compose([
        transforms.Resize(int(image_size)),
        transforms.CenterCrop(int(image_size)-20),
        #transforms.Grayscale(3),      
        transforms.ToTensor(),
        transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  
        std=[0.229, 0.224, 0.225]    
        )]),
    
    'test': transforms.Compose([
        transforms.CenterCrop(int(image_size-20)),  # Crop the image to (image_size-20) x (image_size-20) pixels around the center.
        transforms.ToTensor(),       
        transforms.Normalize(        
        mean=[0.485, 0.456, 0.406],  
        std=[0.229, 0.224, 0.225]    
        )]),
}

image_datasets = {
    x: datasets.ImageFolder(
        path[x], 
        transform=data_transforms[x]
    )
    for x in folder
}

dataloaders = {
    x: torch.utils.data.DataLoader(
        image_datasets[x], batch_size=8,
        shuffle=True, num_workers=4
    )
    for x in folder
}

dataset_sizes = {x: len(image_datasets[x]) for x in folder}

for x in folder :
    print("Loaded {} images under {}".format(dataset_sizes[x], x))
    
print("Classes: ")
class_names = image_datasets['train'].classes
print(image_datasets['train'].classes)

# Downloading the model
models = {} # get model's names
models['vgg16'] = torchvision.models.vgg16(pretrained=True)
#models['alex'] = torchvision.models.alexnet(pretrained=True)
models['vgg'] = torchvision.models.vgg16(pretrained=True)
#models['mob'] = torchvision.models.mobilenet_v2(pretrained=True)
#models['res'] = torchvision.models.resnext101_32x8d(pretrained=True)

# Freeze training for all layers
for param in models['vgg16'].features.parameters():
    param.require_grad = False

# Newly created modules have require_grad=True by default
lay_ = len(os.listdir(path['train']))
num_features = models['vgg16'].classifier[6].out_features
features = list(models['vgg16'].classifier.children())
features.extend([nn.Linear(num_features, lay_)]) # Add our layer with 10 outputs
models['vgg16'].classifier = nn.Sequential(*features) # Replace the model classifier
print(models['vgg16'])
models['vgg16_gray'] = models['vgg16']
resume_training = False

# Loadind a previous model
if resume_training:
    print("Loading pretrained model..")
    try:
        models['vgg16'].load_state_dict(torch.load(model_path)) #on GPU
    except:
        models['vgg16'].load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))) #on CPU
    print("Loaded!")
    
# to set training variables     
criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.SGD(models['vgg16'].parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

# put the models to the device
for name in models.keys():
    models[name].to(device)
