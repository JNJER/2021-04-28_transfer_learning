
from DCNN_training_benchmark.init import *

# VGG-16 datasets initialisation
def datasets_transforms(image_size=args.image_size, p=0):
    data_transforms = {

        'train': transforms.Compose([
            transforms.Resize((int(image_size), int(image_size))),
            transforms.AutoAugment(), # https://pytorch.org/vision/master/transforms.html#torchvision.transforms.AutoAugment
            transforms.RandomGrayscale(p=p),
            transforms.ToTensor(),      # Convert the image to PyTorch Tensor data type.
            transforms_norm ]),

        'val': transforms.Compose([
            transforms.Resize((int(image_size), int(image_size))),
            transforms.AutoAugment(), # https://pytorch.org/vision/master/transforms.html#torchvision.transforms.AutoAugment
            transforms.RandomGrayscale(p=p),
            transforms.ToTensor(),
            transforms_norm ]),

        'test': transforms.Compose([
            transforms.Resize((int(image_size), int(image_size))),
            transforms.RandomGrayscale(p=p),
            transforms.ToTensor(),       
            transforms_norm ]),
    }

    image_datasets = {
        x: datasets.ImageFolder(
            paths[x], 
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

    return(dataset_sizes, dataloaders, image_datasets, data_transforms)

(dataset_sizes, dataloaders, image_datasets, data_transforms) = datasets_transforms(image_size=image_size)

for x in folder :
    print(f"Loaded {dataset_sizes[x]} images under {x}")
    
print("Classes: ")
class_names = image_datasets['train'].classes
print(image_datasets['train'].classes)
lay_ = len(os.listdir(paths['train']))

models = {}
opt = {}
scheduler = {}
df_train = {}

# Downloading the model
for name in train_modes:
    model_paths[name] = str(model_path+name+'.pt')
    models[name] = torchvision.models.vgg16(pretrained=True)
    df_train[name] = pd.DataFrame([], columns=['epoch', 'avg_loss', 'avg_acc', 'avg_loss_val', 'avg_acc_val', 'image_size', 'device', 'model']) 

    # Freeze training for all layers
    for param in models[name].features.parameters():
        param.require_grad = False 
        

    # Newly created modules have require_grad=True by default
    if name == 'vgg16_lin':   
        num_features = models[name].classifier[6].out_features
        features = list(models[name].classifier.children())
        features.extend([nn.Linear(num_features, lay_)])
        models[name].classifier = nn.Sequential(*features)
        opt[name] = optim.SGD(models[name].parameters(), lr=0.001, momentum=0.9) # to set training variables
        
    else : 
        num_features = models[name].classifier[6].in_features
        features = list(models[name].classifier.children())[:-1] # Remove last layer
        features.extend([nn.Linear(num_features, lay_)]) # Add our layer with 10 outputs
        models[name].classifier = nn.Sequential(*features) # Replace the model classifier     
        opt[name] = optim.SGD(models[name].parameters(), lr=0.001, momentum=0.9) # to set training variables

criterion = nn.CrossEntropyLoss()       
print("Loading pretrained model..")
print("Resume_training : ", resume_training)


# Loadind a previous model
if resume_training:
    for name in model_paths:
        try:
            models[name].load_state_dict(torch.load(model_paths[name])) #on GPU
        except:
            models[name].load_state_dict(torch.load(model_paths[name], map_location=torch.device('cpu'))) #on CPU
            
models['vgg'] = torchvision.models.vgg16(pretrained=True)
{print("Loaded : " , x) for x in models}
