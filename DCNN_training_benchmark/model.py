
from DCNN_training_benchmark.init import *
from sklearn.metrics import multilabel_confusion_matrix
import seaborn as sns
import sklearn.metrics
from scipy import stats
from scipy.special import logit, expit

# VGG-16 datasets initialisation
def datasets_transforms(image_size=args.image_size, p=0, num_workers=1, batch_size=8):
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((int(image_size), int(image_size))),
            transforms.AutoAugment(), # https://pytorch.org/vision/master/transforms.html#torchvision.transforms.AutoAugment
            transforms.RandomGrayscale(p=p),
            transforms.ToTensor(),      # Convert the image to pyTorch Tensor data type.
            transforms_norm ]),

        'val': transforms.Compose([
            transforms.Resize((int(image_size), int(image_size))),
            transforms.AutoAugment(), # https://pytorch.org/vision/master/transforms.html#torchvision.transforms.AutoAugment
            transforms.RandomGrayscale(p=p),
            transforms.ToTensor(),      # Convert the image to pyTorch Tensor data type.
            transforms_norm ]),

        'test': transforms.Compose([
            transforms.Resize((int(image_size), int(image_size))),
            transforms.RandomGrayscale(p=p),
            transforms.ToTensor(),      # Convert the image to pyTorch Tensor data type.
            transforms_norm ]),
    }

    image_datasets = {
        folder: datasets.ImageFolder(
            paths[folder], 
            transform=data_transforms[folder]
        )
        for folder in folders
    }

    dataloaders = {
        folder: torch.utils.data.DataLoader(
            image_datasets[folder], batch_size=batch_size,
            shuffle=True, num_workers=num_workers
        )
        for folder in folders
    }

    dataset_sizes = {folder: len(image_datasets[folder]) for folder in folders}

    return(dataset_sizes, dataloaders, image_datasets, data_transforms)

(dataset_sizes, dataloaders, image_datasets, data_transforms) = datasets_transforms(image_size=image_size)

for folder in folders :
    print(f"Loaded {dataset_sizes[folder]} images under {folder}")
    
print("Classes: ")
class_names = image_datasets['train'].classes
print(image_datasets['train'].classes)
lay_ = len(os.listdir(paths['train']))

models = {}
opt = {}
scheduler = {}
df_train = {}

# Downloading the model
for model_name in model_names:
    model_paths[model_name] = model_path + model_name + '.pt'
    models[model_name] = torchvision.models.vgg16(pretrained=True)
    df_train[model_name] = pd.DataFrame([], columns=['epoch', 'avg_loss', 'avg_acc', 'avg_loss_val', 'avg_acc_val', 'image_size', 'device', 'model']) 

    # Freeze training for all layers
    for param in models[model_name].features.parameters():
        param.require_grad = False 
        

    # Newly created modules have require_grad=True by default
    if model_name == 'vgg16_lin':   
        num_features = models[model_name].classifier[6].out_features
        features = list(models[model_name].classifier.children())
        features.extend([nn.Linear(num_features, lay_)])
        models[model_name].classifier = nn.Sequential(*features)
        opt[model_name] = optim.SGD(models[model_name].parameters(), lr=0.001, momentum=0.9) # to set training variables
        
    else : 
        num_features = models[model_name].classifier[6].in_features
        features = list(models[model_name].classifier.children())[:-1] # Remove last layer
        features.extend([nn.Linear(num_features, lay_)]) # Add our layer with 10 outputs
        models[model_name].classifier = nn.Sequential(*features) # Replace the model classifier     
        opt[model_name] = optim.SGD(models[model_name].parameters(), lr=0.001, momentum=0.9) # to set training variables

criterion = nn.CrossEntropyLoss()       
print("Loading pretrained model..")
print("Resume_training : ", resume_training)


# Loadind a previous model
if resume_training:
    for model_name in model_paths:
        try:
            models[model_name].load_state_dict(torch.load(model_paths[model_name])) #on GPU
        except:
            models[model_name].load_state_dict(torch.load(model_paths[model_name], map_location=torch.device('cpu'))) #on CPU
            
models['vgg'] = torchvision.models.vgg16(pretrained=True)

{print("Loaded : " , model) for model in models.keys()}
