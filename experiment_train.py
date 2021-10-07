from DCNN_training_benchmark.model import *
def train_model(model,# model_name, #df_train, criterion, optimizer, i_trials, 
                num_epochs, lr=args.lr, momentum=args.momentum, log_interval=100):
    model.to(device)
    if momentum == 0.:
        optimizer = optim.Adam(model.parameters(), lr=lr)#, betas=(beta1, beta2), amsgrad=amsgrad)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum) # to set training variables

    df_train = pd.DataFrame([], columns=['epoch', 'avg_loss', 'avg_acc', 'avg_loss_val', 'avg_acc_val', 'image_size', 'device_type']) 

    for epoch in range(num_epochs):
        loss_train = 0
        acc_train = 0
        for i, (images, labels) in enumerate(dataloaders['train']):
            images, labels = images.to(device), labels.to(device)   
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            loss_train += loss.item() * images.size(0)
            _, preds = torch.max(outputs.data, 1)
            acc_train += torch.sum(preds == labels.data)
            
        avg_loss = loss_train / dataset_sizes['train']
        avg_acc = acc_train / dataset_sizes['train']
           
        with torch.no_grad():
            loss_val = 0
            acc_val = 0
            for i, (images, labels) in enumerate(dataloaders['val']):
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                loss_val += loss.item() * images.size(0)
                _, preds = torch.max(outputs.data, 1)
                acc_val += torch.sum(preds == labels.data)
        
            avg_loss_val = loss_val / dataset_sizes['val']
            avg_acc_val = acc_val / dataset_sizes['val']        
        
        df_train.loc[epoch] = {'epoch':epoch, 'avg_loss':avg_loss, 'avg_acc':float(avg_acc),
                                  'avg_loss_val':avg_loss_val, 'avg_acc_val':float(avg_acc_val), 'image_size': image_size,
                                  'device_type':device.type}
        print(f"Epoch {epoch+1}/{num_epochs} : train= loss: {avg_loss:.4f} / acc : {avg_acc:.4f} - val= loss : {avg_loss_val:.4f} / acc : {avg_acc_val:.4f}")

    model.cpu()
    torch.cuda.empty_cache()
    return (model, df_train) #, i_trials)
criterion = nn.CrossEntropyLoss()       

# Training and saving the network

models_vgg = {}
opt = {}
#df_train = {}

models_vgg['vgg'] = torchvision.models.vgg16(pretrained=True)

# Downloading the model
for model_name in model_names:
    model_filenames[model_name] = model_path + model_name + '.pt'
    filename = f'results/{datetag}_{HOST}_train_{model_name}.json'

    models_vgg[model_name] = torchvision.models.vgg16(pretrained=True)
    # Freeze training for all layers
    # Newly created modules have require_grad=True by default
    for param in models_vgg[model_name].features.parameters():
        param.require_grad = False 

    if model_name == 'vgg16_lin':   
        num_features = models_vgg[model_name].classifier[6].out_features
        features = list(models_vgg[model_name].classifier.children())
        features.extend([nn.Linear(num_features, lay_)])
        models_vgg[model_name].classifier = nn.Sequential(*features)

    else : 
        num_features = models_vgg[model_name].classifier[6].in_features
        features = list(models_vgg[model_name].classifier.children())[:-1] # Remove last layer
        features.extend([nn.Linear(num_features, lay_)]) # Add our layer with 10 outputs
        models_vgg[model_name].classifier = nn.Sequential(*features) # Replace the model classifier     

    if os.path.isfile(model_filenames[model_name]):
        print("Loading pretrained model for..", model_name)
        #print("Resume_training : ", resume_training)

        if device.type == 'cuda':
            models_vgg[model_name].load_state_dict(torch.load(model_filenames[model_name])) #on GPU
        else:
            models_vgg[model_name].load_state_dict(torch.load(model_filenames[model_name], map_location=torch.device('cpu'))) #on CPU

    else:
        print("Re-training pretrained model...", model_filenames[model_name])
        since = time.time()

        p = 1 if model_name == 'vgg16_gray' else 0
        if model_name =='vgg16_scale':
            df_train = None
            for image_size_ in image_sizes: # starting with low resolution images 
                print(f"Traning {model_name}, image_size = {image_size_}, p (Grayscale) = {p}")
                (dataset_sizes, dataloaders, image_datasets, data_transforms) = datasets_transforms(image_size=image_size_, p=p)
                models_vgg[model_name], df_train_ = train_model(models_vgg[model_name], num_epochs=num_epochs)#, df_train[model_name], criterion, optimizer=opt[model_name], i_trials=i_trials, num_epochs=num_epochs) # need more epochs per scale
                if df_train is None:
                    df_train = df_train_ 
                else: 
                    df_train = df_train.append(df_train_)
        else :
            print(f"Traning {model_name}, image_size = {image_size}, p (Grayscale) = {p}")
            (dataset_sizes, dataloaders, image_datasets, data_transforms) = datasets_transforms(image_size=image_size, p=p)
            models_vgg[model_name], df_train = train_model(models_vgg[model_name], num_epochs=num_epochs)#, df_train[model_name], criterion, optimizer=opt[model_name], i_trials=i_trials, num_epochs=num_epochs)
        torch.save(models_vgg[model_name].state_dict(), model_filenames[model_name])
        df_train.to_json(filename)
        elapsed_time = time.time() - since
        print(f"Training completed in {elapsed_time // 60:.0f}m {elapsed_time % 60:.0f}s")
        print()
            
