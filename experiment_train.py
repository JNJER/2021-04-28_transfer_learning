
from DCNN_training_benchmark.model import *

# Training function
def train_model(model, df_train, criterion, optimizer, i_trials, num_epochs=10):
    model.to(device)
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    avg_loss = 0
    avg_acc = 0
    avg_loss_val = 0
    avg_acc_val = 0
    
    train_batches = len(dataloaders['train'])
    val_batches = len(dataloaders['val'])
    
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print('-' * 10)
        
        loss_train = 0
        loss_val = 0
        acc_train = 0
        acc_val = 0
        
        model.train(True)
        
        for i, data in enumerate(dataloaders['train']):
            if i % 100 == 0:
                print(f"\rTraining batch {i}/{train_batches}", end='', flush=True)
                
            # Use all the training dataset
            if i >= train_batches :
                break
                
            inputs, labels = data
            with torch.no_grad():
                inputs, labels = inputs.to(device), labels.to(device)   
            
            optimizer.zero_grad()

            outputs = model(inputs)

            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            loss_train += loss.item() * inputs.size(0)
            acc_train += torch.sum(preds == labels.data)

            del inputs, labels, outputs, preds
            torch.cuda.empty_cache()
            
        print()
        avg_loss = loss_train / dataset_sizes['train']
        avg_acc = acc_train / dataset_sizes['train']
        
        model.train(False)
        model.eval()
            
        for i, data in enumerate(dataloaders['val']):
            if i % 100 == 0:
                print(f"\rValidation batch {i}/{val_batches}", end='', flush=True)
                
            inputs, labels = data
            
            with torch.no_grad():
                inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)

            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)

            loss_val += loss.item() * inputs.size(0)
            acc_val += torch.sum(preds == labels.data)

            del inputs, labels, outputs, preds
            torch.cuda.empty_cache()
            
        avg_loss_val = loss_val / dataset_sizes['val']
        avg_acc_val = acc_val / dataset_sizes['val']
        
        df_train.loc[i_trials] = {'epoch':i_trials, 'avg_loss':avg_loss, 'avg_acc':float(avg_acc),
                                  'avg_loss_val':avg_loss_val, 'avg_acc_val':float(avg_acc_val), 'image_size': image_size,
                                  'device':str(device), 'model':str(name)}
        i_trials +=1
        print()
        print(f"Epoch {epoch+1} result: ")
        print(f"Avg loss (train): {avg_loss:.4f}")
        print(f"Avg acc (train): {avg_acc:.4f}")
        print(f"Avg loss (val): {avg_loss_val:.4f}")
        print(f"Avg acc (val): {avg_acc_val:.4f}")
        print('-' * 10)
        print()
        
        if avg_acc_val > best_acc:
            best_acc = avg_acc_val
            best_model_wts = copy.deepcopy(model.state_dict())
        
    elapsed_time = time.time() - since
    print()
    print(f"Training completed in {elapsed_time // 60:.0f}m {elapsed_time % 60:.0f}s")
    print(f"Best acc: {best_acc:.4f}")
    
    model.load_state_dict(best_model_wts)
    model.cpu()
    return(model, df_train, i_trials)

# Training and saving the network
#pgray = np.arange(0, 1.1, 0.1)
pgray = np.linspace(0, 1, 10, endpoint=True)

print( f'Train scale : {train_scale}, Train gray : {train_gray}, Train basic : {train_base}')    
for name in train_modes:
    filename = f'results/{datetag}_{HOST}_train_{name}.json'
    i_trials = 0
    if name == 'vgg16_gray':
        if train_gray :
            print(f"Traning {name}, image_size = {image_size}, p (Grayscale) = 1, i_trials = {i_trials}")
            (dataset_sizes, dataloaders, image_datasets, data_transforms) = datasets_transforms(image_size=image_size, p=1)
            models[name], df_train[name], i_trials = train_model(models[name], df_train[name], criterion=criterion, optimizer=opt[name], i_trials=i_trials, num_epochs=num_epochs)
            torch.save(models[name].state_dict(), model_paths[name])
            torch.cuda.empty_cache()
        df_train[name].to_json(filename)

    elif name =='vgg16_scale':
        if train_scale :
            for image_size in image_sizes:
                print(f"Traning {name}, image_size = {image_size}, i_trials = {i_trials}")
                (dataset_sizes, dataloaders, image_datasets, data_transforms) = datasets_transforms(image_size=image_size)
                models[name], df_train[name], i_trials = train_model(models[name], df_train[name], criterion, optimizer=opt[name], i_trials=i_trials, num_epochs=45) # need more epochs per scale
                torch.save(models[name].state_dict(), model_paths[name])
                torch.cuda.empty_cache()
            image_size = args.image_size
            df_train[name].to_json(filename)
    else :
        if train_base :
            print(f"Traning {name}, image_size = {image_size}")
            (dataset_sizes, dataloaders, image_datasets, data_transforms) = datasets_transforms(image_size=image_size, p=0)
            models[name], df_train[name], i_trials = train_model(models[name], df_train[name], criterion, optimizer=opt[name], i_trials=i_trials, num_epochs=num_epochs)
            torch.cuda.empty_cache()
            torch.save(models[name].state_dict(), model_paths[name])
            df_train[name].to_json(filename)
