
from DCNN_training_benchmark.model import *
import copy
# LuP simplify by removing i_trials

# Training function
def train_model(model, model_name, df_train, criterion, optimizer, i_trials, num_epochs, log_interval=100):
    model.to(device)
    since = time.time()
    #best_model_wts = copy.deepcopy(model.state_dict()) # LuP : il faut m'expliquer ce point!
    #best_acc = 0.0
    
    #avg_loss = 0
    #avg_acc = 0
    #avg_loss_val = 0
    #avg_acc_val = 0
    
    # train_batches = len(dataloaders['train'])
    # val_batches = len(dataloaders['val'])
    
    
    for epoch in range(num_epochs):
        #print(f"Epoch {epoch+1}/{num_epochs}")
        #print('-' * 10)
                
        #model.train(False)
        #model.eval()
            
        with torch.no_grad():
            loss_val = 0
            acc_val = 0
            for i, (images, labels) in enumerate(dataloaders['val']):
                # if i % 100 == 0: print(f"\rValidation batch {i}/{val_batches}", end='', flush=True)

                images, labels = images.to(device), labels.to(device)

                outputs = model(images)

                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                loss_val += loss.item() * images.size(0)
                acc_val += torch.sum(preds == labels.data)

            #del images, labels, outputs, preds
            #torch.cuda.empty_cache()
            
            avg_loss_val = loss_val / dataset_sizes['val']
            avg_acc_val = acc_val / dataset_sizes['val']
            
        loss_train = 0
        acc_train = 0

        #model.train(True)
        
        for i, (images, labels) in enumerate(dataloaders['train']):
            #if i % log_interval == 0: print(f"\rTraining batch {i}/{train_batches}", end='', flush=True)
                
            # Use all the training dataset
            #if i >= train_batches : # LuP should not happen
            #    break
                
            #with torch.no_grad():
            images, labels = images.to(device), labels.to(device)   
            
            optimizer.zero_grad()

            outputs = model(images)

            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            loss_train += loss.item() * images.size(0)
            acc_train += torch.sum(preds == labels.data)

            #del images, labels, outputs, preds
            #torch.cuda.empty_cache()
            
        avg_loss = loss_train / dataset_sizes['train']
        avg_acc = acc_train / dataset_sizes['train']

        
        df_train.loc[i_trials] = {'epoch':i_trials, 'avg_loss':avg_loss, 'avg_acc':float(avg_acc),
                                  'avg_loss_val':avg_loss_val, 'avg_acc_val':float(avg_acc_val), 'image_size': image_size,
                                  'device':str(device), 'model':model_name}
        i_trials +=1
        #print()
        #print('-' * 10)
        print(f"Epoch {epoch+1}/{num_epochs} : train= loss: {avg_loss:.4f} / acc : {avg_acc:.4f} - val= loss : {avg_loss_val:.4f} / acc : {avg_acc_val:.4f}")
        #print('-' * 10)
        #print()
        
        #if avg_acc_val > best_acc:
        #    best_acc = avg_acc_val
        #    best_model_wts = copy.deepcopy(model.state_dict())
        
    elapsed_time = time.time() - since
    print(f"Training completed in {elapsed_time // 60:.0f}m {elapsed_time % 60:.0f}s")
    print()
    #print(f"Best acc: {best_acc:.4f}")
    
    #model.load_state_dict(best_model_wts)
    model.cpu()
    return (model, df_train, i_trials)

# Training and saving the network
print( f'Train scale : {train_scale}, Train gray : {train_gray}, Train basic : {train_base}')    
for model_name in models.keys():
    filename = f'results/{datetag}_{HOST}_train_{model_name}.json'
    i_trials = 0
    p = 1 if model_name == 'vgg16_gray' else 0
    print(f"Traning {model_name}, image_size = {image_size}, p (Grayscale) = {p}, i_trials = {i_trials}")
    if model_name == 'vgg16_gray':
        if train_gray :            
            (dataset_sizes, dataloaders, image_datasets, data_transforms) = datasets_transforms(image_size=image_size, p=p)
            models[model_name], df_train[model_name], i_trials = train_model(models[model_name], model_name, df_train[model_name], criterion=criterion, optimizer=opt[model_name], i_trials=i_trials, num_epochs=num_epochs)
            torch.save(models[model_name].state_dict(), model_paths[model_name])
            torch.cuda.empty_cache()
        df_train[model_name].to_json(filename)

    elif model_name =='vgg16_scale':
        if train_scale :
            for image_size in image_sizes: # starting with low resolution images 
                (dataset_sizes, dataloaders, image_datasets, data_transforms) = datasets_transforms(image_size=image_size, p=p)
                models[model_name], df_train[model_name], i_trials = train_model(models[model_name], model_name, df_train[model_name], criterion, optimizer=opt[model_name], i_trials=i_trials, num_epochs=num_epochs) # need more epochs per scale
                torch.save(models[model_name].state_dict(), model_paths[model_name])
                torch.cuda.empty_cache()
            image_size = args.image_size
            df_train[model_name].to_json(filename)
    else :
        if train_base :
            (dataset_sizes, dataloaders, image_datasets, data_transforms) = datasets_transforms(image_size=image_size, p=p)
            models[model_name], df_train[model_name], i_trials = train_model(models[model_name], model_name, df_train[model_name], criterion, optimizer=opt[model_name], i_trials=i_trials, num_epochs=num_epochs)
            torch.cuda.empty_cache()
            torch.save(models[model_name].state_dict(), model_paths[model_name])
            df_train[model_name].to_json(filename)
