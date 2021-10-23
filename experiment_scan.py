
#import model's script and set the output file
from experiment_train import train_model
filename = f'results/{datetag}_train_scan_{args.HOST}.json'
print(f'{filename=}')

measure_columns = ['lr', 'avg_loss_val', 'avg_acc_val', 'time']

def main(N_scan=7, base=10, N_avg=10):
    if os.path.isfile(filename):
        df_scan = pd.read_json(filename)
    else:
        i_trial = 0
        df = pd.DataFrame([], columns=measure_columns) 
        for i_trial, lr  in enumerate(args.lr * np.logspace(-1, 1, N_scan, base=base)):

            # Training and saving the network
            models_vgg_ = torchvision.models.vgg16(pretrained=True)
            # Freeze training for all layers
            # Newly created modules have require_grad=True by default
            for param in models_vgg_.features.parameters():
                param.require_grad = False 

            num_features = models_vgg_.classifier[-1].in_features
            features = list(models_vgg_.classifier.children())[:-1] # Remove last layer
            features.extend([nn.Linear(num_features, n_output)]) # Add our layer with `n_output` outputs
            models_vgg_.classifier = nn.Sequential(*features) # Replace the model classifier

            since = time.time()

            (dataset_sizes, dataloaders, image_datasets, data_transforms) = datasets_transforms(image_size=args.image_size, p=0)
            models_vgg_, df_train = train_model(models_vgg_, num_epochs=args.num_epochs//4, dataloaders=dataloaders)

            df_train.to_json(filename)
            elapsed_time = time.time() - since
            print(f"Training completed in {elapsed_time // 60:.0f}m {elapsed_time % 60:.0f}s")

            df_scan.loc[i_trial] = {'lr':lr, 'avg_loss_val':df_train.iloc[-N_avg:-1]['avg_loss_val'].mean(), 
                               'avg_acc_val':df_train.iloc[-N_avg:-1]['avg_acc_val'].mean(), 'time':elapsed_time}
            print(df_scan.loc[i_trial])
            i_trial += 1
        df_scan.to_json(filename)

main()    
