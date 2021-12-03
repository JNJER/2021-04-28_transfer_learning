
#import model's script and set the output file
#from DCNN_transfer_learning.model import *
from experiment_train import *
filename = f'results/{datetag}_results_1_{args.HOST}.json'
print(f'{filename=}')
def main():
    if os.path.isfile(filename):
        df = pd.read_json(filename)
    else:
        i_trial = 0
        df = pd.DataFrame([], columns=['model', 'likelihood', 'fps', 'time', 'label', 'i_label', 'i_image', 'filename', 'device_type', 'top_1']) 
        (dataset_sizes, dataloaders, image_datasets, data_transforms) = datasets_transforms(image_size=args.image_size, batch_size=1)
        
        for i_image, (data, label) in enumerate(dataloaders['test']):            
            data, label = data.to(device), label.to(device)
            
            for model_name in models_vgg.keys():
                model = models_vgg[model_name]
                model = model.to(device)

                with torch.no_grad():
                    i_label_top = reverse_labels[image_datasets['test'].classes[label]]
                    tic = time.time()
                    out = model(data).squeeze(0)
                    _, indices = torch.sort(out, descending=True)
                    if model_name == 'vgg' : # our previous work
                        top_1 = labels[indices[0]]
                        percentage = torch.nn.functional.softmax(out[args.subset_i_labels], dim=0) * 100
                        likelihood = percentage[reverse_subset_i_labels[i_label_top]].item()
                    else :
                        top_1 = subset_labels[indices[0]] 
                        percentage = torch.nn.functional.softmax(out, dim=0) * 100
                        likelihood = percentage[label].item()
                    elapsed_time = time.time() - tic
                    
                print(f'The {model_name} model get {labels[i_label_top]} at {likelihood:.2f} % confidence in {elapsed_time:.3f} seconds, best confidence for : {top_1}')
                df.loc[i_trial] = {'model':model_name, 'likelihood':likelihood, 'time':elapsed_time, 'fps': 1/elapsed_time,
                                   'label':labels[i_label_top], 'i_label':i_label_top, 
                                   'i_image':i_image, 'filename':image_datasets['test'].imgs[i_image][0], 'device_type':device.type, 'top_1':top_1}
                i_trial += 1
        df.to_json(filename)

main()    
