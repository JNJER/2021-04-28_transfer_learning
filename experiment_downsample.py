#import model's script and set the output file
from DCNN_training_benchmark.model import *
filename = f'results/{datetag}_results_2_{args.HOST}.json'

def main():

    # Output's set up
    try:
        df_downsample = pd.read_json(filename)
    except:
        df_downsample = pd.DataFrame([], columns=['model', 'perf', 'fps', 'time', 'label', 'i_label', 'i_image', 'image_size', 'filename', 'device_type', 'top_1']) 
        i_trial = 0

        # image preprocessing
        for image_size_ in args.image_sizes:
            (dataset_sizes, dataloaders, image_datasets, data_transforms) = datasets_transforms(image_size=image_size_, batch_size=1)
            print(f'Résolution de {image_size_=}')
            i_label_top = reverse_labels[image_datasets['test'].classes[label]]
            # Displays the input image of the model 
            for i_image, (data, label) in enumerate(dataloaders['test']):
                data, label = data.to(device), label.to(device)

                for model_name in models_vgg.keys():
                    model = models_vgg[model_name]
                    model = model.to(device)

                    with torch.no_grad():
                        tic = time.time()
                        out = model(data).squeeze(0)
                        _, indices = torch.sort(out, descending=True)
                        if model_name == 'vgg' : # our previous work
                            top_1 = labels[indices[0]]
                            percentage = torch.nn.functional.softmax(out[args.i_labels], dim=0) * 100
                            perf_ = percentage[reverse_i_labels[i_label_top]].item()
                        else :
                            top_1 = reverse_model_labels[indices[0]] 
                            percentage = torch.nn.functional.softmax(out, dim=0) * 100
                            perf_ = percentage[label].item()
                    dt = time.time() - tic
                    df_downsample.loc[i_trial] = {'model':model_name, 'perf':perf_, 'time':dt, 'fps': 1/dt,
                                       'label':labels[i_label_top], 'i_label':i_label_top, 
                                       'i_image':i_image, 'filename':image_datasets['test'].imgs[i_image][0], 'image_size': image_size_, 'device_type':device.type, 'top_1':str(top_1)}
                    print(f'The {model_name} model get {labels[i_label_top]} at {perf_:.2f} % confidence in {dt:.3f} seconds, best confidence for : {top_1}')
                    i_trial += 1
            df_downsample.to_json(filename)

if __name__ == "__main__":
    main()            
