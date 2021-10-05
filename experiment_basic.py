
#import model's script and set the output file
from DCNN_training_benchmark.model import *

filename = f'results/{datetag}_results_1_{HOST}.json'

try:
    df = pd.read_json(filename)
except:
    df = pd.DataFrame([], columns=['model', 'perf', 'fps', 'time', 'label', 'i_label', 'i_image', 'filename', 'device', 'top_1', 'apriori']) 
    i_trial = 0
    apriori = None
    # image preprocessing
    for i_image, (data, label) in enumerate(image_datasets['test']):
        for name in models.keys():
            model = models[name].to(device)
            model.eval()
            i_label_top = reverse_labels[image_datasets['test'].classes[label]]
            tic = time.time()
            out = model(data.unsqueeze(0).to(device)).squeeze(0)
            if name == 'vgg_sub' :
                percentage = torch.nn.functional.softmax(out, dim=0) * 100
                _, indices = torch.sort(out, descending=True)
                top_1 = labels[int(indices[0])]
                percentage = torch.nn.functional.softmax(out[i_labels], dim=0) * 100
                _, indices = torch.sort(out[i_labels], descending=True)
                apriori = reverse_model_labels_vgg[indices[0]]
                perf_ = percentage[reverse_i_labels[i_label_top]].item()
            else :
                percentage = torch.nn.functional.softmax(out, dim=0) * 100
                _, indices = torch.sort(out, descending=True)
                top_1 = reverse_model_labels[int(indices[0])]
                apriori = None
                perf_ = percentage[label].item()
            dt = time.time() - tic            
            df.loc[i_trial] = {'model':name, 'perf':perf_, 'time':dt, 'fps': 1/dt,
                               'label':labels[i_label_top], 'i_label':i_label_top, 
                               'i_image':i_image, 'filename':image_datasets['test'].imgs[i_image][0], 'device':str(device), 'top_1':str(top_1), 
                               'apriori':apriori}
            print(f'The {name} model get {labels[i_label_top]} at {perf_:.2f} % confidence in {dt:.3f} seconds, best confidence for : {top_1}')
            i_trial += 1
    df.to_json(filename)
