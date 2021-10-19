
from DCNN_transfer_learning.init import *  

from requests.exceptions import ConnectionError, ReadTimeout, TooManyRedirects, MissingSchema, InvalidURL

verbose = False 

with open(args.url_loader) as json_file:
    Imagenet_urls_ILSVRC_2016 = json.load(json_file)

def clean_list(list_dir, patterns=['.DS_Store']):
    for pattern in patterns:
        if pattern in list_dir: list_dir.remove('.DS_Store')
    return list_dir
    
def get_image(img_url, class_folder, timeout=3.):
    if verbose:
        print(f'Processing {img_url}')

    img_name = img_url.split('/')[-1]
    # handle strange file names
    img_name = img_name.split("?")[0]
    if ('jpe' in img_name) or ('gif' in img_name)  or (len(img_name) <= 1):
        if verbose :
            print('Bad format for the image')
    else:
        try:
            img_resp = requests.get(img_url, timeout=timeout)
            if not 'content-type' in img_resp.headers :
                if verbose : print('No content-type')
            elif not 'image' in img_resp.headers['content-type'] :
                        if verbose : print('Not an image')
            elif (len(img_resp.content) < 5000) :
                if verbose : print('Content to short')
            else:
                # LuP some files miss the extension?
                if not 'jpg' in img_name : img_name += 'jpg'

                img_file_path = os.path.join(class_folder, img_name)
                if verbose : print('Good URl, now saving...')
                with open(img_file_path, 'wb') as img_f:
                    img_f.write(img_resp.content)
                    list_dir = os.listdir(class_folder)
                return True

        except Exception as e:
            if verbose : print(f"Failed with {e} for url {img_url}")
    return False

if not os.path.isdir(args.root):
    print(f'folder {args.root} did not exist! creating folder..')
    os.makedirs(args.root, exist_ok=True)

iter_ = 0    
for folder in args.folders :
    filename = f'results/{datetag}_dataset_{folder}_{args.HOST}.json'
    # check if the folder exists
    if os.path.isdir(paths[folder]):
        list_dir = list_dir = clean_list(os.listdir(paths[folder]))
        print("The folder", folder, " already exists, it includes: ", list_dir)
    else :
        # no folder, creating one 
        print(f"No existing path match for this folder, creating a folder at {paths[folder]}")
        os.makedirs(paths[folder])

    list_dir = os.listdir(paths[folder])
    list_dir = clean_list(list_dir)
    # if the folder is empty, download the images using the ImageNet-Datasets-Downloader
    if len(list_dir) < N_labels: 
        df_dataset = pd.DataFrame([], columns=['is_flickr', 'dt', 'lab_work', 'class_wnid', 'class_name'])
        tentativ_ = 0
        print(f'The {folder} folder does not have anough classes, downloading some more \n') 
        for class_wnid in id_dl:
            class_name = reverse_id_labels[class_wnid]
            print(f'Scraping images for class \"{class_name}\"')
            class_folder = os.path.join(paths[folder], class_name)
            if not os.path.exists(class_folder):
                os.mkdir(class_folder)                      
            list_dir = os.listdir(class_folder)
            #for i, j in enumerate(Imagenet_urls_ILSVRC_2016[str(class_wnid)]):
            for i in range(iter_, len(Imagenet_urls_ILSVRC_2016[str(class_wnid)]), 1):
                if len(list_dir) < N_images_per_class[folder] :
                    tentativ_ +=1
                    try :
                        resp = Imagenet_urls_ILSVRC_2016[str(class_wnid)][i]
                    except : 
                        break
                    tic = time.time()
                    worked = get_image(resp, class_folder)
                    if not worked:
                        if verbose :
                            print(Imagenet_urls_ILSVRC_2016[str(class_wnid)].pop(i), ' does not work, deleting the url of the database')
                        else :
                            del(Imagenet_urls_ILSVRC_2016[str(class_wnid)][i])

                    dt = time.time() - tic
                    if verbose: 
                        print('is_flickr :', is_flickr,'dt :', dt,'worked :', worked, 'class_wnid : ', class_wnid, 'class_name :', class_name)
                    df_dataset.loc[tentativ_] = {'is_flickr':1 if 'flickr' in resp else 0,'dt':dt,'lab_work':worked, 'class_wnid':class_wnid, 'class_name':class_name}
                    list_dir = os.listdir(class_folder)
                    print(f'\r{len(list_dir)} / {N_images_per_class[folder]}', end='', flush=True)
                else:
                    print(f'\r{len(list_dir)} / {N_images_per_class[folder]}', end='', flush=True)
                    break
            print('\n')
            if len(list_dir) < N_images_per_class[folder] :
                print('Not anough working url to complete the dataset') 
        list_dir = os.listdir(paths[folder])
        if True :
            df_dataset.to_json(filename)


    elif len(os.listdir(paths[folder])) == N_labels :
        pprint(f'The folder already contains : {len(list_dir)} classes')
        
    iter_ += N_images_per_class[folder]
    
if False: 
    # replace the file with that URLs that worked - removes the ones that failed
    print(f'Replacing file {args.url_loader}')
    with open(args.url_loader, 'wt') as f:
        json.dump(Imagenet_urls_ILSVRC_2016, f, indent=4)
