
from DCNN_transfer_learning.init import *  
verbose = False

with open(args.url_loader) as json_file:
    Imagenet_urls_ILSVRC_2016 = json.load(json_file)

def clean_list(list_dir, patterns=['.DS_Store']):
    for pattern in patterns:
        if pattern in list_dir: list_dir.remove('.DS_Store')
    return list_dir

import imageio
def get_image(img_url, timeout=3., min_content=5000, verbose=verbose):
    try:
        img_resp = imageio.imread(img_url)
        if verbose : print(f"Success with url {img_url}")
        return img_resp
    except Exception as e:
        if verbose : print(f"Failed with {e} for url {img_url}")
        return False # did not work

import hashlib # jah.
# root folder
os.makedirs(args.root, exist_ok=True)
# train, val and test folders
for folder in args.folders : 
    os.makedirs(paths[folder], exist_ok=True)
    
list_urls = {}
list_img_name_used = {}
for class_wnid in class_wnids:
    list_urls[class_wnid] =  Imagenet_urls_ILSVRC_2016[str(class_wnid)]
    np.random.shuffle(list_urls[class_wnid])
    list_img_name_used[class_wnid] = []

    # a folder per class in each train, val and test folder
    for folder in args.folders : 
        class_name = reverse_id_labels[class_wnid]
        class_folder = os.path.join(paths[folder], class_name)
        os.makedirs(class_folder, exist_ok=True)
        list_img_name_used[class_wnid] += clean_list(os.listdir(class_folder)) # join two lists
    
# train, val and test folders
for folder in args.folders : 
    print(f'Folder \"{folder}\"')

    filename = f'results/{datetag}_dataset_{folder}_{args.HOST}.json'
    columns = ['img_url', 'img_name', 'is_flickr', 'dt', 'worked', 'class_wnid', 'class_name']
    if os.path.isfile(filename):
        df_dataset = pd.read_json(filename)
    else:
        df_dataset = pd.DataFrame([], columns=columns)

    for class_wnid in class_wnids:
        class_name = reverse_id_labels[class_wnid]
        print(f'Scraping images for class \"{class_name}\"')
        class_folder = os.path.join(paths[folder], class_name)
        while (len(clean_list(os.listdir(class_folder))) < N_images_per_class[folder]) and (len(list_urls[class_wnid]) > 0):

            # pick and remove element from shuffled list 
            img_url = list_urls[class_wnid].pop()
            
            if len(df_dataset[df_dataset['img_url']==img_url])==0 : # we have not yet tested this URL yet

                # Transform URL into filename
                # https://laurentperrinet.github.io/sciblog/posts/2018-06-13-generating-an-unique-seed-for-a-given-filename.html
                img_name = hashlib.sha224(img_url.encode('utf-8')).hexdigest() + '.png'

                if img_url.split('.')[-1] in ['jpe', 'gif']:
                    if verbose: print('Bad extension for the img_url', img_url)
                    worked, dt = False, 0.
                # make sure it was not used in other folders
                elif not (img_name in list_img_name_used[class_wnid]):
                    tic = time.time()
                    img_content = get_image(img_url, verbose=verbose)
                    dt = time.time() - tic
                    
                    worked = img_content is not False
                    if worked:
                        if verbose : print('Good URl, now saving', img_url, ' in', class_folder, ' as', img_name)
                        imageio.imsave(os.path.join(class_folder, img_name), img_content, format='png')
                        list_img_name_used[class_wnid].append(img_name)
                df_dataset.loc[len(df_dataset.index)] = {'img_url':img_url, 'img_name':img_name, 'is_flickr':1 if 'flickr' in img_url else 0, 'dt':dt,
                                'worked':worked, 'class_wnid':class_wnid, 'class_name':class_name}
                df_dataset.to_json(filename)
                print(f'\r{len(clean_list(os.listdir(class_folder)))} / {N_images_per_class[folder]}', end='\n' if verbose else '', flush=not verbose)
            #print('\n')
        if (len(clean_list(os.listdir(class_folder))) < N_images_per_class[folder]) and (len(list_urls[class_wnid]) == 0): 
            print('Not enough working url to complete the dataset') 
    
    df_dataset.to_json(filename)


if False: 
    
    # replace the file with that URLs that worked - removes the ones that failed
    print(f'Replacing file {args.url_loader}')
    with open(args.url_loader, 'wt') as f:
        json.dump(Imagenet_urls_ILSVRC_2016, f, indent=4)
