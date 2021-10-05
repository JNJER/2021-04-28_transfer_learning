
from DCNN_training_benchmark.init import *  

#def del_mac_ds_store(list_dir):
#if '.DS_Store' in list_dir :
 #   cmd ='rm -Rf .DS_Store'
  #  print('Oups need to delete the `.DS_Store` file, running : ', cmd)
   # os.system(cmd) # running it
    #list_dir = os.listdir(path)
     #return list_dir
#else:
    #return list_dir

img = ('jpg', 'jpeg', 'png', 'ppm', 'bmp', 'pgm', 'tif', 'tiff', 'webp')   
verbose = False 
save = False

def get_image(img_url, class_folder):
    worked = 0
    to_save = True 
    if verbose:
        print(f'Processing {img_url}')

    if len(img_url) <= 1:
        to_save = False

    try:
        img_resp = requests.get(img_url, timeout = 1)
        
    except ConnectionError:
        if verbose :
            print(f"Connection Error for url {img_url}")
        return fail_get_image()

    except ReadTimeout:
        if verbose :
            print(f"Read Timeout for url {img_url}")
        return fail_get_image()
    
    except TooManyRedirects:
        if verbose :
            print(f"Too many redirects {img_url}")
        return fail_get_image()
    
    except MissingSchema:
        if verbose :
            print('MissingSchema failure')
        return fail_get_image()
    
    except InvalidURL:
        if verbose :
            print('InvalidURL failure')
        return fail_get_image()

    if not 'content-type' in img_resp.headers and to_save :
        if verbose :
            print('No content-type')
        to_save = False
    elif not 'image' in img_resp.headers['content-type'] and to_save :
        if verbose :
            print('Not an image')
        to_save = False
        
    if (len(img_resp.content) < 5000) and to_save :
        if verbose :
            print('Content to short')
        to_save = False

    img_name = img_url.split('/')[-1]
    img_name = img_name.split("?")[0]

    if (len(img_name) <= 1) and to_save:
        if verbose :
            print('Bad name for the image')
        to_save = False
    
    if 'jpe' in img_name or 'gif' in img_name :
        if verbose :
            print('Bad format for the image')
        to_save = False
    
    if img_name in os.listdir(class_folder):
        worked = 1
        return worked

    if not 'jpg' in img_name :
        img_name += 'jpg'
    
    if to_save:               
        img_file_path = os.path.join(class_folder, img_name)
        if verbose :
            print('Good URl, now saving...')
        with open(img_file_path, 'wb') as img_f:
            img_f.write(img_resp.content)
            list_dir = os.listdir(class_folder)
            worked = 1
        return worked

    else:
        return fail_get_image()

def fail_get_image():
    worked = 0
    if verbose :
        print(Imagenet_urls_ILSVRC_2016[str(class_wnid)].pop(i), 'do not works, deleting the url of the database')
        return worked 
    else :
        del(Imagenet_urls_ILSVRC_2016[str(class_wnid)][i])
        return worked 

if len(root) == 0:
    print("The root is required to run downloader!")
    exit()

if not os.path.isdir(root):
    os.makedirs(root)
    print(f'folder {root} did not exist! creating folder..')

iter_ = 0    
for x in folder :
    filename = f'results/{datetag}_dataset_{x}_{HOST}.json'
    # check if the folder exist
    if os.path.isdir(paths[x]):
        list_dir = os.listdir(paths[x])
        print("The folder " , x, " already exists, it includes: ", list_dir)


    # no folder, creating one 
    else :
        print(f"No existing path match for this folder, creating a folder at {paths[x]}")
        os.makedirs(paths[x])

    list_dir = os.listdir(paths[x])

    # if the folder is empty, download the images using the ImageNet-Datasets-Downloader
    if len(list_dir) < N_labels: 
        df_dataset = pd.DataFrame([], columns=['is_flickr', 'dt', 'lab_work', 'class_wnid', 'class_name'])
        tentativ_ = 0
        print(f'The {x} folder does not have anough classes, downloading some more \n') 
        for class_wnid in id_dl:
            class_name = reverse_id_labels[class_wnid]
            print(f'Scraping images for class \"{class_name}\"')
            class_folder = os.path.join(paths[x], class_name)
            if not os.path.exists(class_folder):
                os.mkdir(class_folder)                      
            list_dir = os.listdir(class_folder)
            #for i, j in enumerate(Imagenet_urls_ILSVRC_2016[str(class_wnid)]):
            for i in range(iter_, len(Imagenet_urls_ILSVRC_2016[str(class_wnid)]), 1):
                is_flickr = 0
                if len(list_dir) < N_images_per_class[x] :
                    tentativ_ +=1
                    try :
                        resp = Imagenet_urls_ILSVRC_2016[str(class_wnid)][i]
                    except : 
                        break
                    tic = time.time()
                    worked = get_image(resp, class_folder)
                    dt = time.time() - tic
                    if 'flickr' in resp:
                        is_flickr = 1
                    if verbose: 
                        print('is_flickr :', is_flickr,'dt :', dt,'worked :', worked, 'class_wnid : ', class_wnid, 'class_name :', class_name)
                    df_dataset.loc[tentativ_] = {'is_flickr':is_flickr,'dt':dt,'lab_work':worked, 'class_wnid':class_wnid, 'class_name':class_name}
                    list_dir = os.listdir(class_folder)
                    print(f'\r{len(list_dir)} / {N_images_per_class[x]}', end='', flush=True)
                else:
                    print(f'\r{len(list_dir)} / {N_images_per_class[x]}', end='', flush=True)
                    break
            print('\n')
            if len(list_dir) < N_images_per_class[x] :
                print('Not anough working url to complete the dataset') 
        list_dir = os.listdir(paths[x])


    elif len(os.listdir(paths[x])) == N_labels :
        pprint(f'The folder already contains : {len(list_dir)} classes')

    else : # if there are to many folders delete some
        print('The folder have to many classes, deleting some')
        for elem in os.listdir(paths[x]):
            contenu = os.listdir(f'{paths[x]}/{elem}')
            if len(os.listdir(paths[x])) > N_labels :
                for y in contenu:
                    os.remove(f'{paths[x]}/{elem}/{y}') # delete exces folders
                try:
                    os.rmdir(f'{paths[x]}/{elem}')
                except:
                    os.remove(f'{paths[x]}/{elem}')
        list_dir = os.listdir(paths[x])
        pprint("Now the folder " + x + " contains :" + os.listdir(paths[x]))
    if save :
        df_dataset.to_json(filename)
    iter_ += N_images_per_class[x]
    
if save : 
    json_fname = 'Imagenet_urls_ILSVRC_2016.json'
    print(f'Creating file {json_fname}')
    with open(json_fname, 'wt') as f:
        json.dump(Imagenet_urls_ILSVRC_2016, f, indent=4)
