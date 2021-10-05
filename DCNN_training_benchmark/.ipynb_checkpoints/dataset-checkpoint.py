
from DCNN_training_benchmark.init import *

for x in folder :
    # check if the folder exist
    if os.path.isdir(path[x]):
        list_dir = os.listdir(path[x])
        print("The folder " , x, " already exists, it includes: ", list_dir)

    # no folder, creating one 
    else :
        print(f"No existing path match for this folder, creating a folder at {path[x]}")
        os.makedirs(path[x])

    list_dir = os.listdir(path[x])

    # if the folder is empty, download the images using the ImageNet-Datasets-Downloader
    if len(list_dir) < N_labels : 
        print('This folder do not have anough classes, downloading some more') 
        cmd =f"python3 ImageNet-Datasets-Downloader/downloader.py -data_root {root} -data_folder {x} -images_per_class {N_images_per_class[x]} -use_class_list True  -class_list {id_dl} -multiprocessing_workers 0"
        pprint('Command to run : '+ cmd)
        os.system(cmd) # running it
        list_dir = os.listdir(path[x])

    elif len(os.listdir(path[x])) == N_labels :
        pprint(f'The folder already contains : {len(list_dir)} classes')

    else : # if there are to many folders delete some
        print('The folder have to many classes, deleting some')
        for elem in os.listdir(path[x]):
            contenu = os.listdir(f'{path[x]}/{elem}')
            if len(os.listdir(path[x])) > N_labels :
                for y in contenu:
                    os.remove(f'{path[x]}/{elem}/{y}') # delete exces folders
                try:
                    os.rmdir(f'{path[x]}/{elem}')
                except:
                    os.remove(f'{path[x]}/{elem}')
        list_dir = os.listdir(path[x])
        pprint("Now the folder " + x + " contains :" + os.listdir(path[x]))
