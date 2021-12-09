# 2020-01-11_fast_and_curious  [![DOI](https://zenodo.org/badge/362495685.svg)](https://zenodo.org/badge/latestdoi/362495685)
Projet de stage master 2 : modélisation du traitement visuel d'une image fovéale 


## getting the submodule to download images

See [doc](https://github.blog/2016-02-01-working-with-submodules/) :

to init:
```
git submodule update --init --recursive
```

to update:
```
git submodule update --recursive --remote
```

## to run without notebook

(for instance on the NVIDIA)

```
python3 experiment_basic.py
python3 experiment_downsample.py
python3 experiment_grayscale.py
python3 experiment_train.py
```
In case you get `OMP: Error #15: Initializing libiomp5.dylib, but found libiomp5.dylib already initialized.`, run :
```
KMP_DUPLICATE_LIB_OK=TRUE python3 experiment_basic.py
KMP_DUPLICATE_LIB_OK=TRUE python3 experiment_downsample.py
KMP_DUPLICATE_LIB_OK=TRUE python3 experiment_grayscale.py
KMP_DUPLICATE_LIB_OK=TRUE python3 experiment_train.py
```
