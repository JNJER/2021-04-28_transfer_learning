default: all

J=jupyter nbconvert  --ExecutePreprocessor.kernel_name=python3 --ExecutePreprocessor.timeout=0 --allow-errors --execute
JN=date; $(J) --to notebook --inplace

all:
	$(JN) benchmark_transfer_learning_VGG.ipynb

clean:
	rm -fr ./data
	rm -fr ./models

install_local:
	python3 -m pip install --user -r requirements.txt

install_global:
	python3 -m pip install -r requirements.txt
