# Text autoencoder and End-To_End
This is the documentation of text autoencoder and Mapping function between image embeddings obtained from [StackGAN-v2 autoencoder](https://github.com/anindyasdas/stackGANautoen) and text embeddings obtained from text autoencoder using MMD-GAN and GAN.
## Additional Dataset that needs to be downloaded along with this code
End-to-End MMD and GAN model and text autoencoder
- Download [Flower images](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz). Unzip 102flowers.zip, Rename the jpg folder to images and put it inside 102flowers folder
- Download and extract *images* folder from [Link](https://drive.google.com/file/d/1yzcR5J0D9pcI2KlZU0zzxl3Hz_C2QgJK/view?usp=sharing)
- Downloand and extract from [Link](http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz), move the previously extracted *images* folder inside birds_dataset/CUB_200_2011
- Create two folders glove.6B and glove.6B_flowers and download the glove embeddings and save the embeddings into both of these folders.
- Download and extract [1-billion daataset](https://www.statmt.org/lm-benchmark/1-billion-word-language-modeling-benchmark-r13output.tar.gz)
## Run The Text Autoencoder
```
python run_text_test.py dataset_type Input_Folder output_file.txt
```
- For Flower Dataset dataset_type=1, for Birds Dataset dataset_type=2
e.g. 
```
python run_text_test.py 2 /home/user/dev/unsup/data_datasets/CUB_200_2011 outbirds_n.txt
```
## Run The GAN-based Mapping Network
```
python MappingImageText.py Dataset_folder
```
e.g.
```
python MappingImageText.py /home/user/dev/unsup/data_datasets/CUB_200_2011
```
## Run MMD-based Mapping Network
```
python mmd_ganTI.py --dataset /home/das/dev/data_datasets/birds_dataset/CUB_200_2011 --gpu_device 0
```
```
python mmd_ganIT.py --dataset /home/das/dev/data_datasets/birds_dataset/CUB_200_2011 --gpu_device 0
```
