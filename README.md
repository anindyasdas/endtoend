# Text autoencoder and End-To_End
This is the documentation of text autoencoder and Mapping function between image embeddings obtained from [StackGAN-v2 autoencoder](https://github.com/anindyasdas/stackGANautoen) and text embeddings obtained from text autoencoder using MMD-GAN and GAN.
## Additional Dataset that needs to be downloaded along with this code
End-to-End MMD and GAN model and text autoencoder
-Download [Flower images](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz). Unzip 102flowers.zip, Rename the jpg folder to images and put it inside 102flowers folder
- Download and extract *images* folder from [Link](https://drive.google.com/file/d/1yzcR5J0D9pcI2KlZU0zzxl3Hz_C2QgJK/view?usp=sharing)
- Downloand and extract from [Link](http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz), move the previously extracted *images* folder inside birds_dataset/CUB_200_2011
- Create two folders glove.6B and glove.6B_flowers and download the glove embeddings and save the embeddings into both of these folders.
- Download and extract [1-billion daataset](https://www.statmt.org/lm-benchmark/1-billion-word-language-modeling-benchmark-r13output.tar.gz)
