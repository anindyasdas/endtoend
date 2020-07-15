#!/usr/bin/env python
# encoding: utf-8


import argparse
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
#import torchvision.utils as vutils
from torch.autograd import Variable
from torchvision import transforms
import torchvision.models as torch_models
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os
import timeit
import sys
import mmd.util as util
import dateutil.tz
import errno
#import numpy as np
import datetime
#from models.utils1 import Logger
from data.resultwriter import ResultWriter
import mmd.base_module as base_module
#from mmd.mmd import mix_rbf_mmd2
#from tensorboardX import SummaryWriter
from models.stack_gan2.model1 import encoder_resnet1, G_NET, MAP_NET_IT1, MAP_NET_TI1
import models.text_auto_models1 as text_models
#from config import cfg
from data.datasets1 import BirdsDataset2, FlowersDataset2
import pickle
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt
#import random
import numpy as np
import time
from sklearn.manifold import TSNE
#import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
from sklearn.decomposition import PCA



#tag='TextToImage' # EVALUATION PLOT
tag='ImageToText' # EVALUATION PLOT
method= 'GAN' #mmd
file_name=''#'class_granular.txt' #''
def norm_ip(img, min1, max1):
    img = img.clamp_(min=min1, max=max1)
    img = img.add_(-min1).div_(max1 - min1 + 1e-5)
    return img

def norm_range(t, range1=None):
    if range1 is not None:
        img1 = norm_ip(t, range1[0], range1[1])
    else:
        img1 = norm_ip(t, float(torch.min(t)), float(torch.max(t)))
    return img1
    
def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise
            

# Get argument
parser = argparse.ArgumentParser()
parser = util.get_args(parser)
args = parser.parse_args()
print(args)



if torch.cuda.is_available():
    args.cuda = True
    torch.cuda.set_device(args.gpu_device)
    gpu_id = str(args.gpu_device)
    s_gpus = gpu_id.split(',')
    gpus = [int(ix) for ix in s_gpus]
    device = torch.device("cuda:0")
    print("Using GPU device", torch.cuda.current_device())
else:
    device = torch.device("cpu")
    raise EnvironmentError("GPU device not available!")

args.manual_seed = 1126
np.random.seed(seed=args.manual_seed)
random.seed(args.manual_seed)
torch.manual_seed(args.manual_seed)
torch.cuda.manual_seed(args.manual_seed)
cudnn.benchmark = True

encoder_path='/home/das/dev/unsup/saved_models/birds/encG_221025.pth' #image encoder
dec_path='/home/das/dev/unsup/saved_models/birds/netG_221025.pth' #image decoder
text_autoencoder_path = '/home/das/dev/unsup/saved_models/birds/AutoEncoderDglove100_newFalse205.pt' # text auto encoder
#GEN_PATH = '/home/das/dev/unsup/output/flowers_2020_06_08_15_55_28/modeldir/netG_110000.pth' #'netG_139.pth' # netG_100.pth #this is Image t TEXt embedding generator , if not restarted
#GEN_PATH = '/home/das/dev/unsup/output/birds_2020_05_10_09_30_27/modeldir/netG_137471.pth' #'netG_139.pth' # netG_100.pth #this is Image t TEXt embedding generator , if not restarted
#GEN_PATH = '/home/das/dev/unsup/output/birds_2020_05_08_21_03_41/modeldir/netG_137471.pth'
#GEN_PATH = '/home/das/dev/unsup/output/birds_2020_07_05_20_01_03/modeldir/netGTI_55400.pth'
GEN_PATH = '/home/das/dev/unsup/output/birds_2020_07_05_20_01_03/modeldir/netGIT_44000.pth'
#from previous fails should be empty
#DIS_PATH= '' #'netD_139.pth' 

dset='birds'#'birds'

if dset=='birds':
    glove_file='glove.6B'
else:
    glove_file='glove.6B_flowers'
    
embedding_matrx_path =glove_file + '/' +'emtrix.obj'
vocab_i2t_path = glove_file + '/' + 'vocab_i2t.obj'
vocab_t2i_path = glove_file + '/' + 'vocab_t2i.obj'
    


model_name = "AutoEncoderD"
embedding_dim = 100
hidden_dim = 100
##################################################################################
data_dir = args.dataset


if tag=='ImageToText':
    ind=1
else:
    ind= None
#############################################################
#now = datetime.datetime.now(dateutil.tz.tzlocal())
#timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
#log_dir = 'output/%s_%s' %(cfg.DATASET_NAME, timestamp)
#log_dir =  'output/flowers_2020_06_13_20_13_28'
#mkdir_p(log_dir)
#model_dir = os.path.join(log_dir, 'modeldir')
#mkdir_p(model_dir)

#sys.stdout = Logger('{}/run.log'.format(log_dir))
print("###############output folder###############################")
#print(os.path.join(os.getcwd(),log_dir))
###############validation set dir#####################
#img_dir_val = os.path.join(log_dir, 'imgdirval')
#txt_img_dir_val = os.path.join(log_dir, 'txtimgdirval')
#results_writer_img_val = ResultWriter(img_dir_val)
#results_writer_txtimg_val = ResultWriter(txt_img_dir_val)
#tensor_board = os.path.join(log_dir, 'tensorboard')
#mkdir_p(tensor_board)
#writer = SummaryWriter(tensor_board)
# NetG is a decoder
# input: batch_size * nz * 1 * 1
# output: batch_size * nc * image_size * image_size
class NetG(nn.Module):
    def __init__(self, decoder):
        super(NetG, self).__init__()
        self.decoder = decoder

    def forward(self, input1):
        output = self.decoder(input1)
        return output


# NetD is an encoder + decoder
# input: batch_size * nc * image_size * image_size
# f_enc_X: batch_size * k * 1 * 1
# f_dec_X: batch_size * nc * image_size * image_size
class NetD(nn.Module):
    def __init__(self, encoder, decoder):
        super(NetD, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input1):
        f_enc_X = self.encoder(input1)
        f_dec_X = self.decoder(f_enc_X)

        #f_enc_X = f_enc_X.view(input.size(0), -1)
       # f_dec_X = f_dec_X.view(input.size(0), -1)
        return f_enc_X, f_dec_X


class ONE_SIDED(nn.Module):
    def __init__(self):
        super(ONE_SIDED, self).__init__()

        main = nn.ReLU()
        self.main = main

    def forward(self, input):
        output = self.main(-input)
        output = -output.mean()
        return output
    

def adjust_padding(cap, len1):
    cap = cap.numpy()
    len1 = len1.numpy()
    max_len = max(len1)
    temp=[]
    for i in cap:
        j = i[0:max_len]
        temp.append(j)
    cap =torch.LongTensor(temp)
    len1= torch.LongTensor(len1)
    return cap, len1



def load_network():
    ####################Image deoder################################
    dec = G_NET()
    dec.apply(base_module.weights_init)
    dec = torch.nn.DataParallel(dec, device_ids=gpus)
    #################################################################
    # construct encoder/decoder modules
    #hidden_dim = args.nz
    if ind !=None:
        G_decoder = MAP_NET_IT1() # This the Actual Generator 
    else:
        G_decoder = MAP_NET_TI1() # This the Actual Generator 
    #D_encoder = MAP_NET_IT2() #Discriminator should be an Auto encoder without noise
    #D_decoder = MAP_NET_TI2()#
    if method =='GAN':
        netG=torch.nn.DataParallel(G_decoder, device_ids=gpus)
    else:
        netG = NetG(G_decoder)

    #netD = NetD(D_encoder, D_decoder)
    #one_sided = ONE_SIDED()
    print("netG:", netG)
    #print("netD:", netD)
    #print("oneSide:", one_sided)

    netG.apply(base_module.weights_init)
    #netD.apply(base_module.weights_init)
    #one_sided.apply(base_module.weights_init)
    Gpath = os.path.join(GEN_PATH)
    checkpoint = torch.load(Gpath)
    netG.load_state_dict(checkpoint['state_dict'])
        #Epath = os.path.join(path, 'encG.pth' )
        #checkpoint = torch.load(Epath)
        #enc.load_state_dict(checkpoint['state_dict'])
    print('Load ', GEN_PATH)
    
    
    
    
            
    if args.cuda:
        dec.cuda()
        netG.cuda()
        #netD.cuda()
        #one_sided.cuda()

    

    
        
    
    
    return dec, netG
         
def initialize_model(model_name, config, embeddings_matrix):
    
    model_ft= text_models.AutoEncoderD(config, embeddings_matrix)
    model_ft = model_ft.to(device)

    
    dec, gen= load_network()
    #############################################################
    enc = torch_models.resnet50(pretrained=True)
    num_ftrs = enc.fc.in_features
    enc.fc = nn.Linear(num_ftrs, 1024)
    enc = enc.to(device)
    ################################################################
    #enc = encoder_resnet1()
    #enc = enc.to(device)
    
    print("=> loading Image encoder from '{}'".format(encoder_path))
    encoder = torch.load(encoder_path)
    enc.load_state_dict(encoder['state_dict'])
    
    
    print("=> loading Image decoder from '{}'".format(dec_path))
    decoder = torch.load(dec_path)
    dec.load_state_dict(decoder['state_dict'])
    
    
    print("=> loading text autoencoder from '{}'".format(text_autoencoder_path))
    text_autoencoder = torch.load(text_autoencoder_path)
    model_ft.load_state_dict(text_autoencoder['state_dict'])
    

    return model_ft, enc, dec, gen




######################################################################
def plot_scatter(x, colors):
    # choose a color palette with seaborn.
    num_classes = len(np.unique(colors))
    unique_classes = np.unique(colors)
    palette = np.array(sns.color_palette("hls", num_classes))
    
    class_to_number = {ni: indi for indi, ni in enumerate(unique_classes)}
    numbers = [class_to_number[ni] for ni in colors]

    # create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    #sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40, c=palette[colors.astype(np.int)])
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40, c=palette[numbers])
    # produce a legend with the unique colors from the scatter
    
    #ax.legend(handles=sc.legend_elements()[0], labels=unique_classes)

    #plt.xlim(-25, 25)
    #plt.ylim(-25, 25)
    plt.title('t-SNE scatter plot')
    plt.xlabel('X - value')
    plt.ylabel('Y - value')
    
    #ax.grid(True)
    #ax.axis('off')
    #ax.legend()
    ax.axis('tight')
    #plt.show()

    # add the labels for each digit corresponding to the label
    #txts = []

    #for i in range(num_classes):
    tsne_dict = dict()
    
    for i in unique_classes:

        # Position of each label at median of data points.

        xtext, ytext = np.median(x[colors == i, :], axis=0)
        tsne_dict[i]=(xtext, ytext)
        print('class:', i, 'center:', tsne_dict[i])
        txt = ax.text(xtext, ytext, str(i), fontsize=5)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        #txts.append(txt)
    f.savefig(tag + '_'+'tsne_plot.png')
    
    #sns.scatterplot(x[:,0], x[:,1], hue=colors, legend='full', palette=palette)
    
    return


def compute_t_SNE(all_list, all_labels):
    RS=123 #random number fro reproducibility
    x_subset= np.asarray(all_list)
    y_subset =np.asarray(all_labels)
    time_start = time.time()
    pca_50 = PCA(n_components=50)
    pca_result_50 = pca_50.fit_transform(x_subset)
    print('PCA with 50 components done! Time elapsed: {} seconds'.format(time.time()-time_start))
    print('Cumulative variance explained by 50 principal components: {}'.format(np.sum(pca_50.explained_variance_ratio_)))
    time_start = time.time()
    output_pca_tsne = TSNE(random_state=RS).fit_transform(pca_result_50)
    print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
    plot_scatter(output_pca_tsne, y_subset)
    #plot_scatter(pca_result_50, y_subset)
    
    

def calculate_class_accuracy(x_list, y_list, labels):
    out_labels =[] #class labels of fake embeddings based on cosine similarity of fake with original
    l=cosine_similarity(x_list,y_list)
    #in1=(in1 == in1.max(axis=1, keepdims=1)).astype(float) #put ones in max position along rows/here for one hot
#
    out_index=l.argmax(axis=1)#returns the arguments which is the sentence number for which cosine similariy is max
    for idx in out_index:
        out_labels.append(labels[idx]) #for the sentences for wich cosine is max ,class_label is noted down
    in_labels =np.asarray(labels)
    out_labels=np.asarray(out_labels)

    tags= np.unique(labels)
    print(classification_report(in_labels, out_labels))
    #print(classification_report(in1.argmax(axis=1), l.argmax(axis=1)))
    #matrix=confusion_matrix(in1.argmax(axis=1), l.argmax(axis=1))
    matrix=confusion_matrix(in_labels, out_labels)
    ig, ax = plt.subplots(figsize=(7,8))
    hmap=sns.heatmap(matrix, annot=True, fmt='d',
            xticklabels=tags, yticklabels=tags)
    hmap.set(xlabel='Predicted Label', ylabel='True Label')
    figure = hmap.get_figure()    
    figure.savefig('heatmap_'+tag+ '.png', dpi=400)
    #print('accuracy:', accuracy_score(in1.argmax(axis=1), l.argmax(axis=1))*100)
    print('accuracy:', accuracy_score(in_labels, out_labels)*100)


def compute_ranking_recall(x,y,l):
    #print(x)
    #print(y)
    x=[i/np.linalg.norm(i) for i in x]#normaliza
    y=[i/np.linalg.norm(i) for i in y]#normalize
    #print(x)
    #print(y)
    y_arr=np.asarray(y)
    x_arr=np.asarray(x)
    npts = x_arr.shape[0]
    ranks = np.zeros(npts)
    top1 = np.zeros(npts)
    ranks_class = np.zeros(npts)
    for index in range(npts):

        # Get query image
        x_item = x_arr[index].reshape(1, x_arr.shape[1])
        d = np.dot(x_item, y_arr.T).flatten()
        #d = np.dot(x_item, y_arr.T)/(norm(x_item)*norm(y_arr.T))
        inds = np.argsort(d)[::-1]
        #print('inds:',inds)
        # Score
        rank = 1e20
        tmp = np.where(inds == index)[0][0]
        if tmp < rank:
            rank = tmp
        ranks[index] = rank
        top1[index] = inds[0]
         #################Class rank#####
        rank_c = 1e20
        ####################################
        inds_labels=[]
        for i in inds:#removing duplication of class labels such as 3, 3 or 3,5,4,3
            if labels[i] not in inds_labels:
                inds_labels.append(labels[i])
        ##################################
        inds_labels=np.asarray(inds_labels)
        tmp1= np.where(inds_labels == l[index])[0][0]
        if tmp1 < rank_c:
            rank_c = tmp1
        ranks_class[index] = rank_c
    # Compute metrics
    print(ranks)
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    print('recall@1: %.1f'%r1)
    print('recall@5: %.1f'%r5)
    print('recall@10: %.1f'%r10)
    print('median_rank: %.1f'%medr)
    print('mean_rank:% .1f'%meanr)
    r1 = 100.0 * len(np.where(ranks_class < 1)[0]) / len(ranks_class)
    r2 = 100.0 * len(np.where(ranks_class < 2)[0]) / len(ranks_class)
    r3 = 100.0 * len(np.where(ranks_class < 3)[0]) / len(ranks_class)
    r5 = 100.0 * len(np.where(ranks_class < 5)[0]) / len(ranks_class)
    medr = np.floor(np.median(ranks_class)) + 1
    meanr = ranks_class.mean() + 1
    print(ranks_class)
    print(np.unique(ranks_class))
    print('recall@1 class: %.1f'%r1)
    print('recall@2 class: %.1f'%r2)
    print('recall@3 class: %.1f'%r3)
    print('recall@5 class: %.1f'%r5)
    print('median_rank class: %.1f'%medr)
    print('mean_rank class:% .1f'%meanr)






# Get data
#trn_dataset = util.get_data(args, train_flag=True)

data_transforms = transforms.Compose([transforms.ToPILImage(), transforms.Pad(0), transforms.ToTensor(),transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])


print("Initializing Datasets and Dataloaders...")
# Create training and validation data
if dset=='birds':
    text_datasets = {x: BirdsDataset2(os.path.join(data_dir), transform=data_transforms, split=x) for x in ['train', 'val']}
else:
    text_datasets = {x: FlowersDataset2(os.path.join(data_dir), transform=data_transforms, split=x) for x in ['train', 'val']}

ds = text_datasets['train']

vocab = ds.get_vocab_builder()
max_len = ds.max_sent_length
#############################################################
#####################################################################
print("Loading vocabulary, embedding matrix from trained text model.....")
file_ematrix = open(embedding_matrx_path, 'rb') 
file_vocab_i2t = open(vocab_i2t_path, 'rb')
file_vocab_t2i = open(vocab_t2i_path, 'rb')
embeddings_matrix = pickle.load(file_ematrix)
vocab.i2t= pickle.load(file_vocab_i2t)
vocab.t2i= pickle.load(file_vocab_t2i)
text_datasets['val'].vocab_builder.i2t =vocab.i2t
text_datasets['val'].vocab_builder.t2i = vocab.t2i
############################################################################

# Create training and validation dataloaders
dataloaders_dict = {x: DataLoader(text_datasets[x], batch_size=args.batch_size, shuffle=True, num_workers=int(args.workers)) for x in ['train', 'val']}

config = {  'emb_dim': embedding_dim,
                'hid_dim': hidden_dim//2, #birectional is used so hidden become double
                'n_layers': 1,
                'dropout': 0.0,
                'vocab_size': vocab.vocab_size(),
                'sos': vocab.sos_pos(),
                'eos': vocab.eos_pos(),
                'pad': vocab.pad_pos(),
             }

model, enc, dec, netG = initialize_model(model_name, config, embeddings_matrix)




#criterion = nn.MSELoss()
# sigma for MMD
#base = 1.0
#sigma_list = [1, 2, 4, 8, 16]
#sigma_list = [sigma / base for sigma in sigma_list]

# put variable into cuda device
fixed_noise = torch.cuda.FloatTensor(args.batch_size, args.nz).normal_(0, 1)
noise = Variable(torch.FloatTensor(args.batch_size, args.nz))
one = torch.tensor(1, dtype=torch.float)


if args.cuda:
    dec.cuda()
    enc.cuda()
    model.cuda()
    netG.cuda()
    #netD.cuda()
    #one_sided.cuda()
    #criterion.cuda()
    
    noise, fixed_noise, one = noise.cuda(), fixed_noise.cuda(), one.cuda()
    
#mone = one * -1 



# setup optimizer
#optimizerG, optimizerD, count = define_optimizers(netG, netD, model_dir)



######################Evaluate###################################
        
ecount=0
        ######setting in eval mode##########################
enc.eval()
dec.eval()
model.eval()
netG.eval()
        ############################################################

        
   
x_list=[] #conraining input embeddings
y_list=[] #containing corresnponding fake embeddings
labels=[] #class labels of input embeddings
all_list=[] #containing both input embeddings and fakembeddings
all_labels=[] #containing labels both both fake and original, eg. for class no 1, fake class labels
#is f_1, origianl is o_1

#index_to_class=dict()
#l_tags_in=[]
#l_tags_out=[]
index_no=0
granular_class=dict()
if file_name!='':
    f1=open(file_name,'r')
    lines= f1.readlines()
    for line in lines:
        line = line.strip().split('\t')
        granular_class[line[1]]=line[2]
    #print(line[0],line[1],line[2])
    print(granular_class)
for uinputs, inputs,_, _, captions, lengths, class_label in dataloaders_dict['val']:
    #print('class_label shape:', class_label.shape)
    #print('class_label:', class_label)
    inp0 = uinputs[0]
    inp0= inp0.to(device)
    N = inp0.size(0)
    n1 = fixed_noise[:N]
    captions, lengths= adjust_padding(captions, lengths)
    captions = captions.to(device)
    lengths = lengths.to(device)
    with torch.no_grad():        
        img_embedding = enc(inp0)
        #img_embedding, _, _ = enc(inp0)
        #print('img emb shape:', img_embedding.shape)
        text_embedding = model.rnn(pass_type='encode',batch_positions=captions, text_length=lengths)
        #print('text_embedding.shape', text_embedding[0].shape)
        #text_embedding_fake = netG(img_embedding.detach())
        #img_embedding_fake = netG(text_embedding[0].detach())
        #print('text_embedding_fake shape', text_embedding_fake.shape)
        if ind!=None:
            x = text_embedding[0]
            text_embedding_fake = netG(img_embedding.detach())
            y = text_embedding_fake
        else:
            x=img_embedding
            img_embedding_fake = netG(text_embedding[0].detach())
            y=img_embedding_fake
        #y=x
        for ix, iy, c in zip(x,y,class_label):
            x_list.append(ix.tolist())
            y_list.append(iy.tolist())
            if file_name != '':
                new_class= granular_class[str(c.item())]
            else:
                new_class=c.item()
            labels.append(new_class)
            #index_to_class[index_no]=new_class
            all_list.append(ix.tolist())
            all_list.append(iy.tolist())
            all_labels.append('o_' + str(new_class))
            all_labels.append('f_' + str(new_class))
            #l_tags_in.append(tag+'_'+str(index_no))
            #l_tags_out.append('F_'+tag+'_'+str(index_no))
            index_no+=1
calculate_class_accuracy(x_list, y_list, labels)
compute_ranking_recall(x_list,y_list, labels)
compute_t_SNE(all_list, all_labels)
print("#################Evaluation complete#######################################")
