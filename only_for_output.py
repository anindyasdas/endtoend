from __future__ import print_function
from __future__ import division
import os
import errno
os.environ["HOME"] = str('/ukp-storage-1/das')#setting home environment from default home/das to ukp-storage-1/das
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision
import time
from data.datasets1 import BirdsDataset1, FlowersDataset1
from torchvision import transforms
import torchvision.models as torch_models
from config import cfg
from models.stack_gan2.model1 import encoder_resnet, G_NET1 #G_NET
from data.resultwriter import ResultWriter
from models.utils1 import Logger
from tensorboardX import SummaryWriter
from tensorboardX import summary
from tensorboardX import FileWriter
import datetime
import dateutil.tz
import sys
from pathlib import Path
from tempfile import mkdtemp
import pickle
from copy import deepcopy
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

#print('sys.argv[1]',sys.argv[1])
#print('sys.argv[2]',sys.argv[2])
#print('sys.argv[3]',sys.argv[3])
#print(os.getcwd())


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

###########################################################################
###########################################################################
####################### variables#######################################
##########################################################################
########################################################################
encoder_path='/home/das/dev/StackGAN-v2/output/flowers_3stages_2020_05_28_11_00_44/Model/encG_175800.pth' #image encoder
dec_path='/home/das/dev/StackGAN-v2/output/flowers_3stages_2020_05_28_11_00_44/Model/netG_175800.pth' #image decoder

# Top level data directory
data_dir = sys.argv[1]

# Batch size for training (change depending on how much memory you have)
batch_size = 32
# Number of epochs to train for
num_epochs = 150
#restart_epoch = 1 #restartting from failed step

# Flag for feature extracting. When False, we finetune the whole model, else we only extract features

# mean and std calculated fort the dataset
mean_r= 0.5
mean_g= 0.5
mean_b= 0.5
std_r= 0.5
std_g= 0.5
std_b= 0.5
lr1 = 1e-3
lr2 = 1e-3
wt = 1e-5
##################################################


# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
gpu_id = '0'
s_gpus = gpu_id.split(',')
gpus = [int(ix) for ix in s_gpus]

######################################################################
#########################################################################

#############################################################
now = datetime.datetime.now(dateutil.tz.tzlocal())
timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
log_dir = 'output/%s_%s' %(cfg.DATASET_NAME, timestamp)
#log_dir =  'output/birds_2020_05_11_15_29_24'
mkdir_p(log_dir)
model_dir = os.path.join(log_dir, 'modeldir')
mkdir_p(model_dir)

sys.stdout = Logger('{}/run.log'.format(log_dir))
print("###############output folder###############################")
print(os.path.join(os.getcwd(),log_dir))
###############validation set dir#####################

img_input = os.path.join(log_dir, 'img_input')
img_output = os.path.join(log_dir, 'img_out')
results_writer_input = ResultWriter(img_input)
results_writer_output = ResultWriter(img_output)



#######################################################################
########################################################################
########################################################################
################## Functions and Class defination#########################
#####################################################################



class SimpleAutoencoder(nn.Module):
    #############image autoencoder##############################
    def __init__(self, encoder, decoder):
        super(SimpleAutoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        x, mu, sigma = self.encoder(x)
        #x, _, _ = self.encoder(x)
        x, _ = self.decoder(x)
        return x, mu, sigma



def weights_init(m): # to inotialize weigts of model
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.orthogonal(m.weight.data, 1.0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        nn.init.orthogonal(m.weight.data, 1.0)
        if m.bias is not None:
            m.bias.data.fill_(0.0)
            
def load_params(model, new_param):
    for p, new_p in zip(model.parameters(), new_param):
        p.data.copy_(new_p)


def copy_G_params(model):
    flatten = deepcopy(list(p.data for p in model.parameters()))
    return flatten



 
def load_network(path):
    ####################Image deoder################################
    netG = G_NET1()
    netG.apply(weights_init)
    netG = torch.nn.DataParallel(netG, device_ids=gpus)
    #################################################################

    
    


    
    

    if cfg.CUDA:
        netG.cuda()
        
        
    
    
    return netG



def loss_function(final_img,residual_img,upscaled_img,com_img,orig_img):
#size average false means return sum over all pixel points if set to true average over pixel points returned
  com_loss = nn.MSELoss(size_average=False)(orig_img, final_img)
  rec_loss = nn.MSELoss(size_average=False)(residual_img,orig_img-upscaled_img)
  
  return com_loss + rec_loss


def KL_loss(mu, logvar):
    # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.mean(KLD_element).mul_(-0.5)
    return KLD



        
def load_checkpoint(model, optimizer, filename):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    print("=> loading checkpoint '{}'".format(filename))
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer

def optimizerToDevice(optimizer):
    for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
    return optimizer



        
def initialize_model():
    
    #model_ft= text_models.AutoEncoderD(config, embeddings_matrix)
    #model_ft = model_ft.to(device)

    
    dec= load_network(model_dir)
    
    #enc = torch_models.resnet50(pretrained=True)
    #num_ftrs = enc.fc.in_features
    #enc.fc = nn.Linear(num_ftrs, 1024)
    enc = encoder_resnet()
    enc = enc.to(device)
    
    
    print("=> loading Image encoder from '{}'".format(encoder_path))
    encoder = torch.load(encoder_path)
    enc.load_state_dict(encoder['state_dict'])
    
    
    print("=> loading Image decoder from '{}'".format(dec_path))
    decoder = torch.load(dec_path)
    dec.load_state_dict(decoder['state_dict'])
    
    
    
    

    return enc, dec



def save_results(imgs_input, imgs_output):
    
    for ii, io in zip(imgs_input, imgs_output):
        ii = norm_range(ii)#normalize to (0,1)
        io = norm_range(io)#normalize to (0,1)
        ii = ii.cpu()
        io = io.detach().cpu()
        results_writer_input.write_images1(ii)
        results_writer_output.write_images1(io)
        
    
 

class ImageTextTrainer(object):
    def __init__(self, enc, dec, dataloaders, num_epochs, log_dir):
        self.enc = enc
        self.dec = dec
        self.dataloaders = dataloaders
        self.num_epochs = num_epochs
        self.criterion = nn.BCELoss()
        self.batch_size = batch_size
        self.max_epoch= num_epochs
        self.num_batches = len(self.dataloaders['train'])
        self.log_dir = log_dir
        self.tensor_board = os.path.join(self.log_dir, 'tensorboard')
        #self.model_dir = os.path.join(self.log_dir, 'modeldir')
        self.model_dir = model_dir
        mkdir_p(self.tensor_board)
        #mkdir_p(self.model_dir)
        self.writer = SummaryWriter(self.tensor_board)
        self.train_dis1 = True
        self.train_dis2 = True
        self.train_gen1 = False
        self.train_gen2 = False
        
     
    def evaluate(self):
        ######setting in eval mode##########################
        self.enc.eval()
        self.dec.eval()
        
        ############################################################
        nz = cfg.GAN.Z_DIM
        #fixed_noise1 = Variable(torch.FloatTensor(self.batch_size, nz).normal_(0, 1))
        fixed_noise2 = Variable(torch.FloatTensor(self.batch_size, nz).normal_(0, 1))
        #fixed_noise3 = Variable(torch.FloatTensor(self.batch_size, 20).normal_(0, 1))
        if cfg.CUDA:
            #fixed_noise1 = fixed_noise1.cuda()
            fixed_noise2 = fixed_noise2.cuda()
            #fixed_noise3 = fixed_noise3.cuda()
        for uinputs, inputs,_, labels, captions, lengths in self.dataloaders['val']:
            with torch.no_grad():
                inp0 = uinputs[0]
                inp0= inp0.to(device)
                N = inp0.size(0)
                #n1 = fixed_noise1[:N]
                n2 = fixed_noise2[:N]
                #n3 = fixed_noise3[:N]
                #captions, lengths= adjust_padding(captions, lengths)
                #captions = captions.to(device)
                #lengths = lengths.to(device)
            
                #self.img_embedding = self.enc(inp0)
                self.img_embedding, _, _ = self.enc(inp0)
                #self.text_embedding = self.model.rnn(pass_type='encode',batch_positions=captions, text_length=lengths)

            
                #self.text_embedding_fake = self.genIT(n1, self.img_embedding.detach())
                #self.text_embedding_fake = self.genIT(self.img_embedding.detach())
                #temb = self.text_embedding_fake,
            
            ###################generated text from Image##################################
                #length1 = [max_len]*N #taking maximum length
                #length1= torch.LongTensor(length1)
                #_, indices_g = self.model.rnn(pass_type ='generate', hidden=temb, text_length=length1, batch_size=N)
            ########################below verification for original encoded output#####################
                #_, indices_o = self.model.rnn(pass_type ='generate', hidden=self.text_embedding, text_length=length1, batch_size=N)
            ###############We can use original image or output of image decoder#############
                fake_imgs_o, _, _ = self.dec(n2, self.img_embedding.detach())
                fake_imgs_o = self.dec(n2, self.img_embedding.detach())
                #self.img_embedding_fake = self.genTI(n3, self.text_embedding[0].detach())
                #self.img_embedding_fake = self.genTI(self.text_embedding[0].detach())
                #fake_imgs_g, _, _ = self.dec(n2, self.img_embedding_fake.detach())
                
                save_results(inputs[2], fake_imgs_o[2])
                
                
        print("#################Evaluation complete#######################################")

        

    
        
    

    



    
    





##########################################################################
#########################################################################
#####################################################MAIN###############
########################################################################
# Initialize the model for this run() second one if want to load weights from previous check point
#model_ft, netG, netsD, num_Ds = initialize_model(encoder_name, feature_vector_dim, feature_extract, use_pretrained=False, vae=var_ae, use_finetuned= finetuned)
#model_ft, input_size = initialize_model(model_name, feature_vector_dim, feature_extract, use_pretrained=True, vae=var_ae, use_finetuned='checkpoint.pt')
#############################################################################
# Print the model we just instantiated
im_size = 256
#data_transforms2 = transforms.Compose([transforms.ToPILImage(), transforms.Pad(0), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
data_transforms = transforms.Compose([transforms.ToPILImage(), transforms.Pad(0), transforms.ToTensor(),transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
inv_normalize = transforms.Normalize(mean=[-mean_r/std_r, -mean_g/std_g, -mean_b/std_b],std=[1/std_r, 1/std_g, 1/std_b])
# Data augmentation and normalization for training
# Just normalization for validation
# data_transforms = {
#     'train': transforms.Compose([
#         transforms.RandomResizedCrop(im_size),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ]),
#     'val': transforms.Compose([
#         transforms.Resize(im_size),
#         transforms.CenterCrop(im_size),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ]),
# }

print("Initializing Datasets and Dataloaders...")
# Create training and validation data
#text_datasets = {x: BirdsDataset1(os.path.join(data_dir), transform=data_transforms, split=x) for x in ['train', 'val']}
text_datasets = {x: FlowersDataset1(os.path.join(data_dir), transform=data_transforms, split=x) for x in ['train', 'val']}



############################################################################

# Create training and validation dataloaders
dataloaders_dict = {x: DataLoader(text_datasets[x], batch_size=batch_size, shuffle=True, num_workers=0) for x in ['train', 'val']}



enc, dec = initialize_model()

#defining optimizers for Generator and discriminator





# Setup the loss fxn
#criterion = loss_function

IT_model = ImageTextTrainer(enc, dec, dataloaders_dict, num_epochs, log_dir)

# Train and evaluate
IT_model.evaluate()
