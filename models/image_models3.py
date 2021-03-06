from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torchvision
#import argparse
from torch.autograd import Variable
from torchvision import models
from models.utils1 import sample_z
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import torch.nn.functional as F

#args = parser.parse_args()
#args.cuda = not args.no_cuda and torch.cuda.is_available()
### Torchvision models ###
ngf=32 # based on specification 32*32*32 image bottle neck cant be changes
indim=1024# can be changed
model_names = ['resnet', 'alexnet', 'vgg', 'squeezenet', 'densenet', 'inception']
class GLU(nn.Module):
    def __init__(self):
        super(GLU, self).__init__()

    def forward(self, x):
        nc = x.size(1)
        assert nc % 2 == 0, 'channels dont divide 2!'
        nc = int(nc/2)
        return x[:, :nc] * F.sigmoid(x[:, nc:])

def conv3x3(in_planes, out_planes):
    "3x3 convolution with padding" #only changes number of channels
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                     padding=1, bias=False)

# Upsale the spatial size by a factor of 2
def upBlock(in_planes, out_planes):
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        conv3x3(in_planes, out_planes * 2),
        nn.BatchNorm2d(out_planes * 2),
        GLU()
    )
    return block


# Downsale the spatial size by a factor of 2
def downBlock(in_planes, out_planes):
    block = nn.Sequential(
        nn.Conv2d(in_planes, out_planes, 4, 2, 1, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.LeakyReLU(0.2, inplace=True)
    )
    return block

class INIT_STAGE_G(nn.Module):
    def __init__(self, ngf):
        super(INIT_STAGE_G, self).__init__()
        self.gf_dim = ngf
        self.in_dim = indim
        self.define_module()

    def define_module(self):
        in_dim = self.in_dim
        ngf = self.gf_dim
        self.fc = nn.Sequential(
            nn.Linear(in_dim, ngf * 4 * 4 * 2, bias=False),
            nn.BatchNorm1d(ngf * 4 * 4 * 2),
            GLU())


        self.upsample1 = upBlock(ngf, ngf // 2)
        self.upsample2 = upBlock(ngf // 2, ngf // 4)
        self.upsample3 = upBlock(ngf // 4, ngf // 8)
        

    def forward(self, z_code):
        
        in_code = z_code
        # state size 8ngf x 4 x 4
        out_code = self.fc(in_code)
        out_code = out_code.view(-1, self.gf_dim, 4, 4)
        # state size 4ngf x 8 x 8
        out_code = self.upsample1(out_code)
        # state size 2ngf x 16 x 16
        out_code = self.upsample2(out_code)
        # state size ngf x 32 x 32
        out_code = self.upsample3(out_code)
       

        return out_code
    
class GET_IMAGE_G(nn.Module):
    def __init__(self, ngf):
        super(GET_IMAGE_G, self).__init__()
        self.gf_dim = ngf
        self.img = nn.Sequential(
            conv3x3(ngf, ngf),
            nn.Tanh()
        )

    def forward(self, h_code):
        out_img = self.img(h_code)
        return out_img

class G_NET(nn.Module):
    def __init__(self):
        super(G_NET, self).__init__()
        self.gf_dim = ngf
        self.define_module()

    def define_module(self):
            self.h_net1 = INIT_STAGE_G(self.gf_dim * 8)
            self.img_net1 = GET_IMAGE_G(self.gf_dim)
    def forward(self, text_embedding):
        #if cfg.GAN.B_CONDITION and text_embedding is not None:
        #    c_code, mu, logvar = self.ca_net(text_embedding)
        #else:
        #    c_code, mu, logvar = z_code, None, None
        h_code1 = self.h_net1(text_embedding)
        fake_img1 = self.img_net1(h_code1)
        return fake_img1
    

class CAE32(nn.Module):
    """
    This AE module will be fed 3x128x128 patches from the original image
    Shapes are (batch_size, channels, height, width)
    Latent representation: 32x32x32 bits per patch => 240KB per image (for 720p)
    """

    def __init__(self):
        super(CAE32, self).__init__()

        self.encoded = None

        # ENCODER

        # 64x64x64
        self.e_conv_1 = nn.Sequential(
            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(5, 5), stride=(2, 2)),
            nn.LeakyReLU()
        )

        # 128x32x32
        self.e_conv_2 = nn.Sequential(
            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5, 5), stride=(2, 2)),
            nn.LeakyReLU()
        )

        # 128x32x32
        self.e_block_1 = nn.Sequential(
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
            nn.LeakyReLU(),

            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
        )

        # 128x32x32
        self.e_block_2 = nn.Sequential(
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
            nn.LeakyReLU(),

            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
        )

        # 128x32x32
        self.e_block_3 = nn.Sequential(
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
            nn.LeakyReLU(),

            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
        )

        # 32x32x32
        self.e_conv_3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.Tanh()
        )

        # DECODER

        # 128x64x64
        self.d_up_conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1)),
            nn.LeakyReLU(),

            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.ConvTranspose2d(in_channels=64, out_channels=128, kernel_size=(2, 2), stride=(2, 2))
        )

        # 128x64x64
        self.d_block_1 = nn.Sequential(
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
            nn.LeakyReLU(),

            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
        )

        # 128x64x64
        self.d_block_2 = nn.Sequential(
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
            nn.LeakyReLU(),

            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
        )

        # 128x64x64
        self.d_block_3 = nn.Sequential(
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
            nn.LeakyReLU(),

            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
        )

        # 256x128x128
        self.d_up_conv_2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(3, 3), stride=(1, 1)),
            nn.LeakyReLU(),

            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.ConvTranspose2d(in_channels=32, out_channels=256, kernel_size=(2, 2), stride=(2, 2))
        )

        # 3x128x128
        self.d_up_conv_3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=16, kernel_size=(3, 3), stride=(1, 1)),
            nn.LeakyReLU(),

            nn.ReflectionPad2d((2, 2, 2, 2)),
            nn.Conv2d(in_channels=16, out_channels=3, kernel_size=(3, 3), stride=(1, 1)),
            nn.Tanh()
        )

    
    def encode(self,x):
        ec1 = self.e_conv_1(x)
        ec2 = self.e_conv_2(ec1)
        eblock1 = self.e_block_1(ec2) + ec2
        eblock2 = self.e_block_2(eblock1) + eblock1
        eblock3 = self.e_block_3(eblock2) + eblock2
        ec3 = self.e_conv_3(eblock3)  # in [-1, 1] from tanh activation

        # stochastic binarization
        with torch.no_grad():
            rand = torch.rand(ec3.shape).cuda()
            prob = (1 + ec3) / 2
            eps = torch.zeros(ec3.shape).cuda()
            eps[rand <= prob] = (1 - ec3)[rand <= prob]
            eps[rand > prob] = (-ec3 - 1)[rand > prob]

        # encoded tensor
        self.encoded = 0.5 * (ec3 + eps + 1)  # (-1|1) -> (0|1)
        return self.encoded


    def decode(self, encoded):
        y = encoded * 2.0 - 1  # (0|1) -> (-1|1)

        uc1 = self.d_up_conv_1(y)
        dblock1 = self.d_block_1(uc1) + uc1
        dblock2 = self.d_block_2(dblock1) + dblock1
        dblock3 = self.d_block_3(dblock2) + dblock2
        uc2 = self.d_up_conv_2(dblock3)
        dec = self.d_up_conv_3(uc2)

        return dec
    
    def forward(self, x):
        x = self.encode(x)
        return self.decode(x)

class CAE16(nn.Module):
    """
    This AE module will be fed 3x128x128 patches from the original image
    Shapes are (batch_size, channels, height, width)
    Latent representation: 16x16x16 bits per patch => 30KB per image (for 720p)
    """

    def __init__(self):
        super(CAE16, self).__init__()

        self.encoded = None

        # ENCODER

        # 64x64x64
        self.e_conv_1 = nn.Sequential(
            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(5, 5), stride=(2, 2)),
            nn.LeakyReLU()
        )

        # 128x32x32
        self.e_conv_2 = nn.Sequential(
            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5, 5), stride=(2, 2)),
            nn.LeakyReLU()
        )

        # 128x32x32
        self.e_block_1 = nn.Sequential(
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
            nn.LeakyReLU(),

            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
        )

        # 128x32x32
        self.e_block_2 = nn.Sequential(
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
            nn.LeakyReLU(),

            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
        )

        # 128x32x32
        self.e_block_3 = nn.Sequential(
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
            nn.LeakyReLU(),

            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
        )

        # 16x16x16
        self.e_conv_3 = nn.Sequential(
            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(in_channels=128, out_channels=16, kernel_size=(5, 5), stride=(2, 2)),
            nn.Tanh()
        )

        # DECODER

        # 128x32x32
        self.d_up_conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=(3, 3), stride=(1, 1)),
            nn.LeakyReLU(),

            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.ConvTranspose2d(in_channels=64, out_channels=128, kernel_size=(2, 2), stride=(2, 2))
        )

        # 128x32x32
        self.d_block_1 = nn.Sequential(
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
            nn.LeakyReLU(),

            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
        )

        # 128x32x32
        self.d_block_2 = nn.Sequential(
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
            nn.LeakyReLU(),

            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
        )

        # 128x32x32
        self.d_block_3 = nn.Sequential(
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
            nn.LeakyReLU(),

            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
        )

        # 256x64x64
        self.d_up_conv_2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(3, 3), stride=(1, 1)),
            nn.LeakyReLU(),

            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.ConvTranspose2d(in_channels=32, out_channels=256, kernel_size=(2, 2), stride=(2, 2))
        )

        # 3x128x128
        self.d_up_conv_3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=16, kernel_size=(3, 3), stride=(1, 1)),
            nn.LeakyReLU(),

            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.ConvTranspose2d(in_channels=16, out_channels=3, kernel_size=(2, 2), stride=(2, 2)),
            nn.Tanh()
        )

    def encode(self, x):
        ec1 = self.e_conv_1(x)
        ec2 = self.e_conv_2(ec1)
        eblock1 = self.e_block_1(ec2) + ec2
        eblock2 = self.e_block_2(eblock1) + eblock1
        eblock3 = self.e_block_3(eblock2) + eblock2
        ec3 = self.e_conv_3(eblock3)  # in [-1, 1] from tanh activation

        # stochastic binarization
        with torch.no_grad():
            rand = torch.rand(ec3.shape).cuda()
            prob = (1 + ec3) / 2
            eps = torch.zeros(ec3.shape).cuda()
            eps[rand <= prob] = (1 - ec3)[rand <= prob]
            eps[rand > prob] = (-ec3 - 1)[rand > prob]

        # encoded tensor
        self.encoded = 0.5 * (ec3 + eps + 1)  # (-1|1) -> (0|1)

        return self.encoded

    def decode(self, encoded):
        y = encoded * 2.0 - 1  # (0|1) -> (-1|1)

        uc1 = self.d_up_conv_1(y)
        dblock1 = self.d_block_1(uc1) + uc1
        dblock2 = self.d_block_2(dblock1) + dblock1
        dblock3 = self.d_block_3(dblock2) + dblock2
        uc2 = self.d_up_conv_2(dblock3)
        dec = self.d_up_conv_3(uc2)

        return dec
    
    def forward(self, x):
        x = self.encode(x)
        return self.decode(x)
    


class CAE8(nn.Module):
    """
    This AE module will be fed 3x128x128 patches from the original image
    Shapes are (batch_size, channels, height, width)
    Latent representation: 16x8x8 bits per patch => 7.5KB per image (for 720p)
    """

    def __init__(self):
        super(CAE8, self).__init__()

        self.encoded = None

        # ENCODER

        # 64x64x64
        self.e_conv_1 = nn.Sequential(
            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(5, 5), stride=(2, 2)),
            nn.LeakyReLU()
        )

        # 128x32x32
        self.e_conv_2 = nn.Sequential(
            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5, 5), stride=(2, 2)),
            nn.LeakyReLU()
        )

        # 128x32x32
        self.e_block_1 = nn.Sequential(
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
            nn.LeakyReLU(),

            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
        )

        # 128x16x16
        self.e_pool_1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )

        # 128x16x16
        self.e_block_2 = nn.Sequential(
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
            nn.LeakyReLU(),

            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
        )

        # 128x16x16
        self.e_block_3 = nn.Sequential(
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
            nn.LeakyReLU(),

            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
        )

        # 16x8x8
        self.e_conv_3 = nn.Sequential(
            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(in_channels=128, out_channels=16, kernel_size=(5, 5), stride=(2, 2)),
            nn.Tanh()
        )

        # DECODER

        # 128x16x16
        self.d_up_conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=(3, 3), stride=(1, 1)),
            nn.LeakyReLU(),

            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.ConvTranspose2d(in_channels=64, out_channels=128, kernel_size=(2, 2), stride=(2, 2))
        )

        # 128x16x16
        self.d_block_1 = nn.Sequential(
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
            nn.LeakyReLU(),

            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
        )

        # 128x32x32
        self.d_up_1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest')
        )

        # 128x32x32
        self.d_block_2 = nn.Sequential(
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
            nn.LeakyReLU(),

            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
        )

        # 128x32x32
        self.d_block_3 = nn.Sequential(
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
            nn.LeakyReLU(),

            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
        )

        # 256x64x64
        self.d_up_conv_2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(3, 3), stride=(1, 1)),
            nn.LeakyReLU(),

            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.ConvTranspose2d(in_channels=32, out_channels=256, kernel_size=(2, 2), stride=(2, 2))
        )

        # 3x128x128
        self.d_up_conv_3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=16, kernel_size=(3, 3), stride=(1, 1)),
            nn.LeakyReLU(),

            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.ConvTranspose2d(in_channels=16, out_channels=3, kernel_size=(2, 2), stride=(2, 2)),
            nn.Tanh()
        )

    def encode(self, x):
        # ENCODE
        ec1 = self.e_conv_1(x)
        ec2 = self.e_conv_2(ec1)
        eblock1 = self.e_block_1(ec2)
        eblock1 = self.e_pool_1(ec2 + eblock1)
        eblock2 = self.e_block_2(eblock1) + eblock1
        eblock3 = self.e_block_3(eblock2) + eblock2
        ec3 = self.e_conv_3(eblock3)  # in [-1, 1] from tanh activation

        # stochastic binarization
        with torch.no_grad():
            rand = torch.rand(ec3.shape).cuda()
            prob = (1 + ec3) / 2
            eps = torch.zeros(ec3.shape).cuda()
            eps[rand <= prob] = (1 - ec3)[rand <= prob]
            eps[rand > prob] = (-ec3 - 1)[rand > prob]

        # encoded tensor
        self.encoded = 0.5 * (ec3 + eps + 1)  # (-1|1) -> (0|1)

        return self.encoded

    def decode(self, enc):
        y = enc * 2.0 - 1  # (0|1) -> (-1, 1)

        uc1 = self.d_up_conv_1(y)
        dblock1 = self.d_block_1(uc1) + uc1
        dup1 = self.d_up_1(dblock1)
        dblock2 = self.d_block_2(dup1) + dup1
        dblock3 = self.d_block_3(dblock2) + dblock2
        uc2 = self.d_up_conv_2(dblock3)
        dec = self.d_up_conv_3(uc2)

        return dec
    def forward(self, x):
        x = self.encode(x)
        return self.decode(x)


class CAEB(nn.Module):
    """
    This AE module will be fed 3x128x128 patches from the original image
    Shapes are (batch_size, channels, height, width)
    Latent representation: 32x32x32 bits per patch => 240KB per image (for 720p)
    """

    def __init__(self):
        super(CAEB, self).__init__()

        self.encoded = None

        # ENCODER

        # 64x64x64
        self.e_conv_1 = nn.Sequential(
            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(5, 5), stride=(2, 2)),
            nn.LeakyReLU()
        )

        # 128x32x32
        self.e_conv_2 = nn.Sequential(
            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5, 5), stride=(2, 2)),
            nn.LeakyReLU()
        )

        # 128x32x32
        self.e_block_1 = nn.Sequential(
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
            nn.LeakyReLU(),

            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
        )

        # 128x32x32
        self.e_block_2 = nn.Sequential(
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
            nn.LeakyReLU(),

            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
        )

        # 128x32x32
        self.e_block_3 = nn.Sequential(
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
            nn.LeakyReLU(),

            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
        )

        # 32x32x32
        self.e_conv_3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.Tanh()
        )
        
        self.bottle_neck1 =nn.Sequential(
            nn.Linear(32*32*32, 32*32*16),
            nn.LeakyReLU())
        self.bottle_neck2 =nn.Sequential(
            nn.Linear(32*32*16, 32*32*4),
            nn.LeakyReLU())
            
        self.bottle_neck3 =nn.Sequential(
            nn.Linear(32*32*4, indim),
            nn.Tanh())
        self.reverse_bottle = G_NET()
        # DECODER

        # 128x64x64
        self.d_up_conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1)),
            nn.LeakyReLU(),

            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.ConvTranspose2d(in_channels=64, out_channels=128, kernel_size=(2, 2), stride=(2, 2))
        )

        # 128x64x64
        self.d_block_1 = nn.Sequential(
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
            nn.LeakyReLU(),

            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
        )

        # 128x64x64
        self.d_block_2 = nn.Sequential(
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
            nn.LeakyReLU(),

            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
        )

        # 128x64x64
        self.d_block_3 = nn.Sequential(
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
            nn.LeakyReLU(),

            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
        )

        # 256x128x128
        self.d_up_conv_2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(3, 3), stride=(1, 1)),
            nn.LeakyReLU(),

            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.ConvTranspose2d(in_channels=32, out_channels=256, kernel_size=(2, 2), stride=(2, 2))
        )

        # 3x128x128
        self.d_up_conv_3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=16, kernel_size=(3, 3), stride=(1, 1)),
            nn.LeakyReLU(),

            nn.ReflectionPad2d((2, 2, 2, 2)),
            nn.Conv2d(in_channels=16, out_channels=3, kernel_size=(3, 3), stride=(1, 1)),
            nn.Tanh()
        )

    
        
    def encode(self,x):
        ec1 = self.e_conv_1(x)
        ec2 = self.e_conv_2(ec1)
        eblock1 = self.e_block_1(ec2) + ec2
        eblock2 = self.e_block_2(eblock1) + eblock1
        eblock3 = self.e_block_3(eblock2) + eblock2
        ec3 = self.e_conv_3(eblock3)  # in [-1, 1] from tanh activation

        # stochastic binarization
        with torch.no_grad():
            rand = torch.rand(ec3.shape).cuda()
            prob = (1 + ec3) / 2
            eps = torch.zeros(ec3.shape).cuda()
            eps[rand <= prob] = (1 - ec3)[rand <= prob]
            eps[rand > prob] = (-ec3 - 1)[rand > prob]

        # encoded tensor
        self.encoded = 0.5 * (ec3 + eps + 1)  # (-1|1) -> (0|1)

        #return self.decode(self.encoded)
        return self.bottle_func(self.encoded)
            
    def bottle_func(self, encoded):
        #print("encoded.shape", encoded.shape)
        x= torch.flatten(encoded,1)
        #print("b1.shape", x.shape)
        x= self.bottle_neck1(x)
        #print("b2.shape", x.shape)
        x= self.bottle_neck2(x)
        #print("b3.shape", x.shape)
        comp= self.bottle_neck3(x)
        #print("comp.shape", comp.shape)
        
        #print("x.shape", x.shape)
        return comp

    def decode(self, encoded):
        y = encoded * 2.0 - 1  # (0|1) -> (-1|1)

        uc1 = self.d_up_conv_1(y)
        dblock1 = self.d_block_1(uc1) + uc1
        dblock2 = self.d_block_2(dblock1) + dblock1
        dblock3 = self.d_block_3(dblock2) + dblock2
        uc2 = self.d_up_conv_2(dblock3)
        dec = self.d_up_conv_3(uc2)

        return dec
    
    def forward(self, x):
        x = self.encode(x)
        x= self.reverse_bottle(x)
        return self.decode(x)
    
class CAEBo(nn.Module):
    """
    This AE module will be fed 3x128x128 patches from the original image
    Shapes are (batch_size, channels, height, width)
    Latent representation: 32x32x32 bits per patch => 240KB per image (for 720p)
    """

    def __init__(self):
        super(CAEBo, self).__init__()

        self.encoded = None

        # ENCODER

        # 64x64x64
        self.e_conv_1 = nn.Sequential(
            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(5, 5), stride=(2, 2)),
            nn.LeakyReLU()
        )

        # 128x32x32
        self.e_conv_2 = nn.Sequential(
            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5, 5), stride=(2, 2)),
            nn.LeakyReLU()
        )

        # 128x32x32
        self.e_block_1 = nn.Sequential(
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
            nn.LeakyReLU(),

            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
        )

        # 128x32x32
        self.e_block_2 = nn.Sequential(
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
            nn.LeakyReLU(),

            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
        )

        # 128x32x32
        self.e_block_3 = nn.Sequential(
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
            nn.LeakyReLU(),

            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
        )

        # 32x32x32
        self.e_conv_3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.LeakyReLU()
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((16, 16))
        self.bottle_neck =nn.Sequential(
            nn.Linear(32*16*16, indim),
            nn.LeakyReLU())
            
        self.reverse_bottle = G_NET()
        # DECODER

        # 128x64x64
        self.d_up_conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1)),
            nn.LeakyReLU(),

            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.ConvTranspose2d(in_channels=64, out_channels=128, kernel_size=(2, 2), stride=(2, 2))
        )

        # 128x64x64
        self.d_block_1 = nn.Sequential(
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
            nn.LeakyReLU(),

            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
        )

        # 128x64x64
        self.d_block_2 = nn.Sequential(
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
            nn.LeakyReLU(),

            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
        )

        # 128x64x64
        self.d_block_3 = nn.Sequential(
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
            nn.LeakyReLU(),

            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
        )

        # 256x128x128
        self.d_up_conv_2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(3, 3), stride=(1, 1)),
            nn.LeakyReLU(),

            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.ConvTranspose2d(in_channels=32, out_channels=256, kernel_size=(2, 2), stride=(2, 2))
        )

        # 3x128x128
        self.d_up_conv_3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=16, kernel_size=(3, 3), stride=(1, 1)),
            nn.LeakyReLU(),

            nn.ReflectionPad2d((2, 2, 2, 2)),
            nn.Conv2d(in_channels=16, out_channels=3, kernel_size=(3, 3), stride=(1, 1)),
            nn.Tanh()
        )

    def encode(self, x):
        ec1 = self.e_conv_1(x)
        ec2 = self.e_conv_2(ec1)
        eblock1 = self.e_block_1(ec2) + ec2
        eblock2 = self.e_block_2(eblock1) + eblock1
        eblock3 = self.e_block_3(eblock2) + eblock2
        ec3 = self.e_conv_3(eblock3)  # in [-1, 1] from tanh activation

        # stochastic binarization
        with torch.no_grad():
            rand = torch.rand(ec3.shape).cuda()
            prob = (1 + ec3) / 2
            eps = torch.zeros(ec3.shape).cuda()
            eps[rand <= prob] = (1 - ec3)[rand <= prob]
            eps[rand > prob] = (-ec3 - 1)[rand > prob]

        # encoded tensor
        self.encoded = 0.5 * (ec3 + eps + 1)  # (-1|1) -> (0|1)

        #return self.decode(self.encoded)
        return self.bottle_func(self.encoded)
            
    def bottle_func(self, encoded):
        #print("encoded.shape", encoded.shape)
        x= self.avgpool(encoded)
        #print("avg.shape", x.shape)
        x= torch.flatten(x,1)
        #print("b1.shape", x.shape)
        comp= self.bottle_neck(x)
        return comp
        
        

    def decode(self, encoded):
        y = encoded * 2.0 - 1  # (0|1) -> (-1|1)

        uc1 = self.d_up_conv_1(y)
        dblock1 = self.d_block_1(uc1) + uc1
        dblock2 = self.d_block_2(dblock1) + dblock1
        dblock3 = self.d_block_3(dblock2) + dblock2
        uc2 = self.d_up_conv_2(dblock3)
        dec = self.d_up_conv_3(uc2)

        return dec
    def forward(self, x):
        x = self.encode(x)
        x= self.reverse_bottle(x)
        return self.decode(x)

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, bn=False):
        super(ResBlock, self).__init__()

        if mid_channels is None:
            mid_channels = out_channels

        layers = [
            nn.ReLU(),
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0)]
        if bn:
            layers.insert(2, nn.BatchNorm2d(out_channels))
        self.convs = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.convs(x)


class vai(nn.Module):
    def __init__(self, d=64, kl_coef=0.1):
        super(vai, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, d // 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(d // 2, d, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
            ResBlock(d, d, bn=True),
            nn.BatchNorm2d(d),
            ResBlock(d, d, bn=True),
        )
        self.decoder = nn.Sequential(
            ResBlock(d, d, bn=True),
            nn.BatchNorm2d(d),
            ResBlock(d, d, bn=True),
            nn.BatchNorm2d(d),

            nn.ConvTranspose2d(d, d // 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d//2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(d // 2, 3, kernel_size=4, stride=2, padding=1, bias=False),
        )
        self.f = 32 # size of bottle neck image 32*32
        self.d = d
        self.fc11 = nn.Linear(d * self.f ** 2, (d//8) * self.f ** 2)
        self.fc12 = nn.Linear(d * self.f ** 2, (d//8) * self.f ** 2)
        self.fc13 = nn.Linear((d//8) * self.f ** 2,d * self.f ** 2)
        self.kl_coef = kl_coef
        self.kl_loss = 0
        self.mse = 0
    def encode(self, x):
        h1 = self.encoder(x)
        h1 = h1.view(-1, self.d * self.f ** 2)
        return self.fc11(h1), self.fc12(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        z = self.fc13(z)
        z = z.view(-1, self.d, self.f, self.f)
        h3 = self.decoder(z)
        return torch.tanh(h3)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    

class VAE(nn.Module):
    #This VAE is based on https://github.com/bhpfelix/Variational-Autoencoder-PyTorch/blob/master/src/vanila_vae.py
    def __init__(self, nc, ngf, ndf, latent_variable_size):
        super(VAE, self).__init__()

        self.nc = nc
        self.ngf = ngf
        self.ndf = ndf
        self.latent_variable_size = latent_variable_size

        # encoder
        self.e1 = nn.Conv2d(nc, ndf, 4, 2, 1)
        self.bn1 = nn.BatchNorm2d(ndf)

        self.e2 = nn.Conv2d(ndf, ndf*2, 4, 2, 1)
        self.bn2 = nn.BatchNorm2d(ndf*2)

        self.e3 = nn.Conv2d(ndf*2, ndf*4, 4, 2, 1)
        self.bn3 = nn.BatchNorm2d(ndf*4)

        self.e4 = nn.Conv2d(ndf*4, ndf*8, 4, 2, 1)
        self.bn4 = nn.BatchNorm2d(ndf*8)

        self.e5 = nn.Conv2d(ndf*8, ndf*8, 4, 2, 1)
        self.bn5 = nn.BatchNorm2d(ndf*8)

        self.fc1 = nn.Linear(ndf*8*4*4, latent_variable_size)
        self.fc2 = nn.Linear(ndf*8*4*4, latent_variable_size)

        # decoder
        self.d1 = nn.Linear(latent_variable_size, ngf*8*2*4*4)

        self.up1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd1 = nn.ReplicationPad2d(1)
        self.d2 = nn.Conv2d(ngf*8*2, ngf*8, 3, 1)
        self.bn6 = nn.BatchNorm2d(ngf*8, 1.e-3)

        self.up2 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd2 = nn.ReplicationPad2d(1)
        self.d3 = nn.Conv2d(ngf*8, ngf*4, 3, 1)
        self.bn7 = nn.BatchNorm2d(ngf*4, 1.e-3)

        self.up3 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd3 = nn.ReplicationPad2d(1)
        self.d4 = nn.Conv2d(ngf*4, ngf*2, 3, 1)
        self.bn8 = nn.BatchNorm2d(ngf*2, 1.e-3)

        self.up4 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd4 = nn.ReplicationPad2d(1)
        self.d5 = nn.Conv2d(ngf*2, ngf, 3, 1)
        self.bn9 = nn.BatchNorm2d(ngf, 1.e-3)

        self.up5 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd5 = nn.ReplicationPad2d(1)
        self.d6 = nn.Conv2d(ngf, nc, 3, 1)

        self.leakyrelu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h1 = self.leakyrelu(self.bn1(self.e1(x)))
        h2 = self.leakyrelu(self.bn2(self.e2(h1)))
        h3 = self.leakyrelu(self.bn3(self.e3(h2)))
        h4 = self.leakyrelu(self.bn4(self.e4(h3)))
        h5 = self.leakyrelu(self.bn5(self.e5(h4)))
        h5 = h5.view(-1, self.ndf*8*4*4)
        mu = self.fc1(h5)
        logvar = self.fc2(h5)
        z = self.reparametrize(mu, logvar)

        return z, mu, logvar

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        #eps = torch.cuda.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h1 = self.relu(self.d1(z))
        h1 = h1.view(-1, self.ngf*8*2, 4, 4)
        h2 = self.leakyrelu(self.bn6(self.d2(self.pd1(self.up1(h1)))))
        h3 = self.leakyrelu(self.bn7(self.d3(self.pd2(self.up2(h2)))))
        h4 = self.leakyrelu(self.bn8(self.d4(self.pd3(self.up3(h3)))))
        h5 = self.leakyrelu(self.bn9(self.d5(self.pd4(self.up4(h4)))))

        return self.sigmoid(self.d6(self.pd5(self.up5(h5))))

    def get_latent_var(self, x):
        mu, logvar = self.encode(x.view(-1, self.nc, self.ndf, self.ngf))
        z = self.reparametrize(mu, logvar)
        return z

    def forward(self, x):
        z, mu, logvar = self.encode(x.view(-1, self.nc, self.ndf, self.ngf))
        #z = self.reparametrize(mu, logvar)
        res = self.decode(z)
        return res, mu, logvar

  


########################################################################
#parameters
CHANNELS = 3
HEIGHT = 64
WIDTH = 64
EPOCHS = 20
LOG_INTERVAL = 500
HIDDEN = 256
#######################################################################
class Interpolate(nn.Module):
    def __init__(self, size, mode):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode
        
    def forward(self, x):
        x = self.interp(x, size=self.size, mode=self.mode, align_corners=False)
        return x

class encoder1(nn.Module):
    def __init__(self):
        super(encoder1, self).__init__()
        self.define_module()
    
    # Encoder
    # TODO : try with padding = 0
    def define_module(self):
        self.conv1 = nn.Conv2d(CHANNELS, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64, affine=True)
        self.conv3 = nn.Conv2d(64, CHANNELS, kernel_size=3, stride=1, padding=1)
        self.bottle_neck =nn.Sequential(
            nn.Linear(3*(HEIGHT//2)*(WIDTH//2), (HEIGHT//2)*(WIDTH//2)),
            nn.LeakyReLU(0.2))
        self.relu = nn.LeakyReLU(0.2)
        self.Tanh = nn.Tanh()
    def forward(self, x):
        out = self.relu(self.conv1(x))
        #out = nn.ReLU(self.conv1(x))
        out = self.relu(self.conv2(out))
        #out = nn.ReLU(self.conv2(out))
        out = self.bn1(out)
        out = self.conv3(out)
        out = torch.flatten(out,1)
        out = self.bottle_neck(out)
        return out

class com_enc(nn.Module):
    def __init__(self):
        super(com_enc, self).__init__()
    
    # Encoder
    # TODO : try with padding = 0
        self.conv1 = nn.Conv2d(CHANNELS, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64, affine=False)
        self.conv3 = nn.Conv2d(64, CHANNELS, kernel_size=3, stride=1, padding=1)
        self.bottle_neck =nn.Sequential(
            nn.Linear(3*(HEIGHT//2)*(WIDTH//2), HIDDEN),
            nn.ReLU())
        self.relu = nn.ReLU()
        self.Tanh = nn.Tanh()
    def encode(self, x):
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.bn1(out)
        out = self.conv3(out)
        out = torch.flatten(out,1)
        out = self.bottle_neck(out)
        return out
class com_dec(nn.Module):
    def __init__(self):
        super(com_dec, self).__init__()
        self.up_neck =nn.Sequential(
            nn.Linear(HIDDEN, 8*(HEIGHT//2)*(WIDTH//2)),
            nn.LeakyReLU())
        self.deconv0 = nn.Conv2d(8, CHANNELS, kernel_size=3, stride=1, padding=1)
        self.interpolate = Interpolate(size=HEIGHT, mode='bicubic')
        self.deconv1 = nn.Conv2d(CHANNELS, 64, 3, stride=1, padding=1)
        self.deconv2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64, affine=False)
    
        self.deconv_n = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.bn_n = nn.BatchNorm2d(64, affine=False)

    
        self.deconv3 = nn.ConvTranspose2d(64, CHANNELS, 3, stride=1, padding=1)
    
        self.relu = nn.ReLU()
        self.Tanh = nn.Tanh()
    def decode(self, z):
        z = self.up_neck(z)
        z = z.view(-1, 8, (HEIGHT//2), (WIDTH//2))
        z = self.deconv0(z)
        upscaled_image = self.interpolate(z)
        out = self.relu(self.deconv1(upscaled_image))
        out = self.relu(self.deconv2(out))
        out = self.bn2(out)
        for _ in range(10):
            out = self.relu(self.deconv_n(out))
            out = self.bn_n(out)
        out = self.deconv3(out)
        final = upscaled_image + out
        return final,out,upscaled_image, z
        
class comrec1(nn.Module):
  def __init__(self):
    super(comrec1, self).__init__()
    
    # Encoder
    # TODO : try with padding = 0
    self.enco = com_enc()
    self.deco = com_dec()
    
    
  def forward(self, x):
    com = self.enco.encode(x)
    final,out,upscaled_image, com_img = self.deco.decode(com)
    return final, out, upscaled_image, com_img, x
########################################################################

class com_enc2(nn.Module):
    def __init__(self):
        super(com_enc2, self).__init__()
    
    # Encoder
    # TODO : try with padding = 0
        self.conv1 = nn.Conv2d(CHANNELS, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64, affine=False)
        self.conv3 = nn.Conv2d(64, CHANNELS, kernel_size=3, stride=1, padding=1)
        self.bottle_neck1 =nn.Sequential(
            nn.Linear(3*(HEIGHT//2)*(WIDTH//2), HIDDEN),
            nn.Tanh())
        self.bottle_neck2 =nn.Sequential(
            nn.Linear(3*(HEIGHT//2)*(WIDTH//2), HIDDEN),
            nn.ReLU())
        self.relu = nn.ReLU()
        self.Tanh = nn.Tanh()
    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu
    def encode(self, x):
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.bn1(out)
        out = self.conv3(out)
        out = torch.flatten(out,1)
        mu = self.bottle_neck1(out)
        logvar = self.bottle_neck2(out)
        out = self.reparameterize(mu, logvar)
        return out, mu, logvar
class com_dec2(nn.Module):
    def __init__(self):
        super(com_dec2, self).__init__()
        self.up_neck =nn.Sequential(
            nn.Linear(HIDDEN, 8*(HEIGHT//2)*(WIDTH//2)),
            nn.LeakyReLU())
        self.deconv0 = nn.Conv2d(8, CHANNELS, kernel_size=3, stride=1, padding=1)
        self.interpolate = Interpolate(size=HEIGHT, mode='bicubic')
        self.deconv1 = nn.Conv2d(CHANNELS, 64, 3, stride=1, padding=1)
        self.deconv2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64, affine=False)
    
        self.deconv_n = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.bn_n = nn.BatchNorm2d(64, affine=False)

    
        self.deconv3 = nn.ConvTranspose2d(64, CHANNELS, 3, stride=1, padding=1)
    
        self.relu = nn.ReLU()
        self.Tanh = nn.Tanh()
    def decode(self, z):
        z = self.up_neck(z)
        z = z.view(-1, 8, (HEIGHT//2), (WIDTH//2))
        z = self.deconv0(z)
        upscaled_image = self.interpolate(z)
        out = self.relu(self.deconv1(upscaled_image))
        out = self.relu(self.deconv2(out))
        out = self.bn2(out)
        for _ in range(10):
            out = self.relu(self.deconv_n(out))
            out = self.bn_n(out)
        out = self.deconv3(out)
        final = upscaled_image + out
        return final,out,upscaled_image, z
        
class comrec2(nn.Module):
  def __init__(self):
    super(comrec2, self).__init__()
    
    # Encoder
    # TODO : try with padding = 0
    self.enco = com_enc2()
    self.deco = com_dec2()
    
    
  def forward(self, x):
    com, mu, logvar = self.enco.encode(x)
    final,out,upscaled_image, com_img = self.deco.decode(com)
    return final, out, upscaled_image, com_img, x, mu, logvar
########################################################################

    
class comrecCNN(nn.Module):
  def __init__(self):
    super(comrecCNN, self).__init__()
    
    # Encoder
    # TODO : try with padding = 0
    #CHANNEL X H X W 
    self.conv1 = nn.Conv2d(CHANNELS, 64, kernel_size=3, stride=1, padding=1)
    #64 X H X W 
    self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
    #64 X H/2 X W/2 
    self.bn1 = nn.BatchNorm2d(64, affine=False)
    #64 X H/2 X W/2 
    self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
    self.bn2 = nn.BatchNorm2d(64, affine=False)
    #64 X H/4 X W/4 
    self.conv4 = nn.Conv2d(64, CHANNELS, kernel_size=3, stride=1, padding=1)
    #CHANNEL X H/4 X W/4 
    
    # Decoder
    #TODO : try ConvTranspose2d
    #CHANNEL X H/4 X W/4 
    self.deconv0 = nn.ConvTranspose2d(CHANNELS, CHANNELS, 2, stride=2, padding=0)
    #CHANNEL X H/2 X W/2 
    self.interpolate = Interpolate(size=HEIGHT, mode='bilinear')
    #CHANNEL X H X W 
    self.deconv1 = nn.Conv2d(CHANNELS, 64, 3, stride=1, padding=1)
    #64 X H X W 
    self.deconv2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
    self.bn2 = nn.BatchNorm2d(64, affine=False)
    
    self.deconv_n = nn.Conv2d(64, 64, 3, stride=1, padding=1)
    self.bn_n = nn.BatchNorm2d(64, affine=False)

    
    self.deconv3 = nn.ConvTranspose2d(64, CHANNELS, 3, stride=1, padding=1)
    #CHANNEL X H X W 
    self.relu = nn.ReLU()
    self.Tanh = nn.Tanh()
    
  def encode(self, x):
    out = self.relu(self.conv1(x))
    out = self.relu(self.conv2(out))
    out = self.bn1(out)
    out = self.relu(self.conv3(out))
    out = self.bn2(out)
    return self.conv4(out)
    
  
  def reparameterize(self, mu, logvar):
    pass
  
  def decode(self, z):
    z = self.deconv0(z)
    upscaled_image = self.interpolate(z)
    out = self.relu(self.deconv1(upscaled_image))
    out = self.relu(self.deconv2(out))
    out = self.bn2(out)
    for _ in range(10):
      out = self.relu(self.deconv_n(out))
      out = self.bn_n(out)
    out = self.deconv3(out)
    final = upscaled_image + out
    return final,out,upscaled_image

    
  def forward(self, x):
    com_img = self.encode(x)
    final,out,upscaled_image = self.decode(com_img)
    return final, out, upscaled_image, com_img, x

#######################################
    
class auto_enc(nn.Module):
    def __init__(self):
        super(auto_enc, self).__init__()
        self.gf_dim = 64# cfg.GAN.GF_DIM
        self.define_module()

    # Encoder
    # TODO : try with padding = 0
    def define_module(self):
        ngf = 16*self.gf_dim
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(3, ngf//64, kernel_size=3, stride=1, padding=1)
        self.downsample1 = downBlock(ngf//64, ngf // 32)
        self.downsample2 = downBlock(ngf//32, ngf // 16)
        self.downsample3 = downBlock(ngf//16, ngf // 8)
        self.downsample4 = downBlock(ngf//8, ngf//4)
        self.downsample5 = downBlock(ngf//4, ngf // 2)
        self.downsample6 = downBlock(ngf//2, ngf)
        
        
        
    def forward(self, x):
        out = self.relu(self.conv1(x))
        
        out = self.downsample1(out)
        
        out = self.downsample2(out)
        
        out = self.downsample3(out)
        
        out = self.downsample4(out)
        
        
        out = self.downsample5(out)
        
        
        out = self.downsample6(out)
        
        out = out.view(out.size(0), -1)
        
        
        return out
    
class auto_dec(nn.Module):
    def __init__(self):
        super(auto_dec, self).__init__()
        self.gf_dim = 64
        self.define_module()

    def define_module(self):
        ngf = 16*self.gf_dim


        self.upsample1 = upBlock(ngf, ngf // 2)
        self.upsample2 = upBlock(ngf // 2, ngf // 4)
        self.upsample3 = upBlock(ngf // 4, ngf // 8)
        self.upsample4 = upBlock(ngf // 8, ngf // 16)
        self.upsample5 = upBlock(ngf // 16, ngf // 32)
        self.upsample6 = upBlock(ngf // 32, ngf // 64)
    
        self.img = nn.Sequential(
            conv3x3(ngf//64, 3),
            nn.Tanh()
        )

    def forward(self, in_code):
    
        # state size ngf x 1 x 1
        out_code = in_code.view(-1, 16*self.gf_dim, 1, 1)
        # state size ngf//2 x 2 x 2
        out_code = self.upsample1(out_code)
        # state size ngf//4 x 4 x 4
        out_code = self.upsample2(out_code)
        # state size ngf//8 x 8 x 8
        out_code = self.upsample3(out_code)
        # state size ngf//16 x 16 x 16
        out_code = self.upsample4(out_code)
        # state size ngf//32 x 32 x 32
        out_code = self.upsample5(out_code)
        # state size ngf//64 x 64 x 64
        out_code = self.upsample6(out_code)
        out_code = self.img(out_code)
        return out_code
        
        

class autoencdec(nn.Module):
  def __init__(self):
    super(autoencdec, self).__init__()
    
    # Encoder
    # TODO : try with padding = 0
    self.enco = auto_enc()
    self.deco = auto_dec()
    
    
  def forward(self, x):
    encoded = self.enco(x)
    decoded = self.deco(encoded)
    return decoded





###################This is identitx function for NN###########
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x
#################################################################

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


class ImageEncoder(nn.Module):
    def __init__(self, model, device, vae=False):
        super(ImageEncoder, self).__init__()

        self.model = model
        self.vae = vae
        self.device=device

        # if self.vae:
        #     self.vae_transform = nn.Sequential(nn.Linear(self.hid_dim, self.hid_dim * 2),
        #                                            nn.Tanh())

    def forward(self, x):
        enc = self.model(x)
        #print('encoder size:', enc.size())

        if self.vae:
            # z = self.vae_transform(enc)
            mu = enc[:,:enc.size()[1]//2]
            #print('mu size:', mu.size())
            log_var = enc[:,enc.size()[1]//2:]
            #print('log size:', log_var.size())

            if self.training:
                #print('training mode; taking samples')
                enc = sample_z(mu, log_var, self.device)
            else:
                #print('testing mode; taking mean')
                enc = mu
        else:
            mu, log_var = None, None
        #print('new encode size:', enc.size())

        return enc, mu, log_var


def initialize_torchvision_model(model_name, output_dim, feature_extract, device, use_pretrained=True, vae=False):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet18":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.avgpool = Identity()##############average pool replaced by identity####
        #num_ftrs = model_ft.fc.in_features
        num_ftrs = model_ft.fc.in_features*7*7 # as average pooling is replaced by Identity, infeatures are multipleid with kernel size
        if vae: #vae changes: reparameterization at vae size down samples by 2 so we multiply by 2 to maintain consistency in the pipeline
            model_ft.fc = nn.Linear(num_ftrs, output_dim*2) #vae changes
        else:#vae chages
            model_ft.fc = nn.Linear(num_ftrs, output_dim) #vae changes
        #model_ft.fc = nn.Linear(num_ftrs, output_dim) #vae changes
        input_size = 224
        
    elif model_name == "comrecCNN":
        """ comrecCNN
        """
        model_ft = comrecCNN()
        set_parameter_requires_grad(model_ft, feature_extract)
        input_size = 224
    elif model_name == "vai":
        """ vai
        """
        model_ft = vai()
        set_parameter_requires_grad(model_ft, feature_extract)
        input_size = 224
    elif model_name == "VAE":
        """ VAE
        """
        model_ft = VAE(nc=3, ngf=128, ndf=128, latent_variable_size=1024)
        set_parameter_requires_grad(model_ft, feature_extract)
        input_size = 224
    elif model_name == "comrec1":
        """ comrec
        """
        model_ft = comrec1()
        set_parameter_requires_grad(model_ft, feature_extract)
        input_size = 224
    elif model_name == "encoder1":
        """ encoder1
        """
        model_ft = encoder1()
        set_parameter_requires_grad(model_ft, feature_extract)
        input_size = 224
    elif model_name == "comrec2":
        """ comrec2
        """
        model_ft = comrec2()
        set_parameter_requires_grad(model_ft, feature_extract)
        input_size = 224
        
    elif model_name == "CAE8":
        """ CAE8
        """
        model_ft = CAE8()
        set_parameter_requires_grad(model_ft, feature_extract)
        input_size = 224
    elif model_name == "CAE16":
        """ CAE16
        """
        model_ft = CAE16()
        set_parameter_requires_grad(model_ft, feature_extract)
        input_size = 224
    elif model_name == "CAE32":
        """ CAE32
        """
        model_ft = CAE32()
        set_parameter_requires_grad(model_ft, feature_extract)
        input_size = 224
    elif model_name == "CAEBo":
        """ CAEBo
        """
        model_ft = CAEBo()
        set_parameter_requires_grad(model_ft, feature_extract)
        input_size = 224
    elif model_name == "AEC":
        """ AEC
        """
        model_ft = autoencdec()
        set_parameter_requires_grad(model_ft, feature_extract)
        input_size = 224
        
    elif model_name == "resnet50":
        """ Resnet50
        """
        model_ft = models.resnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        if vae: #vae changes: reparameterization at vae size down samples by 2 so we multiply by 2 to maintain consistency in the pipeline
            model_ft.fc = nn.Linear(num_ftrs, output_dim*2) #vae changes
        else:#vae chages
            model_ft.fc = nn.Linear(num_ftrs, output_dim) #vae changes
        #model_ft.fc = nn.Linear(num_ftrs, output_dim) #vae changes
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, output_dim)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, output_dim)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, output_dim, kernel_size=(1, 1), stride=(1, 1))
        model_ft.num_classes = output_dim
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, output_dim)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, output_dim)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, output_dim)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()


    #model_ft = ImageEncoder(model_ft, vae=vae, device=device)

    return model_ft, input_size

