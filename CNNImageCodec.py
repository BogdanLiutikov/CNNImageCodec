#This code is the simplest example of image compression based on neural networks
#Comparison with JPEG is provided as well
#It is a demonstation for Information Theory course
#Written by Evgeny Belyaev, February 2024.
import os
import math
import random
import numpy
from matplotlib import pyplot as plt
from PIL import Image
import imghdr

import torch
from torch import nn
from torch import optim
from torchvision import transforms
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


#import C-implementation of Witten&Neal&Cleary-1987 arithmetic coding as a external module
from EntropyCodec import *

#source folder with test images
testfolder = './test/'
#source folder with train images
trainfolder = './train/'
#size of test and train images
w=128
h=128
#If 0, then the training will be started, otherwise the model will be readed from a file
LoadModel = 1
#Training parameters
batch_size = 4
#Number of bits for representation of the layers sample in the training process
bt = 2
epochs = 100
#Model parameters
n1=128
n2=32
n3=16

#Number of images to be compressed and shown from the test folder
NumImagesToShow = 5

#Number of bits for representation of the layers sample
b = 2

lr=1e-3

model_name = f'model_b2_epochs100'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device =', device)

#Compute PSNR in RGB domain
def PSNR_RGB(image1,image2):
    width, height = image1.size
    I1 = numpy.array(image1.getdata()).reshape(image1.size[0], image1.size[1], 3)
    I2 = numpy.array(image2.getdata()).reshape(image2.size[0], image2.size[1], 3)
    I1 = numpy.reshape(I1, width * height * 3)
    I2 = numpy.reshape(I2, width * height * 3)
    I1 = I1.astype(float)
    I2 = I2.astype(float)
    mse = numpy.mean((I1 - I2) ** 2)
    if (mse == 0):  # MSE is zero means no noise is present in the signal .
        psnr=100.0
    else:
        max_pixel = 255.0
        psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    #print("PSNR = %5.2f dB" % psnr)
    return psnr

#Compute PSNR between two vectors
def PSNR(y_true, y_pred):
    max_pixel = 1.0
    return (10.0 * (1.0 / math.log(10)) * torch.log((max_pixel ** 2) / (torch.mean(torch.square(torch.tensor(y_pred - y_true)))))).item()

class CustomImageDataset(Dataset):
    def __init__(self, foldername, transform=None):
        self.foldername = foldername
        self.transform = transform
        self.image_files = self._load_image_files()
        self.w = w
        self.h = h

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.foldername, img_name)
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        return image

    def _load_image_files(self):
        dir_list = os.listdir(self.foldername)
        image_files = []
        for name in dir_list:
            fullname = os.path.join(self.foldername, name)
            filetype = imghdr.what(fullname)
            if filetype is not None:
                image_files.append(name)
        return image_files

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.gelu = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        out = self.conv2(self.gelu(self.conv1(x)))
        out += residual
        return out

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, n1, kernel_size=3, padding=1, stride=2),
            nn.LeakyReLU(),
            ResidualBlock(n1),

            nn.Conv2d(n1, n2, kernel_size=3, padding=1, stride=2),
            nn.LeakyReLU(),
            ResidualBlock(n2),

            nn.Conv2d(n2, n3, kernel_size=3, padding=1, stride=2),
            nn.LeakyReLU(),
        )
        
        self.decoder = nn.Sequential(
            ResidualBlock(n3),
            nn.ConvTranspose2d(n3, n2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(),

            ResidualBlock(n2),
            nn.ConvTranspose2d(n2, n1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(),

            ResidualBlock(n1),
            nn.ConvTranspose2d(n1, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
        
        self.bt = bt
        self.w = w
        self.h = h

    def forward(self, x):
        # Encoder
        e3 = self.encoder(x)
        # Adding noise during training
        if self.training:
            noise = torch.rand_like(e3) * (torch.max(e3) / (2 ** (self.bt + 1)))
            e3 = e3 + noise

        # Decoder
        x = self.decoder(e3)
        return x

def ImageCodecModel(trainfolder):
    model = Autoencoder().to(device)

    if LoadModel == 0:
        model.train()
        # Load images from folder and preprocess
        transform = transforms.Compose([
            transforms.Resize((w, h)),
            transforms.ToTensor()
        ])
        # Assuming CustomImageDataset is a function that loads images as a tensor
        dataset = CustomImageDataset(trainfolder, transform=transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        sheduler = optim.lr_scheduler.StepLR(optimizer, step_size=250, gamma=0.1)

        for epoch in range(epochs):
            running_loss = 0.0
            for data in dataloader:
                inputs = data
                inputs = inputs.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, inputs)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(f"Epoch {epoch+1}, Loss: {running_loss / len(dataloader)}")

        torch.save(model.state_dict(), f'model/{model_name}.pth')
    else:
        model.load_state_dict(torch.load(f'model/{model_name}.pth'))
        model.eval()

    return model.encoder, model.decoder


#Compresses input layer by multi-alphabet arithmetic coding using memoryless source model
def EntropyEncoder (filename,enclayers,size_z,size_h,size_w):
    temp = numpy.zeros((size_z, size_h, size_w), numpy.uint8, 'C')
    for z in range(size_z):
        for h in range(size_h):
            for w in range(size_w):
                temp[z][h][w] = enclayers[z][h][w]
    maxbinsize = (size_h * size_w * size_z)
    bitstream = numpy.zeros(maxbinsize, numpy.uint8, 'C')
    StreamSize = numpy.zeros(1, numpy.int32, 'C')
    HiddenLayersEncoder(temp, size_w, size_h, size_z, bitstream, StreamSize)
    name = filename
    path = './'
    fp = open(os.path.join(path, name), 'wb')
    out = bitstream[0:StreamSize[0]]
    out.astype('uint8').tofile(fp)
    fp.close()

#Decompresses input layer by multi-alphabet arithmetic coding using memoryless source model
def EntropyDecoder (filename,size_z,size_h,size_w):
    fp = open(filename, 'rb')
    bitstream = fp.read()
    fp.close()
    bitstream = numpy.frombuffer(bitstream, dtype=numpy.uint8)
    declayers = numpy.zeros((size_z, size_h, size_w), numpy.uint8, 'C')
    FrameOffset = numpy.zeros(1, numpy.int32, 'C')
    FrameOffset[0] = 0
    HiddenLayersDecoder(declayers, size_w, size_h, size_z, bitstream, FrameOffset)
    return declayers

#This function is searching for the JPEG quality factor (QF)
#which provides neares compression to TargetBPP
def JPEGRDSingleImage(X,TargetBPP,i):
    X = X*255
    image = Image.fromarray(X.astype('uint8'), 'RGB')
    width, height = image. size
    realbpp = 0
    realpsnr = 0
    realQ = 0
    for Q in range(101):
        image.save('test.jpeg', "JPEG", quality=Q)
        image_dec = Image.open('test.jpeg')
        bytesize = os.path.getsize('test.jpeg')
        bpp = bytesize*8/(width*height)
        psnr = PSNR_RGB(image, image_dec)
        if abs(realbpp-TargetBPP)>abs(bpp-TargetBPP):
            realbpp=bpp
            realpsnr=psnr
            realQ = Q
    JPEGfilename = 'image%i.jpeg' % i
    image.save(JPEGfilename, "JPEG", quality=realQ)
    return realQ, realbpp, realpsnr

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# Main function
if __name__ == '__main__':
    set_random_seed(42)

    transform = transforms.Compose([
        transforms.Resize((w, h)),
        transforms.ToTensor()
    ])
    test_dataset = CustomImageDataset(testfolder, transform=transform)
    test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

    for images in test_dataloader:
        xtest = images.to(device)
        break

    #Train the model
    encoder, decoder = ImageCodecModel(trainfolder)

    #Run the model for first NumImagesToShow images from the test set
    encoded_layers = encoder(xtest).detach().cpu().numpy()
    max_encoded_layers = numpy.zeros(NumImagesToShow, numpy.float16, 'C')

    #normalization the layer to interval [0,1)
    for i in range(NumImagesToShow):
        max_encoded_layers[i] = numpy.max(encoded_layers[i])
        encoded_layers[i] = encoded_layers[i] / max_encoded_layers[i]

    #Quantization of layer to b bits
    encoded_layers1 = (torch.clip(torch.tensor(encoded_layers), 0, 0.9999999) * pow(2, b)).to(torch.int32).numpy()

    #Encoding and decoding of each quantized layer by arithmetic coding
    bpp = numpy.zeros(NumImagesToShow, numpy.float16, 'C')
    declayers = numpy.zeros((NumImagesToShow, n3, 16, 16), numpy.uint8, 'C')
    for i in range(NumImagesToShow):
        binfilename = 'image%i.bin' % i
        EntropyEncoder(binfilename, encoded_layers1[i], n3, 16, 16)
        bytesize = os.path.getsize(binfilename)
        bpp[i] = bytesize * 8 / (w * h)
        declayers[i] = EntropyDecoder(binfilename,  n3, 16, 16)

    #Dequantization and denormalization of each layer
    print(bpp)
    shift = 1.0/pow(2, b+1)
    declayers = torch.tensor(declayers, dtype=torch.float32) / pow(2, b)
    declayers = declayers + shift
    encoded_layers_quantized = numpy.zeros((NumImagesToShow, n3, 16, 16), dtype=np.float32)
    for i in range(NumImagesToShow):
        encoded_layers_quantized[i] = declayers[i] * max_encoded_layers[i]
        encoded_layers[i] = encoded_layers[i] * max_encoded_layers[i]
    
    encoded_layers_tensor = torch.tensor(encoded_layers, dtype=torch.float32).to(device)
    encoded_layers_quantized_tensor = torch.tensor(encoded_layers_quantized, dtype=torch.float32).to(device)

    decoded_imgs = decoder(encoded_layers_tensor)
    decoded_imgsQ = decoder(encoded_layers_quantized_tensor)

    #Shows NumImagesToShow images from the test set
    #For each image the following results are presented
    #Original image
    #Image, represented by the model (without quantization)
    #Image, represented by the model with quantization and compression of the layers samples
    #Corresponding JPEG image at the same compression level
    
    xtest = xtest.detach().cpu().numpy().transpose((0, 2, 3, 1))
    decoded_imgs = decoded_imgs.detach().cpu().numpy().transpose((0, 2, 3, 1))
    decoded_imgsQ = decoded_imgsQ.detach().cpu().numpy().transpose((0, 2, 3, 1))
    
    for i in range(NumImagesToShow):
        title = ''
        plt.subplot(4, NumImagesToShow, i + 1).set_title(title, fontsize=10)
        plt.imshow(xtest[i, :, :, :], interpolation='nearest')
        plt.axis(False)
    for i in range(NumImagesToShow):
        psnr = PSNR(xtest[i, :, :, :], decoded_imgs[i, :, :, :])
        title = '%2.2f' % psnr
        plt.subplot(4, NumImagesToShow, NumImagesToShow + i + 1).set_title(title, fontsize=10)
        plt.imshow(decoded_imgs[i, :, :, :], interpolation='nearest')
        plt.axis(False)
    for i in range(NumImagesToShow):
        psnr = PSNR(xtest[i, :, :, :], decoded_imgsQ[i, :, :, :])
        title = '%2.2f %2.2f' % (psnr, bpp[i])
        plt.subplot(4, NumImagesToShow, 2*NumImagesToShow + i + 1).set_title(title, fontsize=10)
        plt.imshow(decoded_imgsQ[i, :, :, :], interpolation='nearest')
        plt.axis(False)
    for i in range(NumImagesToShow):
        JPEGQP,JPEGrealbpp, JPEGrealpsnr = JPEGRDSingleImage(xtest[i, :, :, :], bpp[i],i)
        JPEGfilename = 'image%i.jpeg' % i
        JPEGimage = Image.open(JPEGfilename)
        title = '%2.2f %2.2f' % (JPEGrealpsnr,JPEGrealbpp)
        plt.subplot(4, NumImagesToShow, 3*NumImagesToShow + i + 1).set_title(title, fontsize=10)
        plt.imshow(JPEGimage, interpolation='nearest')
        plt.axis(False)
    plt.savefig(f'result_{model_name}.png')
    plt.show()