import torch
import torchvision
import torch.nn as nn
from glob import glob
from platform import processor
import os
from models.unet import *
from loader.filesets import bouget21
from loader.meteonet import MeteonetDataset
from loader.samplers import meteonet_random_oversampler, meteonet_sequential_sampler
from torch.utils.data import DataLoader
from tqdm import tqdm
from os.path import join
import argparse
import matplotlib.pyplot as plt
from models.trajGRU import *
from models.refnet import *
from torchvision.transforms.functional import to_pil_image
import numpy as np
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from loader.utilities import calculate_CT, calculate_BS, map_to_classes



# args
class Args(argparse.Namespace):
    data_dir = 'data'
    wind_dir = None
    thresholds = [0.1,1,2.5]
    run_dir = 'runs'
    epochs = 20
    batch_size = 128
    lr_wd = ['0:8e-4,1e-5', '4:1e-4,5e-5'] # format is: [epoch:Learning rate, weight decay]
    num_workers = 8
    oversampling = 0.9
    snapshot_step = 5

args = Args()

## user parameters
# default value is commented
input_len    = 12 # 12
time_horizon = 6  # 6
stride       = input_len  # input_len
clip_grad    = 0.1  # 0.1

thresholds   = [100*k/12 for k in args.thresholds] #  unit: CRF over 5 minutes in 1/100 of mm (as meteonet data)
model_size   = 8 # to do
lr_wd = dict()
for a in args.lr_wd:
    k,u = a.split(':')
    a,b = u.split(',')
    lr_wd[int(k)]=float(a),float(b)

if torch.backends.cuda.is_built() and torch.cuda.is_available():
    device   = 'cuda'
elif torch.backends.mps.is_built():
    device    = 'mps'
else:
    device   = 'cpu'

print(f"""
Data params:
   {input_len = } (history of {input_len*5} minutes)
   {time_horizon = } (nowcasting at {time_horizon*5} minutes)
   {stride = }
   model = Unet classif
   model_size = ?
   {args.data_dir = }
   {args.wind_dir = }
   {len(thresholds)} classes ({thresholds=})

Train params:
   {args.epochs = }
   {args.batch_size = }
   {lr_wd = }
   {clip_grad = }

Others params:
   {device = }
   {args.snapshot_step = }
   {args.num_workers = }
   {args.run_dir = }
""")

# split in validation/test sets according to Section 4.1 from [1]
train_files, val_files, _ = bouget21(join(args.data_dir, 'rainmaps'))

# datasets
indexes = [join(args.data_dir,'train.npz'), join(args.data_dir,'val.npz')]
train_ds = MeteonetDataset( train_files, input_len, input_len + time_horizon, stride, wind_dir=args.wind_dir, cached=indexes[0], tqdm=tqdm)
val_ds   = MeteonetDataset( val_files, input_len, input_len + time_horizon, stride, wind_dir=args.wind_dir, cached=indexes[1], tqdm=tqdm)
val_ds.norm_factors = train_ds.norm_factors

# samplers for dataloaders
train_sampler = meteonet_random_oversampler( train_ds, thresholds[-1], args.oversampling)
val_sampler   = meteonet_sequential_sampler( val_ds)

# dataloaders
train_loader = DataLoader(train_ds, args.batch_size, sampler=train_sampler, num_workers=args.num_workers, pin_memory=True)
val_loader   = DataLoader(val_ds, args.batch_size, sampler=val_sampler, num_workers=args.num_workers, pin_memory=True)


print(f"""
size of train files/items/batch
     {len(train_files)} {len(train_ds)} {len(train_loader)}
size of  files/items/batch
     {len(val_files)} {len(val_ds)} {len(val_loader)}
""")

from collections import OrderedDict

myencoder_params = [
    [   # [in_channels, out_channels, kernel_size, stride, padding]
        OrderedDict({'conv1_leaky_1': [1, 8, 5, 3, 1]}),     # [1,128,128] -> [8,42,42]
        OrderedDict({'conv2_leaky_1': [64, 192, 5, 3, 1]}),  # [64,42,42]  -> [192,14,14]
        OrderedDict({'conv3_leaky_1': [192, 192, 3, 2, 1]}), # [192,14,14] -> [192,7,7]
    ],

    [
        TrajGRU(input_channel=8, num_filter=64, b_h_w=(args.batch_size, 42, 42), zoneout=0.0, L=13,
                i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                h2h_kernel=(5, 5), h2h_dilate=(1, 1),
                act_type=activation('leaky', negative_slope=0.2, inplace=True)),

        TrajGRU(input_channel=192, num_filter=192, b_h_w=(args.batch_size, 14, 14), zoneout=0.0, L=13,
                i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                h2h_kernel=(5, 5), h2h_dilate=(1, 1),
                act_type=activation('leaky', negative_slope=0.2, inplace=True)),
        TrajGRU(input_channel=192, num_filter=192, b_h_w=(args.batch_size, 7, 7), zoneout=0.0, L=9,
                i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                h2h_kernel=(3, 3), h2h_dilate=(1, 1),
                act_type=activation('leaky', negative_slope=0.2, inplace=True))
    ]
]

myforecaster_params = [
    [
        OrderedDict({'deconv1_leaky_1': [192, 192, 4, 2, 1]}),
        OrderedDict({'deconv2_leaky_1': [192, 64, 5, 3, 1]}),
        OrderedDict({'deconv3_leaky_1': [64, 8, 5, 3, 0],
                     'conv3_leaky_2': [8, 8, 3, 1, 1],
                     'conv3_3': [8, 1, 1, 1, 0]
        }),
    ],

    [
        TrajGRU(input_channel=192, num_filter=192, b_h_w=(args.batch_size, 7, 7), zoneout=0.0, L=13,
                i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                h2h_kernel=(3, 3), h2h_dilate=(1, 1),
                act_type=activation('leaky', negative_slope=0.2, inplace=True)),

        TrajGRU(input_channel=192, num_filter=192, b_h_w=(args.batch_size, 14, 14), zoneout=0.0, L=13,
                i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                h2h_kernel=(5, 5), h2h_dilate=(1, 1),
                act_type=activation('leaky', negative_slope=0.2, inplace=True)),
        TrajGRU(input_channel=64, num_filter=64, b_h_w=(args.batch_size, 42, 42), zoneout=0.0, L=9,
                i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                h2h_kernel=(5, 5), h2h_dilate=(1, 1),
                act_type=activation('leaky', negative_slope=0.2, inplace=True))
    ]
]

encoder = Encoder(myencoder_params[0], myencoder_params[1]).to(device)
forecaster = Forecaster(myforecaster_params[0], myforecaster_params[1])
encoder_forecaster = EF(encoder, forecaster).to(device)

#refnet = RefNet(1)

loss = nn.MSELoss()
loss.to(device)

train_losses = []
val_losses = []
val_f1, val_bias, val_ts = [], [], []

writer = SummaryWriter(log_dir='runs_2s')


encoder_forecaster.train()
train_loss = 0
N = 0
optimizer = Adam(encoder_forecaster.parameters(), lr=1e-4)




"""
for batch in tqdm(val_loader, unit=' batches'):
    x, y = batch['inputs'], batch['target']
    print("x:", x.size())
    print("y:", y.size())
    x = x[:, :, None, :,]
    y = y[:, None, :, :,]

    x = x.transpose(1,0)

    x,y = x.to(device), y.to(device)

    print("after x:", x.size())
    print("after y:", y.size())

    output = encoder_forecaster(x)

    print("out:", output.size())
"""


for epoch in range(args.epochs):
    train_loss = 0
    N = 0
    encoder_forecaster.train()
    #refnet.train()

    for batch in tqdm(train_loader, unit=' batches'):
        x,y = batch['inputs'], batch['target']

        x = x[:, :, None, :,] # add the channel dim=1
        y = y[:, None, :, :,] # add the channel dim=1

        x = x.transpose(1,0)

        x,y = x.to(device), y.to(device)

        optimizer.zero_grad()


        print(x.size())
        output = encoder_forecaster(x)[0]
        print(output.size())
        

        l = loss(output, y)
        l.backward()
        torch.nn.utils.clip_grad_value_(encoder_forecaster.parameters(), clip_value=50.0)
        optimizer.step()

        train_loss += l.item()

        N += x.shape[0]

    if N != 0:
        train_loss /= N
        print(epoch+1, train_loss)
        train_losses.append(train_loss)

    encoder_forecaster.eval()
    val_loss = 0
    CT_pred = 0
    RMSE_pred = 0
    N = 0


    with torch.no_grad():
        for batch in tqdm(val_loader, unit=' batches'):
            x,y = batch['inputs'], batch['target']
            x = x[:, :, None, :,]
            y = y[:, None, :, :,]
            x = x.transpose(1,0)
            x,y = x.to(device), y.to(device)

            output = encoder_forecaster(x)[0]

        l = loss(output, y)
        val_loss += l.item()
        print("val loss:", l)
        #CT_pred += calculate_CT(map_to_classes(y_hat, thresholds), map_to_classes(y, thresholds))
        #RMSE_pred += ((y-y_hat)**2).mean()
        N += x.shape[0]

    #f1_pred, bias, ts =  calculate_BS( CT_pred, ['F1','BIAS','TS'])

    if N != 0:
        RMSE_pred = ((RMSE_pred)/N)*0.5
        val_loss /= N
        val_losses.append(val_loss)
        print(epoch+1, val_loss)

    #val_f1.append(f1_pred)
    #val_bias.append(bias)
    #val_ts.append(ts)

    print(f'epoch {epoch+1} {val_loss=} ') #{f1_pred=} {f1_pers=}')

    writer.add_scalar('train', train_loss, epoch)
    writer.add_scalar('val', val_loss, epoch)
    #for c in range(len(thresholds)):
    #    writer.add_scalar(f'F1_C{c+1}', f1_pred[c], epoch)
    #    writer.add_scalar(f'TS_C{c+1}', ts[c], epoch)
    #    writer.add_scalar(f'BIAS_C{c+1}', bias[c], epoch)
    #    writer.add_scalar('RMSE', RMSE_pred, epoch)

torch.save(encoder_forecaster.state_dict(), join('runs_2s', "model_last_epoch_ref.pt"))