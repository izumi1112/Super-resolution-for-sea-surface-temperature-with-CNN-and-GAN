import torch
import logging
import models.modules.SRResNet_arch as SRResNet_arch
import models.modules.discriminator_vgg_arch as SRGAN_arch
import models.modules.RRDBNet_arch as RRDBNet_arch
import models.modules.RRDBNet_arch_input_missing_filter as RRDBNet_arch_input_missing_filter
import models.modules.SRCNN_arch as SRCNN_arch

import models.modules.RCAN_arch as RCAN_arch
import models.modules.SANSISR_arch as SANSISR_arch
# import models.modules.SAN_arch as SAN_arch
logger = logging.getLogger('base')


####################
# define network
####################
#### Generator
def define_G(opt):
    opt_net = opt['network_G']
    which_model = opt_net['which_model_G']

    if which_model == 'MSRResNet':
        netG = SRResNet_arch.MSRResNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                       nf=opt_net['nf'], nb=opt_net['nb'], upscale=opt_net['scale'])
    elif which_model == 'SRCNN':
        netG = SRCNN_arch.SRCNN(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                       nf=opt_net['nf'], nb=opt_net['nb'], upscale=opt_net['scale'])
    elif which_model == 'RRDBNet':
        netG = RRDBNet_arch.RRDBNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                    nf=opt_net['nf'], nb=opt_net['nb'])
    elif which_model == 'RRDBNet_input_missing_filter':
        netG = RRDBNet_arch_input_missing_filter.RRDBNet_input_missing_filter(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                    nf=opt_net['nf'], nb=opt_net['nb'])
    # elif which_model == 'SAN':
    #     netG = SAN_arch.SAN(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
    #                                 nf=opt_net['nf'], nb=opt_net['nb'], upscale=opt_net['scale'])
    elif which_model == 'RCAN':
        netG = RCAN_arch.RCAN(num_in_ch=opt_net['in_nc'], num_out_ch=opt_net['out_nc'],
                                    num_feat=opt_net['nf'], num_block=opt_net['nb'], upscale=opt_net['scale'])
    elif which_model == 'SANSISR':
        netG = SANSISR_arch.SAN()
    # elif which_model == 'sft_arch':  # SFT-GAN
    #     netG = sft_arch.SFT_Net()
    else:
        raise NotImplementedError('Generator model [{:s}] not recognized'.format(which_model))
    return netG


#### Discriminator
def define_D(opt):
    opt_net = opt['network_D']
    which_model = opt_net['which_model_D']

    if which_model == 'discriminator_vgg_128':
        netD = SRGAN_arch.Discriminator_VGG_128(in_nc=opt_net['in_nc'], nf=opt_net['nf'])
    else:
        raise NotImplementedError('Discriminator model [{:s}] not recognized'.format(which_model))
    return netD


#### Define Network used for Perceptual Loss
def define_F(opt, use_bn=False):
    gpu_ids = opt['gpu_ids']
    device = torch.device('cuda' if gpu_ids else 'cpu')
    # PyTorch pretrained VGG19-54, before ReLU.
    if use_bn:
        feature_layer = 49
    else:
        feature_layer = 34
    netF = SRGAN_arch.VGGFeatureExtractor(feature_layer=feature_layer, use_bn=use_bn,
                                          use_input_norm=True, device=device)
    netF.eval()  # No need to train
    return netF
