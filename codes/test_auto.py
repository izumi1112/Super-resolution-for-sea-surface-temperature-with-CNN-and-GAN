import os.path as osp
import logging
import time
import argparse
from collections import OrderedDict

import options.options as option
import utils.util as util
from data.util import bgr2ycbcr
from data import create_dataset, create_dataloader
from models import create_model

#### options
parser = argparse.ArgumentParser()
parser.add_argument('-opt', type=str, required=True, help='Path to options YMAL file.')
opt = option.parse(parser.parse_args().opt, is_train=False)
opt = option.dict_to_nonedict(opt)

########################################################################################
import os
import numpy as np
opt_name = opt['name']
train_name = opt['name'].replace('test', 'train')
experiments_path = '../experiments/' + train_name
experiments_list = os.listdir(experiments_path)
for e in experiments_list:
    if 'val' in e and '.log' in e:
        log_file = e
log_file_path = os.path.join(experiments_path, log_file)
with open(log_file_path) as f:
    lines = f.readlines()
    best_score = np.inf
    best_iters = 0
    for l in lines:
        rmse_score = float(l.split(' ')[-3])
        if rmse_score <= best_score:
            best_score = rmse_score
            best_iters = int(l.split(' ')[-9].replace(',', ''))
best_model = os.path.join(experiments_path, 'models', f'{best_iters}_G.pth')
opt['path']['pretrain_model_G'] = best_model
# print(best_model)
# exit()
#######################################################################################

util.mkdirs(
    (path for key, path in opt['path'].items()
     if not key == 'experiments_root' and 'pretrain_model' not in key and 'resume' not in key))
util.setup_logger('base', opt['path']['log'], 'test_' + opt['name'], level=logging.INFO,
                  screen=True, tofile=True)
logger = logging.getLogger('base')
logger.info(option.dict2str(opt))

#### Create test dataset and dataloader
test_loaders = []
for phase, dataset_opt in sorted(opt['datasets'].items()):
    test_set = create_dataset(dataset_opt)
    test_loader = create_dataloader(test_set, dataset_opt)
    logger.info('Number of test images in [{:s}]: {:d}'.format(dataset_opt['name'], len(test_set)))
    test_loaders.append(test_loader)

model = create_model(opt)
for test_loader in test_loaders:
    test_set_name = test_loader.dataset.opt['name']
    logger.info('\nTesting [{:s}]...'.format(test_set_name))
    test_start_time = time.time()
    dataset_dir = osp.join(opt['path']['results_root'], test_set_name)
    util.mkdir(dataset_dir)

    test_results = OrderedDict()
    test_results['rmse'] = []  # Add !
    test_results['psnr'] = []
    test_results['ssim'] = []
    test_results['psnr_y'] = []
    test_results['ssim_y'] = []
    test_results['rmse_umi'] = []  # Add !
    test_results['rmse_kaigan'] = []  # Add !

    for data in test_loader:
        need_GT = False if test_loader.dataset.opt['dataroot_GT'] is None else True
        model.feed_data(data, need_GT=need_GT)
        img_path = data['GT_path'][0] if need_GT else data['LQ_path'][0]
        img_name = osp.splitext(osp.basename(img_path))[0]

        model.test()
        visuals = model.get_current_visuals(need_GT=need_GT)

        sr_img = util.tensor2img(visuals['SR'])  # uint8

        # save images
        suffix = opt['suffix']
        if suffix:
            #save_img_path = osp.join(dataset_dir, img_name + suffix + '.png')
            save_img_path = osp.join(dataset_dir, img_name + suffix)
        else:
            #save_img_path = osp.join(dataset_dir, img_name + '.png')
            save_img_path = osp.join(dataset_dir, img_name)
        util.save_img(sr_img, save_img_path)

        # calculate PSNR and SSIM
        if need_GT:
            gt_img = util.tensor2img(visuals['GT'])
            gt_img = gt_img# / 255.
            sr_img = sr_img# / 255.

            crop_border = opt['crop_border'] if opt['crop_border'] else opt['scale']
            if crop_border == 0:
                cropped_sr_img = sr_img
                cropped_gt_img = gt_img
            else:
                cropped_sr_img = sr_img[crop_border:-crop_border, crop_border:-crop_border]
                cropped_gt_img = gt_img[crop_border:-crop_border, crop_border:-crop_border]
                #cropped_sr_img = sr_img[crop_border:-crop_border, crop_border:-crop_border, :]
                #cropped_gt_img = gt_img[crop_border:-crop_border, crop_border:-crop_border, :]

            rmse = util.calculate_rmse(cropped_sr_img, cropped_gt_img) / 255. * 44.
            psnr = util.calculate_psnr(cropped_sr_img, cropped_gt_img)
            ssim = util.calculate_ssim(cropped_sr_img, cropped_gt_img)
            rmse_umi, rmse_kaigan = util.calculate_rmse_2(cropped_sr_img, cropped_gt_img)

            #psnr = util.calculate_psnr(cropped_sr_img * 255, cropped_gt_img * 255)
            #ssim = util.calculate_ssim(cropped_sr_img * 255, cropped_gt_img * 255)

            test_results['rmse'].append(rmse)
            test_results['psnr'].append(psnr)
            test_results['ssim'].append(ssim)
            test_results['rmse_umi'].append(rmse_umi)
            test_results['rmse_kaigan'].append(rmse_kaigan)

            #if gt_img.shape[2] == 3:  # RGB image
            if False:
                sr_img_y = bgr2ycbcr(sr_img, only_y=True)
                gt_img_y = bgr2ycbcr(gt_img, only_y=True)
                if crop_border == 0:
                    cropped_sr_img_y = sr_img_y
                    cropped_gt_img_y = gt_img_y
                else:
                    cropped_sr_img_y = sr_img_y[crop_border:-crop_border, crop_border:-crop_border]
                    cropped_gt_img_y = gt_img_y[crop_border:-crop_border, crop_border:-crop_border]
                psnr_y = util.calculate_psnr(cropped_sr_img_y * 255, cropped_gt_img_y * 255)
                ssim_y = util.calculate_ssim(cropped_sr_img_y * 255, cropped_gt_img_y * 255)
                test_results['psnr_y'].append(psnr_y)
                test_results['ssim_y'].append(ssim_y)
                logger.info(
                    '{:20s} - PSNR: {:.6f} dB; SSIM: {:.6f}; PSNR_Y: {:.6f} dB; SSIM_Y: {:.6f}.'.
                    format(img_name, psnr, ssim, psnr_y, ssim_y))
            else:
                logger.info('{:20s} - RMSE: {:.6f}; RMSE_UMI: {:.6f}; RMSE_KAIGAN: {:.6f}; PSNR: {:.6f} dB; SSIM: {:.6f}.'.format(img_name, rmse, rmse_umi, rmse_kaigan, psnr, ssim))
        else:
            logger.info(img_name)

    if need_GT:  # metrics
        # Average PSNR/SSIM results
        ave_rmse = sum(test_results['rmse']) / len(test_results['rmse'])  # Add !
        ave_rmse_umi = sum(test_results['rmse_umi']) / len(test_results['rmse_umi'])  # Add !
        ave_rmse_kaigan = sum(test_results['rmse_kaigan']) / len(test_results['rmse_kaigan'])  # Add !
        ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
        ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
        logger.info(
            '----Average PSNR/SSIM results for {}----\n\tRMSE: {:.6f}; RMSE_UMI: {:.6f}; RMSE_KAIGAN: {:.6f}; PSNR: {:.6f} dB; SSIM: {:.6f}\n'.format(
                test_set_name, ave_rmse, ave_rmse_umi, ave_rmse_kaigan, ave_psnr, ave_ssim))
        if test_results['psnr_y'] and test_results['ssim_y']:
            ave_psnr_y = sum(test_results['psnr_y']) / len(test_results['psnr_y'])
            ave_ssim_y = sum(test_results['ssim_y']) / len(test_results['ssim_y'])
            logger.info(
                '----Y channel, average PSNR/SSIM----\n\tPSNR_Y: {:.6f} dB; SSIM_Y: {:.6f}\n'.
                format(ave_psnr_y, ave_ssim_y))
