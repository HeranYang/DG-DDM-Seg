import time

import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
from core.wandb_logger import WandbLogger
from tensorboardX import SummaryWriter
import os
import numpy as np

join = os.path.join

if __name__ == "__main__":
    # print(torch.cuda.is_available())
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/Our_test_mp.json', help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'val'], default='train', help='Run either train(training) or val(generation)')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_wandb_ckpt', action='store_true')
    parser.add_argument('-log_eval', action='store_true')

    # parse configs
    args = parser.parse_args()
    # args.phase = 'val'
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)

    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    Logger.setup_logger(None, opt['path']['log'], 'train', level=logging.INFO, screen=True)
    Logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))
    tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])

    # Initialize WandbLogger
    if opt['enable_wandb']:
        import wandb

        wandb_logger = WandbLogger(opt)
        wandb.define_metric('validation/val_step')
        wandb.define_metric('epoch')
        wandb.define_metric("validation/*", step_metric="val_step")
        val_step = 0
    else:
        wandb_logger = None

    # dataset
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train' and args.phase != 'val':
            train_set = Data.create_dataset(dataset_opt, phase)
            train_loader = Data.create_dataloader(train_set, dataset_opt, phase)
        elif phase == 'val':
            val_set = Data.create_dataset(dataset_opt, phase)
            val_loader = Data.create_dataloader(val_set, dataset_opt, phase)
    logger.info('Initial Dataset Finished')

    # model
    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')

    # Train
    current_step = diffusion.begin_step
    current_epoch = diffusion.begin_epoch
    n_iter = opt['train']['n_iter']
    n_epoch = opt['train']['n_epoch']

    if opt['path']['resume_state']:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(current_epoch, current_step))

    num_classes = 2
    logger.info('Begin Model Evaluation.')
    avg_dice = np.zeros(num_classes - 1,)

    idx = 0
    result_dir = os.path.join(opt['path']['results'], 'evaluation')
    os.makedirs(result_dir, exist_ok=True)

    for _, val_data in enumerate(val_loader):
        
        start_time = time.time()
        idx += 1

        GLA_img1 = val_data['GLA1']
        LLA_img1 = val_data['LLA1']
        GLA_img2 = val_data['GLA2']
        LLA_img2 = val_data['LLA2']
        lbl      = val_data['target']
        plabel   = val_data['pesudolabel']
        Index    = val_data['Index']

        bs_size, c, w, h = GLA_img1.shape

        input_data = {'ori_aug1': GLA_img1.view(bs_size, -1),
                      'ori_aug2': GLA_img2.view(bs_size, -1),
                      'target': lbl.view(bs_size, -1),
                      'pesudolabel': plabel.view(bs_size, -1),
                      'Index': Index}

        diffusion.feed_data(input_data)
        diffusion.generate_content(diffusion.data,
                         batch_size=bs_size,
                         filter_ratio=0.,
                         sample_type="top0.85r,fast50",
                         )

        end_time = time.time()
        print("Runtime:" + str(end_time - start_time) + " sec")

        visuals = diffusion.get_out_visuals()

        img_size = opt['model']['diffusion_config']['embed_opt']['img_size']
        k_slice = opt['model']['diffusion_config']['embed_opt']['initconv_in_chans']

        ori_aug1 = Metrics.tensor2img_mp(visuals['ori_aug1'].view(bs_size, k_slice, img_size, img_size), min_max=(-10, 10))
        ori_aug1_int = (ori_aug1 * 255.0).round()
        ori_aug2 = Metrics.tensor2img_mp(visuals['ori_aug2'].view(bs_size, k_slice, img_size, img_size), min_max=(-10, 10))
        ori_aug2_int = (ori_aug2 * 255.0).round()

        est_label_aug1 = Metrics.tensor2img_mp(visuals['est_label_aug1'].view(bs_size, img_size, img_size), min_max=(0, num_classes))
        est_label_aug2 = Metrics.tensor2img_mp(visuals['est_label_aug2'].view(bs_size, img_size, img_size), min_max=(0, num_classes))

        gt_label = Metrics.tensor2img_mp(visuals['gt_label'].view(bs_size, img_size, img_size), min_max=(0, num_classes))
        gt_label_int = (gt_label * 255.0).round()
        pesudolabel = Metrics.tensor2img_mp(visuals['pesudolabel'].view(bs_size, img_size, img_size), min_max=(0, num_classes))
        pesudolabel_int = (pesudolabel * 255.0).round()

        for ibatch in range(bs_size):

            val_id = val_data['Index'][ibatch]
            path = val_set.ori_path[val_id]
            image_name = os.path.basename(path)

            print('{}/epoch{:0>3d}_fake_{}'.format(result_dir, current_epoch, image_name))

            label_dtype = (est_label_aug1[ibatch, :, :] * num_classes).dtype
            priority = np.arange(num_classes-1, 0, -1, dtype=label_dtype)
            fused_label = np.zeros_like(est_label_aug1[ibatch, :, :])
            for cls in priority:
                fused_label[est_label_aug1[ibatch, :, :] * num_classes == cls] = cls
                fused_label[est_label_aug2[ibatch, :, :] * num_classes == cls] = cls
            np.save('{}/{}_fake.npy'.format(result_dir, image_name), fused_label)
            np.save('{}/{}_ori.npy'.format(result_dir, image_name), ori_aug1[ibatch, :, :, 1])
            np.save('{}/{}_tg.npy'.format(result_dir, image_name), gt_label[ibatch, :, :] * num_classes)
            np.save('{}/{}_pl.npy'.format(result_dir, image_name), pesudolabel[ibatch, :, :] * num_classes)

            # visualization
            fused_label_int = (fused_label / num_classes * 255.0).round()
            visual_img = np.concatenate((ori_aug1_int[ibatch, :, :, 1],
                                         fused_label_int,
                                         gt_label_int[ibatch, :, :],
                                         pesudolabel_int[ibatch, :, :]), axis=1)
            Metrics.save_img(visual_img, '{}/{}_fake.png'.format(result_dir, image_name))

        final_time = time.time()
        print("Runtime:" + str(final_time - start_time) + " sec")

