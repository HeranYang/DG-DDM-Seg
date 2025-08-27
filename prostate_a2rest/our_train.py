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
    parser.add_argument('-c', '--config', type=str, default='config/Our_train.json', help='JSON file for configuration')
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

    
    
    while current_epoch < n_epoch:
        
        current_epoch += 1
        ## ================================== Train ======================================
        for _, train_data in enumerate(train_loader):
            current_step += 1
            
            # diffusion.feed_data(train_data)
            diffusion.optimize_parameters(train_data, current_epoch, current_step)

            if current_step % opt['train']['print_freq'] == 0:
                logs = diffusion.get_current_log()
                message = '<epoch:{:3d}, iter:{:8,d}> '.format(current_epoch, current_step)

                for k, v in logs.items():
                    message += '{:s}: {:.4e} '.format(k, v)
                    tb_logger.add_scalar(k, v, current_step)
                logger.info(message)

                if wandb_logger:
                    wandb_logger.log_metrics(logs)
        ## ===============================================================================

        ## ================================== Valid ======================================
        if current_epoch % opt['train']['val_freq'] == 0:
            num_classes = 5

            logger.info('epoch: {}, iter: {}, begin validation'.format(current_epoch, current_step))
            avg_dice = np.zeros(num_classes - 1, )

            idx = 0
            result_dir = opt['path']['results']
            os.makedirs(result_dir, exist_ok=True)

            for _, val_data in enumerate(val_loader):
                idx += 1

                GLA_img1 = val_data['GLA1']
                LLA_img1 = val_data['LLA1']
                GLA_img2 = val_data['GLA2']
                LLA_img2 = val_data['LLA2']
                lbl      = val_data['target']
                plabel   = val_data['pesudolabel']
                Index    = val_data['Index']

                b, c, w, h = GLA_img1.shape

                input_data = {'ori_aug1': GLA_img1.view(b, -1),
                              'ori_aug2': GLA_img2.view(b, -1),
                              'target': lbl.view(b, -1),
                              'pesudolabel': plabel.view(b, -1),
                              'Index': Index}

                diffusion.feed_data(input_data)
                diffusion.sample(diffusion.data,
                                 batch_size=1,
                                 filter_ratio=0,
                                 return_logits=False,
                                 )

                visuals = diffusion.get_current_visuals()

                img_size = opt['model']['diffusion_config']['embed_opt']['img_size']
                k_slice = opt['model']['diffusion_config']['embed_opt']['initconv_in_chans']

                ori_aug1 = Metrics.tensor2img(visuals['ori_aug1'].view(k_slice, img_size, img_size), min_max=(-10, 10))
                ori_aug1_int = (ori_aug1 * 255.0).round()
                est_label_aug1 = Metrics.tensor2img(visuals['est_label_aug1'].view(img_size, img_size), min_max=(0, num_classes))
                est_label_aug1_int = (est_label_aug1 * 255.0).round()
                est_pl1 = Metrics.tensor2img(visuals['est_pl1'].view(img_size, img_size), min_max=(0, num_classes))
                est_pl1_int = (est_pl1 * 255.0).round()

                ori_aug2 = Metrics.tensor2img(visuals['ori_aug2'].view(k_slice, img_size, img_size), min_max=(-10, 10))
                ori_aug2_int = (ori_aug2 * 255.0).round()
                est_label_aug2 = Metrics.tensor2img(visuals['est_label_aug2'].view(img_size, img_size), min_max=(0, num_classes))
                est_label_aug2_int = (est_label_aug2 * 255.0).round()
                est_pl2 = Metrics.tensor2img(visuals['est_pl2'].view(img_size, img_size), min_max=(0, num_classes))
                est_pl2_int = (est_pl2 * 255.0).round()

                gt_label = Metrics.tensor2img(visuals['gt_label'].view(img_size, img_size), min_max=(0, num_classes))
                gt_label_int = (gt_label * 255.0).round()
                pesudolabel = Metrics.tensor2img(visuals['pesudolabel'].view(img_size, img_size), min_max=(0, num_classes))
                pesudolabel_int = (pesudolabel * 255.0).round()

                # ======================================================================================================
                # save visual results.
                tmp0 = np.concatenate((ori_aug1_int[:, :, 1],
                                       est_label_aug1_int,
                                       gt_label_int,
                                       pesudolabel_int,
                                       est_pl1_int), axis=1)
                tmp1 = np.concatenate((ori_aug2_int[:, :, 1],
                                       est_label_aug2_int,
                                       gt_label_int,
                                       pesudolabel_int,
                                       est_pl2_int), axis=1)
                visual_img = np.concatenate((tmp0, tmp1), axis=0)

                # generation
                Metrics.save_img(visual_img, '{}/epoch{:0>3d}_{}_{}.png'.format(result_dir, current_epoch,
                                                                                current_step, idx))
                # ======================================================================================================

                # ======================================================================================================
                # save visual features.
                bs_size = 12
                ncount = 2
                bs_sizeh = bs_size // ncount

                img_conv1 = visuals['img_conv1'].view(bs_size, img_size, img_size)
                img_conv2 = visuals['img_conv2'].view(bs_size, img_size, img_size)
                img_conv1, img_conv2 = Metrics.tovisualimg(img_conv1, img_conv2)
                img_conv1_int = (img_conv1 * 255.0).round()
                img_conv2_int = (img_conv2 * 255.0).round()
                diff_conv1 = img_conv1 - np.repeat(img_conv1[:, :, 0:1], bs_size, axis=2)
                diff_conv2 = img_conv2 - np.repeat(img_conv2[:, :, 0:1], bs_size, axis=2)
                diff_conv1_int = (diff_conv1 * 255.0).round()
                diff_conv2_int = (diff_conv2 * 255.0).round()

                pre_conv1 = visuals['pre_conv1'].view(bs_size, img_size, img_size)
                pre_conv2 = visuals['pre_conv2'].view(bs_size, img_size, img_size)
                pre_conv1, pre_conv2 = Metrics.tovisualimg(pre_conv1, pre_conv2)
                pre_conv1_int = (pre_conv1 * pre_conv1 * 255.0).round()
                pre_conv2_int = (pre_conv2 * pre_conv2 * 255.0).round()

                for i in range(ncount):

                    feat_tmp0 = np.concatenate((img_conv1_int[:, :, i * bs_sizeh + 0],
                                                img_conv1_int[:, :, i * bs_sizeh + 1],
                                                img_conv1_int[:, :, i * bs_sizeh + 2],
                                                img_conv1_int[:, :, i * bs_sizeh + 3],
                                                img_conv1_int[:, :, i * bs_sizeh + 4],
                                                img_conv1_int[:, :, i * bs_sizeh + 5]), axis=1)
                    feat_tmp1 = np.concatenate((img_conv2_int[:, :, i * bs_sizeh + 0],
                                                img_conv2_int[:, :, i * bs_sizeh + 1],
                                                img_conv2_int[:, :, i * bs_sizeh + 2],
                                                img_conv2_int[:, :, i * bs_sizeh + 3],
                                                img_conv2_int[:, :, i * bs_sizeh + 4],
                                                img_conv2_int[:, :, i * bs_sizeh + 5]), axis=1)
                    feat_tmp = np.concatenate((feat_tmp0, feat_tmp1), axis=0)

                    prefeat_tmp0 = np.concatenate((pre_conv1_int[:, :, i * bs_sizeh + 0],
                                                   pre_conv1_int[:, :, i * bs_sizeh + 1],
                                                   pre_conv1_int[:, :, i * bs_sizeh + 2],
                                                   pre_conv1_int[:, :, i * bs_sizeh + 3],
                                                   pre_conv1_int[:, :, i * bs_sizeh + 4],
                                                   pre_conv1_int[:, :, i * bs_sizeh + 5]), axis=1)
                    prefeat_tmp1 = np.concatenate((pre_conv2_int[:, :, i * bs_sizeh + 0],
                                                   pre_conv2_int[:, :, i * bs_sizeh + 1],
                                                   pre_conv2_int[:, :, i * bs_sizeh + 2],
                                                   pre_conv2_int[:, :, i * bs_sizeh + 3],
                                                   pre_conv2_int[:, :, i * bs_sizeh + 4],
                                                   pre_conv2_int[:, :, i * bs_sizeh + 5]), axis=1)
                    prefeat_tmp = np.concatenate((prefeat_tmp0, prefeat_tmp1), axis=0)

                    difffeat_tmp0 = np.concatenate((diff_conv1_int[:, :, i * bs_sizeh + 0],
                                                    diff_conv1_int[:, :, i * bs_sizeh + 1],
                                                    diff_conv1_int[:, :, i * bs_sizeh + 2],
                                                    diff_conv1_int[:, :, i * bs_sizeh + 3],
                                                    diff_conv1_int[:, :, i * bs_sizeh + 4],
                                                    diff_conv1_int[:, :, i * bs_sizeh + 5]), axis=1)
                    difffeat_tmp1 = np.concatenate((diff_conv2_int[:, :, i * bs_sizeh + 0],
                                                    diff_conv2_int[:, :, i * bs_sizeh + 1],
                                                    diff_conv2_int[:, :, i * bs_sizeh + 2],
                                                    diff_conv2_int[:, :, i * bs_sizeh + 3],
                                                    diff_conv2_int[:, :, i * bs_sizeh + 4],
                                                    diff_conv2_int[:, :, i * bs_sizeh + 5]), axis=1)
                    difffeat_tmp = np.concatenate((difffeat_tmp0, difffeat_tmp1), axis=0)

                    if i == 0:
                        feat_img = feat_tmp
                        prefeat_img = prefeat_tmp
                        difffeat_img = difffeat_tmp
                    else:
                        feat_img = np.concatenate((feat_img, feat_tmp), axis=0)
                        prefeat_img = np.concatenate((prefeat_img, prefeat_tmp), axis=0)
                        difffeat_img = np.concatenate((difffeat_img, difffeat_tmp), axis=0)

                save_path = r'./experiments/visual_features'
                if not os.path.exists(save_path):
                    os.makedirs(save_path)

                Metrics.save_img(feat_img, '{}/epoch{:0>3d}_{}_{}_feat.png'.format(save_path, current_epoch,
                                                                                   current_step, idx))
                Metrics.save_img(prefeat_img, '{}/epoch{:0>3d}_{}_{}_prefeat.png'.format(save_path, current_epoch,
                                                                                   current_step, idx))
                Metrics.save_img(difffeat_img, '{}/epoch{:0>3d}_{}_{}_difffeat.png'.format(save_path, current_epoch,
                                                                                         current_step, idx))
                # ======================================================================================================

                # compute validation score.
                avg_dice += Metrics.calculate_dice(gt_label, est_label_aug1, num_classes)
                avg_dice += Metrics.calculate_dice(gt_label, est_label_aug2, num_classes)

                if wandb_logger:
                    wandb_logger.log_image(f'validation_{idx}', visual_img)


            avg_dice = avg_dice / idx / 2.

            dice_str = ' '.join(['{:.2f}'.format(d) for d in avg_dice])
            logger.info('# Validation # Dice: ' + dice_str)
            logger_val = logging.getLogger('val')  # validation logger
            logger_val.info('<epoch:{:3d}, iter:{:8,d}> dice: {}'.format(current_epoch,
                                                                         current_step,
                                                                         dice_str))
            # tensorboard logger
            tb_logger.add_scalar('dice', np.mean(avg_dice), current_step)

            if wandb_logger:
                wandb_logger.log_metrics({
                    'validation/val_dice': np.mean(avg_dice),
                    'validation/val_step': val_step
                })
                val_step += 1

            logger.info('validation completed')
        ## ===============================================================================

        if current_epoch % opt['train']['save_checkpoint_freq'] == 0:
            logger.info('Saving models and training states.')
            diffusion.save_network(current_epoch, current_step)

            if wandb_logger and opt['log_wandb_ckpt']:
                wandb_logger.log_checkpoint(current_epoch, current_step)

        if wandb_logger:
            wandb_logger.log_metrics({'epoch': current_epoch - 1})

    # save model
    logger.info('End of training.')