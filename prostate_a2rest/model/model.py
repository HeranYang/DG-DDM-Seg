import logging
import torch
import torch.nn as nn
import numpy as np
import cv2
import os
from collections import OrderedDict
from torch.autograd import Variable

import model.networks as networks
from data.saliency_balancing_fusion import get_SBF_map

logger = logging.getLogger('base')


class VQDM(nn.Module):
    def __init__(self, opt):

        super().__init__()
        self.opt = opt
        self.truncation_forward = False

        self.begin_step = 0
        self.begin_epoch = 0

        # define network and load pretrained models
        self.device = torch.device('cuda' if opt['gpu_ids'] is not None else 'cpu')
        self.netG = self.set_device(networks.define_G(opt))

        if self.opt['phase'] == 'train':
            self.netG.train()

            # find the parameters to optimize
            # optim_params = filter(lambda p: p.requires_grad, self.netG.parameters())
            optim_params = self.netG.parameters()

            # define optimizer
            self.optG = torch.optim.AdamW(
                optim_params,
                lr=opt['train']['optimizer']['lr'],
                betas=tuple(opt['train']['optimizer']['betas']),
                weight_decay=opt['train']['optimizer']['weight_decay']
            )
            self.log_dict = OrderedDict()

        self.load_network()
        self.print_network()

    def feed_data(self, data):
        self.data = self.set_device(data)

    def set_device(self, x):
        if isinstance(x, dict):
            for key, item in x.items():
                if item is not None:
                    x[key] = item.to(self.device)
        elif isinstance(x, list):
            for item in x:
                if item is not None:
                    item = item.to(self.device)
        else:
            x = x.to(self.device)
        return x

    def optimize_parameters(self, train_data, current_epoch, current_step):

        self.optG.zero_grad()

        GLA_img1 = train_data['GLA1']
        LLA_img1 = train_data['LLA1']
        GLA_img2 = train_data['GLA2']
        LLA_img2 = train_data['LLA2']
        lbl      = train_data['target']
        plabel   = train_data['pesudolabel']
        Index    = train_data['Index']

        b, c, w, h = GLA_img1.shape
        grid_size = self.opt['model']['diffusion_config']['grid_size']

        # =========================================================================================
        # 1st forward & backward.
        input_var1 = Variable(GLA_img1.view(b, -1), requires_grad=True)
        input_var2 = Variable(GLA_img2.view(b, -1), requires_grad=True)

        input_data = {'ori_aug1': input_var1,
                      'ori_aug2': input_var2,
                      'target': lbl.view(b, -1),
                      'pesudolabel': plabel.view(b, -1),
                      'Index': Index}
        self.data = self.set_device(input_data)

        output = self.netG(self.data)
        output['loss'].backward(retain_graph=True)
        # =========================================================================================

        # produce mixed_images based on saliency_map.
        input_var1_grad = torch.reshape(input_var1.grad, (b, c, w, h))
        grad1 = torch.sqrt(torch.mean(input_var1_grad ** 2, dim=1, keepdim=True)).detach()
        saliency1 = get_SBF_map(grad1.to(self.device), grid_size)
        mixed_img1 = GLA_img1.detach().to(self.device) * saliency1 + LLA_img1.to(self.device) * (1 - saliency1)

        input_var2_grad = torch.reshape(input_var2.grad, (b, c, w, h))
        grad2 = torch.sqrt(torch.mean(input_var2_grad ** 2, dim=1, keepdim=True)).detach()
        saliency2 = get_SBF_map(grad2.to(self.device), grid_size)
        mixed_img2 = GLA_img2.detach().to(self.device) * saliency2 + LLA_img2.to(self.device) * (1 - saliency2)

        # =========================================================================================
        # 2nd forward & backward.
        aug_var1 = Variable(mixed_img1.view(b, -1), requires_grad=True)
        aug_var2 = Variable(mixed_img2.view(b, -1), requires_grad=True)

        input_data = {'ori_aug1': aug_var1,
                      'ori_aug2': aug_var2,
                      'target': lbl.view(b, -1),
                      'pesudolabel': plabel.view(b, -1),
                      'Index': Index}
        self.data = self.set_device(input_data)

        output = self.netG(self.data)
        output['loss'].backward()
        # =========================================================================================

        # update parameters.
        self.optG.step()

        # set log
        self.log_dict['ori_loss'] = output['loss'].item()
        self.log_dict['aug1_loss'] = output['aug1_loss'].item()
        self.log_dict['aug2_loss'] = output['aug2_loss'].item()
        self.log_dict['rp_loss'] = output['rp_loss'].item()
        self.log_dict['aug1_pl_loss'] = output['aug1_pl_loss'].item()
        self.log_dict['aug2_pl_loss'] = output['aug2_pl_loss'].item()

        # =========================================================================================
        # visualization code.
        if current_step % self.opt['train']['print_freq'] == 0:
            save_path = r'./experiments/visual_images'
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            tmp0 = np.concatenate((GLA_img1.detach().cpu().numpy()[0, 0],
                                   LLA_img1.detach().cpu().numpy()[0, 0],
                                   saliency1.detach().cpu().numpy()[0, 0],
                                   mixed_img1.detach().cpu().numpy()[0, 0]), axis=1)
            tmp1 = np.concatenate((GLA_img1.detach().cpu().numpy()[0, 1],
                                   LLA_img1.detach().cpu().numpy()[0, 1],
                                   saliency1.detach().cpu().numpy()[0, 0],
                                   mixed_img1.detach().cpu().numpy()[0, 1]), axis=1)
            tmp2 = np.concatenate((GLA_img1.detach().cpu().numpy()[0, 2],
                                   LLA_img1.detach().cpu().numpy()[0, 2],
                                   saliency1.detach().cpu().numpy()[0, 0],
                                   mixed_img1.detach().cpu().numpy()[0, 2]), axis=1)
            tmp3 = np.concatenate((GLA_img2.detach().cpu().numpy()[0, 0],
                                   LLA_img2.detach().cpu().numpy()[0, 0],
                                   saliency2.detach().cpu().numpy()[0, 0],
                                   mixed_img2.detach().cpu().numpy()[0, 0]), axis=1)
            tmp4 = np.concatenate((GLA_img2.detach().cpu().numpy()[0, 1],
                                   LLA_img2.detach().cpu().numpy()[0, 1],
                                   saliency2.detach().cpu().numpy()[0, 0],
                                   mixed_img2.detach().cpu().numpy()[0, 1]), axis=1)
            tmp5 = np.concatenate((GLA_img2.detach().cpu().numpy()[0, 2],
                                   LLA_img2.detach().cpu().numpy()[0, 2],
                                   saliency2.detach().cpu().numpy()[0, 0],
                                   mixed_img2.detach().cpu().numpy()[0, 2]), axis=1)
            tmp = np.concatenate((tmp0, tmp1, tmp2, tmp3, tmp4, tmp5), axis=0)

            visual_img = (tmp - np.min(tmp)) / (np.max(tmp) - np.min(tmp))  # to range [0,1]
            visual_img_int = (visual_img * 255.0).round()

            cv2.imwrite('{}/visual_epoch{:0>4d}_{}.png'.format(save_path, current_epoch, current_step),
                        cv2.cvtColor(visual_img_int, cv2.COLOR_RGB2BGR))
        # =========================================================================================


    def get_ema_model(self):
        return self.netG

    def p_sample_with_truncation(self, func, sample_type):
        truncation_rate = float(sample_type.replace('q', ''))
        def wrapper(*args, **kwards):
            out, out_pl, img_conv, pre_conv = func(*args, **kwards)
            import random
            if random.random() < truncation_rate:
                out, out_pl, img_conv, pre_conv = func(out, args[1], args[2], **kwards)
            return out, out_pl, img_conv, pre_conv
        return wrapper


    def predict_start_with_truncation(self, func, sample_type):
        if sample_type[-1] == 'p':
            truncation_k = int(sample_type[:-1].replace('top', ''))
            def wrapper(*args, **kwards):
                out, out_pl, img_conv, pre_conv = func(*args, **kwards)
                val, ind = out.topk(k = truncation_k, dim=1)
                probs = torch.full_like(out, -70)
                probs.scatter_(1, ind, val)
                return probs, out_pl, img_conv, pre_conv
            return wrapper
        elif sample_type[-1] == 'r':
            truncation_r = float(sample_type[:-1].replace('top', ''))
            def wrapper(*args, **kwards):
                out, out_pl, img_conv, pre_conv = func(*args, **kwards)
                # notice for different batches, out are same, we do it on out[0]
                temp, indices = torch.sort(out, 1, descending=True)
                temp1 = torch.exp(temp)
                temp2 = temp1.cumsum(dim=1)
                temp3 = temp2 < truncation_r
                new_temp = torch.full_like(temp3[:,0:1,:], True)
                temp6 = torch.cat((new_temp, temp3), dim=1)
                temp3 = temp6[:,:-1,:]
                temp4 = temp3.gather(1, indices.argsort(1))
                temp5 = temp4.float()*out+(1-temp4.float())*(-70)
                probs = temp5
                return probs, out_pl, img_conv, pre_conv
            return wrapper

        else:
            print("wrong sample type")


    @torch.no_grad()
    def generate_content(
            self,
            input,
            batch_size,
            filter_ratio=0.,
            sample_type="top0.85r",
    ):
        self.netG.eval()

        if len(sample_type.split(',')) > 1:
            if sample_type.split(',')[1][:1] == 'q':
                self.netG.p_sample = self.p_sample_with_truncation(self.netG.p_sample, sample_type.split(',')[1])

        if sample_type.split(',')[0][:3] == "top" and self.truncation_forward == False:
            self.netG.predict_start = self.predict_start_with_truncation(self.netG.predict_start,
                                                                         sample_type.split(',')[0])
            self.truncation_forward = True

        if len(sample_type.split(',')) == 2 and sample_type.split(',')[1][:4] == 'fast':
            trans_out = self.netG.sample_fast(input=input,
                                              batch_size=batch_size,
                                              filter_ratio=filter_ratio,
                                              return_logits=False,
                                              skip_step=int(sample_type.split(',')[1][4:]))

        else:
            trans_out = self.netG.sample(input=input,
                                         batch_size=batch_size,
                                         filter_ratio=filter_ratio,
                                         return_logits=False)

        content_token1 = trans_out['content_token1']  # (B,1,192,192)
        content_token2 = trans_out['content_token2']  # (B,1,192,192)

        self.netG.train()

        out = {
            'content_token1': content_token1,
            'content_token2': content_token2
        }
        self.out = out

        return out

    def get_out_visuals(self):
        out_dict = OrderedDict()

        out_dict['est_label_aug1'] = self.out['content_token1'].detach().float().cpu()
        out_dict['est_label_aug2'] = self.out['content_token2'].detach().float().cpu()

        out_dict['ori_aug1'] = self.data['ori_aug1'].detach().float().cpu()
        out_dict['ori_aug2'] = self.data['ori_aug2'].detach().float().cpu()
        out_dict['gt_label'] = self.data['target'].detach().float().cpu()
        out_dict['pesudolabel'] = self.data['pesudolabel'].detach().float().cpu()

        return out_dict


    @torch.no_grad()
    def sample(self,
               input,
               batch_size,
               filter_ratio = 0,
               return_logits=False,
               ):

        self.netG.eval()

        content_samples = {'ori_aug1': input['ori_aug1']}
        content_samples['ori_aug2'] = input['ori_aug2']
        content_samples['pesudolabel'] = input['pesudolabel']
        content_samples['gt_label'] = input['target']

        trans_out = self.netG.sample(input=input,
                                     batch_size=batch_size,
                                     filter_ratio=filter_ratio,
                                     return_logits=return_logits,
                                     )

        content_samples['est_label_aug1'] = trans_out['content_token1']
        content_samples['est_label_aug2'] = trans_out['content_token2']
        content_samples['est_pl1'] = trans_out['pl_token1']
        content_samples['est_pl2'] = trans_out['pl_token2']

        content_samples['img_conv1'] = trans_out['img_conv1']
        content_samples['img_conv2'] = trans_out['img_conv2']
        content_samples['pre_conv1'] = trans_out['pre_conv1']
        content_samples['pre_conv2'] = trans_out['pre_conv2']

        if return_logits:
            content_samples['logits1'] = trans_out['logits1']
            content_samples['logits2'] = trans_out['logits2']

        self.content_samples = content_samples

        self.netG.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self):
        out_dict = OrderedDict()

        out_dict['ori_aug1'] = self.content_samples['ori_aug1'].detach().float().cpu()
        out_dict['est_label_aug1'] = self.content_samples['est_label_aug1'].detach().float().cpu()
        out_dict['est_pl1'] = self.content_samples['est_pl1'].detach().float().cpu()

        out_dict['ori_aug2'] = self.content_samples['ori_aug2'].detach().float().cpu()
        out_dict['est_label_aug2'] = self.content_samples['est_label_aug2'].detach().float().cpu()
        out_dict['est_pl2'] = self.content_samples['est_pl2'].detach().float().cpu()

        out_dict['img_conv1'] = self.content_samples['img_conv1'].detach().float().cpu()
        out_dict['img_conv2'] = self.content_samples['img_conv2'].detach().float().cpu()
        out_dict['pre_conv1'] = self.content_samples['pre_conv1'].detach().float().cpu()
        out_dict['pre_conv2'] = self.content_samples['pre_conv2'].detach().float().cpu()

        out_dict['gt_label'] = self.data['target'].detach().float().cpu()
        out_dict['pesudolabel'] = self.data['pesudolabel'].detach().float().cpu()

        return out_dict


    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)

        logger.info(
            'Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        logger.info(s)

    def get_network_description(self, network):
        """Get the string and total parameters of the network"""
        if isinstance(network, nn.DataParallel):
            network = network.module
        s = str(network)
        n = sum(map(lambda x: x.numel(), network.parameters()))
        return s, n

    def save_network(self, epoch, iter_step):

        gen_path = os.path.join(
            self.opt['path']['checkpoint'], 'I{}_E{}_gen.pth'.format(iter_step, epoch))
        opt_path = os.path.join(
            self.opt['path']['checkpoint'], 'I{}_E{}_opt.pth'.format(iter_step, epoch))

        network = self.netG
        if isinstance(self.netG, nn.parallel.DistributedDataParallel):
            network = network.module

        # save network.
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, gen_path)

        # save network and optimizer.
        opt_state = {'epoch': epoch,
                     'iter': iter_step,
                     'scheduler': None,
                     'optimizer': self.optG.state_dict()}
        torch.save(opt_state, opt_path)

        logger.info('Saved model in [{:s}] ...'.format(gen_path))

    def load_network(self):
        load_path = self.opt['path']['resume_state']

        if load_path is not None:
            logger.info('Loading pretrained model for G [{:s}] ...'.format(load_path))
            gen_path = '{}_gen.pth'.format(load_path)
            opt_path = '{}_opt.pth'.format(load_path)

            # load network for testing.
            network = self.netG
            if isinstance(self.netG, nn.parallel.DistributedDataParallel):
                network = network.module
            network.load_state_dict(torch.load(gen_path), strict=False)

            # load optimizer for training.
            if self.opt['phase'] == 'train':
                opt = torch.load(opt_path)
                self.optG.load_state_dict(opt['optimizer'])

                self.begin_step = opt['iter']
                self.begin_epoch = opt['epoch']
