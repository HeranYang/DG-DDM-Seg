# ------------------------------------------
# VQ-Diffusion
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# written By Shuyang Gu
# ------------------------------------------

import torch
from torch import nn
import torch.nn.functional as F

import numpy as np
from torch.cuda.amp import autocast

eps = 1e-8

def sum_except_batch(x, num_dims=1):
    return x.reshape(*x.shape[:num_dims], -1).sum(-1)

def l1_norm(x, y):
    L1_none = nn.L1Loss(reduction='none')
    return L1_none(x, y)

def log_1_min_a(a):
    return torch.log(1 - a.exp() + 1e-40)

def log_add_exp(a, b):
    maximum = torch.max(a, b)
    return maximum + torch.log(torch.exp(a - maximum) + torch.exp(b - maximum))

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def log_categorical(log_x_start, log_prob):
    return (log_x_start.exp() * log_prob).sum(dim=1)

def index_to_log_onehot(x, num_classes):
    assert x.max().item() < num_classes, \
        f'Error: {x.max().item()} >= {num_classes}'
    x_onehot = F.one_hot(x, num_classes)
    permute_order = (0, -1) + tuple(range(1, len(x.size())))
    x_onehot = x_onehot.permute(permute_order)
    log_x = torch.log(x_onehot.float().clamp(min=1e-30))
    return log_x

def log_onehot_to_index(log_x):
    return log_x.argmax(1)

def alpha_schedule(time_step, N=100, att_1 = 0.99999, att_T = 0.000009, ctt_1 = 0.000009, ctt_T = 0.99999):
    # att: alpha_t hat
    att = np.arange(0, time_step)/(time_step-1)*(att_T - att_1) + att_1
    att = np.concatenate(([1], att))
    # at: alpha_t
    at = att[1:]/att[:-1]

    # ctt: gamma_t hat
    ctt = np.arange(0, time_step)/(time_step-1)*(ctt_T - ctt_1) + ctt_1
    ctt = np.concatenate(([0], ctt))
    # ct: gamma_t
    one_minus_ctt = 1 - ctt
    one_minus_ct = one_minus_ctt[1:] / one_minus_ctt[:-1]
    ct = 1-one_minus_ct

    # bt: beta_t
    bt = (1-at-ct)/N
    att = np.concatenate((att[1:], [1]))
    ctt = np.concatenate((ctt[1:], [0]))
    # btt: beta_t hat
    btt = (1-att-ctt)/N

    return at, bt, ct, att, btt, ctt

class DiffusionTransformer(nn.Module):
    def __init__(
            self,
            transformer,
            content_seq_len=36864,
            num_classes=4,
            diffusion_step=100,
            auxiliary_loss_weight=0,
            adaptive_auxiliary_loss=False,
            mask_weight=[1,1],
    ):
        super().__init__()

        self.transformer = transformer
        self.content_seq_len = content_seq_len
        self.amp = False

        self.num_classes = num_classes + 1  # added 1 represents the MASK label.

        self.shape = content_seq_len
        self.num_timesteps = diffusion_step

        self.auxiliary_loss_weight = auxiliary_loss_weight
        self.adaptive_auxiliary_loss = adaptive_auxiliary_loss
        self.mask_weight = mask_weight

        self.loss_type = 'vb_stochastic'
        self.parametrization = 'x0'

        at, bt, ct, att, btt, ctt = alpha_schedule(self.num_timesteps, N=self.num_classes-1)

        at = torch.tensor(at.astype('float64'))
        bt = torch.tensor(bt.astype('float64'))
        ct = torch.tensor(ct.astype('float64'))
        log_at = torch.log(at)
        log_bt = torch.log(bt)
        log_ct = torch.log(ct)
        att = torch.tensor(att.astype('float64'))
        btt = torch.tensor(btt.astype('float64'))
        ctt = torch.tensor(ctt.astype('float64'))
        log_cumprod_at = torch.log(att)
        log_cumprod_bt = torch.log(btt)
        log_cumprod_ct = torch.log(ctt)

        log_1_min_ct = log_1_min_a(log_ct)
        log_1_min_cumprod_ct = log_1_min_a(log_cumprod_ct)

        assert log_add_exp(log_ct, log_1_min_ct).abs().sum().item() < 1.e-5
        assert log_add_exp(log_cumprod_ct, log_1_min_cumprod_ct).abs().sum().item() < 1.e-5

        self.diffusion_acc_list = [0] * self.num_timesteps
        self.diffusion_keep_list = [0] * self.num_timesteps
        # Convert to float32 and register buffers.
        self.register_buffer('log_at', log_at.float())
        self.register_buffer('log_bt', log_bt.float())
        self.register_buffer('log_ct', log_ct.float())
        self.register_buffer('log_cumprod_at', log_cumprod_at.float())
        self.register_buffer('log_cumprod_bt', log_cumprod_bt.float())
        self.register_buffer('log_cumprod_ct', log_cumprod_ct.float())
        self.register_buffer('log_1_min_ct', log_1_min_ct.float())
        self.register_buffer('log_1_min_cumprod_ct', log_1_min_cumprod_ct.float())

        self.register_buffer('Lt_history', torch.zeros(self.num_timesteps))
        self.register_buffer('Lt_count', torch.zeros(self.num_timesteps))
        self.zero_vector = None


    def multinomial_kl(self, log_prob1, log_prob2):   # compute KL loss on log_prob
        kl = (log_prob1.exp() * (log_prob1 - log_prob2)).sum(dim=1)
        return kl

    def q_pred_one_timestep(self, log_x_t, t):         # q(xt|xt_1)
        log_at = extract(self.log_at, t, log_x_t.shape)             # at
        log_bt = extract(self.log_bt, t, log_x_t.shape)             # bt
        log_ct = extract(self.log_ct, t, log_x_t.shape)             # ct
        log_1_min_ct = extract(self.log_1_min_ct, t, log_x_t.shape)          # 1-ct

        log_probs = torch.cat(
            [
                log_add_exp(log_x_t[:,:-1,:]+log_at, log_bt),
                log_add_exp(log_x_t[:, -1:, :] + log_1_min_ct, log_ct)
            ],
            dim=1
        )

        return log_probs

    def q_pred(self, log_x_start, t):           # q(xt|x0)
        # log_x_start can be onehot or not
        t = (t + (self.num_timesteps + 1))%(self.num_timesteps + 1)
        log_cumprod_at = extract(self.log_cumprod_at, t, log_x_start.shape)         # at~
        log_cumprod_bt = extract(self.log_cumprod_bt, t, log_x_start.shape)         # bt~
        log_cumprod_ct = extract(self.log_cumprod_ct, t, log_x_start.shape)         # ct~
        log_1_min_cumprod_ct = extract(self.log_1_min_cumprod_ct, t, log_x_start.shape)       # 1-ct~
        

        log_probs = torch.cat(
            [
                log_add_exp(log_x_start[:,:-1,:]+log_cumprod_at, log_cumprod_bt),
                log_add_exp(log_x_start[:,-1:,:]+log_1_min_cumprod_ct, log_cumprod_ct)
            ],
            dim=1
        )

        return log_probs

    def predict_start(self, log_x_t, image, pesudolabel, t):          # p(x0|xt)
        x_t = log_onehot_to_index(log_x_t)
        if self.amp == True:
            with autocast():
                out, out_pl, img_conv, pre_conv = self.transformer(x_t.float(), image, pesudolabel, t)
        else:
            out, out_pl, img_conv, pre_conv = self.transformer(x_t.float(), image, pesudolabel, t)

        assert out.size(0) == x_t.size(0)
        assert out.size(1) == self.num_classes-1
        assert out.size()[2:] == x_t.size()[1:]
        log_pred = F.log_softmax(out.double(), dim=1).float()
        batch_size = log_x_t.size()[0]
        if self.zero_vector is None or self.zero_vector.shape[0] != batch_size:
            self.zero_vector = torch.zeros(batch_size, 1, self.content_seq_len).type_as(log_x_t)- 70
        log_pred = torch.cat((log_pred, self.zero_vector), dim=1)
        log_pred = torch.clamp(log_pred, -70, 0)

        # ======================================================================================================
        log_pred_pl = F.log_softmax(out_pl.double(), dim=1).float()
        log_pred_pl = torch.cat((log_pred_pl, self.zero_vector), dim=1)
        log_pred_pl = torch.clamp(log_pred_pl, -70, 0)
        # ======================================================================================================

        return log_pred, log_pred_pl, img_conv, pre_conv

    def q_posterior(self, log_x_start, log_x_t, t):            # p_theta(xt_1|xt) = sum(q(xt-1|xt,x0')*p(x0'))
        # notice that log_x_t is onehot
        assert t.min().item() >= 0 and t.max().item() < self.num_timesteps
        batch_size = log_x_start.size()[0]
        onehot_x_t = log_onehot_to_index(log_x_t)
        mask = (onehot_x_t == self.num_classes-1).unsqueeze(1) 
        log_one_vector = torch.zeros(batch_size, 1, 1).type_as(log_x_t)
        log_zero_vector = torch.log(log_one_vector+1.0e-30).expand(-1, -1, self.content_seq_len)

        log_qt = self.q_pred(log_x_t, t)                                  # q(xt|x0)
        # log_qt = torch.cat((log_qt[:,:-1,:], log_zero_vector), dim=1)
        log_qt = log_qt[:,:-1,:]
        log_cumprod_ct = extract(self.log_cumprod_ct, t, log_x_start.shape)         # ct~
        ct_cumprod_vector = log_cumprod_ct.expand(-1, self.num_classes-1, -1)
        # ct_cumprod_vector = torch.cat((ct_cumprod_vector, log_one_vector), dim=1)
        log_qt = (~mask)*log_qt + mask*ct_cumprod_vector
        

        log_qt_one_timestep = self.q_pred_one_timestep(log_x_t, t)        # q(xt|xt_1)
        log_qt_one_timestep = torch.cat((log_qt_one_timestep[:,:-1,:], log_zero_vector), dim=1)
        log_ct = extract(self.log_ct, t, log_x_start.shape)         # ct
        ct_vector = log_ct.expand(-1, self.num_classes-1, -1)
        ct_vector = torch.cat((ct_vector, log_one_vector), dim=1)
        log_qt_one_timestep = (~mask)*log_qt_one_timestep + mask*ct_vector
        
        # log_x_start = torch.cat((log_x_start, log_zero_vector), dim=1)
        # q = log_x_start - log_qt
        q = log_x_start[:,:-1,:] - log_qt
        q = torch.cat((q, log_zero_vector), dim=1)
        q_log_sum_exp = torch.logsumexp(q, dim=1, keepdim=True)
        q = q - q_log_sum_exp
        log_EV_xtmin_given_xt_given_xstart = self.q_pred(q, t-1) + log_qt_one_timestep + q_log_sum_exp
        return torch.clamp(log_EV_xtmin_given_xt_given_xstart, -70, 0)

    def p_pred(self, log_x, image, pesudolabel, t):             # if x0, first p(x0|xt), than sum(q(xt-1|xt,x0)*p(x0|xt))
        if self.parametrization == 'x0':
            log_x_recon, log_x_recon_pl, img_conv, pre_conv = self.predict_start(log_x, image, pesudolabel, t)
            log_model_pred = self.q_posterior(
                log_x_start=log_x_recon, log_x_t=log_x, t=t)
        elif self.parametrization == 'direct':
            log_model_pred, log_x_recon_pl, img_conv, pre_conv = self.predict_start(log_x, image, pesudolabel, t)
        else:
            raise ValueError
        return log_model_pred, log_x_recon_pl, img_conv, pre_conv

    @torch.no_grad()
    def p_sample(self, log_x, image, pesudolabel, t):               # sample q(xt-1) for next step from  xt, actually is p(xt-1|xt)
        model_log_prob, log_x_recon_pl, img_conv, pre_conv = self.p_pred(log_x, image, pesudolabel, t)
        out = self.log_sample_categorical(model_log_prob)
        out_pl = self.log_sample_categorical(log_x_recon_pl)
        return out, out_pl, img_conv, pre_conv

    def log_sample_categorical(self, logits):           # use gumbel to sample onehot vector from log probability
        uniform = torch.rand_like(logits)
        gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
        sample = (gumbel_noise + logits).argmax(dim=1)
        log_sample = index_to_log_onehot(sample, self.num_classes)
        return log_sample

    def q_sample(self, log_x_start, t):                 # diffusion step, q(xt|x0) and sample xt
        log_EV_qxt_x0 = self.q_pred(log_x_start, t)

        log_sample = self.log_sample_categorical(log_EV_qxt_x0)

        return log_sample

    def sample_time(self, b, device, method='uniform'):
        if method == 'importance':
            if not (self.Lt_count > 10).all():
                return self.sample_time(b, device, method='uniform')

            Lt_sqrt = torch.sqrt(self.Lt_history + 1e-10) + 0.0001
            Lt_sqrt[0] = Lt_sqrt[1]  # Overwrite decoder term with L1.
            pt_all = Lt_sqrt / Lt_sqrt.sum()

            t = torch.multinomial(pt_all, num_samples=b, replacement=True)

            pt = pt_all.gather(dim=0, index=t)

            return t, pt

        elif method == 'uniform':
            t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

            pt = torch.ones_like(t).float() / self.num_timesteps
            return t, pt
        else:
            raise ValueError

    def _train_loss(self, x, img_aug1, img_aug2, pesudolabel, is_train=True):                       # get the KL loss
        b, device = x.size(0), x.device

        assert self.loss_type == 'vb_stochastic'
        x_start = x
        t, pt = self.sample_time(b, device, 'uniform')

        log_x_start = index_to_log_onehot(x_start, self.num_classes)
        log_xt = self.q_sample(log_x_start=log_x_start, t=t)
        xt = log_onehot_to_index(log_xt)

        mask_region = (xt == self.num_classes - 1).float()
        mask_weight = mask_region * self.mask_weight[0] + (1. - mask_region) * self.mask_weight[1]

        mask = (t == torch.zeros_like(t)).float()

        ## =============================================================================================================
        # PATH 1: for img_aug1
        log_x0_recon_aug1, log_x_recon_pl_aug1, img_conv_aug1, pre_conv_aug1 = self.predict_start(log_xt,
                                                                                                  img_aug1,
                                                                                                  pesudolabel,
                                                                                                  t=t)  # B x out_cls x 192*192, B x 144 x 256
        log_model_prob_aug1 = self.q_posterior(log_x_start=log_x0_recon_aug1,
                                               log_x_t=log_xt,
                                               t=t)  # go through q(xt_1|xt,x0)

        decoder_nll_aug1 = -log_categorical(log_x_start, log_model_prob_aug1)
        decoder_nll_aug1 = sum_except_batch(decoder_nll_aug1)

        kl_aux_aug1 = self.multinomial_kl(log_x_start[:, :-1, :], log_x0_recon_aug1[:, :-1, :])
        kl_aux_aug1 = kl_aux_aug1 * mask_weight
        kl_aux_aug1 = sum_except_batch(kl_aux_aug1)
        kl_aux_aug1_loss = mask * decoder_nll_aug1 + (1. - mask) * kl_aux_aug1

        aug1_loss = self.auxiliary_loss_weight * kl_aux_aug1_loss / pt  # B


        kl_aux_pl_aug1 = self.multinomial_kl(log_x_start[:, :-1, :], log_x_recon_pl_aug1[:, :-1, :])
        kl_aux_pl_aug1 = kl_aux_pl_aug1 * mask_weight
        kl_aux_pl_aug1 = sum_except_batch(kl_aux_pl_aug1)

        aug1_pl_loss = self.auxiliary_loss_weight * kl_aux_pl_aug1 / pt  # B
        ## =============================================================================================================

        ## =============================================================================================================
        # PATH 2: for img_aug2
        log_x0_recon_aug2, log_x_recon_pl_aug2, img_conv_aug2, pre_conv_aug2 = self.predict_start(log_xt,
                                                                                                  img_aug2,
                                                                                                  pesudolabel,
                                                                                                  t=t)  # P_theta(x0|xt), B x 144 x 256
        log_model_prob_aug2 = self.q_posterior(log_x_start=log_x0_recon_aug2,
                                               log_x_t=log_xt,
                                               t=t)  # go through q(xt_1|xt,x0)

        decoder_nll_aug2 = -log_categorical(log_x_start, log_model_prob_aug2)
        decoder_nll_aug2 = sum_except_batch(decoder_nll_aug2)

        kl_aux_aug2 = self.multinomial_kl(log_x_start[:, :-1, :], log_x0_recon_aug2[:, :-1, :])
        kl_aux_aug2 = kl_aux_aug2 * mask_weight
        kl_aux_aug2 = sum_except_batch(kl_aux_aug2)
        kl_aux_aug2_loss = mask * decoder_nll_aug2 + (1. - mask) * kl_aux_aug2

        aug2_loss = self.auxiliary_loss_weight * kl_aux_aug2_loss / pt  # i.e., 2e-3 * 1.0 * kl_aux_loss / pt


        kl_aux_pl_aug2 = self.multinomial_kl(log_x_start[:, :-1, :], log_x_recon_pl_aug2[:, :-1, :])
        kl_aux_pl_aug2 = kl_aux_pl_aug2 * mask_weight
        kl_aux_pl_aug2 = sum_except_batch(kl_aux_pl_aug2)

        aug2_pl_loss = self.auxiliary_loss_weight * kl_aux_pl_aug2 / pt  # i.e., 2e-3 * 1.0 * kl_aux_loss / pt
        ## =============================================================================================================

        ## =============================================================================================================
        # loss between aug1 and aug2.
        rp_loss = l1_norm(img_conv_aug1, img_conv_aug2)
        rp_loss = self.auxiliary_loss_weight * sum_except_batch(rp_loss) / pt  # 2e-3 * rp_loss / pt
        ## =============================================================================================================

        vb_loss = aug1_loss \
                  + aug2_loss \
                  + rp_loss      * 0.0002 \
                  + aug1_pl_loss * 0.1 \
                  + aug2_pl_loss * 0.1

        return log_model_prob_aug1, vb_loss, aug1_loss, aug2_loss, rp_loss, aug1_pl_loss, aug2_pl_loss


    @property
    def device(self):
        return self.transformer.to_logits[-1].weight.device

    def parameters(self, recurse=True, name=None):
        """
        Following minGPT:
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """
        return super().parameters(recurse=True)

    def forward(
            self, 
            input,
            **kwargs):

        if kwargs.get('autocast') == True:
            self.amp = True

        # 1) prepare input
        sample_label = input['target']
        image_aug1 = input['ori_aug1']
        image_aug2 = input['ori_aug2']
        pesudolabel = input['pesudolabel']

        # now we get cond_emb and sample_image
        log_model_prob, loss, aug1_loss, aug2_loss, rp_loss, aug1_pl_loss, aug2_pl_loss = self._train_loss(sample_label,
                                                                                                           image_aug1,
                                                                                                           image_aug2,
                                                                                                           pesudolabel)
        loss = loss.sum() / (sample_label.size()[0] * sample_label.size()[1]) / 2.

        # 4) get output, especially loss
        out = {}
        out['logits'] = torch.exp(log_model_prob)
        out['loss'] = loss
        out['aug1_loss'] = aug1_loss.sum() / (sample_label.size()[0] * sample_label.size()[1]) / 2.
        out['aug2_loss'] = aug2_loss.sum() / (sample_label.size()[0] * sample_label.size()[1]) / 2.
        out['rp_loss'] = rp_loss.sum() / (sample_label.size()[0] * sample_label.size()[1]) / 2. * 0.0002
        out['aug1_pl_loss'] = aug1_pl_loss.sum() / (sample_label.size()[0] * sample_label.size()[1]) / 2. * 0.1
        out['aug2_pl_loss'] = aug2_pl_loss.sum() / (sample_label.size()[0] * sample_label.size()[1]) / 2. * 0.1

        self.amp = False
        return out


    def sample(
            self,
            input,
            batch_size,
            filter_ratio = 0.5,
            return_logits = False,
            ):
    
        device = self.log_at.device
        start_step = int(self.num_timesteps * filter_ratio)

        # 1) prepare input
        sample_label = input['target']
        image_aug1 = input['ori_aug1']
        image_aug2 = input['ori_aug2']
        pesudolabel = input['pesudolabel']

        if start_step == 0:
            # use full mask sample
            zero_logits = torch.zeros((batch_size, self.num_classes - 1, self.shape), device=device)
            one_logits = torch.ones((batch_size, 1, self.shape), device=device)
            mask_logits = torch.cat((zero_logits, one_logits), dim=1)
            log_z1 = torch.log(mask_logits)
            log_z2 = torch.log(mask_logits)
            start_step = self.num_timesteps
            with torch.no_grad():
                for diffusion_index in range(start_step - 1, -1, -1):
                    t = torch.full((batch_size,), diffusion_index, device=device, dtype=torch.long)
                    log_z1, log_pl1, img_conv1, pre_conv1 = self.p_sample(log_z1, image_aug1, pesudolabel, t)  # log_z is log_onehot
                    log_z2, log_pl2, img_conv2, pre_conv2 = self.p_sample(log_z2, image_aug2, pesudolabel, t)  # log_z is log_onehot

        else:
            t = torch.full((batch_size,), start_step - 1, device=device, dtype=torch.long)
            log_x_start = index_to_log_onehot(pesudolabel.to(torch.int64), self.num_classes)
            log_xt = self.q_sample(log_x_start=log_x_start, t=t)
            log_z1 = log_xt
            log_z2 = log_xt
            with torch.no_grad():
                for diffusion_index in range(start_step - 1, -1, -1):
                    t = torch.full((batch_size,), diffusion_index, device=device, dtype=torch.long)
                    log_z1, log_pl1, img_conv1, pre_conv1 = self.p_sample(log_z1, image_aug1, pesudolabel, t)  # log_z is log_onehot
                    log_z2, log_pl2, img_conv2, pre_conv2 = self.p_sample(log_z2, image_aug2, pesudolabel, t)  # log_z is log_onehot

        content_token1 = log_onehot_to_index(log_z1)
        content_token2 = log_onehot_to_index(log_z2)

        pl_token1 = log_onehot_to_index(log_pl1)
        pl_token2 = log_onehot_to_index(log_pl2)

        output = {'content_token1': content_token1,
                  'content_token2': content_token2,
                  'pl_token1': pl_token1,
                  'pl_token2': pl_token2,
                  'img_conv1': img_conv1,
                  'img_conv2': img_conv2,
                  'pre_conv1': pre_conv1,
                  'pre_conv2': pre_conv2}

        if return_logits:
            output['logits1'] = torch.exp(log_z1)
            output['logits2'] = torch.exp(log_z2)

        return output

    def sample_fast(
            self,
            input,
            batch_size,
            filter_ratio = 0.5,
            return_logits = False,
            skip_step = 1,
            ):

        device = self.log_at.device
        start_step = int(self.num_timesteps * filter_ratio)

        # 1) prepare input
        sample_label = input['target']
        image_aug1 = input['ori_aug1']
        image_aug2 = input['ori_aug2']
        pesudolabel = input['pesudolabel']

        assert start_step == 0
        zero_logits = torch.zeros((batch_size, self.num_classes - 1, self.shape), device=device)
        one_logits = torch.ones((batch_size, 1, self.shape), device=device)
        mask_logits = torch.cat((zero_logits, one_logits), dim=1)
        log_z1 = torch.log(mask_logits)
        log_z2 = torch.log(mask_logits)
        start_step = self.num_timesteps

        with torch.no_grad():
            # skip_step = 1
            diffusion_list = [index for index in range(start_step-1, -1, -1-skip_step)]
            if diffusion_list[-1] != 0:
                diffusion_list.append(0)
            # for diffusion_index in range(start_step-1, -1, -1):
            for diffusion_index in diffusion_list:

                t = torch.full((batch_size,), diffusion_index, device=device, dtype=torch.long)
                log_x_recon1, log_pl1, img_conv1, pre_conv1 = self.predict_start(log_z1, image_aug1, pesudolabel, t)
                log_x_recon2, log_pl2, img_conv2, pre_conv2 = self.predict_start(log_z2, image_aug2, pesudolabel, t)

                if diffusion_index > skip_step:
                    model_log_prob1 = self.q_posterior(log_x_start=log_x_recon1, log_x_t=log_z1, t=t - skip_step)
                    model_log_prob2 = self.q_posterior(log_x_start=log_x_recon2, log_x_t=log_z2, t=t - skip_step)
                else:
                    model_log_prob1 = self.q_posterior(log_x_start=log_x_recon1, log_x_t=log_z1, t=t)
                    model_log_prob2 = self.q_posterior(log_x_start=log_x_recon2, log_x_t=log_z2, t=t)

                log_z1 = self.log_sample_categorical(model_log_prob1)
                log_z2 = self.log_sample_categorical(model_log_prob2)

        content_token1 = log_onehot_to_index(log_z1)
        content_token2 = log_onehot_to_index(log_z2)

        pl_token1 = log_onehot_to_index(log_pl1)
        pl_token2 = log_onehot_to_index(log_pl2)

        output = {'content_token1': content_token1,
                  'content_token2': content_token2,
                  'pl_token1': pl_token1,
                  'pl_token2': pl_token2,
                  'img_conv1': img_conv1,
                  'img_conv2': img_conv2,
                  'pre_conv1': pre_conv1,
                  'pre_conv2': pre_conv2}

        if return_logits:
            output['logits1'] = torch.exp(log_z1)
            output['logits2'] = torch.exp(log_z2)

        return output