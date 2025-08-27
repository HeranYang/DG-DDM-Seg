
import logging
import torch
import torch.nn as nn
logger = logging.getLogger('base')


####################
# define network
####################


# Generator
def define_G(opt):
    model_opt = opt['model']

    from .vqdm_modules import diffusion_transformer, transformer_utils, embedding

    embed_opt = model_opt['diffusion_config']['embed_opt']
    content_emb = embedding.InputEmbedding(
        img_size = embed_opt['img_size'],       # 192,
        patch_size = embed_opt['patch_size'],   # 16,
        in_chans = embed_opt['in_chans'],       # 1,
        embed_dim = embed_opt['embed_dim'],     # 256,
        depth = embed_opt['depth'],             # 6,
        num_heads = embed_opt['num_heads'],     # 16,
        mlp_ratio = embed_opt['mlp_ratio'],     # 4.0,
        out_chans = embed_opt['out_chans'],     # 256,
        qkv_bias = embed_opt['qkv_bias'],       # True,
        use_rel_pos = embed_opt['use_rel_pos'], # True,
    )

    plabel_emb = embedding.InputEmbedding(
        img_size=embed_opt['img_size'],  # 192,
        patch_size=embed_opt['patch_size'],  # 16,
        in_chans=embed_opt['in_chans'],  # 1,
        embed_dim=embed_opt['embed_dim'],  # 256,
        depth=embed_opt['depth'],  # 6,
        num_heads=embed_opt['num_heads'],  # 16,
        mlp_ratio=embed_opt['mlp_ratio'],  # 4.0,
        out_chans=embed_opt['out_chans'],  # 256,
        qkv_bias=embed_opt['qkv_bias'],  # True,
        use_rel_pos=embed_opt['use_rel_pos'],  # True,
    )

    image_emb = embedding.InputEmbedding(
        img_size=embed_opt['img_size'],  # 192,
        patch_size=embed_opt['patch_size'],  # 16,
        in_chans=embed_opt['cond_in_chans'],  # 12,
        embed_dim=embed_opt['cond_embed_dim'],  # 256,
        depth=embed_opt['cond_depth'],  # 6,
        num_heads=embed_opt['num_heads'],  # 16,
        mlp_ratio=embed_opt['mlp_ratio'],  # 4.0,
        out_chans=embed_opt['out_chans'],  # 256,
        qkv_bias=embed_opt['qkv_bias'],  # True,
        use_rel_pos=embed_opt['use_rel_pos'],  # True,
    )

    init_conv = embedding.ImageConv(
        img_size=embed_opt['img_size'],  # 192,
        in_chans=embed_opt['initconv_in_chans'],  # 3,
        out_chans=embed_opt['cond_in_chans'],  # 12,
    )

    # config in_channel and out_channel.
    transformer_opt = model_opt['diffusion_config']['transformer_config']
    transformer = transformer_utils.Condition2LabelTransformer(
        content_emb=content_emb,
        plabel_emb=plabel_emb,
        image_emb=image_emb,
        init_conv=init_conv,
        out_cls=model_opt['diffusion_config']['num_classes'],
        n_layer=transformer_opt['n_layer'],
        n_embd=transformer_opt['n_embd'],
        n_head=transformer_opt['n_head'],
        attn_pdrop=transformer_opt['attn_pdrop'],
        resid_pdrop=transformer_opt['resid_pdrop'],
        mlp_hidden_times=transformer_opt['mlp_hidden_times'],
        block_activate=transformer_opt['block_activate'],
        diffusion_step=model_opt['diffusion_config']['diffusion_step'],
        timestep_type=transformer_opt['timestep_type'],
    )

    diffusion_opt = model_opt['diffusion_config']
    netG = diffusion_transformer.DiffusionTransformer(
        transformer=transformer,
        content_seq_len=diffusion_opt['content_seq_len'],
        num_classes=diffusion_opt['num_classes'],
        diffusion_step=diffusion_opt['diffusion_step'],
        auxiliary_loss_weight=diffusion_opt['auxiliary_loss_weight'],
        adaptive_auxiliary_loss=diffusion_opt['adaptive_auxiliary_loss'],
        mask_weight=diffusion_opt['mask_weight']
    )

    if opt['gpu_ids'] and opt['distributed']:
        assert torch.cuda.is_available()
        netG = nn.DataParallel(netG)

    return netG
