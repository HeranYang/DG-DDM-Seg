'''create dataset and dataloader'''
import logging
import torch.utils.data


def create_dataloader(dataset, dataset_opt, phase):
    '''create dataloader '''
    if phase == 'train':
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=dataset_opt['batch_size'],
            shuffle=dataset_opt['use_shuffle'],
            num_workers=dataset_opt['num_workers'],
            pin_memory=True)
    elif phase == 'val':
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=dataset_opt['batch_size'],
            shuffle=False,
            num_workers=dataset_opt['num_workers'],
            pin_memory=True)
    else:
        raise NotImplementedError(
            'Dataloader [{:s}] is not found.'.format(phase))


def create_dataset(dataset_opt, phase):
    """create dataset"""
    from data.MRI_dataset import MRIDataset as D
    dataset = D(dataroot=dataset_opt['dataroot'],
                data_sequence={'cond_image': dataset_opt['cond_image_sequence'], 
                               'output_label': dataset_opt['output_label_sequence'],
                               'cond_pesudolabel': dataset_opt['cond_pesudolabel_sequence']},
                nclass=dataset_opt['nclass'],
                thres=dataset_opt['thres'],
                randnum_pl=dataset_opt['randnum_pl'],
                split=phase,
                data_len=dataset_opt['data_len'])
    logger = logging.getLogger('base')
    logger.info('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__,
                                                           dataset_opt['name']))
    return dataset
