import os.path
import random

import numpy as np
import torch
from torch.utils.data import Dataset
import data.util as Util


class MRIDataset(Dataset):
    def __init__(self, dataroot, data_sequence, nclass, thres, randnum_pl, split='train', data_len=-1):
        self.data_sequence = data_sequence
        self.data_len = data_len
        self.split = split
        self.nclass = nclass
        self.thres = thres
        self.randnum_pl = randnum_pl  # we produce 160 pesudolabels for each train slice.

        self.ori_path = Util.get_paths_from_images(dataroot, data_sequence['cond_image'])
        self.tg_path = Util.get_paths_from_images(dataroot, data_sequence['output_label'])
        self.pl_path = Util.get_paths_from_images(dataroot, data_sequence['cond_pesudolabel'])
        
        self.dataset_len = len(self.ori_path)
        if self.data_len <= 0:
            self.data_len = self.dataset_len
        else:
            self.data_len = min(self.data_len, self.dataset_len)
            

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):

        ## ------------------------------------------------
        # prepare 3-slice names.
        tmp_ori_path = self.ori_path[index]
        (orifilepath, orifilename) = os.path.split(tmp_ori_path)
        real_ori_path = orifilepath + '/'  # /home/.../label/

        orifilename_head = orifilename[:-6]
        orifilename_sliceid = orifilename[-6:-4]
        orifilename_end = orifilename[-4:]

        sliceid_int = int(orifilename_sliceid)
        sliceid_plus_str = '{:0>2d}'.format(sliceid_int + 1)
        orifilename_plus = orifilename_head + sliceid_plus_str + orifilename_end
        sliceid_minor_str = '{:0>2d}'.format(sliceid_int - 1)
        orifilename_minor = orifilename_head + sliceid_minor_str + orifilename_end

        filenameList = []
        if orifilename_sliceid == r'00':
            filenameList.append(orifilename_head + r'00' + orifilename_end)
            filenameList.append(orifilename_head + r'00' + orifilename_end)
            filenameList.append(orifilename_head + r'01' + orifilename_end)

        elif os.path.exists(os.path.join(orifilepath, orifilename_plus)):
            filenameList.append(orifilename_minor)
            filenameList.append(orifilename)
            filenameList.append(orifilename_plus)

        else:
            filenameList.append(orifilename_minor)
            filenameList.append(orifilename)
            filenameList.append(orifilename)
        ## ------------------------------------------------

        ## ======================================================================
        # target.
        tmp_tg_path = self.tg_path[index]
        (tgfilepath, tgfilename) = os.path.split(tmp_tg_path)  # /home/.../label, MSCMRSeg19-id08-slice05.npy

        real_tg_path = tgfilepath + '/'  # /home/.../label/
        ## ======================================================================

        ## ======================================================================
        # randomly select the pesudolabel from #160 selections.
        tmp_pl_path = self.pl_path[index]
        (plfilepath, plfilename) = os.path.split(tmp_pl_path)  # /home/.../pesudolabel/1, MSCMRSeg19-id08-slice05.npy
        plfilepathroot = os.path.dirname(plfilepath)  # /home/.../pesudolabel

        rand_plfolder_id = random.randint(1, self.randnum_pl)

        real_pl_path = plfilepathroot + '/{}/'.format(rand_plfolder_id)  # /home/.../pesudolabel/RANDOM_ID/
        ## ======================================================================

        ## ------------------------------------------------
        # load 3-slice image.
        ori0 = np.load(real_ori_path + filenameList[0])
        ori1 = np.load(real_ori_path + filenameList[1])
        ori2 = np.load(real_ori_path + filenameList[2])
        ori = np.stack([ori0,
                        ori1,
                        ori2], axis=-1)
        # load 1-slice label.
        tg = np.load(real_tg_path + filenameList[1])
        # load 1-slice pesudolabel.
        pre_pl = np.load(real_pl_path + filenameList[1])
        ## ------------------------------------------------

        ## ------------------------------------------------
        # load 3-slice label, for producing LLA.
        tg0 = np.load(real_tg_path + filenameList[0])
        tg2 = np.load(real_tg_path + filenameList[2])
        mask = np.stack([tg0,
                         tg,
                         tg2], axis=-1)
        ## ------------------------------------------------

        [GLA1, LLA1, GLA2, LLA2, tg, pl] = Util.transform_augment([ori, tg, pre_pl, mask],
                                                                  self.nclass,
                                                                  thres=self.thres,
                                                                  split=self.split)

        tg = tg.to(torch.int64)

        return {'GLA1': GLA1,
                'LLA1': LLA1,
                'GLA2': GLA2,
                'LLA2': LLA2,
                'target': tg,
                'pesudolabel': pl,
                'Index': index}
