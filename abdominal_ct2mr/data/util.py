import os
import torchvision
import numpy as np
import data.transform_utils as trans

from .location_scale_augmentation import LocationScaleAugmentation

join = os.path.join
IMG_EXTENSIONS = ['.npy', '.nii.gz']


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def get_paths_from_images(dir, sequence):
    assert os.path.isdir(dir), '{:s} is not a valid directory'.format(dir)
    image_path = []
    if sequence == "all":
        idx = 0  # index of sequence of images
        for folder_name in sorted(os.listdir(dir)):
            img_dir = join(dir, folder_name)
            image_path.append([])
            for img_name in sorted(os.listdir(img_dir)):
                if img_name.endswith('.npy'):
                    image_path[idx].append(join(img_dir, img_name))
            idx += 1
    else:
        img_dir = join(dir, sequence)
        assert os.path.isdir(dir), '{:s} is not a valid name for sequence images'.format(sequence)
        for img_name in sorted(os.listdir(img_dir)):
            if is_image_file(img_name):
                image_path.append(join(img_dir, img_name))

    assert image_path, '{:s} has no valid image file'.format(dir)
    return sorted(image_path)



totensor = torchvision.transforms.ToTensor()
tr_func  = trans.transform_with_label(trans.tr_aug)

def transform_augment(img_list, nclass, thres=0.9, split='val'):

    if split == 'train':
        image = img_list[0]
        label = img_list[1]
        pre_pesudolabel = img_list[2]
        mask = img_list[3]

        pesudolabel = prepare_pesudolabel(pre_pesudolabel, thres)

        # ==============================================================================================================
        vmax = image.max()
        vmin = image.min()
        image = (image - vmin) / (vmax - vmin)

        location_scale1 = LocationScaleAugmentation(vrange=(0., 1.), background_threshold=0.01)
        GLA1 = location_scale1.Global_Location_Scale_Augmentation(image.copy())
        GLA1 = GLA1 * (vmax - vmin) + vmin
        LLA1 = location_scale1.Local_Location_Scale_Augmentation(image.copy(), mask.astype(np.int32))
        LLA1 = LLA1 * (vmax - vmin) + vmin

        location_scale2 = LocationScaleAugmentation(vrange=(0., 1.), background_threshold=0.01)
        GLA2 = location_scale2.Global_Location_Scale_Augmentation(image.copy())
        GLA2 = GLA2 * (vmax - vmin) + vmin
        LLA2 = location_scale2.Local_Location_Scale_Augmentation(image.copy(), mask.astype(np.int32))
        LLA2 = LLA2 * (vmax - vmin) + vmin
        # ==============================================================================================================
        
        comp = np.stack( [GLA1[..., 0], GLA1[..., 1], GLA1[..., 2],
                          LLA1[..., 0], LLA1[..., 1], LLA1[..., 2],
                          GLA2[..., 0], GLA2[..., 1], GLA2[..., 2],
                          LLA2[..., 0], LLA2[..., 1], LLA2[..., 2],
                          label,
                          pesudolabel], axis = -1 )

        timg1, timg2, label, pesudolabel = tr_func(comp,
                                                     c_img=12,
                                                     c_label=1,
                                                     c_plabel=1,
                                                     nclass=nclass, is_train=True, use_onehot=False)
        GLA1, LLA1 = np.split(timg1, 2, -1)
        GLA2, LLA2 = np.split(timg2, 2, -1)

        img1     = np.float32(GLA1)
        aug_img1 = np.float32(LLA1)
        img2     = np.float32(GLA2)
        aug_img2 = np.float32(LLA2)
        label = np.int64(label)
        pesudolabel = np.float32(pesudolabel)

        imgs = [img1, aug_img1, img2, aug_img2, label, pesudolabel]
        imgs = [totensor(img) for img in imgs]
        
    elif split == 'val':

        image = img_list[0]
        label = img_list[1]
        pre_pesudolabel = img_list[2]
        mask = img_list[3]

        img1     = np.float32(image)
        aug_img1 = np.float32(image)
        img2     = np.float32(image)
        aug_img2 = np.float32(image)
        label = np.int64(label)
        pesudolabel = np.float32(pre_pesudolabel)

        imgs = [img1, aug_img1, img2, aug_img2, label, pesudolabel]
        imgs = [totensor(img) for img in imgs]
        
    return imgs


def prepare_pesudolabel(pre_pesudolabel, thres):
    pre_pesudolabel = np.squeeze(pre_pesudolabel)
    c, h, w = np.shape(pre_pesudolabel)
    # define an all-zero volume for output.
    pesudolabel = np.zeros((h, w))
    # for each pixel in (jh, kw).
    for jh in range(h):
        for kw in range(w):
            classVec = np.arange(c-1) + 1  # (c-1)-length class vector, without considering background in class-0.
            # for each forground class in 1-to-c.
            for iclass in classVec:
                # if the probability larger than the thres, then take it;
                #   otherwise, skip to next pixel.
                if pre_pesudolabel[iclass, jh, kw] > thres:
                    pesudolabel[jh, kw] = iclass
    return pesudolabel