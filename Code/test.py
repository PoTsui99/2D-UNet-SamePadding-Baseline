import warnings
from pathlib import Path

import SimpleITK as sitk
import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import Dataset
from unet import Unet
from metrics import dice_coef
from argparser import parse_args

warnings.filterwarnings("ignore")

# global variable
wt_dices = []
tc_dices = []
et_dices = []


def evaluate(wt_pred, wt_gt, tc_pred, tc_gt, et_pred, et_gt):
    wt_dice = dice_coef(wt_pred, wt_gt)
    wt_dices.append(wt_dice)

    tc_dice = dice_coef(tc_pred, tc_gt)
    tc_dices.append(tc_dice)

    et_dice = dice_coef(et_pred, et_gt)
    et_dices.append(et_dice)


def get_slice_info(slice_path):
    """
    Fetch the name and slice_idx of the slice.
    """
    fname = slice_path.name
    fname_without_suffix = fname[:-4]  # eliminate suffix: .npy
    parts = fname_without_suffix.split('_')
    patient_name = parts[0] + '_' + parts[1] + '_' + parts[2] + '_' + parts[3]
    return int(parts[-1]), patient_name  # (slice_idx, sample name)


if __name__ == '__main__':
    # name corresponds the file name of training weight folder
    # only the parameters --name and --batch-size makes sense
    args = parse_args()
    batch_size = args.batch_size
    name = args.name
    args = joblib.load('../models/%s/args.pkl' % args.name)  # load args from local disk
    args.name = name
    args.batch_size = batch_size

    output_path = Path('output/%s' % args.name)
    if not output_path.exists():
        output_path.mkdir(parents=True)  # = mkdir -p

    model = Unet()

    # sort to make slices from certain sample put continually
    test_data_paths = sorted(list(Path('test_data_float16').iterdir()))
    test_ground_truth_paths = sorted(list(Path('test_ground_truth_float16').iterdir()))

    # load weight and switch to evaluation mode
    model = model.cuda()
    model.load_state_dict(torch.load('../models/%s/model.pth' % args.name))
    model.eval()

    test_dataset = Dataset(test_data_paths, test_ground_truth_paths)

    # shuffle=False, for sequence counts, slices of same sample should abut each other
    test_iter = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=False)

    with torch.no_grad():
        is_first_slice = True  # this flag aims to initialize processing_sample and pred&gt arrays
        for batch_idx, (data, target) in tqdm(enumerate(test_iter), total=len(test_iter)):
            data = data.cuda()
            output = model(data)
            output = torch.sigmoid(output).data.cpu().numpy()  # shape: (idx_in_this_batch, wt/tc/et, width, height)
            target = target.data.cpu().numpy()

            img_paths = test_data_paths[batch_size * batch_idx: batch_size * (batch_idx + 1)]

            for i in range(output.shape[0]):  # transverse this batch
                # get slice_idx and sample name
                cur_slice_idx, cur_sample_name = get_slice_info(img_paths[i])

                # process the first slice of one segmentation
                if is_first_slice:
                    is_first_slice = False
                    processing_sample_name = cur_sample_name  # sample name being processed

                    # initialization of prediction and ground truth arrays
                    # predictive and ground truth segmentation
                    # Def: SHARED BLOCK 1
                    seg_pred = np.zeros([155, 160, 160], dtype=np.uint8)
                    seg_gt = np.zeros([155, 160, 160], dtype=np.uint8)
                    # predictive WT(whole tumor), TC(tumor nucleus) and ET(enhancing tumor)
                    wt_pred = np.zeros([155, 160, 160], dtype=np.uint8)
                    tc_pred = np.zeros([155, 160, 160], dtype=np.uint8)
                    et_pred = np.zeros([155, 160, 160], dtype=np.uint8)
                    # ground truth WT, TC and ET
                    wt_gt = np.zeros([155, 160, 160], dtype=np.uint8)
                    tc_gt = np.zeros([155, 160, 160], dtype=np.uint8)
                    et_gt = np.zeros([155, 160, 160], dtype=np.uint8)

                    # output.shape: (slice_idx_of_batch, wt/tc/et, width, height)
                    # Def: SHARED BLOCK 2
                    for idx in range(160):
                        for idy in range(160):
                            if output[i, 0, idx, idy] > 0.5:
                                wt_pred[cur_slice_idx, idx, idy] = 1
                            if output[i, 1, idx, idy] > 0.5:
                                tc_pred[cur_slice_idx, idx, idy] = 1
                            if output[i, 2, idx, idy] > 0.5:
                                et_pred[cur_slice_idx, idx, idy] = 1

                    # generate ground-truth WT, TC, ET
                    # SHARED BLOCK 3
                    wt_gt[cur_slice_idx, :, :] = target[i, 0, :, :]
                    tc_gt[cur_slice_idx, :, :] = target[i, 1, :, :]
                    et_gt[cur_slice_idx, :, :] = target[i, 2, :, :]

                    continue

                else:
                    # all slices of last sample is already processed, generate predictive segmentation refering to predictive WT, TC, ET
                    if cur_sample_name != processing_sample_name:
                        # calculate metrics
                        evaluate(wt_pred, wt_gt, tc_pred, tc_gt, et_pred, et_gt)

                        # domain={0, 1}, shape=(155, 160, 160) -> domain={0, 1, 2, 4}, shape=(155, 240, 240) -> .nii.gz
                        # SHARED BLOCK 4
                        for slice_idx in range(155):
                            for idx in range(160):
                                for idy in range(160):
                                    if (wt_pred[slice_idx, idx, idy] == 1):
                                        seg_pred[slice_idx, idx, idy] = 2
                                    if (tc_pred[slice_idx, idx, idy] == 1):
                                        seg_pred[slice_idx, idx, idy] = 1
                                    if (et_pred[slice_idx, idx, idy] == 1):
                                        seg_pred[slice_idx, idx, idy] = 4
                        sample_seg_original_size = np.zeros([155, 240, 240], dtype=np.uint8)
                        sample_seg_original_size[:, 40:200, 40:200] = seg_pred[:, :, :]

                        # SHARED BLOCK 5
                        sitk_img = sitk.GetImageFromArray(sample_seg_original_size)
                        sitk.WriteImage(sitk_img, str(output_path / (processing_sample_name + ".nii.gz")))

                        processing_sample_name = cur_sample_name

                        # SHARED BLOCK 1
                        seg_pred = np.zeros([155, 160, 160], dtype=np.uint8)
                        seg_gt = np.zeros([155, 160, 160], dtype=np.uint8)
                        wt_pred = np.zeros([155, 160, 160], dtype=np.uint8)
                        tc_pred = np.zeros([155, 160, 160], dtype=np.uint8)
                        et_pred = np.zeros([155, 160, 160], dtype=np.uint8)
                        wt_gt = np.zeros([155, 160, 160], dtype=np.uint8)
                        tc_gt = np.zeros([155, 160, 160], dtype=np.uint8)
                        et_gt = np.zeros([155, 160, 160], dtype=np.uint8)

                        # SHARED BLOCK 2
                        for idx in range(160):
                            for idy in range(160):
                                if output[i, 0, idx, idy] > 0.5:
                                    wt_pred[cur_slice_idx, idx, idy] = 1
                                if output[i, 1, idx, idy] > 0.5:
                                    tc_pred[cur_slice_idx, idx, idy] = 1
                                if output[i, 2, idx, idy] > 0.5:
                                    et_pred[cur_slice_idx, idx, idy] = 1

                        # SHARED BLOCK 3
                        wt_gt[cur_slice_idx, :, :] = target[i, 0, :, :]
                        tc_gt[cur_slice_idx, :, :] = target[i, 1, :, :]
                        et_gt[cur_slice_idx, :, :] = target[i, 2, :, :]

                    # processing current slice of the sample
                    elif cur_sample_name == processing_sample_name:
                        # SHARE BLOCK 2
                        for idx in range(160):
                            for idy in range(160):
                                if output[i, 0, idx, idy] > 0.5:
                                    wt_pred[cur_slice_idx, idx, idy] = 1
                                if output[i, 1, idx, idy] > 0.5:
                                    tc_pred[cur_slice_idx, idx, idy] = 1
                                if output[i, 2, idx, idy] > 0.5:
                                    et_pred[cur_slice_idx, idx, idy] = 1

                        # SHARED BLOCK 3
                        wt_gt[cur_slice_idx, :, :] = target[i, 0, :, :]
                        tc_gt[cur_slice_idx, :, :] = target[i, 1, :, :]
                        et_gt[cur_slice_idx, :, :] = target[i, 2, :, :]

                    # inference of whole test dataset has finished
                    if batch_idx == len(test_iter) - 1 and i == output.shape[0] - 1:
                        evaluate(wt_pred, wt_gt, tc_pred, tc_gt, et_pred, et_gt)

                        # SHARED BLOCK 4
                        for slice_idx in range(155):
                            for idx in range(160):
                                for idy in range(160):
                                    if (wt_pred[slice_idx, idx, idy] == 1):
                                        seg_pred[slice_idx, idx, idy] = 2
                                    if (tc_pred[slice_idx, idx, idy] == 1):
                                        seg_pred[slice_idx, idx, idy] = 1
                                    if (et_pred[slice_idx, idx, idy] == 1):
                                        seg_pred[slice_idx, idx, idy] = 4
                        sample_seg_original_size = np.zeros([155, 240, 240], dtype=np.uint8)
                        sample_seg_original_size[:, 40:200, 40:200] = seg_pred[:, :, :]

                        # SHARED BLOCK 5
                        sitk_img = sitk.GetImageFromArray(sample_seg_original_size)
                        sitk.WriteImage(sitk_img, str(output_path / (processing_sample_name + ".nii.gz")))

        torch.cuda.empty_cache()

    # mean metrics of prediction on test dataset
    print('WT Dice: %.4f' % np.mean(wt_dices))
    print('TC Dice: %.4f' % np.mean(tc_dices))
    print('ET Dice: %.4f' % np.mean(et_dices))


