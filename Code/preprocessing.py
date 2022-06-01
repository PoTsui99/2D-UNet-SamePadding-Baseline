# 只运行一次, 生成./train_data_float16 ./train_ground_truth_float16 ./test_data ./test_ground_truth
from pathlib import Path  # OO范式的路径, 易于迁移
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

home = Path('.')
train_data_path = Path('../dataset/MICCAI_BraTS_2018_Data_Training')
hgg_path = train_data_path / 'HGG'
lgg_path = train_data_path / 'LGG'

output_data_path = home / 'train_data_float16'
output_ground_truth_path = home / 'train_ground_truth_float16'
output_test_data_path = home / 'test_data_float16'
output_test_ground_truth_path = home / 'test_ground_truth_float16'

if not output_data_path.exists():
    output_data_path.mkdir()
if not output_ground_truth_path.exists():
    output_ground_truth_path.mkdir()
if not output_test_data_path.exists():
    output_test_data_path.mkdir()
if not output_test_ground_truth_path.exists():
    output_test_ground_truth_path.mkdir()

hgg_folder_list = list(hgg_path.iterdir())
lgg_folder_list = list(lgg_path.iterdir())
folder_list = hgg_folder_list + lgg_folder_list  # 285 samples
train_folder_list = folder_list[:228]  # 0.8×285 = 228 samples as training data
test_folder_list = folder_list[228:]  # the rest as test data


def normalize(slice, max=99, min=1):
    """
    剔除异常值, 进行z-score标准化
    :param max: slice取值的ceil
    :param min: slice取值的floor
    :return: z-score标准化后的slice
    """
    ceil = np.percentile(slice, max)  # 返回max%的数据
    floor = np.percentile(slice, min)
    slice = np.clip(slice, floor, ceil)

    image_notzero = slice[np.nonzero(slice)]  # 80%以上的背景不参与统计
    if np.std(slice) == 0 or np.std(image_notzero) == 0:  # 全背景切片, 之后会筛除
        return slice
    else:
        img_normalized = (slice - np.mean(image_notzero)) / np.std(image_notzero)
        img_normalized[abs(img_normalized - img_normalized.min()) <= 1e-3] = -10  # -10来标记背景
        return img_normalized


def crop_out_center(img, height, width):
    ori_height, ori_width = img[0].shape
    starth = ori_height // 2 - (height // 2)
    startw = ori_width // 2 - (width // 2)
    return img[:, starth:starth + height, startw:startw + width]

def nii2array(img_path):
    """
    .nii.gz文件转换为numpy array
    :param img_path: .nii.gz文件路径
    :return: numpy.ndarray
    """
    return sitk.GetArrayFromImage(sitk.ReadImage(str(img_path), sitk.sitkInt16))


def preprocess_folder(folder_list, output_data_path, output_ground_truth_path, is_test=False, data_type='float16'):
    for i in tqdm(range(len(folder_list))):

        # 路径
        flair_image_path = folder_list[i] / (folder_list[i].name + '_flair.nii.gz')
        t1_image_path = folder_list[i] / (folder_list[i].name + '_t1.nii.gz')
        t1ce_image_path = folder_list[i] / (folder_list[i].name + '_t1ce.nii.gz')
        t2_image_path = folder_list[i] / (folder_list[i].name + '_t2.nii.gz')
        seg_image_path = folder_list[i] / (folder_list[i].name + '_seg.nii.gz')

        # nii.gz转换为numpy array
        flair_array = nii2array(flair_image_path)
        t1_array = nii2array(t1_image_path)
        t1ce_array = nii2array(t1ce_image_path)
        t2_array = nii2array(t2_image_path)
        seg_array = nii2array(seg_image_path)

        # 对四个模态分别进行标准化,由于它们对比度不同, ground_truth不用
        flair_array_normalized = normalize(flair_array)
        t1_array_normalized = normalize(t1_array)
        t1ce_array_normalized = normalize(t1ce_array)
        t2_array_normalized = normalize(t2_array)

        # 裁剪(被2整除), 防止背景过多, 导致类别不均衡
        flair_crop = crop_out_center(flair_array_normalized, 160, 160)
        t1_crop = crop_out_center(t1_array_normalized, 160, 160)
        t1ce_crop = crop_out_center(t1ce_array_normalized, 160, 160)
        t2_crop = crop_out_center(t2_array_normalized, 160, 160)
        seg_crop = crop_out_center(seg_array, 160, 160)
        # if not is_test:
        #     flair_crop = crop_out_center(flair_array_normalized, 160, 160)
        #     t1_crop = crop_out_center(t1_array_normalized, 160, 160)
        #     t1ce_crop = crop_out_center(t1ce_array_normalized, 160, 160)
        #     t2_crop = crop_out_center(t2_array_normalized, 160, 160)
        #     seg_crop = crop_out_center(seg_array, 160, 160)
        # else:
        #     flair_crop = flair_array_normalized
        #     t1_crop = t1_array_normalized
        #     t1ce_crop = t1ce_array_normalized
        #     t2_crop = t2_array_normalized
        #     seg_crop = seg_array
        # print('processing: ', folder_list[i].name)

        # 切片处理

        for slice in range(flair_crop.shape[0]):  # 总共155个slice
            if not is_test and np.max(seg_crop[slice, :, :]) == 0:  # 对于训练集合(train/validation), 只选取存在病变的slice
                    continue

            seg_slice_img = seg_crop[slice, :, :]

            four_modal_slices_array = np.zeros((4, flair_crop.shape[1], flair_crop.shape[2]), dtype=data_type)

            flair_slice = flair_crop[slice, :, :].astype(dtype=data_type)
            t1_slice = t1_crop[slice, :, :].astype(dtype=data_type)
            t1ce_slice = t1ce_crop[slice, :, :].astype(dtype=data_type)
            t2_slice = t2_crop[slice, :, :].astype(dtype=data_type)
            four_modal_slices_array[0, :, :] = flair_slice
            four_modal_slices_array[1, :, :] = t1_slice
            four_modal_slices_array[2, :, :] = t1ce_slice
            four_modal_slices_array[3, :, :] = t2_slice

            four_modal_slice_path = output_data_path / (folder_list[i].name + "_" + str(slice) + ".npy")
            seg_slice_path = output_ground_truth_path / (folder_list[i].name + "_" + str(slice) + ".npy")
            np.save(str(four_modal_slice_path), four_modal_slices_array)  # (4, 160,160)
            np.save(str(seg_slice_path), seg_slice_img)  # (160, 160) dtype('uint8') 值为0 1 2 4


if __name__ == '__main__':
    # 节省空间, 使用float16保存
    preprocess_folder(train_folder_list, output_data_path, output_ground_truth_path, data_type='float16')
    preprocess_folder(test_folder_list, output_test_data_path, output_test_ground_truth_path, is_test=True, data_type='float16')
