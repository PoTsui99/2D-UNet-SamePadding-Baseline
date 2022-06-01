import numpy as np
import torch.utils.data


class Dataset(torch.utils.data.Dataset):  # Implementation: __init__, __len__, __getitem__

    # aug参数单独列出, validation的时候为不增强, 所有默认为False
    def __init__(self, data_paths, target_paths):
        """
        :param data_paths: 预处理后的.npy切片(4, 160, 160)
        :param target_paths: 预处理后的.npy分割切片(1, 160, 160) 0, 1, 2, 4构成
        :param args: configuration
        """
        self.data_paths = data_paths
        self.target_paths = target_paths

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        data_path = self.data_paths[idx]
        target_path = self.target_paths[idx]

        data = np.load(data_path).astype('float32')
        segmentation = np.load(target_path)

        # NET(non-enhancing tumor): 1
        # ED(peritumoral edema): 2
        # ET(enhancing tumor): 4
        # WT = ED(2) + ET(4) + NET(1), TC = ET(4) + NET(1), ET(4)
        # Turn target from (160, 160), domain={1, 2, 4} to (3, 160, 160), domain={0, 1}
        wt = np.zeros_like(segmentation, dtype='float32')
        tc = np.zeros_like(segmentation, dtype='float32')
        et = np.zeros_like(segmentation, dtype='float32')

        wt[segmentation == 1] = 1.
        wt[segmentation == 2] = 1.
        wt[segmentation == 4] = 1.
        tc[segmentation == 1] = 1.
        tc[segmentation == 4] = 1.
        et[segmentation == 4] = 1.

        target = np.empty((3, 160, 160), dtype='float32')
        target[0, :, :] = wt
        target[1, :, :] = tc
        target[2, :, :] = et

        return data, target
