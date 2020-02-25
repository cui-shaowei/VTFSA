import os
from PIL import Image
import cv2

import numpy as np
from numpy import loadtxt
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms

# import matplotlib.pyplot as plt

img_transforms = transforms.Compose([transforms.ToPILImage(),
                                     # transforms.Resize(256),
                                     # transforms.RandomCrop(224),
                                     # transforms.RandomHorizontalFlip(p=0.5),
                                     # transforms.ToTensor(),
                                     ])
to_tensor = transforms.Compose([transforms.Resize(224),
                                transforms.ToTensor()])


class TsingHuaVTDataset(Dataset):
    """
        For TsingHua Visual-Tactile Dataset downloaded from
          https://github.com/tsinghua-rll/Visual-Tactile_Dataset
        """
    def __init__(self, root_path, train=True):
        self.data_paths = []
        assert os.path.isdir(root_path), "{} is not exist.".format(root_path)
        for root, dirs, files in os.walk(root_path):
            for f in files:
                if f == "label.txt":
                    tmp = os.listdir(root)
                    jpg_tmp = [i for i in tmp if i.endswith(".jpg")]
                    # mp4_tmp = [i for i in tmp if i.endswith(".mp4")]
                    if len(jpg_tmp) == 4:
                        self.data_paths.append(root)
                        # if train:
                        #     if len(mp4_tmp) > 0:
                        #         self.data_paths.append(root)
                        # else:
                        #     if len(mp4_tmp) == 0:
                        #         self.data_paths.append(root)
        if train:
            self.data_paths = self.data_paths[:int(0.8 * len(self.data_paths))]
        else:
            self.data_paths = self.data_paths[int(0.8 * len(self.data_paths)):]

    def __getitem__(self, item):
        data_files = os.listdir(self.data_paths[item])
        data_files.sort()
        img_data = []
        label_data, tactile_data, time_node = None, None, None
        for data_file in data_files:
            data_file = os.path.join(self.data_paths[item], data_file)
            if data_file.endswith(".jpg"):
                img_data.append(img_transforms(cv2.imread(data_file)))
            if data_file.endswith("label.txt"):
                label_data = loadtxt(data_file)# 1:for success; 0:for failure
            if data_file.endswith("tactile.txt"):
                tactile_data = loadtxt(data_file)

        merge_data = [to_tensor(self._img_merge(img_data[0], img_data[2])),
                      to_tensor(self._img_merge(img_data[1], img_data[3]))]

        assert label_data is not None
        assert tactile_data is not None
        # x = [i for i in range(int(label_data[0]), int(label_data[-2]))]
        # y = np.array([tactile_data[i, :] for i in range(int(label_data[0]), int(label_data[-2]))])
        # for i in range(16):
        #     plt.plot(x, y[:, i], linewidth=8)
        # plt.show()
        return merge_data, tactile_data[int(label_data[2])-40:int(label_data[2]), :]/10000, int(label_data[-1])

    def __len__(self):
        return len(self.data_paths)

    def _img_merge(self, img0, img1):
        new_img = Image.new("RGB", (640, 480*2))
        new_img.paste(img0, (0, 0, 640, 480))
        new_img.paste(img1, (0, 480, 640, 960))

        return new_img
# default_collate_err_msg_format = (
#     "default_collate: batch must contain tensors, numpy arrays, numbers, "
#     "dicts or lists; found {}")
# np_str_obj_array_pattern = re.compile(r'[SaUO]')

# def tsinghua_collate_fn(batch):
#     """Puts each data field into a tensor with outer dimension batch size"""
#
#     r"""Puts each data field into a tensor with outer dimension batch size"""
#
#     elem = batch[0]
#     elem_type = type(elem)
#     if isinstance(elem, torch.Tensor):
#         out = None
#         if torch.utils.data.get_worker_info() is not None:
#             # If we're in a background process, concatenate directly into a
#             # shared memory tensor to avoid an extra copy
#             numel = sum([x.numel() for x in batch])
#             storage = elem.storage()._new_shared(numel)
#             out = elem.new(storage)
#         return torch.stack(batch, 0, out=out)
#     elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
#             and elem_type.__name__ != 'string_':
#         elem = batch[0]
#         if elem_type.__name__ == 'ndarray':
#             # array of string classes and object
#             if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
#                 raise TypeError(default_collate_err_msg_format.format(elem.dtype))
#
#             tensor_batch = [b.tolist() for b in batch]
#             # return default_collate([torch.as_tensor(b) for b in batch])
#             return torch.tensor(tensor_batch, dtype=torch.float32)
#         elif elem.shape == ():  # scalars
#             return torch.as_tensor(batch)
#     elif isinstance(elem, float):
#         return torch.tensor(batch, dtype=torch.float64)
#     elif isinstance(elem, int_classes):
#         return torch.tensor(batch)
#     elif isinstance(elem, string_classes):
#         return batch
#     elif isinstance(elem, container_abcs.Mapping):
#         return {key: tsinghua_collate_fn([d[key] for d in batch]) for key in elem}
#     elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
#         return elem_type(*(tsinghua_collate_fn(samples) for samples in zip(*batch)))
#     elif isinstance(elem, container_abcs.Sequence):
#         transposed = zip(*batch)
#         return [tsinghua_collate_fn(samples) for samples in transposed]
#
#     raise TypeError(default_collate_err_msg_format.format(elem_type))

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    test_test = TsingHuaVTDataset("../../Visual-Tactile_Dataset")
    test_dataloader = DataLoader(test_test, batch_size=1, shuffle=True)

    for i, data in enumerate(test_dataloader):
        img_data, tactile_data, label_data = data
        print("###############################")

    print("hold on")