import os.path
import random
from data.base_dataset import BaseDataset, get_params, get_transform
import torchvision.transforms as transforms
from data.image_folder import make_dataset
from PIL import Image


class HolopixDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        # 2D image path
        self.dir_A = os.path.join(opt.dataroot, "Left")  # get the image directory
        self.dir_B = os.path.join(opt.dataroot, "multiview_disparity")
        # Read Index text
        self.index_dir = os.path.join("./data_index", opt.phase + '_' + opt.dataset_num + ".txt")
        with open(self.index_dir) as f:
            img_disp_pairs = f.readlines()
        img_disp_pairs = [x.strip() for x in img_disp_pairs]
        random.shuffle(img_disp_pairs)
        self.A_paths = []
        self.B_paths = []
        for img_disp_pair in img_disp_pairs:
            img_name, disp_name = img_disp_pair.split(",")
            self.A_paths.append(os.path.join(self.dir_A, img_name))
            self.B_paths.append(os.path.join(self.dir_B, disp_name))
        self.A_paths = sorted(self.A_paths)
        self.B_paths = sorted(self.B_paths)
        assert(len(self.A_paths) == len(self.B_paths) and len(self.A_paths) > 0)
        self.As = []
        self.Bs = []
        for i in range(len(self.A_paths)):
            assert (os.path.isfile(self.A_paths[i]) and os.path.isfile(self.B_paths[i]))
            self.As.append(Image.open(self.A_paths[i]).convert('RGB'))
            self.Bs.append(Image.open(self.B_paths[i]).convert('L'))
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        # A_path = self.A_paths[index]
        # B_path = self.B_paths[index]
        # A = Image.open(A_path).convert('RGB')
        # B = Image.open(B_path).convert('L')
        A = self.As[index]
        B = self.Bs[index]

        # apply the same transform to both A and B
        transform_params = get_params(self.opt, A.size)
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))

        A = A_transform(A)
        B = B_transform(B)

        return {'A': A, 'B': B, 'A_paths': self.A_paths[index], 'B_paths': self.B_paths[index]}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.A_paths)
