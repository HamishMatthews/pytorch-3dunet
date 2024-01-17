import glob
import os
from itertools import chain

import pytorch3dunet.augment.transforms as transforms
from pytorch3dunet.datasets.utils import get_slice_builder, ConfigDataset, calculate_stats
from pytorch3dunet.unet3d.utils import get_logger

import numpy as np
import tifffile

logger = get_logger('FolderDataset')

class FolderDataset(ConfigDataset):
    def __init__(self, file_path, image_folder, label_folder, phase, slice_builder_config, transformer_config, weight_folder=None, global_normalization=False):
        self.file_path = file_path
        self.image_path = os.path.join(file_path, image_folder)
        self.label_path = os.path.join(file_path, label_folder)
        self.phase = phase


        # Load images and labels as NumPy arrays
        print("Loading images from {}".format(self.image_path))
        self.raw = self.load_dataset(self.image_path)

        stats = calculate_stats(self.raw, global_normalization)
        self.transformer = transforms.Transformer(transformer_config, stats)
        self.raw_transform = self.transformer.raw_transform()

        if phase != 'test':
            self.label_transform = self.transformer.label_transform()
            self.label = self.load_dataset(self.label_path)
            
            # Load weight maps if provided
            self.weight_map = None
            if weight_folder is not None:
                weight_path = os.path.join(file_path, weight_folder)
                self.weight_map = self.load_dataset(weight_path)
                self.weight_transform = self.transformer.weight_transform()
            else:
                self.weight_map = None
            # Check volume sizes
            self._check_volume_sizes(self.raw, self.label)
        else:
            self.label = None
            self.weight_map = None

        # Build slice indices for raw, label, and weight datasets
        self.slice_builder = get_slice_builder(self.raw, self.label, self.weight_map, slice_builder_config)
        self.raw_slices = self.slice_builder.raw_slices
        self.label_slices = self.slice_builder.label_slices
        self.weight_slices = self.slice_builder.weight_slices if self.weight_map is not None else None
        self.patch_count = len(self.raw_slices)

    @staticmethod
    def load_dataset(directory):
        """
        Load and stack images from a directory into a single numpy array using tifffile library.
        
        Args:
            directory (str): Path to the directory containing images. Assumes that all images in the directory are
                of the same dimension and have '.tif' extension.

        Returns:
            numpy.ndarray: 3D or 4D stacked array of images.
        """

        # check if directory exists
        assert os.path.isdir(directory), f"Directory '{directory}' does not exist."

        # check if the directory contains tif files
        assert len(glob.glob(os.path.join(directory, '*.tif'))) > 0, \
            f"Directory '{directory}' does not contain any '.tif' files."
        
        image_paths = sorted(glob.glob(os.path.join(directory, '*.tif')))
        images = [tifffile.imread(img_path) for img_path in image_paths]
        
        ds = np.stack(images)

        assert ds.ndim in [3, 4], \
            f"Invalid dataset dimension: {ds.ndim}. Supported dataset formats: (C, Z, Y, X) or (Z, Y, X)"
        return ds
    
    def __len__(self):
        return self.patch_count

    def __getitem__(self, idx):
        if idx >= len(self):
            raise StopIteration

        # get the slice for a given index 'idx'
        raw_idx = self.raw_slices[idx]
        # get the raw data patch for a given slice
        raw_patch_transformed = self.raw_transform(self.raw[raw_idx])

        if self.phase == 'test':
            # discard the channel dimension in the slices: predictor requires only the spatial dimensions of the volume
            if len(raw_idx) == 4:
                raw_idx = raw_idx[1:]
            return raw_patch_transformed, raw_idx
        else:
            # get the slice for a given index 'idx'
            label_idx = self.label_slices[idx]
            label_patch_transformed = self.label_transform(self.label[label_idx])
            if self.weight_map is not None:
                weight_idx = self.weight_slices[idx]
                weight_patch_transformed = self.weight_transform(self.weight_map[weight_idx])
                return raw_patch_transformed, label_patch_transformed, weight_patch_transformed
            # return the transformed raw and label patches
            return raw_patch_transformed, label_patch_transformed

    @staticmethod
    def _check_volume_sizes(raw, label):
        def _volume_shape(volume):
            if volume.ndim == 3:
                return volume.shape
            return volume.shape[1:]

        assert raw.ndim in [3, 4], 'Raw dataset must be 3D (DxHxW) or 4D (CxDxHxW)'
        assert label.ndim in [3, 4], 'Label dataset must be 3D (DxHxW) or 4D (CxDxHxW)'

        assert _volume_shape(raw) == _volume_shape(label), 'Raw and labels have to be of the same size'

    @classmethod
    def create_datasets(cls, dataset_config, phase):
        phase_config = dataset_config[phase]

        # Load data augmentation and slice builder configurations
        transformer_config = phase_config['transformer']
        slice_builder_config = phase_config['slice_builder']
        
        # Load base directories to process
        file_paths = phase_config['file_paths']

        # Extract common folder names for images and labels from the dataset configuration
        image_folder = dataset_config['image_folder']
        label_folder = dataset_config['label_folder']
        weight_folder = dataset_config.get('weight_folder', None)
        global_normalization = dataset_config.get('global_normalization', False)

        datasets = []
        for file_path in file_paths:
            try:
                logger.info(f'Loading {phase} set from: {file_path}...')
                dataset = cls(file_path=file_path,
                              image_folder=image_folder,
                              label_folder=label_folder,
                              phase=phase,
                              slice_builder_config=slice_builder_config,
                              transformer_config=transformer_config,
                              weight_folder=weight_folder,
                              global_normalization=global_normalization)
                datasets.append(dataset)
            except Exception:
                logger.error(f'Skipping {phase} set: {file_path}', exc_info=True)
        return datasets
    