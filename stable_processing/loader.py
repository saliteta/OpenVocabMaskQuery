from torch.utils.data import Dataset
import torch.nn.functional as F

from PIL import Image
import numpy as np
import torch
import os

import torch
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
from typing import Tuple

from open_clip.model import CLIP
from open_clip.tokenizer import SimpleTokenizer

import open_clip


OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)


def simplified_transform():
    mean = OPENAI_DATASET_MEAN  # Placeholder, replace with actual values
    std = OPENAI_DATASET_STD   # Placeholder, replace with actual values

    return Compose([
        Resize(224, interpolation=InterpolationMode.BICUBIC, antialias=True),
        CenterCrop(224),
        Normalize(mean=mean, std=std)
    ])

class Image_Mask_Dataset(Dataset):
    def __init__(self, original_image_directory: str, refined_masks:str, refined_labels: str, transform:Compose, device: str):
        """
        Args:
            original_image_directory (string): Directory with all the images.
            refined_masks (string): for storing refined mask, there might be multiple mask for one image
            device (string): Device to perform computations on.
        """
        self.image_dirctory = original_image_directory
        self.refined_masks = refined_masks
        self.refined_labels = refined_labels
        self.transform = transform
        self.device = device
        self.images = [os.path.join(self.image_dirctory, img) for img in sorted(os.listdir(self.image_dirctory)) if img.endswith(('.png', '.jpg', '.jpeg'))]

        # notice that we load a dictionary
        self.masks = np.load(self.refined_masks) # notice that this mask is a 256*256 masks filter out the padding prompt 
        self.labels = np.load(self.refined_labels) # notice that this will load the global labels for each masks

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, str]:

        # we need to first get the original image
        # we need to preprocess the original image and the masks 
        # return pre-processed original image and masks as a torch.Tensor

        img_path = self.images[idx]
        basename = os.path.basename(img_path)
        
        image = Image.open(img_path).convert('RGB')


        # obtained numpy array
        masks = self.masks[basename] # this are only logits in the shape of b,h,w
        labels = self.labels[basename] # this are the labels in b

        masks = torch.tensor(masks).to(self.device)
        labels = torch.tensor(labels).to(self.device)

        h, w = image.size

        # call the following process
        masks = self.logits_to_mask(masks) # get the real tensors

        masks = self.align_mask_shape(h, w, masks) # get the mask shape

        batched_masked_images = self.apply_masks_and_transform(image, masks) # multiply image and masks

        return batched_masked_images, labels, basename
    
    def apply_masks_and_transform(self, image: Image.Image, masks: torch.Tensor) -> torch.Tensor:
        # Assume image is a PIL Image and masks is a torch Tensor of shape [N, H, W]
        image_tensor = ToTensor()(image).to(self.device)  # Convert image to tensor [C, H, W]
        image_tensor = image_tensor.unsqueeze(0)  # [1, C, H, W]

        # Expand masks to match image tensor shape [N, C, H, W]
        masks = masks.unsqueeze(1)  # [N, 1, H, W]
        masks = masks.expand(-1, image_tensor.shape[1], -1, -1)  # Match the color channels

        # Apply masks
        masked_images = image_tensor * masks  # Broadcast multiplication [N, C, H, W]

        # Apply transformations
        image_batch =  torch.cat((image_tensor, masked_images), dim=0)

        return self.transform(image_batch)

    def align_mask_shape(self, H: int, W: int, masks: torch.Tensor) -> torch.Tensor:
        """
        Adjust the mask tensor to match the original image dimensions.

        Args:
        - H (int): Height of the original image.
        - W (int): Width of the original image.
        - masks (torch.Tensor): Tensor of shape [N, 1, S, S] where N is the number of masks and S is the original dimension of the square masks.

        Returns:
        - torch.Tensor: Tensor of masks adjusted to shape [N, H, W].
        """

        if H >= W:
            # Interpolate to (H, H)
            masks_resized = F.interpolate(masks, size=(H, H), mode='bilinear', align_corners=False).squeeze(1)
            # Crop to (H, W)
            masks_cropped = masks_resized[:, :W, :]
        else:
            # Interpolate to (W, W)
            masks_resized = F.interpolate(masks, size=(W, W), mode='bilinear', align_corners=False).squeeze(1)
            # Crop to (H, W)
            masks_cropped = masks_resized[:, :, :H]

        return masks_cropped

    def logits_to_mask(self, logits:torch.Tensor, threshold = 0.5) -> torch.Tensor:
        '''
            Basic process stratgy that can smooth the segmentation. 
        '''
        # Create a simple averaging kernel
        kernel_size = 4  # Size of the kernel (3x3)
        kernel = torch.ones((kernel_size, kernel_size)).cuda() / (kernel_size ** 2)
        kernel = kernel.expand((1, 1, kernel_size, kernel_size))  # Expand for conv2d compatibility

        # Ensure that the kernel is a floating point tensor
        kernel = kernel.type(torch.float32)

        logits = logits.unsqueeze(1)
        filtered_logits = F.conv2d(logits, kernel, padding=1, stride=1)
        filtered_logits = filtered_logits

        binary_tensor = torch.where(filtered_logits > threshold, torch.tensor(0.8), torch.tensor(0.0))
        return binary_tensor

def load_dataset(img_directory, mask_location, label_location, num_workers, device, batch_size = 1):
    # Initialize the image transformation class
    transform = simplified_transform()
    
    # Create dataset
    dataset = Image_Mask_Dataset(
        original_image_directory = img_directory,
        refined_masks = mask_location,
        refined_labels = label_location,
        transform = transform,
        device = device
    )
    
    # Create data loader
    
    return dataset


def load_text_to_list(file_location: str) -> Tuple[str]:
    """
    Load text from a file into a Python list where each element contains a word.

    Args:
        file_location (str): The location of the text file.

    Returns:
        Tuple[str]: A tuple containing words from the text file.
    """
    word_list = []

    # Open the file and read its content
    with open(file_location, 'r') as file:
        content = file.read()

    # Split the content into words based on spaces
    words = content.split(',')

    # Append each word to the list
    word_list.extend(words)

    # Return the list as a tuple
    return tuple(word_list)


def load_visualization_data(
    image_location:str, 
    feature_location:str, 
    masks_location: str, 
    text_discription: str,
    device: str
    ) -> Tuple[Image.Image, torch.Tensor, torch.Tensor, Tuple[str]]:
    '''
        Load image
        Load Features
        Load Masks
        Load Text
    '''
    image = Image.open(image_location).convert('RGB')

    basename = os.path.basename(image_location)

    features = torch.Tensor(np.load(feature_location)[basename]).to(device)
    masks =  torch.Tensor(np.load(masks_location)[basename]).to(device)
    text = load_text_to_list(text_discription)

    return image, features, masks, text

def load_model(
    clip_version: str,
    clip_checkpoint: str,
    device: str
) -> Tuple[CLIP, SimpleTokenizer]:
    
    model, _, _ = open_clip.create_model_and_transforms(clip_version, pretrained=clip_checkpoint)
    tokenizer = open_clip.get_tokenizer(clip_version)

    model.to(device)
    return model, tokenizer