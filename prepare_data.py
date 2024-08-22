'''
    This is a shitty code, we want to convert masks from npz, to a sequence of npz file
    We also want to convert feature into a sequence of npz file
    Notice that the masks should be in the same size as original image. That means we need to filter out the embeddings
    Notice we also need to have a full masks
'''
import argparse
import os
from typing import Tuple

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch._tensor import Tensor
from stable_processing.loader import Image_Mask_Dataset
from stable_processing.color_logging import print_with_color
import cuml
from stable_processing.metrics import yuv_to_rgb
from PIL import Image

def parser():
    parser = argparse.ArgumentParser("", add_help=True)
    parser.add_argument("--colmap_location", "-o", type=str, default="outputs", required=True, help="output directory")
    parser.add_argument("--mask_location", type=str, required=True, help="path to mask file")
    parser.add_argument("--label_location", type=str, required=True, help="path to label file")
    parser.add_argument("--features_location", type=str, required=True, help="path to label file")
    args = parser.parse_args()
    return args

debugging = True
def get_color_map(mask_array, color_array):
    mask_extended = np.expand_dims(mask_array, axis=-1)  # Now shape is (B, H, W, 1)
    mask_extended = np.tile(mask_extended, (1, 1, 1, 3))  # Now shape is (B, H, W, 3)
    # Step 2: Extend the color_array to shape (B, H, W, 3)
    color_extended = np.expand_dims(color_array, axis=1)  # Now shape is (B, 1, 3)
    color_extended = np.expand_dims(color_extended, axis=1)  # Now shape is (B, 1, 1, 3)
    color_extended = np.tile(color_extended, (1, mask_array.shape[1], mask_array.shape[2], 1))  # Now shape is (B, H, W, 3)
    # Step 3: Element-wise multiplication of mask_extended and color_extended
    weighted_color = mask_extended * color_extended  # Shape is still (B, H, W, 3)
    # Step 4: Sum across the first axis (B axis)
    result = np.sum(weighted_color, axis=0)  # Shape is now (H, W, 3)
    # Step 5: Compute mask_sum by summing mask_array along the B axis
    mask_sum = np.sum(mask_array, axis=0)  # Shape is (H, W)
    # Step 6: Extend mask_sum to shape (H, W, 3)
    mask_sum_extended = np.expand_dims(mask_sum, axis=-1)  # Now shape is (H, W, 1)
    mask_sum_extended = np.tile(mask_sum_extended, (1, 1, 3))  # Now shape is (H, W, 3)
    # Step 7: Divide result by mask_sum_extended (handle division by zero by adding a small epsilon)
    epsilon = 1e-10  # Small value to avoid division by zero
    final_result = result / (mask_sum_extended + epsilon)  # Shape is (H, W, 3)

    return final_result

def _get_color_map(mask_array:torch.Tensor, color_array: torch.Tensor):
    # Convert inputs to torch tensors and move to GPU
    mask_tensor = mask_array.to('cuda')
    color_tensor = color_array.to('cuda')
    # Extend mask tensor to (B, H, W, 3)
    mask_extended = mask_tensor.unsqueeze(-1).expand(-1, -1, -1, 3)
    # Extend color tensor to (B, H, W, 3)
    color_extended = color_tensor.unsqueeze(1).unsqueeze(1).expand(-1, mask_array.shape[1], mask_array.shape[2], -1)
    # Element-wise multiplication of mask_extended and color_extended
    weighted_color = mask_extended * color_extended
    # Sum across the first axis (batch dimension)
    result = weighted_color.sum(0)
    # Sum the mask along the batch dimension to compute the mask sum
    mask_sum = mask_tensor.sum(0)
    # Extend mask_sum to (H, W, 3)
    mask_sum_extended = mask_sum.unsqueeze(-1).expand(-1, -1, 3)
    # Add a small epsilon to avoid division by zero
    epsilon = 1e-10
    final_result = result / (mask_sum_extended + epsilon)
    # Optionally move the tensor back to CPU if further non-GPU processing is needed
    return final_result


class mask_label_dataset(Image_Mask_Dataset):
    def __getitem__(self, idx) -> Tuple[Tensor, Tensor, str]:
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


        return masks, labels, basename


if __name__ == '__main__':
    args = parser()
    mask_location = args.mask_location
    label_location = args.label_location
    features_location = args.features_location
    output_dir = args.colmap_location

    assert os.path.exists(output_dir), "The output directory must exist, and it's structure should like a colmap structure with images\ colmap\\"
    assert os.path.exists(os.path.join(output_dir, 'colmap/sparse/0')), "there should be a colmap/sparse/0 folder in your colmap location"
    assert os.path.exists(os.path.join(output_dir, 'images')), "there should be a images folder in your colmap location"
    
    mask_folder = os.path.join(output_dir, 'masks')
    feature_folder = os.path.join(output_dir, 'features')
    color_folder = os.path.join(output_dir, 'colors')

    if os.path.exists(mask_folder):
        print_with_color(f'{mask_folder} already exists, skip folder creationg', 'YELLOW')
    else:
        os.makedirs(mask_folder)
        print_with_color(f'created folder {mask_folder}', 'GREEN')

    if os.path.exists(feature_folder):
        print_with_color(f'{feature_folder} already exists, skip folder creationg', 'YELLOW')
    else:
        os.makedirs(feature_folder)
        print_with_color(f'created folder {feature_folder}', 'GREEN')

    if os.path.exists(color_folder):
        print_with_color(f'{color_folder} already exists, skip folder creationg', 'YELLOW')
    else:
        os.makedirs(color_folder)
        print_with_color(f'created folder {color_folder}', 'GREEN')

    dataset = mask_label_dataset(os.path.join(output_dir,'images'), refined_masks=mask_location, refined_labels=label_location, transform=None, device='cuda')
    
    features_dict = np.load(features_location)

    feature_list = []
    mask_list = []


    for (mask_torch, label, name) in tqdm(dataset, desc='saving dataset'):
        # Step 1: Create a tensor of ones with the same height and width as mask_torch
        ones_tensor = torch.ones((1, mask_torch.shape[1], mask_torch.shape[2]), dtype=mask_torch.dtype, device=mask_torch.device)

        # Step 2: Concatenate the ones_tensor with mask_torch along the batch dimension (dim=0)
        concatenated_tensor = torch.cat((ones_tensor, mask_torch), dim=0)
        # Step 3: Convert the concatenated tensor to a NumPy array
        masks:np.ndarray = concatenated_tensor.cpu().numpy()

        features:np.ndarray = features_dict[name]

        assert features.shape[0] == masks.shape[0], "The shape of masks and features should be the same, see if we missing a full image mask here"

        name = name.split('.')[0] + '.npz'
        np.savez_compressed(os.path.join(output_dir, 'masks', name), masks.round().astype(np.int8))
        np.savez_compressed(os.path.join(output_dir, 'features', name), features.astype(np.float32))

        feature_list.append(torch.tensor(features).cuda()) # (B, 512)

    ############################ yuv stratgy ############################################
    all_data = torch.cat(feature_list, dim=0)
    reducer = cuml.TSNE(n_components=2, random_state=5)
    reduced_data = reducer.fit_transform(all_data)
    # # Move reduced data back to CPU for plotting
    reduced_data = reduced_data.get()
    x = reduced_data[:, 0]
    y = reduced_data[:, 1]
    fixed_luminance = 0.8
    u = (x - np.min(x)) / (np.ptp(x)) - 0.5  # Normalize and shift to range [-0.5, 0.5]
    v = (y - np.min(y)) / (np.ptp(y)) - 0.5  # Normalize and shift to range [-0.5, 0.5]
    yuv_colors = np.stack([np.full_like(u, fixed_luminance), u, v], axis=-1)  
    rgb_colors:np.ndarray = yuv_to_rgb(yuv_colors)
    ########################## yuv stratgy ############################################

    count = 0
    for (mask_torch, label, name) in tqdm(dataset, desc='saving dataset'):
        # Step 1: Create a tensor of ones with the same height and width as mask_torch
        ones_tensor = torch.ones((1, mask_torch.shape[1], mask_torch.shape[2]), dtype=mask_torch.dtype, device=mask_torch.device)

        # Step 2: Concatenate the ones_tensor with mask_torch along the batch dimension (dim=0)
        concatenated_tensor = torch.cat((ones_tensor, mask_torch), dim=0)

        # Step 3: Convert the concatenated tensor to a NumPy array
        masks = concatenated_tensor.cpu().numpy().round().astype(np.int8)

        if debugging:
            array = rgb_colors[count:masks.shape[0]+count]
            print(array)
            result_image = _get_color_map(mask_array=torch.tensor(masks), color_array=torch.tensor(array.astype(np.float32)))
            rgb_array_uint8 = (result_image * 255).cpu().numpy().astype(np.uint8)

            # Step 2: Create an image object from the numpy array
            image = Image.fromarray(rgb_array_uint8)
            image.save(f'/home/butian/workspace/outputs/outputs/images/{name}')
            
        name = name.split('.')[0] + '.npz'

        np.savez_compressed(os.path.join(output_dir, 'colors', name), rgb_colors[count:masks.shape[0]+count].astype(np.float32))

        count += masks.shape[0]





        
        
