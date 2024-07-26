import torch
import torch.nn.functional as F
from PIL import Image

from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
import os
from stable_processing.color_logging import print_with_color


class Feature_merger:
    def __init__(self, height: int, width: int):
        self.height = height
        self.width = width

    def _logits_to_mask(self, logits:torch.Tensor, threshold = 0.5) -> torch.Tensor:
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

    def _align_mask_shape(self, masks: torch.Tensor) -> torch.Tensor:
        """
        Adjust the mask tensor to match the original image dimensions.

        Args:
        - H (int): Height of the original image.
        - W (int): Width of the original image.
        - masks (torch.Tensor): Tensor of shape [N, 1, S, S] where N is the number of masks and S is the original dimension of the square masks.

        Returns:
        - torch.Tensor: Tensor of masks adjusted to shape [N, H, W].
        """

        if self.height >= self.width:
            # Interpolate to (H, H)
            masks_resized = F.interpolate(masks, size=(self.height, self.height), mode='bilinear', align_corners=False).squeeze(1)
            # Crop to (H, W)
            masks_cropped = masks_resized[:, :self.width, :]
        else:
            # Interpolate to (W, W)
            masks_resized = F.interpolate(masks, size=(self.width, self.width), mode='bilinear', align_corners=False).squeeze(1)
            # Crop to (H, W)
            masks_cropped = masks_resized[:, :, :self.height]

        return masks_cropped
    
    def merge(self, masks:torch.Tensor, features:torch.Tensor) -> torch.Tensor:
        '''
            mask is in the shape of B H' W' and it is a logits
            features is in the shape of B,C

            return a H, W, C feature vector, where each C is a unit vector
        '''
        masks = self._logits_to_mask(masks)
        masks = self._align_mask_shape(masks)
        _,C = features.shape
        _, W, H = masks.shape
        masks = masks.unsqueeze(-1).expand(-1,-1,-1, C)
        features = features.unsqueeze(1).unsqueeze(1).expand(-1, W, H, -1)
        
        
        
        # overlaying global information to the feature
        ones_mask = torch.ones_like(features).to(masks.device)
        ones_mask[1:] = masks
        masks = ones_mask
        # overlaying global information to the feature

        attention_map = (features * masks).sum(dim=0)

        norm = attention_map.norm(dim=-1, keepdim=True)

        sphere_feature = attention_map/norm

        return sphere_feature

    def cross_attetion(self, spherical_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor: 
        '''
            spherical_features: H, W, C
            text_features: B, C
            Where each C is a unit vector
            
            return a B, H, W where each element of B,H,W is a attention score rangin from -1 to 1
        '''
        B,C = text_features.shape
        H, W, _ = spherical_features.shape
        

        spherical_features =spherical_features.view(-1, C)

        # remember, dot product


        attention_map = (100.0 * spherical_features @ text_features.T).softmax(dim=-1) 
        attention_map = attention_map.reshape(H,W, -1).permute(2,0,1)
        return attention_map
    
def text_image_attention_display(original_image: Image.Image, cross_attention_map: torch.Tensor) -> Tuple[Image.Image]:
    '''
    Original image is a PIL Image with the shape of (H, W, 3).
    cross_attention_map is a torch tensor with the shape of (B, H, W), elements range from -1 to 1.
    
    Returns a tuple of images displaying the overlay heatmap of cross_attention_map on the original image in batch.
    '''
    # Normalize the attention map to range 0 to 1
    norm_attention_maps = (cross_attention_map - cross_attention_map.min()) / (cross_attention_map.max() - cross_attention_map.min())
    
    # Convert original image to numpy array
    img_array = np.array(original_image)
    
    # Initialize list to hold the resulting images
    result_images = []

    for attention_map in norm_attention_maps:
        # Resize attention map to match image dimensions if needed
        attention_map = torch.nn.functional.interpolate(attention_map.unsqueeze(0).unsqueeze(0), size=img_array.shape[:2], mode='bilinear', align_corners=False).squeeze()

        # Convert the normalized attention map to a heatmap
        cmap = plt.get_cmap('jet')
        heatmap = cmap(attention_map.cpu().numpy())  # This produces an RGBA image

        # Convert RGBA heatmap to RGB by ignoring alpha channel
        heatmap = (heatmap[..., :3] * 255).astype(np.uint8)
        
        # Blend the heatmap with the original image
        blended_image = 0.5 * img_array + 0.5 * heatmap
        blended_image = blended_image.astype(np.uint8)


        # Convert blended image to PIL Image and add to results
        result_images.append(Image.fromarray(blended_image))

    return tuple(result_images)


def only_attention(norm_attention_maps: torch.Tensor, text, img_size = (1006, 755)) -> Tuple[Image.Image]:
    count = 0
    for attention_map in norm_attention_maps:
        # Resize attention map to match desired image dimensions
        attention_map = torch.nn.functional.interpolate(
            attention_map.unsqueeze(0).unsqueeze(0), 
            size=img_size, 
            mode='bilinear', 
            align_corners=False
        ).squeeze()

        # Convert the normalized attention map to a heatmap
        cmap = plt.get_cmap('jet')
        heatmap = cmap(attention_map.cpu().numpy())  # This produces an RGBA image
        
        # Display the heatmap with a color bar
        plt.figure(figsize=(6, 4))
        plt.imshow(heatmap, aspect='auto')  # 'aspect' can be 'auto' or 'equal'
        plt.colorbar(plt.cm.ScalarMappable(cmap=cmap), ax=plt.gca(), orientation='vertical')
        plt.title('Attention Heatmap with Color Bar')
        plt.axis('off')  # Hide axes
        plt.savefig(f'{text[count]}.jpg')
        count += 1

def store_visualization_result(
        images: Tuple[Image.Image],
        output_directory: str,
        file_names: Tuple[str]
) -> None:
    '''
    Input a list of images and the directory where they should be stored.
    Stores the images in the specified location.
    
    :param images: A tuple of PIL Image objects.
    :param output_directory: The directory where images should be stored.
    '''
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        print_with_color(f'{output_directory} does not exist, created new one', 'YELLOW')

    for i, image in enumerate(images):
        post_fix = file_names[i].replace(" ", "_")
        image_path = os.path.join(output_directory, f'image_{post_fix}.png')
        image.save(image_path)
    
    print_with_color(f'Image Saved to {output_directory}. Done', 'GREEN')

