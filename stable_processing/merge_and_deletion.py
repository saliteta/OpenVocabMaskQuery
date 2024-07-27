import numpy as np
import torch


import cuml
import matplotlib.pyplot as plt

from stable_processing.metrics import Consistancy_metrics
import random
from tqdm import tqdm

import os



class mask_refinement(Consistancy_metrics):
    '''
        This class is used for refine the semantic masks. 
        Different label of classes some times represent nonsense. 
        We need to merge that nonsense semantic class with others
    '''

    def __init__(self, labels_location: str, features_location: str, image_location: str, mask_location: str, device: str = 'cuda'):
        super().__init__(labels_location, features_location, image_location, mask_location, device)

    
    def all_visualization(
            self,
            output_dir: str
    ):
        os.makedirs(output_dir, exist_ok=True)
        for batched_masked_images, labels, basename in self.dataset:
            count = 0
            os.makedirs(os.path.join(output_dir, basename.split('.')[0]), exist_ok=True)
            for image in batched_masked_images:
                if count == 0:
                    count += 1
                    continue
                else:
                    dir = os.path.join(output_dir, basename.split('.')[0])
                    base = f'{labels[count-1]}.jpg'
                    self._plot_sample(image, 
                                      f'cluster {labels[count-1]}', 
                                      os.path.join(dir, base)
                                      )
                    count += 1
    
    
    def _plot_sample(self, image, title:str, location:str):
        plt.figure(figsize=(8, 8))  # Set the size of the figure (optional)
        image = image.permute(1,2,0).cpu().numpy()
        plt.imshow(image)  # 'hot' colormap goes from black to red to yellow to white
        plt.colorbar()  # Show color scale
        plt.title(title)
        plt.savefig(location)
        plt.close()


if __name__ == '__main__':
    metrics = mask_refinement(
        labels_location='/home/planner/xiongbutian/ignores/output/refined_label.npz',
        features_location='/home/planner/xiongbutian/ignores/clip_result/semantic_features.npz',
        image_location='/home/planner/xiongbutian/ignores/images',
        mask_location='/home/planner/xiongbutian/ignores/output/refined_mask.npz',
        device='cuda'
    )

    metrics.all_visualization(
        output_dir='/home/planner/xiongbutian/ignores/output/masks_visual'
    )