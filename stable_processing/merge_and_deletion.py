import numpy as np
import torch



from stable_processing.metrics import Consistancy_metrics
from tqdm import tqdm

import os

from typing import Tuple


class mask_refinement(Consistancy_metrics):
    '''
        This class is used for refine the semantic masks. 
        Different label of classes some times represent nonsense. 
        We need to merge that nonsense semantic class with others
    '''

    def __init__(self, labels_location: str, features_location: str, image_location: str, mask_location: str, device: str = 'cuda'):
        super().__init__(labels_location, features_location, image_location, mask_location, device)
        self.overall_similarity, self.pairwise_similarity, self.labels = self.get_inter_label_similarity()
        self.mask_dict = np.load(mask_location)

    def deletion(self, store_coeficient: float = 1.0)-> Tuple[dict, dict]:
        '''
            According to the overall independent metrics to delete several masks
            Args:
            - store_coeficient: This threshould means we will remove all similarity above (\mu + threshold * \std) labels, default is 1.0
        '''
        # detemine which labels is noisy labels
        mu = self.overall_similarity.mean()
        std = self.overall_similarity.std()
        threshold = mu + store_coeficient * std
        removed_lable_index = torch.where(self.overall_similarity>threshold)[0].long()
        to_be_filtered_labels = torch.tensor(self.labels).cuda()[removed_lable_index].cpu().numpy() # this are the labels need to be filter out

        new_label_dict = {}
        new_mask_dict = {}
        for key in tqdm(self.labels_dict.keys(), desc='delete noisy labels and masks'):
            '''
                to be filtered labels here
            '''
            current_labels = self.labels_dict[key]
            current_mask = self.mask_dict[key]

            # Create a mask for elements to keep (those not in to_be_filtered_labels)
            keep_mask = np.isin(current_labels, to_be_filtered_labels, invert=True)
            true_indices = np.where(keep_mask)[0]

            # Filter out the labels and corresponding masks
            new_label_dict[key] = current_labels[true_indices]
            new_mask_dict[key] = current_mask[true_indices]
        ### We need to save all the result, and recalculate the CLIP feature correlations

        ### We also need to refine the similarity metrics
        self.labels_dict = new_label_dict
        self.mask_dict = new_mask_dict
        self.delete_similarit_matrix(removed_lable_index)


        return new_label_dict, new_mask_dict

    def delete_similarit_matrix(self, deletion:torch.Tensor):
        mask = torch.ones(self.pairwise_similarity.size(0), dtype=bool)
        mask[deletion] = False

        # Use the mask to select rows and columns
        self.pairwise_similarity = self.pairwise_similarity[mask][:, mask]
        self.labels = torch.tensor(self.labels)[mask]

    def merge(self, threshold = 0.6) -> Tuple[dict, dict]:
        '''
            This would be an algorithm that merges pair-wise related cluster
        '''
        triu_indices = torch.triu_indices(self.pairwise_similarity.size(0), self.pairwise_similarity.size(1), offset=0)
        self.pairwise_similarity[triu_indices[0], triu_indices[1]] = 0
        
        high_corr_indices = torch.where(self.pairwise_similarity > threshold)
        pairs_to_merge_index = list(zip(high_corr_indices[0].tolist(), high_corr_indices[1].tolist()))
        self.labels = np.array(self.labels)
        
        pairs_to_merge = {}
        
        for i, merged_tuple in enumerate(pairs_to_merge_index):
            for label_index in merged_tuple:
                pairs_to_merge[self.labels[label_index]] = min(self.labels[np.array(merged_tuple)])
        new_label_dict = {}
        new_mask_dict = {}
        for key in tqdm(self.labels_dict.keys(), desc='merge noisy labels and masks'):
            current_labels = self.labels_dict[key] # if current label in the key of pairs to merge
            current_mask = self.mask_dict[key]
            new_label_dict[key] = []
            new_mask_dict[key] = []
            for i, label in enumerate(current_labels):
                mask = current_mask[i]
                new_label = pairs_to_merge.get(label, label) 
                # if label in the merged mask, we need to merge
                # else it is not in, then it stay where they are
                if new_label in new_label_dict[key]: # The masks one need to merge appears at same image
                   # Then merge two mask
                   new_mask_dict[key][new_label_dict[key].index(new_label)] = self.merge_masks(new_mask_dict[key][new_label_dict[key].index(new_label)], mask)
                else: # if we do not need to merge, we solve it normally
                    new_mask_dict[key].append(mask)
                    new_label_dict[key].append(new_label)
            
            new_label_dict[key] = np.array(new_label_dict[key])
            new_mask_dict[key] = np.array(new_mask_dict[key])
        return new_label_dict, new_mask_dict

    def merge_masks(self, mask1:np.ndarray, mask2:np.ndarray) -> np.ndarray:
        stacked_masks = torch.tensor(np.array([mask1,mask2]))
        max_result, _ = stacked_masks.max(dim=0)

        return max_result.numpy()


                




if __name__ == '__main__':
    metrics = mask_refinement(
        labels_location='/home/planner/xiongbutian/ignores/output/refined_label.npz',
        features_location='/home/planner/xiongbutian/ignores/clip_result/pictures/semantic_features.npz',
        image_location='/home/planner/xiongbutian/ignores/images',
        mask_location='/home/planner/xiongbutian/ignores/output/refined_mask.npz',
        device='cuda'
    )

    label, masks = metrics.deletion()

    label, masks = metrics.merge()
    np.savez_compressed('/home/planner/xiongbutian/ignores/clip_result/refined_mask_labels/labels_dict_merged_0.npz', **label)
    np.savez_compressed('/home/planner/xiongbutian/ignores/clip_result/refined_mask_labels/masks_merged_0.npz', **masks)
