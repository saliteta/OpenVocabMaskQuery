import numpy as np
import torch


import cuml
import matplotlib.pyplot as plt

from stable_processing.loader import Image_Mask_Dataset
from stable_processing.color_logging import print_with_color
import random
from tqdm import tqdm

import os

class Consistancy_metrics:
    def __init__(
            self,
            labels_location: str,
            features_location: str,
            image_location: str,
            mask_location: str,
            device: str = 'cuda'
    ):
        '''
            Args: 
            - labels_location(str): is a npz file path that store a dict. Each element
            contains each masks corresponding labels
            - features_location(str): is a npz file path that store a dict. Each element
            contains a features corresponds to the masks
            - image_location(str): is a image folder path that store a dict. Each element
            contains a features corresponds to the masks
        '''
        self.dataset = Image_Mask_Dataset(
            original_image_directory=image_location,
            refined_masks=mask_location,
            refined_labels=labels_location,
            transform=None,
            device=device
            )
        self.labels_dict = np.load(labels_location)
        self.features_dict = np.load(features_location)
        self.device = device
        self.image_location = image_location
        self._set_up_feature_label_correlation() # this will construct a dict key is the label, element is all the features

    def _set_up_feature_label_correlation(self):
        '''
            Construct a dictionary whose key is label
            element is features
        '''
        self.feature_label_correlation = {element: [] for key in self.labels_dict for element in self.labels_dict[key]}
        for key in tqdm(self.labels_dict):
            for i, element in enumerate(self.labels_dict[key]):
                self.feature_label_correlation[element].append(self.features_dict[key][i + 1])
        for key in self.feature_label_correlation:
            self.feature_label_correlation[key] = torch.Tensor(np.array(self.feature_label_correlation[key])).to(self.device)

    def get_intra_label_coherence(self) -> dict:
        """ Calculate mean and variance within each label. """
        intra_label_var = {}
        for label, features in self.feature_label_correlation.items():
            mean = features.mean(dim=0)
            distances = torch.norm(features - mean, dim=1)
            variance = distances.var(unbiased=True)
            intra_label_var[label] = variance
        return intra_label_var

    def get_inter_label_difference(self) -> torch.Tensor:
        """ Calculate the differences in means between labels. """
        means = []
        labels = list(self.feature_label_correlation.keys())
        for label in labels:
            mean = self.feature_label_correlation[label].mean(dim=0)
            means.append(mean)
        
        means = torch.stack(means)
        mean_diffs = torch.cdist(means, means, p=2)  # Compute pairwise Euclidean distances between means
        return mean_diffs.mean().item()

    def visualize_cluster(self, save_path:str) :
        """ Use PCA or t-SNE from cuml to reduce dimensions to 2D and visualize the clusters on GPU. """
        '''
        
            The saved result will be in save_fig path
        '''
        all_data = []
        labels = []
        colors = plt.cm.rainbow(torch.linspace(0, 1, len(self.feature_label_correlation)))

        # Gather all data and labels
        for i, (label, features) in enumerate(self.feature_label_correlation.items()):
            all_data.append(features)
            labels.extend([i] * features.shape[0])  # Assign an integer label for each class

        # Convert list of tensors to a single GPU tensor
        all_data = torch.cat(all_data, dim=0).cuda()
        labels = torch.tensor(labels).cuda()

        # Dimensionality reduction with t-SNE using cuml
        reducer = cuml.TSNE(n_components=2, random_state=5)
        reduced_data = reducer.fit_transform(all_data)

        # Move reduced data back to CPU for plotting
        reduced_data = reduced_data.get()

        # Plotting
        plt.figure(figsize=(10, 7))
        scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels.cpu().numpy(), cmap='rainbow', alpha=0.6)
        plt.colorbar(scatter, ticks=range(len(self.feature_label_correlation)))
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.title('Cluster Visualization')
        plt.grid(True)

        plt.savefig(save_path)
    
    def plot_intra_label_variance(self, save_path:str):
        """ Plot the variance for each label and annotate with average inter-label mean difference. """
        stats = self.get_intra_label_coherence()
        labels = list(stats.keys())
        variances = torch.tensor([stats[label] for label in labels]).cpu().numpy()  # Directly use scalar variance

        # Get average of mean differences
        mean_diff_avg = self.get_inter_label_difference()

        # Create bar plot
        plt.figure(figsize=(8, 6))
        plt.bar(labels, variances, color='skyblue')
        plt.xlabel('Labels')
        plt.ylabel('Average Variance')
        plt.title('Intra-label Variance with Inter-label Mean Difference Annotation')
        plt.grid(True, linestyle='--', alpha=0.6)

        # Annotate with mean difference average
        plt.figtext(0.99, 0.01, f'Avg. Inter-label Mean Diff: {mean_diff_avg:.2f}', 
                    horizontalalignment='right', fontsize=12, verticalalignment='bottom')

        plt.savefig(save_path)

    def hint_cluster_generation(self, save_path):
        '''
            When the cluster give out its label in numbers, we usually cannot find what are they corresponds to. 
            Using this function we can generate a folder that contains the number of labels picture. 
            In each picture, it contains some example of that labels
        '''

        random_index = random.sample(range(len(self.dataset)), 100)
        if os.path.exists(save_path):
            print_with_color(f'{save_path} exists, skip creating folder', 'YELLOW')
        else:
            os.makedirs(save_path)

        hint_dict = {}

        for i in tqdm(random_index, desc='generate hint'):
            batched_masked_images, labels, basename = self.dataset[i]
            for index, label in enumerate(labels):
                if int(label) in hint_dict.keys():
                    hint_dict[int(label)].append(batched_masked_images[index+1])
                else: 
                    hint_dict[int(label)] = [batched_masked_images[index+1]]
        
        for label in hint_dict:
            title = f'for cluster {int(label)}'
            self._four_sample_plot(
                image_list = hint_dict[label],
                title=title,
                location = os.path.join(save_path, str(int(label))+'.png')
            )
    
    def _four_sample_plot(self, image_list, title:str, location:str):
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        for i in range(4):
            axes[i//2, i%2].imshow(image_list[i].permute(1,2,0).cpu().numpy())
            axes[1, 0].set_title(f'Image {i}')
        fig.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.savefig(location)



if __name__ == '__main__':
    metrics = Consistancy_metrics(
        labels_location='/home/planner/xiongbutian/ignores/output/refined_label.npz',
        features_location='/home/planner/xiongbutian/ignores/clip_result/semantic_features.npz',
        image_location='/home/planner/xiongbutian/ignores/images',
        mask_location='/home/planner/xiongbutian/ignores/output/refined_mask.npz',
        device='cuda'
    )

    metrics.plot_intra_label_variance('magnitude.png')
    metrics.visualize_cluster('cluster.png')
    metrics.hint_cluster_generation('/home/planner/xiongbutian/VLM_text_semantic_response/semantic_latent_retrival/stable_processing/hint')

    
