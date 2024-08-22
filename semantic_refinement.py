'''
    This work can be finished after one extracted semantic feature in the run.sh
'''


from stable_processing.color_logging import print_with_color
from stable_processing.merge_and_deletion import mask_refinement
import argparse
import numpy as np
import os


def parser():
    parser = argparse.ArgumentParser("Semantic refinement", add_help=True)
    parser.add_argument("--image_dir", type=str, required=True, help="path to image file")
    parser.add_argument("--mask_location", type=str, required=True, help="path to mask file")
    parser.add_argument("--features_location", type=str, required=True, help="path to feature file")
    parser.add_argument("--label_location", type=str, required=True, help="path to label file")
    parser.add_argument("--output_dir", "-o", type=str, default="outputs", required=True, help="output directory")
    parser.add_argument("--device", type=str, default="cuda", help="running on cpu only!, default=False")
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = parser()
    image_dir = args.image_dir
    mask_location = args.mask_location
    label_location = args.label_location
    features_location = args.features_location
    device = args.device
    output_dir = args.output_dir

    if os.path.exists(output_dir):
        print_with_color(f'{output_dir} already exists, skip folder creationg', 'YELLOW')
    else:
        os.makedirs(output_dir)
        print_with_color(f'created folder {output_dir}', 'GREEN')

    print_with_color('loading dataset ...', 'YELLOW')
    metrics = mask_refinement(
        labels_location=label_location,
        features_location=features_location,
        image_location=image_dir,
        mask_location=mask_location,
        device='cuda'
    )

    print_with_color('Deleting Noisy Labels ...', 'YELLOW')
    label, masks = metrics.deletion()

    print_with_color('Merge Related Labels ...', 'YELLOW')
    label, masks = metrics.merge()

    print_with_color('Saving Results...', 'GREEN')

<<<<<<< HEAD
    np.savez_compressed(os.path.join(output_dir,'refined_label.npz'), **label)
    np.savez_compressed(os.path.join(output_dir,'refined_mask.npz'), **masks)
=======
    np.savez_compressed(os.path.join(output_dir,'labels_dict_refined.npz'), **label)
    np.savez_compressed(os.path.join(output_dir,'masks_refined.npz'), **masks)
>>>>>>> 893896b9ad8263553cd0f19b797a270a941a3f35


    print_with_color(f'All Done, result stored at {output_dir}', 'GREEN')


    
