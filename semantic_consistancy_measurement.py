'''
    Here is how to determine the consistancy in semantic consistancy
    If we regard it as the same labels, then the inter-label difference should be large
    inter label difference should be large
'''


from stable_processing.color_logging import print_with_color
from stable_processing.metrics import Consistancy_metrics
import argparse
import os


def parser():
    parser = argparse.ArgumentParser("Semantic Consistancy Metrics Visualization", add_help=True)
    parser.add_argument("--image_dir", type=str, required=True, help="path to image file")
    parser.add_argument("--mask_location", type=str, required=True, help="path to mask file")
    parser.add_argument("--features_location", type=str, required=True, help="path to label file")
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

    print_with_color('loading dataset ...', 'YELLOW')
    metrics = Consistancy_metrics(
        labels_location=label_location,
        features_location=features_location,
        image_location=image_dir,
        mask_location=mask_location,
        device='cuda'
    )

    print_with_color('Generate Measurement Metrics and Visualization ...', 'YELLOW')
    metrics.plot_intra_label_variance(output_dir)
    metrics.plot_inter_label_similarity(output_dir)
    metrics.visualize_cluster(output_dir)
    #metrics.hint_cluster_generation(os.path.join(output_dir,'hint'))

    print_with_color(f'All Done, result stored at {output_dir}', 'GREEN')


    
