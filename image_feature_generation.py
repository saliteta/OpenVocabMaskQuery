import torch
import open_clip
from stable_processing.loader import load_dataset
from stable_processing.color_logging import print_with_color

import argparse
from tqdm import tqdm
import numpy as np




def parser():
    parser = argparse.ArgumentParser("SEG-CLIP Generate Semantic Specific Feature Representation", add_help=True)
    parser.add_argument("--clip_version", type=str, default="ViT-B-16", required=True, help="the version of CLIP version, in debugging only use ViT-B-16")
    parser.add_argument("--clip_checkpoint", type=str, required=True, help="path to clip checkpoint file")
    parser.add_argument("--image_dir", type=str, required=True, help="path to image file")
    parser.add_argument("--mask_location", type=str, required=True, help="path to mask file")
    parser.add_argument("--label_location", type=str, required=True, help="path to label file")
    parser.add_argument("--output_dir", "-o", type=str, default="outputs", required=True, help="output directory")
    parser.add_argument("--debugging", type=str, default="False", help="if to save the internal output to image folder")
    parser.add_argument("--device", type=str, default="cuda", help="running on cpu only!, default=False")
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parser()
    clip_version = args.clip_version
    clip_checkpoint = args.clip_checkpoint
    image_dir = args.image_dir
    mask_location = args.mask_location
    label_location = args.label_location
    debugging = args.debugging
    device = args.device
    output_dir = args.output_dir


    model, _, _ = open_clip.create_model_and_transforms(clip_version, pretrained=clip_checkpoint)
    model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active
    tokenizer = open_clip.get_tokenizer(clip_version)
    model = model.cuda()

    print_with_color(f'model type {clip_version}, initialization accomplished, your model is in {device}', 'GREEN')

    dataset = load_dataset(
        img_directory = image_dir,
        mask_location = mask_location,
        label_location = label_location,
        num_workers = 1,
        device = 'cuda'
    )

    # the image dict will saved as: whole image data, each masks corresponding data
    image_feature_dict = {} 
    print_with_color(f'dataset initialization accomplished', 'GREEN')

    print_with_color(f'processing ...', 'YELLOW')

    with torch.no_grad(), torch.cuda.amp.autocast():

        for (batched_masked_images, labels, basename) in tqdm(dataset):
            batched_masked_images = batched_masked_images
            image_features =model.encode_image(batched_masked_images)
            # at here, we get the image features for each batched image
            image_features /= image_features.norm(dim=-1, keepdim=True)

            image_feature_dict[basename] = image_features.cpu().numpy()

        
        np.savez_compressed(f'{output_dir}/semantic_features.npz', **image_feature_dict)
        
        print_with_color(f'Image Feature extracting accomplished', 'GREEN')
        print_with_color(f'file location is at {output_dir}/semantic_features.npz', 'GREEN')

            
