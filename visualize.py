'''
    We have low resolution masks
    We also have original image
    We also have a semantic features for each labels
    We then need to merge different features together

    feature_map = normalize(sum(Binary_Masks * Features))

    attention_map = (text_features * feature_map) 
    
    display
'''

import torch


from stable_processing.feature_merge import Feature_merger, text_image_attention_display, store_visualization_result, only_attention
from stable_processing.loader import load_visualization_data as load_data
from stable_processing.loader import load_model
from typing import Tuple

from stable_processing.logging import print_with_color

import argparse
from PIL import Image


def parser():
    parser = argparse.ArgumentParser("SEG-CLIP Generate Semantic Specific Attention Map", add_help=True)
    parser.add_argument("--clip_version", type=str, default="ViT-B-16", required=True, help="the version of CLIP version, in debugging only use ViT-B-16")
    parser.add_argument("--clip_checkpoint", type=str, required=True, help="path to clip checkpoint file")
    parser.add_argument("--image_location", type=str, required=True, help="path to image file")
    parser.add_argument("--feature_location", type=str, required=True, help="path to mask file")
    parser.add_argument("--mask_location", type=str, required=True, help="path to label file")
    parser.add_argument("--text_discription", type=str, required=True, help="Specify, where is your text description list are")
    parser.add_argument("--output_dir", type=str, required=True, help="Specify, where shall one store the result")
    parser.add_argument("--device", type=str, default="cuda", help="running on cuda only!, default=cuda")
    args = parser.parse_args()
    return args



if __name__ == '__main__':

    args = parser()
    clip_version = args.clip_version
    clip_checkpoint = args.clip_checkpoint
    image_location = args.image_location
    feature_location = args.feature_location
    mask_location = args.mask_location
    text_discription = args.text_discription
    output_dir = args.output_dir
    device = args.device

    image, features, masks, text = load_data(
        image_location=image_location,
        feature_location=feature_location,
        masks_location=mask_location,
        text_discription=text_discription,
        device=device
    )

    print_with_color('loading data accomplsihed', 'GREEN')



    model, tokenizer = load_model(
        clip_version=clip_version,
        clip_checkpoint=clip_checkpoint,
        device=device
    )

    print_with_color('loading model accomplsiehd', 'GREEN')

    text_tokenizer = tokenizer(text)
    text_tokenizer = text_tokenizer.to(device)

    print_with_color('Processing Images and Masks ...', 'YELLOW')
    feature_merger = Feature_merger(
        height=image.size[0],
        width=image.size[1]
    )


    image_per_pixel_features = feature_merger.merge(masks, features)
    with torch.no_grad(), torch.cuda.amp.autocast():
        text_features = model.encode_text(text_tokenizer)
        text_features /= text_features.norm(dim=-1, keepdim=True)


        attention_map = feature_merger.cross_attetion(image_per_pixel_features, text_features)


    print_with_color('Creating Attention Image ...', 'YELLOW')
    attention_images = text_image_attention_display(image,attention_map)

    store_visualization_result(
        images=attention_images,
        output_directory=output_dir,
        file_names=text
    )











    