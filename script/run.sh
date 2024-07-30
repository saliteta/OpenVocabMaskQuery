# CUDA_VISIBLE_DEVICES=2 python image_feature_generation.py \
#     --clip_version ViT-B-16 \
#     --clip_checkpoint /home/planner/xiongbutian/Foundation_Models/CLIP/open_clip_pytorch_model.bin \
#     --image_dir /home/planner/xiongbutian/ignores/images \
#     --mask_location /home/planner/xiongbutian/ignores/output/refined_mask.npz \
#     --label_location /home/planner/xiongbutian/ignores/output/refined_label.npz \
#     --output_dir /home/planner/xiongbutian/ignores/clip_result/pictures \
#     --debugging False \
#     --device cuda 


# CUDA_VISIBLE_DEVICES=2 python image_feature_generation.py \
#     --clip_version ViT-B-16 \
#     --clip_checkpoint /home/planner/xiongbutian/Foundation_Models/CLIP/open_clip_pytorch_model.bin \
#     --image_dir /home/planner/xiongbutian/ignores/images \
#     --mask_location /home/planner/xiongbutian/ignores/clip_result/refined_mask_labels/masks_delete.npz \
#     --label_location /home/planner/xiongbutian/ignores/clip_result/refined_mask_labels/labels_dict_delete.npz \
#     --output_dir /home/planner/xiongbutian/ignores/clip_result/pictures_deletion \
#     --debugging False \
#     --device cuda 


CUDA_VISIBLE_DEVICES=2 python image_feature_generation.py \
    --clip_version ViT-B-16 \
    --clip_checkpoint /home/planner/xiongbutian/Foundation_Models/CLIP/open_clip_pytorch_model.bin \
    --image_dir /home/planner/xiongbutian/ignores/images \
    --mask_location /home/planner/xiongbutian/ignores/clip_result/refined_mask_labels/masks_merged_0.npz \
    --label_location /home/planner/xiongbutian/ignores/clip_result/refined_mask_labels/labels_dict_merged_0.npz \
    --output_dir /home/planner/xiongbutian/ignores/clip_result/pictures_merge_0 \
    --debugging False \
    --device cuda 