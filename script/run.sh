CUDA_VISIBLE_DEVICES=2 python image_feature_generation.py \
    --clip_version ViT-B-16 \
    --clip_checkpoint /home/planner/xiongbutian/Foundation_Models/CLIP/open_clip_pytorch_model.bin \
    --image_dir /home/planner/xiongbutian/ignores/images \
    --mask_location /home/planner/xiongbutian/ignores/output/refined_mask.npz \
    --label_location /home/planner/xiongbutian/ignores/output/refined_label.npz \
    --output_dir /home/planner/xiongbutian/ignores/clip_result \
    --debugging False \
    --device cuda 