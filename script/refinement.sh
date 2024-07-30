CUDA_VISIBLE_DEVICES=2 python semantic_refinement.py \
    --image_dir /home/planner/xiongbutian/ignores/images \
    --mask_location /home/planner/xiongbutian/ignores/output/refined_mask.npz \
    --features_location /home/planner/xiongbutian/ignores/clip_result/pictures/semantic_features.npz \
    --label_location /home/planner/xiongbutian/ignores/output/refined_label.npz \
    --output_dir /home/planner/xiongbutian/ignores/clip_result/pictures_merge \
    --device cuda 