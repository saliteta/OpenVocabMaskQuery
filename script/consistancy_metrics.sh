CUDA_VISIBLE_DEVICES=5 python semantic_consistancy_measurement.py\
    --image_dir /home/planner/xiongbutian/ignores/images\
    --mask_location /home/planner/xiongbutian/ignores/output/refined_mask.npz \
    --label_location /home/planner/xiongbutian/ignores/output/refined_label.npz \
    --features_location /home/planner/xiongbutian/ignores/clip_result/semantic_features.npz \
    --output_dir /home/planner/xiongbutian/ignores/clip_result/pictures \
    --device cuda 