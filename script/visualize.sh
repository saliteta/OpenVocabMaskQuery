CUDA_VISIBLE_DEVICES=2 python visualize.py \
    --clip_version ViT-B-16 \
    --clip_checkpoint /home/planner/xiongbutian/Foundation_Models/CLIP/open_clip_pytorch_model.bin \
    --image_location /home/planner/xiongbutian/ignores/images/frame_00002.jpg \
    --mask_location /home/planner/xiongbutian/ignores/output/refined_mask.npz \
    --feature_location /home/planner/xiongbutian/ignores/clip_result/semantic_features.npz \
    --text_discription /home/planner/xiongbutian/VLM_text_semantic_response/semantic_latent_retrival/discription/test.txt \
    --output_dir /home/planner/xiongbutian/ignores/clip_result/pictures \
    --device cuda 