<<<<<<< HEAD
COARSE_OUTPUT_DIR=/home/butian/workspace/extract-semantics/results/clip_output/coarse/
COARSE_MASK_LABEL=/home/butian/workspace/extract-semantics/results/output/
IMAGE_DIR=/data/10F/images/
CLIP_VERSION=ViT-L-14
CLIP_CHECKPOINT=/data/foundation_models/CLIP/open_clip_pytorch_model.bin
REFINED_OUTPUT_DIR=/home/butian/workspace/extract-semantics/results/clip_output/refined/

# CUDA_VISIBLE_DEVICES=0 python -W ignore image_feature_generation.py \
#     --clip_version ${CLIP_VERSION} \
#     --clip_checkpoint ${CLIP_CHECKPOINT} \
#     --image_dir ${IMAGE_DIR} \
#     --mask_location ${COARSE_MASK_LABEL}refined_mask.npz \
#     --label_location ${COARSE_MASK_LABEL}refined_label.npz \
#     --output_dir ${COARSE_OUTPUT_DIR} \
#     --debugging False \
#     --device cuda 
# 
# CUDA_VISIBLE_DEVICES=0 python -W ignore semantic_consistancy_measurement.py\
#     --image_dir ${IMAGE_DIR} \
#     --mask_location ${COARSE_MASK_LABEL}refined_mask.npz \
#     --label_location ${COARSE_MASK_LABEL}refined_label.npz \
#     --features_location ${COARSE_OUTPUT_DIR}semantic_features.npz\
#     --output_dir ${COARSE_OUTPUT_DIR} \
#     --device cuda 
# 
# CUDA_VISIBLE_DEVICES=0 python -W ignore semantic_refinement.py \
#     --image_dir ${IMAGE_DIR} \
#     --mask_location ${COARSE_MASK_LABEL}refined_mask.npz \
#     --label_location ${COARSE_MASK_LABEL}refined_label.npz \
#     --features_location ${COARSE_OUTPUT_DIR}semantic_features.npz \
#     --output_dir ${REFINED_OUTPUT_DIR}\
#     --device cuda 

######################## Visualize Refinement Result #################################################
######################## Visualize Refinement Result #################################################
######################## Not Neccessary, Just examing it #############################################



# CUDA_VISIBLE_DEVICES=0 python -W ignore image_feature_generation.py \
#     --clip_version ${CLIP_VERSION} \
#     --clip_checkpoint ${CLIP_CHECKPOINT} \
#     --image_dir ${IMAGE_DIR} \
#     --mask_location ${REFINED_OUTPUT_DIR}refined_mask.npz \
#     --label_location ${REFINED_OUTPUT_DIR}refined_label.npz \
#     --output_dir ${REFINED_OUTPUT_DIR} \
#     --debugging False \
#     --device cuda 

# CUDA_VISIBLE_DEVICES=0 python -W ignore semantic_consistancy_measurement.py\
#     --image_dir ${IMAGE_DIR} \
#     --mask_location ${REFINED_OUTPUT_DIR}refined_mask.npz \
#     --label_location ${REFINED_OUTPUT_DIR}refined_label.npz \
#     --features_location ${REFINED_OUTPUT_DIR}semantic_features.npz\
#     --output_dir ${REFINED_OUTPUT_DIR} \
#     --device cuda 

CUDA_VISIBLE_DEVICES=0 python -W ignore prepare_data.py\
    --mask_location ${REFINED_OUTPUT_DIR}refined_mask.npz \
    --label_location ${REFINED_OUTPUT_DIR}refined_label.npz \
    --features_location ${REFINED_OUTPUT_DIR}semantic_features.npz\
    -o /home/butian/workspace/10F
=======
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
>>>>>>> 893896b9ad8263553cd0f19b797a270a941a3f35
