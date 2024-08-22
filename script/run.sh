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
