#!/bin/bash

set -x

declare -a MODELS=(
    alexnet # O
    # Background_Matting # no custom batch size
    # basic_gnn_edgecnn # no custom batch size
    # basic_gnn_gcn # no custom batch size
    # basic_gnn_gin # no custom batch size
    # basic_gnn_sage # no custom batch size
    BERT_pytorch # O
    # cm3leon_generate # compile error
    # DALLE2_pytorch # untrainable
    dcgan
    # demucs # ValueError: Batch size 1 must be divisible by group size 4
    densenet121 # O
    # detectron2_fasterrcnn_r_101_c4 # Too many open files
    # detectron2_fasterrcnn_r_101_dc5 # Too many open files
    # detectron2_fasterrcnn_r_101_fpn # Too many open files
    # detectron2_fasterrcnn_r_50_c4 # Too many open files
    # detectron2_fasterrcnn_r_50_dc5 # Too many open files
    # detectron2_fasterrcnn_r_50_fpn # Too many open files
    # detectron2_fcos_r_50_fpn # Too many open files
    # detectron2_maskrcnn # Too many open files
    # detectron2_maskrcnn_r_101_c4 # Too many open files
    # detectron2_maskrcnn_r_101_fpn # Too many open files
    # detectron2_maskrcnn_r_50_c4 # Too many open files
    # detectron2_maskrcnn_r_50_fpn # Too many open files
    # dlrm # ??
    # doctr_det_predictor # untrainable
    # doctrreco_predictor # untrainable
    # drq # no custom batch size
    # fastNLP_Bert # too complex train
    functorch_dp_cifar10 # O
    # functorch_maml_omniglot # no custom batch size
    hf_Albert # O
    # hf_Bart # uncompilable
    hf_Bert # O
    hf_Bert_large # O
    # hf_BigBird # unhandled fake tensor error (cuda 0 to 3)
    # hf_clip # clip output has no logit -> fixable
    hf_DistilBert # O
    # hf_distil_whisper # untrainable
    hf_GPT2
    hf_GPT2_large
    # hf_Longformer # potential multi-graph: cpu + cuda:3
    # hf_Reformer # potential multi-graph: cpu + cuda:3
    hf_Roberta_base # O
    # hf_T5 # proxy object does not support item assignment (from pickling)
    # hf_T5_base # proxy object does not support item assignment (from pickling)
    # hf_T5_generate # potential multi-graph: cpu + cuda:3
    # hf_T5_large # proxy object does not support item assignment (from pickling)
    # hf_Whisper # potential multi-graph: cpu + cuda:3
    LearningToPaint # O
    lennard_jones # O
    llama # O
    # llama_v2_7b_16h # hf token required
    # llava # untrainable
    # maml # no custom batch size
    # maml_omniglot # no custom batch size
    # microbench_unbacked_tolist_sum # untrainable
    mnasnet1_0 # O
    mobilenet_v2 # O
    # mobilenet_v2_quantized_qat # skip quant model
    mobilenet_v3_large # O
    # moco # inherently requires DDP / cuda
    # moondream # untrainable
    nanogpt # X: incorrect result
    # nvidia_deeprecommender # list index out of range
    # opacus_cifar10 # list index out of range
    phlippe_densenet # O
    phlippe_resnet # O
    # pyhpc_equation_of_state # empty parameter
    # pyhpc_isoneutral_mixing # empty parameter
    # pyhpc_turbulent_kinetic_energy # no custom batch size
    # pytorch_CycleGAN_and_pix2pix # no custom batch size
    # pytorch_stargan # no custom batch size
    pytorch_unet # O
    resnet152 # O
    resnet18 # O 
    resnet50 # O 
    # resnet50_quantized_qat # skip quant model
    resnext50_32x4d # O
    # sam # untrainable
    # sam_fast # untrainable
    shufflenet_v2_x1_0 # O
    # simple_gpt # Model requires cuda
    # simple_gpt_tp_manual # Model requires cuda
    # soft_actor_critic # no custom batch size
    # speech_transformer # no custom batch size
    squeezenet1_1 # O
    # stable_diffusion_text_encoder # HF TOKEN required
    # stable_diffusion_unet # HF TOKEN required
    Super_SloMo # O
    # tacotron2 # Model requires cuda
    # timm_efficientdet # Model requires cuda
    timm_efficientnet # X: incorrect result
    # timm_nfnet # shared weight error
    timm_regnet # O
    timm_resnest # O
    timm_vision_transformer # O
    timm_vision_transformer_large # O
    timm_vovnet # O
    torch_multimodal_clip # O
    # tts_angular # list index out of range
    vgg16 # O
    # vision_maskrcnn # no custom batch size
    # yolov3 # wrong input format / untrainable
)

# Iterate the string array using for loop
cd ../pfeife
for val in ${MODELS[@]}; do
    python mp_launch.py --world_size 2 benchmark_partial.py \
      --device_cnt=2 --batch_size=1 --batch_cnt=2 --repeat=2 \
      --model=$val --loop_cnt=1 --prefetch_fwd=0 --no_bench_optimizer \
      --device_bench profile/stable-nccl-dev2.bench --check_valid
done