#!/bin/bash
# Download 3DDFA-V3 pretrained models and assets from HuggingFace
# Run once before using the model. Creates assets/ directory.

set -e
BASE="https://huggingface.co/datasets/Zidu-Wang/3DDFA-V3/resolve/main"

FILES=(
    face_model.npy
    indices_38365_35709.npy
    indices_53215_35709.npy
    indices_53215_38365.npy
    indices_53490_35709.npy
    large_base_net.pth
    meanshape-106ldms.obj
    meanshape-134ldms.obj
    meanshape-68ldms.obj
    meanshape-parallel.obj
    meanshape-seg.obj
    net_recon.pth
    net_recon_mbnet.pth
    retinaface_resnet50_2020-07-20_old_torch.pth
    similarity_Lm3D_all.mat
)

mkdir -p assets
for f in "${FILES[@]}"; do
    if [ -f "assets/$f" ]; then
        echo "✓ assets/$f (cached)"
    else
        echo "↓ assets/$f"
        wget -q --show-progress "$BASE/assets/$f" -O "assets/$f"
    fi
done

echo "Done — $(ls assets/ | wc -l) files in assets/"
