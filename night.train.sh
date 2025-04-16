#!/bin/bash

set -e 
set -x

echo "Step 1: Extracting visual features with I3D..."
python feature_extraction.py \
    --input_dir augmented_videos \
    --output_dir visual_features \
    --type rgb

python feature_extraction.py \
    --input_dir raw_videos \
    --output_dir visual_features \
    --type rgb


echo "Step 2: Extracting landmark features..."
python landmark_extraction.py \
    --input_dir raw_videos \
    --output_dir landmark_features

python landmark_extraction.py \
    --input_dir augmented_videos \
    --output_dir landmark_features


if [ -d "data/Ours/features" ]; then
    echo "Directory data/Ours/features already exists. Moving it to data/Ours/features_old..."
    mv data/Ours/features data/Ours/features_old
fi


echo "Step 3: Copying visual features to ASFormer input directory..."

mkdir -p data/Ours/features
cp -r visual_features/* data/Ours/features/

echo "Step 4: Training and evaluating ASFormer..."
python train.py --action train --dataset Ours --split 1


echo "Step 5: Copying output results and model..."
mkdir -p visual_results
mkdir -p visual_models
cp -r results/Ours/split_1 visual_results/
cp -r models/Ours/split_1 visual_models/

echo "Step 6: Copying landmark features to ASFormer input directory..."

rm -r data/Ours/featurs/
mkdir -p data/Ours/features
cp -r landmark_features/* data/Ours/features/


echo "Step 7: Training and evaluating ASFormer with landmark features..."
python train.py --action train --dataset Ours --split 1


echo "Step 8: Copying output results and model with landmark features..."
mkdir -p landmark_results
mkdir -p landmark_models
cp -r results/Ours/split_1 landmark_results/
cp -r models/Ours/split_1 landmark_models/



