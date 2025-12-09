#!/bin/bash


## Feature Study Experiments ###
python -m modeling.train --wandb_project FeatureStudy --head feature_study --feature_level 5 --scheduler none --tag AllFeatures --lr 0.01 --epochs 20 || true
python -m modeling.train --wandb_project FeatureStudy --head feature_study --feature_level 4 --scheduler none --tag Feature4 --lr 0.01 --epochs 20 || true
python -m modeling.train --wandb_project FeatureStudy --head feature_study --feature_level 3 --scheduler none --tag Feature3 --lr 0.01 --epochs 20 || true
python -m modeling.train --wandb_project FeatureStudy --head feature_study --feature_level 2 --scheduler none --tag Feature2 --lr 0.01 --epochs 20 || true
python -m modeling.train --wandb_project FeatureStudy --head feature_study --feature_level 1 --scheduler none --tag Feature1 --lr 0.01 --epochs 20 || true

## Max Performance Study Experiments ###
python -m modeling.train --wandb_project MaxPerformanceStudy --head mlp --feature_level 5 --scheduler cosine --tag CADqMLP --lr 0.001 --epochs 10 || true
python -m modeling.train --wandb_project MaxPerformanceStudy --head mlp --feature_level 5 --scheduler cosine --no-freeze --tag MaxPerformanceMlp --lr 0.001 --epochs 5 || true

## Train Final CADq Model ###
python -m modeling.train --wandb_project MaxPerformanceStudy --head mlp --feature_level 5 --scheduler cosine --tag CADqMLP --lr 0.001 --epochs 10 || true
