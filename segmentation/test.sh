CUDA_VISIBLE_DEVICES=0,1,2,3 torchpack dist-run -np 4 python test.py configs/semantic_kitti/linkunet/default.yaml --load_path ../checkpoints/max-iou-val.pt
