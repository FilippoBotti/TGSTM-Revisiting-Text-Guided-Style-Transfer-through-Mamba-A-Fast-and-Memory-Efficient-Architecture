CUDA_VISIBLE_DEVICES=0

python main.py --mode test --content_dir ./data/train_set_small \
    --name exp1 --vgg ./vgg_normalised.pth \
    --text "Watercolor painting with purple brush" --test_dir ./data/test_set_small --max_iter 200 --seed 2 \
    --output_dir ./output --layer_dec_mamba 1  \
    --batch_size 4 --checkpoint_path ./checkpoints/model.pth