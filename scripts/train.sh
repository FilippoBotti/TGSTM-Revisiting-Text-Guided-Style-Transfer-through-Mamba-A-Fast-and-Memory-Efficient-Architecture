CUDA_VISIBLE_DEVICES=0

python main.py --mode train --content_dir ./data/train_set_small \
    --name exp1 --vgg ./vgg_normalised.pth \
    --text "Watercolor painting with purple brush" --test_dir ./data/test_set_small --max_iter 200 \
    --output_dir ./outputs --layer_dec_mamba 1  \
    --batch_size 4
