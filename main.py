import argparse
import torch
import random
from train_model import train
import numpy as np
import torch.backends.cudnn as cudnn
from PIL import Image, ImageFile
import os
from test import test

def set_seed(seed: int = 42):
    # Python
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    # Make cuDNN deterministic
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)

    # Ensure hash-based ops are deterministic
    os.environ["PYTHONHASHSEED"] = str(seed)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Basic options
    parser.add_argument('--content_dir', type=str, default ='./train_set_small')
    parser.add_argument('--style_dir', type=str, default ='/home/filippo/datasets/wikiart')
    parser.add_argument('--test_dir', type=str, default ='./test_set_small_new') 
    parser.add_argument('--hr_dir', type=str)  
    parser.add_argument('--img_dir', type=str, default ='./test_set_small_new')     
    parser.add_argument('--vgg', type=str, default='models/vgg_normalised.pth')
    parser.add_argument('--mode', type=str, required=True,
                        choices=['train', 'test'])

    # training options
    parser.add_argument('--output_dir', default='./output',
                        help='Directory to save the output images')
    parser.add_argument('--checkpoint_path', default='./output/checkpoints/model.pth',
                        help='Checkpoint path')

    parser.add_argument('--text', default='Fire',
                        help='text condition')
    parser.add_argument('--name', default='none',
                        help='name')
    parser.add_argument('--n_threads', type=int, default=1)                   
    parser.add_argument('--num_test', type=int, default=16)
    parser.add_argument('--save_model_interval', type=int, default=500)
    parser.add_argument('--save_img_interval', type=int, default=200)
    parser.add_argument('--decoder', type=str, default='/home/filippo/checkpoints/decoder/decoder.pth')

    # net parameters
    parser.add_argument('--seed',type=int, default=777)
    parser.add_argument('--layer_dec_mamba',type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr_decay', type=float, default=5e-5)
    parser.add_argument('--max_iter', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--crop_size', type=int, default=224)
    parser.add_argument('--thresh', type=float, default=0.7)

    parser.add_argument('--d_state', type=int, default=16)
    parser.add_argument('--hidden_dim', type=int, default=512)

    # loss weights
    parser.add_argument('--content_weight', type=float, default=1.0)           
    parser.add_argument('--clip_weight', type=float, default=10.0)             
    parser.add_argument('--tv_weight', type=float, default=1e-4)               
    parser.add_argument('--glob_weight', type=float, default=1.0)  
    


    args = parser.parse_args()
    set_seed(args.seed)
    if args.mode == 'train':
        # print(args)
        train(args)
    if args.mode == 'test':
        # print(args)
        test(args)
    else:
        print("Specify valid mode")