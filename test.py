import torch
import torch.nn as nn
import torch.utils.data as data
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from utility.sampler import InfiniteSamplerWrapper
from torchvision.utils import save_image
from torchvision.transforms.functional import adjust_contrast
import models.helper as helper
import models.mamba as mamba
import models.Stytr as StyTr
from datasets.flatfolder import FlatFolderDataset
from models.clip_helper import ClipWrapper
from models.losses import StyLosses
import os
from PIL import ImageDraw, ImageFont, Image
import textwrap

import time

def text_to_image(text, width, height, font_size=14):
    img = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except:
        font = ImageFont.load_default()
    avg_char_width = sum(font.getbbox(char)[2] - font.getbbox(char)[0] for char in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ") / 52
    max_chars_per_line = max(1, int(width // avg_char_width))
    lines = textwrap.wrap(text, width=max_chars_per_line)
    y = 5
    for line in lines:
        draw.text((5, y), line, fill='black', font=font)
        bbox = font.getbbox(line)
        line_height = bbox[3] - bbox[1]
        y += line_height + 2
    to_tensor = transforms.ToTensor()
    return to_tensor(img)

def train_transform(crop_size=224):
    transform_list = [
        transforms.RandomCrop(crop_size),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)

def test_transform():
    transform_list = [
        transforms.Resize(size=(512, 512)),                     ##add antialias=True
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)


def adjust_learning_rate(args, optimizer, iteration_count):
    """Imitating the original implementation"""
    lr = args.lr / (1.0 + args.lr_decay * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        

def test(args):
    device = torch.device('cuda')
    vgg = helper.make_vgg()
    vgg.load_state_dict(torch.load(args.vgg))
    vgg = nn.Sequential(*list(vgg.children())[:31]).to(device)

    decoder = helper.make_decoder()
        
    decoder.load_state_dict(torch.load(args.decoder))
    mamba_net = mamba.Mamba(args=args, d_model = 512)

    network = StyTr.Net(mamba_net, decoder, vgg, args)
    network.load_state_dict(torch.load(args.checkpoint_path))
    network.to(device)
    
    clip_model = ClipWrapper()

    test_tf = test_transform()

    test_dataset = FlatFolderDataset(args.test_dir, test_tf)

    test_iter = iter(data.DataLoader(
        test_dataset, batch_size=args.batch_size,
        num_workers=args.n_threads))


    test_images1 = next(test_iter)
    test_images1 = test_images1.cuda()

    with torch.no_grad():
        _, style_features, _  = clip_model.extract_features(args.text)
 

    output_dir = args.output_dir + f"/{args.text.replace(r' ','_')}/test/"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with torch.no_grad():
        Ics_test = network(test_images1, style_features)
        clip_score = clip_model.get_clip_score(Ics_test, args.text).item()
        ssim_score = clip_model.get_ssim(Ics_test, test_images1).item()
        Ics_test = adjust_contrast(Ics_test, 1.5)
        # caption = text_to_image(style_texts[0], Ics_test.size(2), Ics_test.size(3), font_size=20).to(device)
        # Ics_test = torch.cat([style_images, caption.unsqueeze(0), Ics_test], dim=0)
        for h in range(Ics_test.size(0)):
            save_image(Ics_test[h], f"{output_dir}/img_{h}.png", nrow=1, normalize=True, scale_each=True)

    print("Final clip score: ", clip_score)
    print("Final ssim score: ", ssim_score)



