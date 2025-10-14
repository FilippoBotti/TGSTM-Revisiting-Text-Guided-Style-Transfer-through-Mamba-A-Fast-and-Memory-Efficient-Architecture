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
        

def train(args):
    device = torch.device('cuda')
    vgg = helper.make_vgg()
    vgg.load_state_dict(torch.load(args.vgg))
    vgg = nn.Sequential(*list(vgg.children())[:31]).to(device)

    decoder = helper.make_decoder()
        
    decoder.load_state_dict(torch.load(args.decoder))
    mamba_net = mamba.Mamba(args=args, d_model = 512)

    network = StyTr.Net(mamba_net, decoder, vgg, args)
    network.train()
    network.to(device)
    
    clip_model = ClipWrapper()
    loss_helper = StyLosses(clip_model, vgg)

    source = "a Photo"

    content_tf = train_transform(args.crop_size)
    test_tf = test_transform()

    content_dataset = FlatFolderDataset(args.content_dir, content_tf)
    test_dataset = FlatFolderDataset(args.test_dir, test_tf)

    content_iter = iter(data.DataLoader(
        content_dataset, batch_size=args.batch_size,
        sampler=InfiniteSamplerWrapper(content_dataset),
        num_workers=args.n_threads))
    test_iter = iter(data.DataLoader(
        test_dataset, batch_size=args.batch_size,
        num_workers=args.n_threads))

    optimizer = torch.optim.Adam(list(network.decode.parameters()) +
                                list(network.mamba.parameters()),
                                lr=args.lr)

    test_images1 = next(test_iter)
    test_images1 = test_images1.cuda()

    with torch.no_grad():
        style_cls_token, style_features, _  = clip_model.extract_features(args.text, template=False)
        source_cls_token, _, _ = clip_model.extract_features(source, template=True)
 

    output_dir = args.output_dir + f"/{args.text.replace(r' ','_')}/images/"
    checkpoint_dir = args.output_dir + f"/{args.text.replace(r' ','_')}/checkpoints/"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    total_iter_time = 0.0

    with tqdm(range(0, args.max_iter)) as titer:
        for i in titer:
            iter_start = time.perf_counter()
            adjust_learning_rate(args, optimizer, iteration_count=i)
            content_images = next(content_iter).to(device)     
            Ics = network(content_images, style_features)      
            
            loss_c = loss_helper.content_loss(Ics, content_images)
            loss_aug=loss_helper.aug_clip_loss(Ics, content_images, style_cls_token, source_cls_token, args.thresh)
            loss_glob = loss_helper.directional_clip_loss(Ics, content_images, style_cls_token, source_cls_token)
            reg_tv = loss_helper.tv_loss(Ics)

            loss = args.content_weight*loss_c + args.clip_weight*loss_aug +  args.glob_weight*loss_glob + args.tv_weight*reg_tv

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            iter_time = time.perf_counter() - iter_start
            total_iter_time += iter_time
            avg_time = total_iter_time / (i + 1)
            eta = avg_time * (args.max_iter - i - 1)

            clip_score = clip_model.get_clip_score(Ics, args.text).item()
            ssim_score = clip_model.get_ssim(Ics, content_images).item()
            postfix = {
                "clip": f"{clip_score:.3f}",
                "ssim": f"{ssim_score:.3f}",
                "t/it(s)": f"{iter_time:.3f}",
                "ETA(m)": f"{eta/60:.1f}"
            }
            titer.set_postfix(**postfix)
            titer.set_description(f"avg {avg_time:.3f}s")

            if (i + 1) % args.save_img_interval == 0:
                with torch.no_grad():
                    Ics_test = network(test_images1, style_features)
                    clip_score = clip_model.get_clip_score(Ics_test, args.text).item()
                    ssim_score = clip_model.get_ssim(Ics_test, test_images1).item()
                    Ics_test = adjust_contrast(Ics_test, 1.5)
                    # caption = text_to_image(style_texts[0], Ics_test.size(2), Ics_test.size(3), font_size=20).to(device)
                    # Ics_test = torch.cat([style_images, caption.unsqueeze(0), Ics_test], dim=0)
                    for h in range(Ics_test.size(0)):
                        save_image(Ics_test[h], f"{output_dir}/img_{h}.png", nrow=1, normalize=True, scale_each=True)

    
    torch.save(network.state_dict(), f"{checkpoint_dir}/model.pth")

    print("Final clip score: ", clip_score)
    print("Final ssim score: ", ssim_score)



