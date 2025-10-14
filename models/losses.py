import torch.nn as nn
import torch
from utility.function import normal,normal_style
import utility.function as func
from utility.function import normal,normal_style
from utility.function import calc_mean_std
from torch.nn import functional as F

class StyLosses:
    def __init__(self, clip_model, vgg_encoder):
        self.clip_model = clip_model
        enc_layers = list(vgg_encoder.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1
        
        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False
                
        self.mse_loss = nn.MSELoss()

    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(4):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]
    
    def calc_content_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        return self.mse_loss(input, target)
    
    def content_loss(self, Ics, content):
        Ics_feats = self.encode_with_intermediate(Ics)
        content_feats = self.encode_with_intermediate(content)
        loss = self.calc_content_loss(Ics_feats[-1], content_feats[-1]) 

        return loss
    

    def directional_clip_loss(self, stylized_image, content_image, text_features, source_text_features):
        content_feat = self.clip_model.encode_image(func.clip_normalize(content_image))
        stylized_feat = self.clip_model.encode_image(func.clip_normalize(stylized_image))
        delta_img = stylized_feat - content_feat
        delta_text = text_features - source_text_features
        
        loss = (1 - torch.cosine_similarity(delta_img, delta_text, dim=1)).mean()
        
        return loss 
    
    def tv_loss(self, inputs_jit):
        # COMPUTE total variation regularization loss
        diff1 = inputs_jit[:, :, :, :-1] - inputs_jit[:, :, :, 1:]
        diff2 = inputs_jit[:, :, :-1, :] - inputs_jit[:, :, 1:, :]
        diff3 = inputs_jit[:, :, 1:, :-1] - inputs_jit[:, :, :-1, 1:]
        diff4 = inputs_jit[:, :, :-1, :-1] - inputs_jit[:, :, 1:, 1:]
        loss_var_l2 = torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)
        
        return loss_var_l2
    
    def aug_clip_loss(self, stylized_image, content_image, text_features, source_text_features, threeshold=0.5):
        aug_img = []
        loss_patch = 0
        for it in range(16):
            out_aug = self.clip_model.augmentations(stylized_image)                            ##torch.Size([4, 3, 224, 224])
            aug_img.append(out_aug)
        aug_img = torch.cat(aug_img,dim=0)                              ##torch.Size([64, 3, 224, 224])
        source_features = self.clip_model.encode_image(func.clip_normalize(content_image))
        source_features /= (source_features.clone().norm(dim=-1, keepdim=True))

        image_features = self.clip_model.encode_image(func.clip_normalize(aug_img))
        image_features /= (image_features.clone().norm(dim=-1, keepdim=True))
        
        img_direction = (image_features-source_features.repeat(16,1))
        img_direction /= img_direction.clone().norm(dim=-1, keepdim=True)
        
        text_direction = (text_features-source_text_features)
        text_direction /= text_direction.norm(dim=-1, keepdim=True)
        loss_temp = (1- torch.cosine_similarity(img_direction, text_direction, dim=1))#.mean()

        loss_temp[loss_temp<threeshold] =0

        loss_patch+=loss_temp.mean()
        
        return loss_patch
    

