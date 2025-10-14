import clip
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchvision import transforms
import utility.function as func
from utility.template import imagenet_templates

class ClipWrapper():
    def __init__(self, device='cuda'):
        self.clip_model, _ = clip.load('ViT-B/32', device, jit=False)
        self.clip_model.eval()
        for param in self.clip_model.parameters():
            param.requires_grad = False
    
        self.device = device
        
        self.ssim = StructuralSimilarityIndexMeasure().to(device)
        self.augmentations = transforms.Compose([
            transforms.RandomPerspective(fill=0, p=1, distortion_scale=0.5),
            transforms.Resize(224)
        ])

    def encode_text(self, text):
        template_text = func.compose_text_with_templates(text, imagenet_templates)
        tokens = clip.tokenize(template_text).to(self.device)                                ##torch.Size([79, 77])
        text_features = self.clip_model.encode_text(tokens).detach()                         ##torch.Size([79, 512])
        text_features = text_features.mean(axis=0, keepdim=True)                        ##torch.Size([1, 512])
        text_features /= text_features.norm(dim=-1, keepdim=True)

        return text_features

    def encode_image(self, x):
        x = x.to(self.device)
        x = self.clip_model.encode_image(func.clip_normalize(x))
        x = x / x.norm(dim=-1, keepdim=True)
        return x
    
    
    def extract_features(self, text, template=False):
        if template:
            text = func.compose_text_with_templates(text, imagenet_templates)
        text = clip.tokenize(text).to(self.device)
        x = self.clip_model.token_embedding(text)        # [B,77,512]
        x = x + self.clip_model.positional_embedding
        x = x.permute(1, 0, 2)    
        x = x.half()
        for i, block in enumerate(self.clip_model.transformer.resblocks):
            attn_mask = block.attn_mask.half().to(self.device)
            x_attn, attn_probs = block.attn(block.ln_1(x),block.ln_1(x),block.ln_1(x), need_weights=True, attn_mask=attn_mask)  # MultiheadAttention
            x = x + x_attn
            x = x + block.mlp(block.ln_2(x)) # Transformer
        x = x.permute(1, 0, 2)                                       # [B,77,512]
        hidden = self.clip_model.ln_final(x)                                     # LayerNorm
        eos_pos = text.argmax(dim=-1)                            # [B] (EOS idx)
        pooled  = hidden[torch.arange(hidden.shape[0]), eos_pos] # [B,512]
        pooled  = pooled @ self.clip_model.text_projection                                     # final feature

        if template:
            pooled = pooled.mean(axis=0, keepdim=True)
        pooled = pooled / pooled.norm(dim=-1, keepdim=True) 
        hidden = hidden.mean(1, keepdim=True)   # [79, 1, 512]
        if template:
            hidden = hidden.mean(0, keepdim=True)   # [79, 1, 512]
        return pooled, hidden, attn_probs        # hidden → [B,77,512]
    
    
    def get_clip_score(self, img, text):
        img_feature = self.encode_image(img)
        tx_feature = self.encode_text(text)
        score = img_feature @ tx_feature.t()
        
        score = score.mean()
        return score
    
    
    def get_ssim(self, content_images, stylized_images):
        score = self.ssim(stylized_images, content_images)
        return score

