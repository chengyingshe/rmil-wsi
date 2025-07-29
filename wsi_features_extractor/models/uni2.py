# https://huggingface.co/MahmoodLab/UNI
import timm
from torchvision import transforms
import torch
from timm.layers import SwiGLUPacked

    
def get_uni_trans():
    transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    return transform


def get_uni_model(device):
    timm_kwargs = {
                'img_size': 224, 
                'patch_size': 14, 
                'depth': 24,
                'num_heads': 24,
                'init_values': 1e-5, 
                'embed_dim': 1536,
                'mlp_ratio': 2.66667*2,
                'num_classes': 0, 
                'no_embed_class': True,
                'mlp_layer': SwiGLUPacked,
                'act_layer': torch.nn.SiLU, 
                'reg_tokens': 8, 
                'dynamic_img_size': True
            }

    model = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs)
    model.eval()
    return model.to(device)

if __name__ ==  '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_uni_model(device)
