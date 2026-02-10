import torch
import torch.nn as nn

from BEATs import BEATs, BEATsConfig
from ast_models import ASTModel

class AudioClassifier(nn.Module):
    def __init__(self, backbone_name, num_classes, ckpt_path, duration):
        super().__init__()
        self.backbone_name = backbone_name.lower()

        if self.backbone_name == 'beats':
            if ckpt_path:
                ckpt = torch.load(ckpt_path)
                self.backbone = BEATs(BEATsConfig(ckpt['cfg']))
                self.backbone.load_state_dict(ckpt['model'], strict=False)

            else:
                self.backbone = BEATs(BEATsConfig({'encoder_embed_dim': 768}))

            embed_dim = 768
        
        elif self.backbone_name == 'ast':
            target_tdim = int(duration *100)
            self.backbone = ASTModel(
                input_tdim=target_tdim,
                input_fdim=128,
                imagenet_pretrain=True
            )

            embed_dim = 768
        
        else:
            raise ValueError(f"Unknown backbone: {backbone_name}")
        
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        if x.di() == 3 and x.shape[1] == 1:
            x = x.squeeze(1)

        if self.backbone_name == 'beats':
            feats, _ = self.backbone.extract_features(x) 
        elif self.backbone_name == 'ast':
            feats = self.backbone(x)

        if feats.dim() == 3:
            feats = feats.mean(dim=1)

        return self.head(feats)
    

class ClassificationHead(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        if x.dim() == 3:
            x = x.mean(dim=1)

        return self.head(x)