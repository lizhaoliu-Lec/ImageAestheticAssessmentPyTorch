"""
Image Aesthetic Assessment Based on Pairwise Comparison – A Unified
Approach to Score Regression, Binary Classification, and Personalization. ICCV 2019

paper reference: https://openaccess.thecvf.com/content_ICCV_2019/papers/Lee_Image_Aesthetic_Assessment_Based_on_Pairwise_Comparison__A_Unified_ICCV_2019_paper.pdf
"""
import torch
import torch.nn as nn

from model.base import get_base
from model.factory import ModelFactory
from model.utils import freeze_weights, name_size_grad

__all__ = ['UnifiedNet']


class GlobalAndLocalPooling(nn.Module):
    def __init__(self, output_size=(2, 2)):
        super().__init__()
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        self.output_size = output_size
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.local_pool = nn.AdaptiveAvgPool2d(output_size)

    def forward(self, x: torch.Tensor):
        local_pooled = self.local_pool(x)  # (N, C, oh, ow)
        global_pooled = self.global_pool(x)  # (N, C, 1, 1)

        N, C, oh, ow = local_pooled.size()
        local_pooled = local_pooled.permute(0, 2, 3, 1)  # # (N, oh, ow, C)
        local_pooled = local_pooled.reshape((N, oh * ow, C))
        global_pooled = global_pooled.permute(0, 2, 3, 1).reshape((N, 1, C))
        aggregate_pooled = torch.cat([local_pooled, global_pooled], dim=1)
        return aggregate_pooled.reshape((N, -1))


@ModelFactory.register('UnifiedNet')
class UnifiedNet(nn.Module):
    def __init__(self,
                 base_name='resnet50',
                 stage='pretrained',
                 pool_window=(2, 2),
                 fc_dims=(1024, 256),
                 num_classes=2,
                 freeze_base=False,
                 **kwargs):
        super().__init__()
        assert stage in ['pretrained', 'finetune']
        self.base = get_base(base_name=base_name, **kwargs)
        self.head = self.get_head(pool_window, fc_dims, num_classes)

        if freeze_base:
            freeze_weights(self.base)

        self.stage = stage

    def get_head(self, pool_window, fc_dims, num_classes):
        in_dim = self.base.last_dim * (pool_window[0] * pool_window[1] + 1)

        global_and_local_pool = GlobalAndLocalPooling(pool_window)
        fc1 = nn.Linear(in_features=in_dim, out_features=fc_dims[0])
        fc2 = nn.Linear(in_features=fc_dims[0], out_features=fc_dims[1])
        fc3 = nn.Linear(in_features=fc_dims[1], out_features=num_classes)
        return nn.Sequential(global_and_local_pool, fc1, fc2, fc3)

    def forward(self, x):
        return self.head(self.base(x))


if __name__ == '__main__':
    def run_UnifiedNet():
        x = torch.randn((5, 3, 224, 224)).cuda()
        model = UnifiedNet(freeze_base=True).cuda()
        print(model)
        out = model(x)
        print("===> x.size() ", x.size())
        print("===> pool_out.size() ", out.size())
        name_size_grad(model)


    # run_local_pooling()
    run_UnifiedNet()
