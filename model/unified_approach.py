"""
Image Aesthetic Assessment Based on Pairwise Comparison – A Unified
Approach to Score Regression, Binary Classification, and Personalization. ICCV 2019

paper reference: https://openaccess.thecvf.com/content_ICCV_2019/papers/Lee_Image_Aesthetic_Assessment_Based_on_Pairwise_Comparison__A_Unified_ICCV_2019_paper.pdf
"""
import torch
import torch.nn as nn

from model.resnet_based import ResNetBase
from model.factory import ModelFactory

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
class UnifiedNet(ResNetBase):
    def __init__(self,
                 stage='pretrained',
                 pool_window=(2, 2),
                 fc_dims=(1024, 256),
                 num_classes=2,
                 **kwargs):
        super().__init__(**kwargs)
        assert stage in ['pretrained', 'finetune']
        self.stage = stage
        self.global_and_local_pool = GlobalAndLocalPooling(pool_window)
        in_dim = self.last_dim * (pool_window[0] * pool_window[1] + 1)
        self.fc1 = nn.Linear(in_features=in_dim, out_features=fc_dims[0])
        self.fc2 = nn.Linear(in_features=fc_dims[0], out_features=fc_dims[1])
        self.classification = nn.Linear(in_features=fc_dims[1], out_features=num_classes)

    def forward(self, x):
        x = super().forward(x)
        x = self.global_and_local_pool(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.classification(x)


if __name__ == '__main__':
    def run_local_pooling():
        x = torch.randn((5, 3, 224, 224)).cuda()
        model = ResNetBase().cuda()
        local_pool = GlobalAndLocalPooling((2, 2)).cuda()
        out = model(x)
        pool_out = local_pool(out)

        print("===> x.size() ", x.size())
        print("===> out.size() ", out.size())
        print("===> pool_out.size() ", pool_out.size())


    def run_UnifiedNet():
        x = torch.randn((5, 3, 224, 224)).cuda()
        model = UnifiedNet().cuda()
        print(model)
        out = model(x)
        print("===> x.size() ", x.size())
        print("===> pool_out.size() ", out.size())


    # run_local_pooling()
    run_UnifiedNet()
