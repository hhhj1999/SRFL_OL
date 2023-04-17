import torch
from skimage import measure
from torch import nn
import torch.nn.functional as F

from .tiny_vit import *
from .vit_model import *

from torch.utils.tensorboard import SummaryWriter

# from tiny_vit import *
# from vit_model import *

class MainNet(nn.Module):
    def __init__(self, proposalN=None, num_classes=None, channels=None):
        # nn.Module子类的函数必须在构造函数中执行父类的构造函数
        super(MainNet, self).__init__()
        self.model = tiny_vit_21m_224(pretrained=True)
#         self.model = net = vit_base_patch16_224_in21k(num_classes=200)



    def forward(self, x, epoch, batch_idx, status='test', DEVICE='cuda'):
#     def forward(self, x):
        writer = SummaryWriter('log')

        out,weights = self.model(x)

        #
        B,_ = out.shape

        weight_ori = weights
        weights = None
        weight_ori = weight_ori.reshape((B,int(weight_ori.size(0)/B),18,49,49))
#         print('weight_ori',weight_ori.shape)

        M = torch.randn(weight_ori.shape[0], 14,14).cuda()
        for item in range(weight_ori.shape[0]):
            weight = weight_ori[item]
#             print('weight',weight.shape)
            weight = weight.transpose(1, 0)
    
    
#             weight = weight / weight.sum(dim=-1).unsqueeze(-1)
            weight = weight
        
        
            j = torch.zeros(weight.size()).cuda()                                 
            j[0] = weight[0]
#             print(j[0].shape)
            for n in range(1, weight.size(0)):
                # 18个head逐一相加
#                 j[n] = torch.matmul(weight[n], j[n - 1])
                j[n] = torch.add(weight[n], j[n - 1])
                
            v = j[-1]
#             print('v', v.shape)
#             patch = torch.mean(v, dim=1) / (torch.mean(v, dim=1).max())
            patch = torch.mean(v, dim=2)           
#             print('patch', patch.shape)
            patch = patch.view(1, 2, 2, 7, 7)
#             print('patch', patch.shape)
            patch = patch.permute(0, 1, 3, 2, 4).contiguous().view(14, 14)
#             print('patch', patch.shape)
#             print(patch.flatten())
            a = torch.mean(patch.flatten()) * 1
#             print('a',a.shape)
            M[item] = (patch > a).float()
#             print('m',M[item].shape)

        coordinates = []
        for i, m in enumerate(M):
            mask_np = m.cpu().numpy().reshape(14, 14)
            component_labels = measure.label(mask_np, connectivity=2)

            properties = measure.regionprops(component_labels)
            areas = []
            for prop in properties:
                areas.append(prop.area)
            max_idx = areas.index(max(areas))

            bbox = properties[max_idx]['bbox']
            x_lefttop = bbox[0] * 32 - 1
            y_lefttop = bbox[1] * 32 - 1
            x_rightlow = bbox[2] * 32 - 1
            y_rightlow = bbox[3] * 32 - 1
            # for image
            if x_lefttop < 0:
                x_lefttop = 0
            if y_lefttop < 0:
                y_lefttop = 0
            coordinate = [int(x_lefttop), int(y_lefttop), int(x_rightlow), int(y_rightlow)]
            coordinates.append(coordinate)

        coordinates = torch.tensor(coordinates)
        batch_size = len(coordinates)
        local_imgs = torch.zeros([B, 3, 448, 448]).cuda()  # [N, 3, 448, 448]
        # local_imgs = torch.zeros([B, 3, 448, 448])
        for i in range(batch_size):
            [x0, y0, x1, y1] = coordinates[i]
            local_imgs[i:i + 1] = F.interpolate(x[i:i + 1, :, x0:(x1 + 1), y0:(y1 + 1)], size=(448, 448),
                                                mode='bilinear', align_corners=True)
        out2, weights = self.model(local_imgs.detach()) 
        
        if batch_idx < 10:
#             writer.add_images('images', x, 0)
            writer.add_images('images', local_imgs, batch_idx)   
    
        return out,out2



# net = MainNet()
# input = torch.ones((6,3,448,448))
# net(input)

