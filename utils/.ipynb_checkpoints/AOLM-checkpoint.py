import torch
from skimage import measure


def AOLM(fms,fm1):

    # conv5_c
    # A = torch.sum(fms, dim=1, keepdim=True)  # A
    # a = torch.mean(A, dim=[2, 3], keepdim=True) # a平均
    # M = (A > a).float() # 掩膜
    #
    # conv5_b
    A1 = torch.sum(fm1, dim=1, keepdim=True)
    a1 = torch.mean(A1, dim=[2, 3], keepdim=True)
    M1 = (A1 > a1).float()

    # gram-cam
    A = torch.sum(fms, dim=1, keepdim=True)  # A
    a = torch.mean(A, dim=[2, 3], keepdim=True)  # a平均
    M = (A > a).float()  # 掩膜




    # conv5_b 和 conv5_c进行交集
    coordinates = []
    for i, m in enumerate(M1):
        mask_np = m.cpu().numpy().reshape(14, 14)
        component_labels = measure.label(mask_np)

        # properties = measure.regionprops(component_labels)
        # # 区域
        # areas = []
        # for prop in properties:
        #     areas.append(prop.area)
        # max_idx = areas.index(max(areas))
        #
        #
        # intersection = ((component_labels==(max_idx+1)).astype(int) + (M1[i][0].cpu().numpy()==1).astype(int)) ==2
        #prop = measure.regionprops(intersection.astype(int))

        properties = measure.regionprops(component_labels)
        #区域
        areas = []
        for prop in properties:
            areas.append(prop.area)
            
        if areas == []:
            bbox = [0, 0, 14, 14]
            print('there is one img no intersection')
        else:
            max_idx = areas.index(max(areas))
            intersection = ((component_labels==(max_idx+1)).astype(int)+ (M[i][0].cpu().numpy()==1).astype(int)) >=1
            prop = measure.regionprops(intersection.astype(int))
            if len(prop) == 0:
                bbox = [0, 0, 14, 14]
                print('there is one img no intersection')
            else:
                bbox = prop[0].bbox
            
        


        x_lefttop = bbox[0] * 32 - 1
        y_lefttop = bbox[1] * 32 - 1
        x_rightlow = bbox[2] * 32 - 1
        y_rightlow = bbox[3] * 32 - 1
        # for image
        if x_lefttop < 0:
            x_lefttop = 0
        if y_lefttop < 0:
            y_lefttop = 0
        coordinate = [x_lefttop, y_lefttop, x_rightlow, y_rightlow]
        coordinates.append(coordinate)
    return coordinates

