import numpy as np
import cv2
from shapely.geometry import Polygon, LineString
import pyclipper
import math
from concern.config import State
from .data_process import DataProcess


class MakeShrinkMap(DataProcess):
    r'''
    Making binary mask from detection data with ICDAR format.
    Typically following the process of class `MakeICDARData`.
    '''
    min_text_size = State(default=8)
    shrink_ratio = State(default=0.4)
    def __init__(self, **kwargs):
        self.load_all(**kwargs)
    def dist(self, a, b):
        return np.sqrt(np.sum((a - b) ** 2))     
    def process(self, data):
        '''
        requied keys:
            image, polygons, ignore_tags, filename
        adding keys:
            mask
        '''
        # mid_area = 0
        # ints = 0
        image = data['image']
        polygons = data['polygons']
        ignore_tags = data['ignore_tags']
        image = data['image']
        filename = data['filename']
        
        h, w = image.shape[:2]
        # if data['filename'] =='../../dataset/total_text//train_images/img949.jpg':
        #         print(polygon,11)
        if data['is_training']:
            polygons, ignore_tags = self.validate_polygons(
                polygons, ignore_tags, h, w)
        gt = np.zeros((1, h, w), dtype=np.float32)
        short_gt = np.zeros((1, h, w), dtype=np.float32)
        mask = np.ones((h, w), dtype=np.float32)
        for i in range(len(polygons)):
            
            polygon = polygons[i]

            height = max(polygon[:, 1]) - min(polygon[:, 1])
            width = max(polygon[:, 0]) - min(polygon[:, 0])
            if ignore_tags[i] or min(height, width) < self.min_text_size:
                cv2.fillPoly(mask, polygon.astype(
                    np.int32)[np.newaxis, :, :], 0)
                ignore_tags[i] = True
            else:
                middle_width,middle_height = (polygon[0] + polygon[1] + polygon[2] + polygon[3])/4
                poly_shrink = []
                for k in range(polygon.shape[0]):
                    dis_x =  middle_width - polygon[k, 0]
                    dis_y =  middle_height - polygon[k, 1]
                    poly_shrink.append([middle_width - dis_x*0.6, middle_height - dis_y*0.6])
                poly_shrink = np.array(poly_shrink).astype(np.int32)
                lens = polygon.shape[0]
               
                cv2.fillPoly(gt[0], [poly_shrink.astype(np.int32)], 1)

                # cv2.imwrite('xx.png',short_gt[0]*2)
        #print(kwmask.max(),kwmask.min())
        if filename is None:
            filename = ''
        data.update(image=image,
                    polygons=polygons,
                    gt=gt, mask=mask
                   )
        return data

    def validate_polygons(self, polygons, ignore_tags, h, w):
        '''
        polygons (numpy.array, required): of shape (num_instances, num_points, 2)
        '''
        if len(polygons) == 0:
            return polygons, ignore_tags
        assert len(polygons) == len(ignore_tags)
        for polygon in polygons:
            polygon[:, 0] = np.clip(polygon[:, 0], 0, w - 1)
            polygon[:, 1] = np.clip(polygon[:, 1], 0, h - 1)

        for i in range(len(polygons)):
            area = self.polygon_area(polygons[i])
            if abs(area) < 1:
                ignore_tags[i] = True
            if area > 0:
                polygons[i] = polygons[i][::-1, :]
        return polygons, ignore_tags

    def polygon_area(self, polygon):
        edge = 0
        for i in range(polygon.shape[0]):
            next_index = (i + 1) % polygon.shape[0]
            edge += (polygon[next_index, 0] - polygon[i, 0]) * (polygon[next_index, 1] + polygon[i, 1])

        return edge / 2.

