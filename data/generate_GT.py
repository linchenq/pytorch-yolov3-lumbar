import numpy as np
import hdf5storage
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
# from PIL import Image
from matplotlib.image import imsave
import re

from data.generate_xml import Xmlgenerator

class GTGenerator(object):
    def __init__(self, filename):
        super().__init__()
        self.filename = filename
        self.point_dict = {}
        self.xml_dict = {}

        self._generate_dict()

    def _generate_dict(self):
        point, point_addition, point_length = 3, 4, 4
        edge, edge_addition, edge_length = 199, 348, 58
        for index in range(1, 6):
            _list = [i-1 for i in range(point, point+point_length)] + [j-1 for j in range(edge, edge+edge_length*4)]
            self.point_dict['D'+str(index)] = _list
            point += point_addition
            edge += edge_addition
        self.point_dict['S'] = [i-1 for i in range(21, 25)] + [j-1 for j in range(1765, 1997)]

    def _generate_bbox(self, jpeg_path, xml_path):
        mat_data = hdf5storage.loadmat(self.filename)
        raw_image, raw_points = mat_data['ImageShapePair'][:, 0][0], mat_data['ImageShapePair'][:, 1][0]
        ant_points = np.transpose(raw_points[0:2])
        ant_points[:, 0], ant_points[:, 1] = ant_points[:, 0] - 1., ant_points[:, 1] -1.

        assert ant_points.shape == (1996, 2)

        plt.figure()

        # save image to jpeg_path
        ########### Alternative Method ###########
        # save_img = Image.fromarray(raw_image)
        # save_img = save_img.convert('L')
        # save_img.save(jpeg_path)
        ##########################################

        imsave(jpeg_path, raw_image, cmap='gray')

        for key in self.point_dict.keys():
            image = ant_points[self.point_dict[key], :]
            image = image.astype(np.float32)
            x, y, w, h = cv2.boundingRect(image)

            plt.gca().add_patch(patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none'))

            xmin, ymin, xmax, ymax = x, y-h, x+w, y

            self.xml_dict[key] = (xmin, ymin, xmax, ymax)

        plt.imshow(raw_image, cmap='gray')

        # save annotation to Xml
        save_xml = Xmlgenerator(re.split(r'/|\\', jpeg_path)[-1], (512, 512, 1), self.xml_dict)
        save_xml.output(xml_path)

if __name__ == '__main__':
    root_path = 'C:/Research/LumbarSpine/RealImageShapePair/'
    for index in range(30):
        gt_gen = GTGenerator(root_path + 'RealImageShapePair_P' + str(index+1) + '.mat')
        gt_gen._generate_bbox(jpeg_path='C:/Research/LumbarSpine/Github/PyTorch-YOLOv3/data/lumbar/images/' + str(index+1) + '.jpeg',
                              xml_path='C:/Research/LumbarSpine/Github//PyTorch-YOLOv3/data/lumbar/annotations/' + str(index+1) + '.xml')


