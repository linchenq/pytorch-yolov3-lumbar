from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *

import os
import sys
import time
import datetime
import argparse

from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="data/lumbar/samples", help="path to dataset")
    parser.add_argument("--model_def", type=str, default="config/yolov3-lumbar.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="checkpoints/yolov3_ckpt_196.pth", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/lumbar/classes.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.9, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=512, help="size of each image dimension")
    parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("output", exist_ok=True)

    # Set up model
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)

    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))

    model.eval()  # Set in evaluation mode

    dataloader = DataLoader(
        ImageFolder(opt.image_folder, img_size=opt.img_size),
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_cpu,
    )

    classes = load_classes(opt.class_path)  # Extracts class labels from file

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    imgs = []  # Stores image paths
    img_detections = []  # Stores detections for each image index

    print("\nPerforming object detection:")
    prev_time = time.time()
    for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
        # Configure input
        input_imgs = Variable(input_imgs.type(Tensor))

        # Get detections
        with torch.no_grad():
            detections = model(input_imgs)
            detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)

        # Log progress
        current_time = time.time()
        inference_time = datetime.timedelta(seconds=current_time - prev_time)
        prev_time = current_time
        print("\t+ Batch %d, Inference Time: %s" % (batch_i, inference_time))

        # Save image and detections
        imgs.extend(img_paths)
        img_detections.extend(detections)

    # Bounding-box colors
    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]

    print("\nSaving images:")
    # Iterate through images and save plot of detections
    for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):

        print("(%d) Image: '%s'" % (img_i, path))

        # Create plot
        img = np.array(Image.open(path).convert('L'))
        plt.figure()
        # fig, ax = plt.subplots(1)
        # ax.imshow(img)
        plt.imshow(img, cmap='gray')

        # Draw bounding boxes and labels of detections
        if detections is not None:
            # Rescale boxes to original image
            detections = rescale_boxes(detections, opt.img_size, img.shape[:2])
            unique_labels = detections[:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)
            bbox_colors = random.sample(colors, n_cls_preds)

            # TODO: Only modified on the lumbar Dataset
            #### Since each image could only contain one type of unique disks
            unique_dict = {}
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                cur = classes[int(cls_pred)]
                if cur not in unique_dict:
                    unique_dict[cur] = (x1, y1, x2, y2, conf, cls_conf, cls_pred)
                else:
                    if cls_conf > unique_dict[cur][5]:
                        unique_dict[cur] = (x1, y1, x2, y2, conf, cls_conf, cls_pred)
            new_detections = torch.Tensor([])
            for k, v in unique_dict.items():
                add_tensor = torch.Tensor([i for i in v])
                new_detections = torch.cat((new_detections, add_tensor), 0)
            new_detections = new_detections.view(-1, 7)
            #### END

            # TODO: Only modified on the lumbar Dataset
            #### Since each image could only contain one type of unique disks
            # for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in new_detections:
            #### END

                print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))

                box_w = x2 - x1
                box_h = y2 - y1

                color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                # Create a Rectangle patch
                bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
                # Add the bbox to the plot
                # ax.add_patch(bbox)
                plt.gca().add_patch(bbox)
                # Add label
                # 1. Add text
                plt.text(
                    (0.7 * x1),
                    y1,
                    s=classes[int(cls_pred)],
                    color="white",
                    verticalalignment="top"
                    # bbox={"color": color, "pad": 0},
                )
                # 2. OR Add anotation
                # plt.annotate(
                #     classes[int(cls_pred)],
                #     xy=(x1, y1),
                #     xytext=((3.0*x1-x2)/2.0, y1),
                #     color='w',
                #     arrowprops=dict(arrowstyle='->',connectionstyle='arc3',color='white')
                # )

        # Save generated image with detections
        plt.axis("off")
        plt.gca().xaxis.set_major_locator(NullLocator())
        plt.gca().yaxis.set_major_locator(NullLocator())
        filename = path.split("/")[-1].split(".")[0]
        plt.savefig(f"output/{filename}.png", bbox_inches="tight", pad_inches=0.0)
        plt.show()
        plt.close()
