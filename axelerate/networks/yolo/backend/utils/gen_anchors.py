import random
import argparse
import numpy as np
import cv2
from axelerate.networks.yolo.backend.utils.annotation import parse_annotation
import json

## Adapted from https://github.com/experiencor/keras-yolo2/blob/master/gen_anchors.py (MIT license)

def IOU(ann, centroids):
    w, h = ann
    similarities = []

    for centroid in centroids:
        c_w, c_h = centroid

        if c_w >= w and c_h >= h:
            similarity = w*h/(c_w*c_h)
        elif c_w >= w and c_h <= h:
            similarity = w*c_h/(w*h + (c_w-w)*c_h)
        elif c_w <= w and c_h >= h:
            similarity = c_w*h/(w*h + c_w*(c_h-h))
        else: #means both w,h are bigger than c_w and c_h respectively
            similarity = (c_w*c_h)/(w*h)
        similarities.append(similarity) # will become (k,) shape

    return np.array(similarities)

def avg_IOU(anns, centroids):
    n,d = anns.shape
    sum = 0.

    for i in range(anns.shape[0]):
        sum+= max(IOU(anns[i], centroids))

    return sum/n

def to_anchors(centroids):
    anchors = centroids.copy()

    widths = anchors[:, 0]
    sorted_indices = np.argsort(widths)

    r = "anchors: ["
    for i in sorted_indices[:-1]:
        r += '%0.2f,%0.2f, ' % (anchors[i,0], anchors[i,1])

    #there should not be comma after last anchor, that's why
    r += '%0.2f,%0.2f' % (anchors[sorted_indices[-1:],0], anchors[sorted_indices[-1:],1])
    r += "]"
    return r

def run_kmeans(ann_dims, anchor_num):
    ann_num = ann_dims.shape[0]
    iterations = 0
    prev_assignments = np.ones(ann_num)*(-1)
    iteration = 0
    old_distances = np.zeros((ann_num, anchor_num))

    indices = [random.randrange(ann_dims.shape[0]) for i in range(anchor_num)]
    centroids = ann_dims[indices]
    anchor_dim = ann_dims.shape[1]

    while True:
        distances = []
        iteration += 1
        for i in range(ann_num):
            d = 1 - IOU(ann_dims[i], centroids)
            distances.append(d)
        distances = np.array(distances) # distances.shape = (ann_num, anchor_num)

        print("iteration {}: dists = {}".format(iteration, np.sum(np.abs(old_distances-distances))))

        #assign samples to centroids
        assignments = np.argmin(distances,axis=1)

        if (assignments == prev_assignments).all() :
            return centroids

        #calculate new centroids
        centroid_sums=np.zeros((anchor_num, anchor_dim), np.float)
        for i in range(ann_num):
            centroid_sums[assignments[i]]+=ann_dims[i]
        for j in range(anchor_num):
            centroids[j] = centroid_sums[j]/(np.sum(assignments==j) + 1e-6)

        prev_assignments = assignments.copy()
        old_distances = distances.copy()

def generate_anchors(config_dict: dict, num_anchors: int):
    config = config_dict

    train_imgs = parse_annotation(config['train']['train_annot_folder'],
                                                config['train']['train_image_folder'],
                                                config['model']['labels'])

    grid_w = config['model']['input_size']/32
    grid_h = config['model']['input_size']/32

    # run k_mean to find the anchors
    annotation_dims = []
    for image in train_imgs:
        im = cv2.imread(image.fname)

        cell_w = im.shape[1]/grid_w
        cell_h = im.shape[0]/grid_h

        for obj in image.boxes:
            relative_w = (float(obj[2]) - float(obj[0]))/cell_w
            relatice_h = (float(obj[3]) - float(obj[1]))/cell_h
            annotation_dims.append(tuple(map(float, (relative_w,relatice_h))))

    annotation_dims = np.array(annotation_dims)
    centroids = run_kmeans(annotation_dims, num_anchors)

    # write anchors to file
    print('\naverage IOU for', num_anchors, 'anchors:', '%0.2f' % avg_IOU(annotation_dims, centroids))
    return to_anchors(centroids)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations_folder", required=True, type=str)
    parser.add_argument("--images_folder", required=True, type=str)
    parser.add_argument("--labels", required=True, type=str)
    parser.add_argument("--input_size", default=224, type=int)
    parser.add_argument("--num_anchors", default=5, type=int)
    args = parser.parse_args()
    cd = {}
    cd["train"] = {}
    cd["train"]["train_annot_folder"] = args.annotations_folder
    cd["train"]["train_image_folder"] = args.images_folder
    cd["model"] = {}
    cd["model"]["labels"] = [x.strip() for x in args.labels.split(",")]
    cd["model"]["input_size"] = args.input_size
    anch = generate_anchors(cd, args.num_anchors)
    print("***** ANCHORS *****")
    print(anch)


