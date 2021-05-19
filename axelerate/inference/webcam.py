import cv2
import os
import numpy as np
import json
import tensorflow as tf 
from axelerate.networks.yolo.frontend import create_yolo
from axelerate.networks.yolo.backend.utils.box import draw_boxes
import argparse
from pathlib import Path


physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    config = tf.config.experimental.set_memory_growth(physical_devices[0], True)


def get_args():
    a = argparse.ArgumentParser()
    a.add_argument("--config", type=Path, required=True, help="the config file")
    a.add_argument("--weights", type=Path, required=True, help="the weights")
    a.add_argument("--threshold", type=float, default=0.3)
    return a.parse_args()


def prep_image(frame_from_cv2: np.array, size):
    img_rgb = cv2.merge([frame_from_cv2[:,:,2],frame_from_cv2[:,:,1], frame_from_cv2[:,:,0]])
    sy, sx, ch = img_rgb.shape
    diff = sx - sy
    st0 = int(diff/2)
    st1 = int(sx - diff/2)
    img_rgb = img_rgb[:,st0:st1,:]
    img_rgb = cv2.resize(img_rgb, (size,size))
    img_normalized = (img_rgb.astype(float) / 125.5) - 1

    return img_normalized, img_rgb

def to_bgr(frame_from_cv2: np.array) -> np.array:
    img_rgb = cv2.merge([frame_from_cv2[:,:,2],frame_from_cv2[:,:,1], frame_from_cv2[:,:,0]])
    return img_rgb



def main():
    args = get_args()
    config = json.load(open(args.config, "r"))
    weights = args.weights
    try:
        input_size = config['model']['input_size'][:]
    except:
        input_size = [config['model']['input_size'],config['model']['input_size']]
    yolo = create_yolo(config['model']['architecture'],
                           config['model']['labels'],
                           input_size,
                           config['model']['anchors'])
    yolo.load_weights(weights)
    threshold = args.threshold

    cap = cv2.VideoCapture(0)
    video_idx = 0
    writing_video = False
    writer = None
    while True:
        ret, frame = cap.read()
        frame, vizframe = prep_image(frame, input_size[0])
        
        print(frame.shape)
        height, width = frame.shape[:2]
        prediction_time, boxes, probs = yolo.predict(frame[np.newaxis,...], height, width, float(threshold))
        labels = np.argmax(probs, axis=1) if len(probs) > 0 else []
        n_copy = draw_boxes(vizframe, boxes, probs, config['model']['labels'])
        n_c_v = to_bgr(n_copy)
        if writing_video:
            writer.write(n_c_v)
        cv2.imshow("annotated", n_c_v)
        ky = cv2.waitKey(1) & 0xFF
        if ky == ord('s'):
            if not writing_video:
                fcc = cv2.VideoWriter_fourcc(*'VP90') # VP90 MJPG
                writer = cv2.VideoWriter(f'output-{video_idx}.avi',fcc,25.0, (224,224))
                print(f" START writing output-{video_idx}.avi")
                writing_video = True
            else:
                print(f" STOP writing output-{video_idx}.avi")
                writer.release()
                writing_video = False
                video_idx += 1
        if ky == ord('q'):
            if writing_video:
                writer.release()
            break

    cap.release()


if __name__ == "__main__":
    main()
