import os

pose_path = "D:\paddledeneme\yolov7-pose-estimation"
if not os.path.exists(pose_path):
    print('yolov7-pose-estimation not found, to install => git clone https://github.com/RizwanMunawar/yolov7-pose-estimation')
    exit()
import time
import os
from tqdm import tqdm
import argparse

import cv2
import numpy as np
import onnxruntime as ort
import matplotlib.pyplot as plt 
import matplotlib.path as mplPath
import torch
import sys
sys.path.append(pose_path)
from utils.torch_utils import select_device
from models.experimental import attempt_load
from utils.plots import output_to_keypoint
from utils.general import non_max_suppression_kpt, strip_optimizer

from sort import *
#.... Initialize SORT .... 
#......................... 
sort_max_age = 1 # def 1
sort_min_hits = 3 # def 3
sort_iou_thresh = 0.3 # 0.3
sort_tracker = Sort(max_age=sort_max_age,
								min_hits=sort_min_hits,
								iou_threshold=sort_iou_thresh) 

# static values
left_color = (255, 0, 0)
right_color = (0, 255, 0)
roi_color = (0, 0, 255)
font = cv2.FONT_HERSHEY_SIMPLEX
thickness = 2
person_info = {}


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, r, (dw, dh)

def preprocess(img, new_shape):
    img, ratio, dwdh = letterbox(img.copy(), new_shape=new_shape, auto=False)
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, 0)
    img = np.ascontiguousarray(img)
    img = img.astype(np.float32)
    img /= 255
    return img, ratio, dwdh

def plot(img, box, label, color):

    # circle
    center = box[:2] + ((box[2:] - box[:2]) / 2)
    img = cv2.circle(img, center=center.astype(np.int32), radius=3, color=color, thickness=thickness * 2)

    # custom edges
    x1, y1, x2, y2 = box
    r, d = 10, 10
    # Top left
    img = cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    img = cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    img = cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
    # Top right
    img = cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    img = cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    img = cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
    # Bottom left
    img = cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    img = cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    img = cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
    # Bottom right
    img = cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    img = cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    img = cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)

    # label
    (w, h), _ = cv2.getTextSize(label, font, 0.6, thickness)
    img = cv2.rectangle(img, (x1, y1 - 20), (x1 + w + 20, y1), color, -1)
    img = cv2.putText(img, label, (x1, y1 - 2) ,font, 0.75, [225, 255, 255], thickness)

    return img

roi = np.array([
    [10, 610],
    [2030, 1200],
    [1980, 1320],
    [5, 720]
])
mpl_roi = mplPath.Path(roi)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='best.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default="D:\paddledeneme\dataguess_proj\TEST.mp4", help='source')  # file/folder, 0 for webcam
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--thres', type=float, default=0.1, help='object confidence threshold')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    opt = parser.parse_args()

    if not os.path.exists(opt.source):
        print('source not found')
        exit()

    device = opt.device if torch.cuda.is_available() else 'cpu'
    device = select_device(device)
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    model_path = os.path.join(pose_path, 'yolov7-w6-pose.pt')
    model_shape = 1280
    model = attempt_load(model_path, map_location=device).eval()


    cap = cv2.VideoCapture(opt.source)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    w = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if not opt.nosave:
        folder, file_name = os.path.split(opt.source)
        file_name = os.path.splitext(file_name)[0] + '_result.mp4'
        video = cv2.VideoWriter(
            os.path.join(folder, file_name), 
            cv2.VideoWriter_fourcc(*'XVID'), 
            fps, 
            (h, w)
        )

    for i in tqdm(range(num_frames)):

        ret, img = cap.read()
        if not ret:
            cap.release()
            if not opt.nosave:
                video.release()
            print('\ncant read video, exiting...')
            exit()

        frame, ratio, dwdh = preprocess(
            img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
            new_shape=model_shape
        )

        img = cv2.polylines(img, [roi], True, roi_color, thickness)

        frame = torch.from_numpy(frame).to(device)

        with torch.no_grad():
            pred, _ = model(frame)

            pred = non_max_suppression_kpt(pred, 
                                    0.30, 
                                    0.65, 
                                    nc=model.yaml['nc'], 
                                    nkpt=model.yaml['nkpt'], 
                                    kpt_label=True)[0]

            pred = pred.detach().cpu().numpy()
            results = pred[:, :6]

        dets_to_sort = np.empty((0, 6))
        
        for x0, y0, x1, y1, conf, cls in results:

            if cls != 0 or conf < opt.thres:
                continue

            box = np.array([x0, y0, x1, y1])
            box -= np.array(dwdh*2)
            box /= ratio
            box = box.round().astype(int)         

            img = plot(img, box, f'Person', (0, 255, 0))

            center = (int(box[0] + (box[2]-box[0])/2), int(box[1] + (box[3]-box[1])/2))
            if not mpl_roi.contains_point(center):
                continue

            dets_to_sort = np.vstack((dets_to_sort, np.hstack((box, conf, cls))))

        tracked_dets = sort_tracker.update(dets_to_sort)
        tracks =sort_tracker.getTrackers()
        
        reference_vector = np.array([1, 0])
        for track in tracks:
         
            p1 = np.asarray([track.centroidarr[0][0], track.centroidarr[0][1]])
            p2 = np.asanyarray([track.centroidarr[-1][0], track.centroidarr[-1][1]])
            vector = p1 - p2
            norm = np.linalg.norm(vector)
            direction = vector / norm if norm != 0 else vector
            cross_product = np.cross(reference_vector, direction)
            
            if track.id not in person_info.keys():
                person_info[track.id] = []

            if cross_product < 0:
                person_info[track.id].append("Left")
            elif cross_product > 0:
                person_info[track.id].append("Right")
            else:
                person_info[track.id].append("Straight")

            cv2.putText(img, person_info[track.id][-1], p2.astype(int), font, 1.5, (0, 0, 255), thickness)

            [cv2.line(img, (int(track.centroidarr[i][0]),
                                int(track.centroidarr[i][1])), 
                                (int(track.centroidarr[i+1][0]),
                                int(track.centroidarr[i+1][1])),
                                (25,25,25), thickness=2) 
                                for i,_ in  enumerate(track.centroidarr) 
                                    if i < len(track.centroidarr)-1]
            
        total_left = len([key for key, value in person_info.items() if max(set(value), key=value.count) == 'Left'])
        total_right = len([key for key, value in person_info.items() if max(set(value), key=value.count) == 'Right'])

        img[0:200:2, img.shape[1] - 275:img.shape[1]:2]	= np.multiply(img[0:200:2, img.shape[1] - 275:img.shape[1]:2], 0.25)
        img = cv2.putText(img, f'Left: {total_left}', (img.shape[1] - 260, 75), font, 1.5, (255, 255, 255), thickness)
        img = cv2.putText(img, f'Right: {total_right}', (img.shape[1] - 260, 125), font, 1.5, (255, 255, 255), thickness)

        if not opt.nosave:
            video.write(img)

        if opt.view_img:
            cv2.imshow('frame', cv2.resize(img, (1280, 720)))

            if cv2.waitKey(1) == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                if not opt.nosave:
                    video.release()
                print('\nexiting...')
                exit()

    if opt.view_img:
        cv2.destroyAllWindows()
    cap.release()
    if not opt.nosave:
        video.release()
        print(f'result video saved to {os.path.join(folder, file_name)}')
    print('done.')


