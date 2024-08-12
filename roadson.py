import os
import argparse
from tqdm import tqdm
from time import time
import cv2
import numpy as np
import onnxruntime as ort
from paddleocr import PaddleOCR
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.path as mplPath
import math
from sort import *



#I strongly recommend using miniconda environment
#
#before running the code, you should enable ->  set KMP_DUPLICATE_LIB_OK=TRUE


# Initialize SORT
sort_max_age = 1
sort_min_hits = 3
sort_iou_thresh = 0.3
sort_tracker = Sort(max_age=sort_max_age, min_hits=sort_min_hits, iou_threshold=sort_iou_thresh)

classes = ['car', 'car_labels', 'moto_labels']
font = cv2.FONT_HERSHEY_SIMPLEX
thickness = 2
roi_color = (0, 0, 255)
ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False)
ratio_123 = 0.05
roi = np.array([
    [0, 500],
    [1400, 500],
    [1400, 1400],
    [0, 1400]
])
mpl_roi = mplPath.Path(roi)

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
    shape = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (dw, dh)

def euclidean_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def preprocess(img, new_shape, np_type):
    img, ratio, dwdh = letterbox(img.copy(), new_shape=new_shape, auto=False)
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, 0)
    img = np.ascontiguousarray(img)
    img = img.astype(np_type)
    img /= 255
    return img, ratio, dwdh

def plot(img, box, label, color, num_cars):
    global previous_centers
    center = box[:2] + ((box[2:] - box[:2]) / 2)
    img = cv2.circle(img, center=center.astype(np.int32), radius=3, color=color, thickness=thickness * 2)
    x1, y1, x2, y2 = box
    r, d = 10, 10
    img = cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    img = cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    img = cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
    img = cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    img = cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    img = cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
    img = cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    img = cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    img = cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
    img = cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    img = cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    img = cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)
    (w, h), _ = cv2.getTextSize(label, font, 0.6, thickness)
    img = cv2.rectangle(img, (x1, y1 - 20), (x1 + w + 20, y1), color, -1)
    img = cv2.putText(img, label, (x1, y1 - 2), font, 0.75, [225, 255, 255], thickness)
    return img

def plot2(img, box, label, color, num_cars):
    x1, y1, x2, y2 = box
    img = cv2.putText(img, f'Number of Cars : {num_cars}', (50, 50), font, 0.75, [255, 255, 255], thickness)
    return img

def plot_plate(img, box, color, Plate):
    x1, y1, x2, y2 = box
    img = cv2.putText(img, f'Plate : {Plate}', (x1, 2*y2-y1), font, 0.75, [255, 255, 255], thickness)
    return img

def read_plate(img, box):
    x1, y1, x2, y2 = box
    plate_region = img[y1:y2, x1:x2]
    plate_img = Image.fromarray(cv2.cvtColor(plate_region, cv2.COLOR_BGR2RGB))
    result = ocr.ocr(np.array(plate_img), cls=True)
    print(f'OCR result: {result}')
    if result is None:
        print("OCR result is None")
        return ''
    try:
        Plate = ''.join([word[1][0] for line in result for word in line])
    except (TypeError, IndexError) as e:
        print(f"Error processing OCR result: {e}")
        return ''
    return Plate

def plot_center(img, box, color):
    center = box[:2] + ((box[2:] - box[:2]) / 2)
    center = center.astype(np.int32)
    img = cv2.circle(img, center=tuple(center), radius=3, color=color, thickness=thickness * 2)
    return img

def calculate_speed(track, fps, ratio_123):
    num_centroids = len(track.centroidarr)
    if num_centroids >= 3:
        dist = 0
        for i in range(1, num_centroids):
            dist += euclidean_distance(track.centroidarr[i - 1], track.centroidarr[i])
        dist /= num_centroids - 1
        dist *= ratio_123
        speed = dist * fps * 3.6  # convert to km/h
    else:
        speed = 0
    return speed

if __name__ == '__main__':
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=r'D:\paddledeneme\dataguess_proj\best.onnx', help='model.onnx path')
    parser.add_argument('--source', type=str, default="D:\paddledeneme\dataguess_proj\TEST.mp4", help='source video')
    parser.add_argument('--thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    opt = parser.parse_args()

    # Check file existence
    if not os.path.exists(opt.weights):
        print('model not found')
        exit()
    if not os.path.exists(opt.source):
        print('source not found')
        exit()

    # Initialize ONNX runtime
    provider = ['CUDAExecutionProvider'] if (ort.get_device() == 'GPU' and opt.device == '0') else ['CPUExecutionProvider']
    session = ort.InferenceSession(opt.weights, providers=provider)
    outname = [i.name for i in session.get_outputs()]
    inname = [i.name for i in session.get_inputs()]
    model_type = np.float16 if 'float16' in session.get_inputs()[0].type else np.float32
    model_shape = session.get_inputs()[0].shape[-1]

    # Initialize video capture
    cap = cv2.VideoCapture(opt.source)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    w = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initialize video writer if saving results
    if not opt.nosave:
        folder, file_name = os.path.split(opt.source)
        file_name = os.path.splitext(file_name)[0] + '_result.avi'
        video = cv2.VideoWriter(
            os.path.join(folder, file_name), 
            cv2.VideoWriter_fourcc(*'XVID'), 
            30, 
            (h, w)
        )

    # Process frames
    # Process frames
for i in tqdm(range(num_frames)):
    num_cars = 0
    Plate = ''
    ret, img = cap.read()

    if not ret:
        cap.release()
        if not opt.nosave:
            video.release()
        print('\ncant read video, exiting...')
        exit()

    img = cv2.polylines(img, [roi], True, roi_color, thickness)
    frame, ratio, dwdh = preprocess(
        img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
        new_shape=model_shape,
        np_type=model_type
    )

    inp = {inname[0]: frame}
    results = session.run(outname, inp)[0]
    dets_to_sort = np.empty((0, 6))

    # Process detections
    for batch_id, x0, y0, x1, y1, cls_id, score in results:
        if score < opt.thres:
            continue

        box = np.array([x0, y0, x1, y1])
        box -= np.array(dwdh * 2)
        box /= ratio
        box = box.round().astype(int)

        if not mpl_roi.contains_point((box[0], box[1])) or not mpl_roi.contains_point((box[2], box[3])):
            continue

        label = classes[int(cls_id)]
        color = (0, 0, 255)
        img = plot(img, box, f'{label} %{int(100 * score)}', color, num_cars)

        center = (int(box[0] + (box[2] - box[0]) / 2), int(box[1] + (box[3] - box[1]) / 2))
        if not mpl_roi.contains_point(center):
            continue

        dets_to_sort = np.vstack((dets_to_sort, np.hstack((box, cls_id, score))))

        if label == 'car_labels':
            Plate = read_plate(img, box)
        if score > opt.thres and label == 'car':
            num_cars += 1
        if score > opt.thres and label == 'car_labels':
            img = plot_plate(img, box, color, Plate)

    # Update and draw tracked objects
    tracked_dets = sort_tracker.update(dets_to_sort)
    tracks = sort_tracker.getTrackers()

    for track in tracks:
        track_id = track.id + 1
        label = f'ID {track_id}'

        if len(track.centroidarr) >= 2:
            speed = calculate_speed(track, fps, ratio_123)
            label += f' Speed: {speed:.2f} km/h'

        # Draw track
        for i, _ in enumerate(track.centroidarr):
            if i < len(track.centroidarr) - 1:
                img = cv2.line(img,
                               (int(track.centroidarr[i][0]), int(track.centroidarr[i][1])),
                               (int(track.centroidarr[i + 1][0]), int(track.centroidarr[i + 1][1])),
                               (25, 25, 25), thickness=2)

        if track.centroidarr:
            center = (int(track.centroidarr[-1][0]), int(track.centroidarr[-1][1]))
            img = cv2.putText(img, label, center, font, 0.75, [0, 0, 255], thickness)

    img = plot2(img, box, f'Number of Cars : {num_cars}', color, num_cars)

    # Save or display results
    if not opt.nosave:
        video.write(img)

    if opt.view_img:
        cv2.imshow('frame', cv2.resize(img, (h // 2, w // 2)))
        if cv2.waitKey(1) == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            if not opt.nosave:
                video.release()
            print('\nexiting...')
            exit()

# Cleanup
if opt.view_img:
    cv2.destroyAllWindows()
cap.release()
if not opt.nosave:
    video.release()
    print(f'result video saved to {os.path.join(folder, file_name)}')
print('done.')

