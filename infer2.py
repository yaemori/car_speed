import os
import argparse
from tqdm import tqdm
from time import time

import cv2 
import torch
import numpy as np
import onnxruntime as ort
import pytesseract

classes = ['car', 'car_labes', 'moto_labels']
font = cv2.FONT_HERSHEY_SIMPLEX
thickness = 2
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


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

def preprocess(img, new_shape, np_type):
    img, ratio, dwdh = letterbox(img.copy(), new_shape=new_shape, auto=False)
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, 0)
    img = np.ascontiguousarray(img)
    img = img.astype(np_type)
    img /= 255
    return img, ratio, dwdh

    

def plot(img, box, label, color, num_cars):

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

def plot2(img, box, label, color, num_cars):

    x1, y1, x2, y2 = box
    #display
    img = cv2.putText(img, f'Number of Cars : {num_cars}', (50, 50), font, 0.75, [255, 255, 255], thickness)

    return img

def plot_plate(img, box, color, Plate):
    x1, y1, x2, y2 = box
    img = cv2.putText(img, f'Plate : {Plate}', (x1, 2*y2-y1), font, 0.75, [255, 255, 255], thickness)
    return img

def read_plate(img, box, Plate):
    x1, y1, x2, y2 = box
    plate_region = img[y1:y2, x1:x2]
    Plate = pytesseract.image_to_string(plate_region, config='--psm 8 --oem 3')
    return Plate

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=r"D:\paddledeneme\dataguess_proj\best.onnx", help='model.onnx path')
    parser.add_argument('--source', type=str, default=r"D:\paddledeneme\dataguess_proj\TEST.mp4" , help='source video')
    parser.add_argument('--thres', type=float, default=0.75, help='object confidence threshold')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    opt = parser.parse_args()


    if not os.path.exists(opt.weights):
        print('model not found')
        exit()

    if not os.path.exists(opt.source):
        print('source not found')
        exit()    

    order = {}

    provider = ['CUDAExecutionProvider'] if (ort.get_device() == 'GPU' and opt.device == '0') else ['CPUExecutionProvider']
    session = ort.InferenceSession(opt.weights, providers=provider)
    outname = [i.name for i in session.get_outputs()]
    inname = [i.name for i in session.get_inputs()]

    model_type = np.float16 if 'float16' in session.get_inputs()[0].type else np.float32
    model_shape = session.get_inputs()[0].shape[-1]

    cap = cv2.VideoCapture(opt.source)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    w = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if not opt.nosave:
        folder, file_name = os.path.split(opt.source)
        file_name = os.path.splitext(file_name)[0] + '_result.avi'
        video = cv2.VideoWriter(
            os.path.join(folder, file_name), 
            cv2.VideoWriter_fourcc(*'XVID'), 
            30,#fps, 
            (h, w)
        )

        
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

        frame, ratio, dwdh = preprocess(
            img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
            new_shape=model_shape,
            np_type=model_type
        )

        inp = {inname[0]:frame}
        results = session.run(outname, inp)[0]   
        
        for batch_id, x0, y0, x1, y1, cls_id, score in results:
            
            if score < opt.thres:
                continue

            box = np.array([x0, y0, x1, y1])
            box -= np.array(dwdh*2)
            box /= ratio
            box = box.round().astype(int)  

            label = classes[int(cls_id)] 
            color = (0, 0, 255)      
            img = plot(img, box, f'{label} %{int(100*score)}', color, num_cars)

            Plate = read_plate(img, box, Plate)
            
            if(score > opt.thres and label == 'car'):
                num_cars += 1
            if(score > opt.thres and label == 'car_labes'):
                img = plot_plate(img, box, color, Plate)
        img = plot2(img, box, f'{label} %{int(100*score)}', color, num_cars)
        
        

        if not opt.nosave:
            video.write(img)

        if opt.view_img:
            cv2.imshow('frame', cv2.resize(img, (h//2, w//2)))

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


