import cv2
import numpy as np
from paddleocr import PaddleOCR, draw_ocr
from PIL import Image, ImageDraw, ImageFont

# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en')  # Set use_angle_cls=True if you want to detect text orientation

# Define input and output video paths
input_video_path = 'input_video.mp4'
output_video_path = 'output_video.mp4'

# Open the video file
cap = cv2.VideoCapture(input_video_path)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert frame to PIL Image format
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    # Perform OCR on the image
    result = ocr.ocr(np.array(image), cls=True)
    
    # Extract boxes, texts, and scores from the OCR result
    boxes = [line[0] for line in result[0]]
    txts = [line[1][0] for line in result[0]]
    scores = [line[1][1] for line in result[0]]
    
    # Draw OCR results on the image
    im_show = draw_ocr(image, boxes, txts, scores, font_path='path_to_font.ttf')
    im_show = np.array(im_show)
    
    # Convert image back to BGR format for OpenCV
    im_show_bgr = cv2.cvtColor(im_show, cv2.COLOR_RGB2BGR)
    
    # Write the frame to the output video
    out.write(im_show_bgr)

# Release video objects
cap.release()
out.release()
cv2.destroyAllWindows()

print("Video processing complete. The output video is saved as", output_video_path)