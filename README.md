
# car_detection-speed
Car Detection with YOLOv7 then using paddleOCR to read plates

To use the code, I strongly recommend using miniconda environment and its prompt shell.

What code does it, we have the trained data model with cars and car labels. It gets their x y coordinates and using paddle OCR reads the plate writes it and check the car and car's label's speed in every 3 frames. This is a project we did in our free time in internship company.

You will need to download  requirements.txt in both files then download any missing libraries. 

Problems you might encounter:
-If you encounter with KMP DUPLICATE error, check roadson.py thats the main python file to run.
-If you see something about DLL error. I strongly create new conda environment and download requirements first. Also you will need torch version compitable with your gpu cuda. You will see it while downloading the requirements.

