# YOLOV5-Custom-data-Fashion-
This repo contain the jupyter notebook to train the custom model using YOLOV5 and also detect the objects based on YOLOV5 Flask application

Data set Used:

https://www.kaggle.com/datasets/nguyngiabol/colorful-fashion-dataset-for-object-detection

Steps:

1. Mount the google drive
from google.colab import drive
drive.mount('/content/drive')

2.Install the requirements
!git clone https://github.com/ultralytics/yolov5  # clone repo
%cd yolov5
%pip install -qr requirements.txt # install dependencies

3. Data set preparation
- 1.Select the small data set e.g 
https://www.kaggle.com/datasets/nguyngiabol/colorful-fashion-dataset-for-object-detection

- 2.Manually split the Train and Test data and save the images e.g fashion/images/train and fashion/images/test folder.

- 3.copy the corresponding annotation.txt files into 
     lables/train and label/test folder
- 4.Dataset does not contain the .txt annotation file.Refer above link for how to convert xml to Yolo5 formated (txt)? and also Train and Test split.

    https://blog.paperspace.com/train-yolov5-custom-data/

- 5.Create the data.ymal file under fashion folder as below

![image](https://user-images.githubusercontent.com/46878296/185730278-3be8898d-853d-42bd-a9ce-31fb9a09b78f.png)


3. Train the custom data
 Navigate to  Yolov5 folder as below
 %cd /content/drive/MyDrive/Ineuron_DeepLearning/Object_detection/yolov5
 
# First time Training command
!python train.py --img 640 \
--batch-size 32 \
--epochs 50 \
--data /content/drive/MyDrive/Ineuron_DeepLearning/Object_detection/Yolood/fashion/data.yaml \
--cfg /content/drive/MyDrive/Ineuron_DeepLearning/Object_detection/yolov5/models/yolov5s.yaml\
 --name /content/drive/MyDrive/Ineuron_DeepLearning/Object_detection/Yolood/fashion\Model
 
 # Resume
 !python train.py --resume
 
4. Performance of the model checking
 %load_ext tensorboard
#
%tensorboard --logdir /content/drive/MyDrive/Ineuron_DeepLearning/Object_detection/yolov5/runs/train/exp

# Inference model

5. Inference model using below command

!python detect.py --conf 0.5 \
--weights /content/drive/MyDrive/Ineuron_DeepLearning/Object_detection/yolov5/runs/train/fashion_model5/weights/best.pt \
--source /content/drive/MyDrive/Ineuron_DeepLearning/Object_detection/Yolood/fashion/Sample_Images

6. Display the result stored into Results saved to runs/detect/exp

#display inference on ALL test images

import glob
from IPython.display import Image, display

for imageName in glob.glob('/content/drive/MyDrive/Ineuron_DeepLearning/Object_detection/yolov5/runs/detect/exp/*.jpg'): #assuming JPG
    display(Image(filename=imageName))
    print("\n")
    
   ![image](https://user-images.githubusercontent.com/46878296/185730669-c8c1c948-477d-434d-a73d-f23da05cbab7.png)


# Flask API:

1. Create the virtual envoriment or conda envoriment
2. Activate the envoriment
3. clone the YOLOV5 & install the requirements

!git clone https://github.com/ultralytics/yolov5  # clone repo
%cd yolov5
%pip install -qr requirements.txt # install dependencie

4. copy the custom weight into yolov5\include\predictor_yolo_detector
5. Downlaod the attached Zip file & Extract
6. Open the root directory in visual code or python IDE
7. Run the flask application as  below


$ python clientApp.py
 * Serving Flask app 'clientApp'
 * Debug mode: off
WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
 * Running on http://127.0.0.1:9501
Press CTRL+C to quit
127.0.0.1 - - [20/Aug/2022 11:09:01] "GET / HTTP/1.1" 200 -
127.0.0.1 - - [20/Aug/2022 11:09:08] "GET /favicon.ico HTTP/1.1" 404 -
YOLOv5 🚀 2022-8-13 Python-3.8.0 torch-1.12.1+cpu CPU

Open the above URL http://127.0.0.1:9501 in browser

8. Input the required image

yolov5\include\predictor_yolo_detector\inference\images


9. Result:

Result will store into yolov5\include\predictor_yolo_detector\inference\output

![image](https://user-images.githubusercontent.com/46878296/185730785-b19fbed6-5914-44ac-9fd6-18428bc824a7.png)

![image](https://user-images.githubusercontent.com/46878296/185730826-605979a6-3893-488e-92b1-6c746efa55c9.png)



# Youtube video inference:

python detect_yolov5.py  --source https://www.youtube.com/watch?v=yPQfcG-eimk --weight ./include/predictor_yolo_detector/best.pt

![image](https://user-images.githubusercontent.com/46878296/185731769-2837925a-e244-41f0-8a2a-a7a28f79430c.png)


Video result will store into yolov5\runs\detect\exp

![image](https://user-images.githubusercontent.com/46878296/185731974-fa2e456b-5152-4b72-8306-e221f9574735.png)


# Reference:
- https://github.com/ultralytics/yolov5/issues/6998
- https://github.com/ultralytics/yolov5
- https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data







 
 
 
