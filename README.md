# YOLOV5-Custom-data- Object Dtecttion : Fashion Dataset
This repo contain the jupyter notebook to train the custom model using YOLOV5 and also detect the objects based on YOLOV5 Flask application

Data URL:

https://www.kaggle.com/datasets/nguyngiabol/colorful-fashion-dataset-for-object-detection

Steps:

1. Mount the google drive

```
 from google.colab import drive
 drive.mount('/content/drive')

```
2.Install the requirements


```
  !git clone https://github.com/ultralytics/yolov5  # clone repo
  %cd yolov5
  %pip install -qr requirements.txt # install dependencies
```
3. Data set preparation

- 1.Select the small data set e.g   https://www.kaggle.com/datasets/nguyngiabol/colorful-fashion-dataset-for-object-detection & download

- 2 Extarct the data  and manually split the Train and Test data 
 
- 3 Save the images e.g fashion/images/train and fashion/images/test folder inside the google drive

- 4 Copy the corresponding annotation.txt files into  lables/train and label/test folder
- 5.Dataset does not contain the .txt annotation file.Refer above link for how to convert xml to Yolo5 formated (txt)? and also Train and Test split.

    - https://blog.paperspace.com/train-yolov5-custom-data/


- 6.Create the data.ymal file under fashion folder as below

![image](https://user-images.githubusercontent.com/46878296/185730278-3be8898d-853d-42bd-a9ce-31fb9a09b78f.png)


3. Train the custom data 
    Navigate to  Yolov5 folder as below
-  %cd /content/drive/MyDrive/Ineuron_DeepLearning/Object_detection/yolov5
 
# First time Training command


```
!python train.py --img 640 \
--batch-size 32 \
--epochs 50 \
--data /content/drive/MyDrive/Ineuron_DeepLearning/Object_detection/Yolood/fashion/data.yaml \
--cfg /content/drive/MyDrive/Ineuron_DeepLearning/Object_detection/yolov5/models/yolov5s.yaml\
 --name /content/drive/MyDrive/Ineuron_DeepLearning/Object_detection/Yolood/fashion\Model

```
 
 # Training  Resume
 
 ```
  !python train.py --resume
 ```
4. Performance of the model checking

```
  %load_ext tensorboard
    %tensorboard --logdir /content/drive/MyDrive/Ineuron_DeepLearning/Object_detection/yolov5/runs/train/exp
```
# Inference model

5. Inference model using below command

```
!python detect.py --conf 0.5 \
--weights /content/drive/MyDrive/Ineuron_DeepLearning/Object_detection/yolov5/runs/train/fashion_model5/weights/best.pt \
--source /content/drive/MyDrive/Ineuron_DeepLearning/Object_detection/Yolood/fashion/Sample_Images

```
6. Display the result stored into Results saved to runs/detect/exp

#display inference on ALL test images


```
import glob
from IPython.display import Image, display

for imageName in glob.glob('/content/drive/MyDrive/Ineuron_DeepLearning/Object_detection/yolov5/runs/detect/exp/*.jpg'): #assuming JPG
    display(Image(filename=imageName))
    print("\n")
```


   ![image](https://user-images.githubusercontent.com/46878296/185730669-c8c1c948-477d-434d-a73d-f23da05cbab7.png)


# Flask API:

1. Create the virtual envoriment or conda envoriment
2. Activate the envoriment
3. clone the YOLOV5 & install the requirements

```
!git clone https://github.com/ultralytics/yolov5  # clone repo
%cd yolov5
%pip install -qr requirements.txt # install dependencie

```

4. copy the custom weight into yolov5\include\predictor_yolo_detector & data.yaml into ./data folder
5. Downlaod the attached Zip file & Extract
6. Open the root directory in visual code or python IDE
7. Run the flask application as  below

![image](https://user-images.githubusercontent.com/46878296/185967430-7b5b4234-5dbf-47da-ae35-aa0aab861e16.png)



Open the above URL http://127.0.0.1:9501 in browser

8. Input the required image

 from sample folder  or yolov5\include\predictor_yolo_detector\inference\images


9. Result:

Result will store into yolov5\include\predictor_yolo_detector\inference\output

![image](https://user-images.githubusercontent.com/46878296/185730785-b19fbed6-5914-44ac-9fd6-18428bc824a7.png)

![image](https://user-images.githubusercontent.com/46878296/185730826-605979a6-3893-488e-92b1-6c746efa55c9.png)



# Youtube video inference:

```
python detect_yolov5.py  --source https://www.youtube.com/watch?v=yPQfcG-eimk --weight ./include/predictor_yolo_detector/best.pt

```
![image](https://user-images.githubusercontent.com/46878296/185731769-2837925a-e244-41f0-8a2a-a7a28f79430c.png)


Video result will store into yolov5\runs\detect\exp

![image](https://user-images.githubusercontent.com/46878296/185731974-fa2e456b-5152-4b72-8306-e221f9574735.png)


# Reference or Original repos:
- https://github.com/ultralytics/yolov5/issues/6998
- https://github.com/ultralytics/yolov5
- https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data







 
 
 
