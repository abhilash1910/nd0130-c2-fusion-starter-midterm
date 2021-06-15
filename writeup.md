# Self-Driving Car Beta Testing Nanodegree 

This is a template submission for the midterm second course in the  [Udacity Self-Driving Car Engineer Nanodegree Program](https://www.udacity.com/course/c-plus-plus-nanodegree--nd213) : 3D Object Detection (Midterm). 


## 3D Object detection

We have used the [Waymo Open Dataset's](https://console.cloud.google.com/storage/browser/waymo_open_dataset_v_1_2_0_individual_files) real-world data and used 3d point cloud for lidar based object detection. 

- Configuring the ranges channel to 8 bit and view the range /intensity image (ID_S1_EX1)
- Use the Open3D library to display the lidar point cloud on a 3d viewer and identifying 10 images from point cloud.(ID_S1_EX2)
- Create Birds Eye View perspective (BEV) of the point cloud,assign lidar intensity values to BEV,normalize the heightmap of each BEV (ID_S2_EX1,ID_S2_EX2,ID_S2_EX3)
- In addition to YOLO, use the [repository](https://review.udacity.com/github.com/maudzung/SFA3D) and add parameters ,instantiate fpn resnet model(ID_S3_EX1)
- Convert BEV coordinates into pixel coordinates and convert model output to bounding box format  (ID_S3_EX2)
- Compute intersection over union, assign detected objects to label if IOU exceeds threshold (ID_S4_EX1)
- Compute false positives and false negatives, precision and recall(ID_S4_EX2,ID_S4_EX3)


The project can be run by running 

```
python loop_over_dataset.py
```
All training/inference is done on GTX 2060 in windows 10 machine.


## Step-1: Compute Lidar point cloud from Range Image

In this we are first previewing the range image and convert range and intensity channels to 8 bit format. After that, we use the openCV library to stack the range and intensity channel vertically to visualize the image.

- Convert "range" channel to 8 bit
- Convert "intensity" channel to 8 bit
- Crop range image to +/- 90 degrees  left and right of forward facing x axis
- Stack up range and intensity channels vertically in openCV

The changes are made in 'loop_over_dataset.py'
![img1](img/id_s1e1.png)

![img1](img/id_s1e12.png)

The changes are made in "objdet_pcl.py"

![img1](img/id_s1e13.png)

The range image sample:

![img1](img/range_img.png)

![img1](img/range_img2.png)



For the next part, we use the Open3D library to display the lidar point cloud on a 3D viewer and identify 10 images from point cloud
- Visualize the point cloud in Open3D
- 10 examples from point cloud  with varying degrees of visibility

The changes are made in 'loop_over_dataset.py'
![img1](img/id_s1e2.png)

The changes are made in "objdet_pcl.py"
![img1](img/id_s1e21.png)


Point cloud images

![img1](img/pc1.png)

![img1](img/pc2.png)

![img1](img/pc3.png)

![img1](img/pc4.png)

![img1](img/pc5.png)

![img1](img/pc6.png)

![img1](img/pc7.png)

![img1](img/pc8.png)

![img1](img/pc9.png)

![img1](img/pc10.png)


Stable features include the tail lights, the rear bumper  majorly. In some cases the additional features include the headover lights, car front lights, rear window shields. These are identified through the intensity channels . The chassis of the car is the most prominent identifiable feature from the lidar perspective. The images are analysed with different settings and the rear lights are the major stable components, also the bounding boxes are correctly assigned to the cars (used from Step-3).


## Step-2: Creaate BEV from Lidar PCL

In this case, we are:
- Converting the coordinates to pixel values
- Assigning lidar intensity values to the birds eye view BEV mapping
- Using sorted and pruned point cloud lidar from the  previous task
- Normalizing the height map in the BEV
- Compute and map the intensity values

The changes are in the 'loop_over_dataset.py'

![img1](img/id_s2e1.png)

The changes are also in the "objdet_pcl.py"

![img1](img/id_s2e12.png)


A sample preview of the BEV:

![img1](img/bev.png)

![img1](img/bev2.png)

A preview of the intensity layer:

The 'lidar_pcl_top' is used in this case, shown in the Figure:

![img1](img/id_s2e2.png)

The corresponding intensity channel:

![img1](img/intensity_layer.png)

The corresponding normalized height channel:


![img1](img/height_channel.png)


## Step-3: Model Based Object Detection in BEV Image

Here we are using the cloned [repo](https://github.com/maudzung/SFA3D) ,particularly the test.py file  and extracting the relevant configurations from 'parse_test_configs()'  and added them in the 'load_configs_model' config structure.

- Instantiating the fpn resnet model from the cloned repository configs
- Extracting 3d bounding boxes from the responses
- Transforming the pixel to vehicle coordinates
- Model output tuned to the bounding box format [class-id, x, y, z, h, w, l, yaw]

The changes are in "loop_over_dataset.py"

![img1](img/id_s3e1.png)

The changes for the detection are inside the "objdet_detect.py" file:

![img1](img/id_s3e12.png)

As the model input is a three-channel BEV map, the detected objects will be returned with coordinates and properties in the BEV coordinate space. Thus, before the detections can move along in the processing pipeline, they need to be converted into metric coordinates in vehicle space.

A sample preview of the bounding box images:

![img1](img/3d_bb.png)

![img1](img/obj_detect.png)


## Step-4: Performance detection for 3D Object Detection

In this step, the performance is computed by getting the IOU  between labels and detections to get the false positive and false negative values.The task is to compute the geometric overlap between the bounding boxes of labels and the detected objects:

- Assigning a detected object to a label if IOU exceeds threshold
- Computing the degree of geometric overlap
- For multiple matches objects/detections pair with maximum IOU are kept
- Computing the false negative and false positive values
- Computing precision and recall over the false positive and false negative values

The changes in the code are:

![img1](img/id_s4e1.png)

The changes for "objdet_eval.py" where the precision and recall are calculated as functions of false positives and negatives:

![img1](img/id_s4e12.png)


![img1](img/id_s4e11.png)


The precision recall curve is plotted showing similar results of precision =0.996 and recall=0.81372

![img1](img/recall.png)

In the next step, we set the 
```python
configs_det.use_labels_as_objects=True
```
 which results in precision and recall values as 1.This is shown in the following image:


![img1](img/pre-recall_2.png)


## Summary of Lidar based 3D Object Detection

From the project, it is understandable that for a stabilized tracking, lidar should be used . The conversion of range data to point cloud through spatial volumes, or points (or CNN networks) are important for further analysis. The usage of resnet/darknet and YOLO to convert these high dimensional point cloud representations to object detections through bounding boxes is essential for 3D object detection. Evaluating the performance with help of maximal IOU mapping ,mAP, and representing the precision/recall of the bounding boxes are essential to understand the effectiveness of Lidar based detection.
