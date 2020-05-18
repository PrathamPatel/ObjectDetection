# Hello, this is a custom object detection program guide to detect custom set of object with Tensorflow Object Detection API.

----
## First things first.
1) Clone or download the Tensorflow Object Detection API from [here](https://github.com/tensorflow/models) and extract it to your project directory.

**Make sure you have these installed as well:**

1. `pip install protobuf`.
2. `pip install pillow`.
3. `pip install lxml`.
4. `pip install Cython`.
5. `pip install jupyter`.
6. `pip install matplotlib`.
7. `pip install pandas`.
8. `pip install opencv-python`.
9. `pip install tensorflow`.

[This](https://www.geeksforgeeks.org/ml-training-image-classifier-using-tensorflow-object-detection-api/) tutorial was really helpful in understanding the process. Yet there were challenges along the way due to different versions, but nothing some extra research can't help with ;)

----
## STEPS I TOOK TO GET IT WORKING:
1. The **object_detection** folder can be found in the **research** directory. Navigate to the **research** directory.
We need to set the environmental path variables:
**run this command and replace _path to folder_ with your path**:

 `set PYTHONPATH=C:\(Path to Project)\models;C:\(Path to Project)\models\research;C:\(Path to Project)\models\research\slim`

2. We need to compile the **Protobuf** files, which are used by TensorFlow to configure the model and the training parameters.

   For Compiling protoc files first we need to get protobuf compiler. I have already included it in the **research** directory.
I have also included a **python script** to compile the **protobuf** files and creates a **name_pb2.py** file from every **name.proto** file in the `object_detection\protos` folder.

 run this command from the **`/research`**directory:
 
  `python use_protobuf.py  .\object_detection\protos\ .\bin\protoc`

3. After that, run the following commands one after the other from the **`\research`** directory:
 
 ` python setup.py build`

 ` python setup.py install`

 With this the installation is finished and a package named **object-detection** is installed. 

4. For testing the Object Detection api, go to **`object_detection`** directory and run the following command:

 `jupyter notebook object_detection_tutorial.ipynb`

5. This opens up the jupyter notebook in the browser.
**Note:** If you have a line `sys.path.append(“..”)` in the first cell of the notebook, remove that line.

 Run all the cells of the notebook and check if you’re getting an output with an image of a dog with bounding boxes around it. If you are, you have successfully configured the API, otherwise, please check if the above steps have been carried out successfully.

## Now to use our own dataset:

1. **Preparing our dataset:** Gather as many different and variety of images consisting of the objects that you want to detect. Create a directory named images inside the **`\research`** directory and store about **80%** of the images into a **`images\train`** directory  and **20%** of the images into **`images\test`** directory. The more images you have, the better the accuracy you shall have.

2. **Labeling Images:** To further our quest, we need a tool called **LabelImg** which can be found [here](https://tzutalin.github.io/labelImg/). Open the application and draw bounding boxes around your images and give them the name you want to train them by. **Remember** to set the default save location to your **`\train`** and **`\test`** directories we made earlier.

 Save each image after labeling which generates a **`xml`** file with the respective image’s name.

3. **Generating records:** We need to generate TFRecords that can be served as input data for training of the object detector. This can be done by 2 helpful scripts (already **included** in this project in the **`\object_detection`** directory) from the **racoon detector** sample which can be found [here](https://github.com/datitran/raccoon_dataset). 

 The scripts we are looking for are the **`xml_to_csv.py`** and **`generatre_tfrecord.py`**.

 We first need to convert our annotated images to CSV format, you can do so by running the following command fromt the **`\object_detection`** directory:

 `python xml_to_csv.py`

 This creates **`test.csv`** and **`train.csv `**files in the **images** folder.

 Next, open the **`generate_tfrecord.py`** file in a text editor and edit the method **`class_text_to_int()`** which can be found arundthe line 30 and edit the **label map** by replacing the `row_label`with your labels.

 Then, generate the **TFRecord** files by issuing these commands one by one from the **`\object_detection`** folder:

    `python generate_tfrecord.py --csv_input=images\train_labels.csv --image_dir=images\train --output_path=train.record`

 `python generate_tfrecord.py --csv_input=images\test_labels.csv --image_dir=images\test --output_path=test.record`

 This creates **`test.record`** and **`train.record`** files in **`object_detection`** directory. 


4. **Training configuration:** we now need to create a label map.
 Create a new directory named **`training`** inside **`object_detection`** directory.

 Use a text editor to create a new file and save it as **`labelmap.pbtxt`** in the **`training`** directory. The label map tells the trainer what each object is by defining a mapping of class names to class ID numbers.

**For example:**

    item {
    id: 1
    name: 'a'   
    }

    item {
    id: 2
    name: 'o'   
    }

Please make sure the label map ID numbers are the same as what was defined in the **`generate_tfrecord.py`** file.

## Next steps for configuration:

5. We need a model i.e, algorithm to train our classifier. In this project we are going to use **`faster_rcnn_inception model`**. Tensorflow’s object detection API comes with a huge number of models. Navigate to **`object_detection\samples\configs`**.
In this location you can find a lot of config files to all the models provided by the API.

2. You can download the **` faster_rcnn_inception_v2_coco`** model [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md). After downloading extract the folder **`faster_rcnn_inception_v2_coco_2018_01_28`** to the **`object_detection`** directory.

 For further understanding of the model, you can take a look at [this](https://medium.com/@smallfishbigsea/faster-r-cnn-explained-864d4fb7e3f8) article.

3. As we are using **`faster_rcnn_inception_v2_coco`** model, copy the **`faster_rcnn_inception_v2_coco.config`** file from **`object_detection\samples\configs`** and paste it in the **training** directory created before.

4. Use a text editor to open the config file and make the following changes:

 **Note: The paths must be entered with single forward slashes (NOT backslashes), or TensorFlow will give a file path error when trying to train the model! Also, the paths must be in double quotation marks ( ” ), not single quotation marks ( ‘ ).**


 **Line 10:** Set the `num_classes` value to the number of objects your classifier is classifying. Example: `num_classes: 2`.

 **Line 107:** Give the absolute path of `model.ckpt` file to the `file_tuning_checkpoint` parameter. **model.ckpt** file is present in the location **`object_detection/faster_rcnn_inception_v2_coco_2018_01_28`**. **_Replace`path to folder` with your path_** 

 **Example:** `fine_tune_checkpoint: “C:/(Path to folder)/models/research/object_detection/faster_rcnn_inception_v2_coco_2018_01_28/model.ckpt”`

 **line 120:** **`train_input_reader section`**. In this section set the **`input_path`** parameter to your **train.record** file. 

 **Example:**
`input_path: “C:/(Path to folder)/models/research/object_detection/train.record”.`

 Set the **`label_map_path`** parameter to the **labelmap.pbtxt** file. 

 **Example**:
`label_map_path: “C:/(Path to folder)/models/research/object_detection/training/labelmap.pbtxt”`


 **line 128:** **`eval config section`** set **`num_examples`** parameter to the number of images present in the **test** directory. 

 **Example:**
num_examples: 10


**line 134:** **`eval_input_reader section`**. Similar to `train_input_reader` section, set the paths to **`test.record`** and **`labelmap.pbtxt`** files. 

**Example: **
**`input_path:`** `“C:/(Path to folder)/models/research/object_detection/test.record”`

**`label_map_path:`** `“C:/(Path to folder)/models/research/object_detection/training/labelmap.pbtxt”`


## Finally, we can train our model:

1. You can find a file named **`train.py`** at the location **`object_detection/legacy/`**.

 Copy the and paste it in the **`object_detection`** directory.

 Navigate to **`object_detection`** directory and run the following command to start training your model with the following command:

 `python train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/faster_rcnn_inception_v2_coco.config`

*It takes around 1min to initialize the setup before the training begins*

Tensorflow creates a checkpoint for every 5 minutes and stores it. You can see all the checkpoints are saved in the **`training`** directory.

Continue the training process until the **loss** is **less than or equal to 0.1**. 

2. This is the last stop before we reach our goal. **YAY!!!**

 Now that we have a trained model we need to generate an **`inference graph`**, which can be used to run the model.

 Navigate to the **`training`** directory and look for the **`model.ckpt`** file with the **biggest** index.

 Then we can create the **inference graph** by running the following command in the command line:

 `python export_inference_graph.py --input_type image_tensor --pipeline_config_path training/faster_rcnn_inception_v2_coco.config --trained_checkpoint_prefix training/model.ckpt-XXXX --output_directory inference_graph`

 Where **`XXXX`** is the highest checkpoint number.

 This creates a **`frozen_inference_graph.pb`** file in the **`\object_detection\inference_graph`** folder. The **.pb** file contains the **object detection classifier**. 


## All that's left to do is test our model out.

In the **`object_detection`** directory there is a file named **`Trial.py`** where you can replace the `IMAGE_NAME` variable with the path to your own image. 

You can run the file from the command line with:
`python Trial.py`

**I HOPE THIS WAS HELPFUL**

----
## THANK YOU :)
