# ML_RoboVision
# Advanced Object Tracking
##Install:
###For Windows 64bit
 1. Install tensorflow version 2.4.0 and tensor flow models library
    > git clone https://github.com/tensorflow/models/
2. Verify install _python -c "import tensorflow as tf;print(tf.reduce_sum(tf.random.normal([1000, 1000])))"_
3. GPU Support - Install CUDA and CuDNN
    1. If _cusolver64_10.dll_ cannot be found, change the dll name from 11 to 10
    2. Add to path
    > <INSTALL_PATH>\NVIDIA GPU Computing Toolkit\CUDA\v10.1\bin
   > 
    ><INSTALL_PATH>\NVIDIA GPU Computing Toolkit\CUDA\v10.1\libnvvp
   > 
    ><INSTALL_PATH>\NVIDIA GPU Computing Toolkit\CUDA\v10.1\extras\CUPTI\libx64
   > 
    ><INSTALL_PATH>\NVIDIA GPU Computing Toolkit\CUDA\v10.1\cuda\bin
   3. Run _python -c "import tensorflow as tf;print(tf.reduce_sum(tf.random.normal([1000, 1000])))"_ to verify that changes were successful
4. Install protobuf
    1. _protoc-3.xx.x-win64.zip_
    2. Move protc.exe under research folder and cd to research
    3. > for /f %i in ('dir /b object_detection\protos\*.proto') do protoc object_detection\protos\%i --python_out=.
5. Check compete install with 
    1. >python object_detection/builders/model_builder_tf2_test.py
   2. Returns 20 tests completes with 1 skipped

##Generate Inference
1. Get images in different lighting need about __100__ per item
2. Separate images into folders test and train. 90% of images in train and rest in test
3. Using LabelImg to draw bounding box and create .xml for each img
    1. __w__ for bounding box and __d__ for next image
    2. Turn __auto save__ on under view
    
4. Partition the data using script
   >python partition_dataset.py -x -i [PATH_TO_IMAGES_FOLDER] -r 0.1
5. Create a label_map.pbtxt under annotations with an item tag for each label
    1. `item {
    id: 1
    name: 'cat'
}`
6. Convert .xml to TFRecords for tensorflow support using script
    > python generate_tfrecord.py [-h] [-x XML_DIR] [-l LABELS_PATH] [-o OUTPUT_PATH] [-i IMAGE_DIR] [-c CSV_PATH]
7. Add skipped steps
8. Start a new training job
    1. > python model_main_tf2.py --model_dir=models/my_ssd_resnet50_v1_fpn --pipeline_config_path=models/my_ssd_resnet50_v1_fpn/pipeline.config2.
    2. This take _a long time_ depending on the hardware
    3. Target under 1 total loss
9. Export model
    1. >python .\exporter_main_v2.py --input_type image_tensor --pipeline_config_path .\models\my_ssd_resnet50_v1_fpn\pipeline.config --trained_checkpoint_dir .\models\my_ssd_resnet50_v1_fpn\ --output_directory .\exported-models\my_model
       
Helps with visuals -> https://www.youtube.com/watch?v=usR2LQuxhL4&t=1521s

From google and the best overall -> https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html

##Object Tracking with Inference
1. Open webcam feed
2. Get each frame from the feed
3. Using a tensorflow model draw bounding box around target
4. Label bounding box and supply additional information
###Object Distance
* Triangle Similarity formula
    * _Focal_ =( _Apparent Pixels_ x _Distance from Camera_ ) / _Known width of object_
    * Should be set globaly or set manualy
* Use bounding box to fill in variables and calculate distance

### Vocab
TFRecords - TensorFlow specific binary data storage. More effective than other files but data needs to be converted into this type
