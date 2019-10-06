# Objects as Points Forked

This repo is used for my own study and experiment. Refer to the [original project CenterNet](https://github.com/xingyizhou/CenterNet)

## Setting environment

**NOTE:** I use **pytorch 1.0**

1.  install python env


    ```sh
    conda create --name CenterNet python=3.6
    conda activate CenterNet
    conda install pytorch=1.0 torchvision cudatoolkit=10.0
    pip install -r requirements.txt
    ```

2.  install cocoapi

    ```sh
    git clone https://github.com/cocodataset/cocoapi.git
    cd cocoapi/PythonAPI
    make
    python setup.py install --user
    ```

3.  install DCNv2

    ```sh
    cd src/lib/models/networks/DCNv2
    ./make.sh
    ```

4. install nms

    ```sh
    cd src/lib/external
    make
    ```

5.  download voc dataset

    ```sh
    cd src/tools
    ./get_pascal_voc.sh
    mv voc ../../data
    ```


## train 

1.  training with `resdcn18`

    ```sh
    cd src
    python main.py ctdet --exp_id pascal_resdcn18_384 --arch resdcn_18 --dataset pascal --num_epochs 70 --lr_step 45,60
    ```

2.  training with `mobilenetv2_ctdet` 

    **NOTE:** This model is experimental. The `deconv layer` part of the model refers to `resdcn18_384`, but it does not use `DCNv2`, but ordinary convolution. I want to use this model to validate and migrate to Tensorflow

    ```sh
    python main.py ctdet --exp_id pascal_mobile_384 --arch mobile_1 --dataset pascal --head_conv 256 --num_epochs 70 --lr_step 45,60
    python test.py ctdet --exp_id pascal_mobile_384 --arch mobile_1 --dataset pascal --resume
    ```
    
    Evaluation Result, I don't think Centernet is suitable for small models. :
    ```
    Evaluating detections
    VOC07 metric? Yes
    AP for aeroplane = 0.0000
    AP for bicycle = 0.0000
    AP for bird = 0.0000
    AP for boat = 0.0000
    AP for bottle = 0.0000
    AP for bus = 0.0000
    AP for car = 0.0000
    AP for cat = 0.0000
    AP for chair = 0.0000
    AP for cow = 0.0000
    AP for diningtable = 0.0000
    AP for dog = 0.0000
    AP for horse = 0.0000
    AP for motorbike = 0.0000
    AP for person = 0.0000
    AP for pottedplant = 0.0000
    AP for sheep = 0.0000
    AP for sofa = 0.0000
    AP for train = 0.0000
    AP for tvmonitor = 0.0000
    Mean AP = 0.0000
    ```
    
## inference

1.  inference with `mobilenetv2_ctdet`

    ```sh
    python demo.py ctdet --exp_id pascal_mobile_384 --arch mobile_1 --dataset pascal --head_conv 256 --demo ../images --load_model ../exp/ctdet/pascal_mobile_384/model_best.pth
    ```
    
