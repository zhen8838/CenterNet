mkdir voc
cd voc
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCdevkit_18-May-2011.tar
tar xvf VOCtrainval_06-Nov-2007.tar
sync
tar xvf VOCtest_06-Nov-2007.tar
sync
tar xvf VOCdevkit_08-Jun-2007.tar
sync
tar xvf VOCtrainval_11-May-2012.tar
sync
tar xvf VOCdevkit_18-May-2011.tar
sync
mkdir images
cp VOCdevkit/VOC2007/JPEGImages/* images/
sync
cp VOCdevkit/VOC2012/JPEGImages/* images/
sync
wget https://storage.googleapis.com/coco-dataset/external/PASCAL_VOC.zip
unzip PASCAL_VOC.zip
sync
mv PASCAL_VOC annotations/
sync
cd ..
python merge_pascal_json.py
