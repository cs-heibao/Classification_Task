# Classification_Task
# Multi-Neural Networks
The project was implemented aims to realize three types rock thin section images classification. And the method was optimized based on GoogLeNet、VGG16、MobileNetV2、ShuffleNetV2.
# Contents
- [1.Environment](#environment)

- [2.Preparation](#preparation)

- [3.Training](#training)

- [4.Demo](#demo)
# Environment
    - Driver: NVIDIA-Linux-x86_64-470.63.01.run
    - CUDA: cuda_11.4.2_470.57.02_linux.run
    - Torch 1.7.0+
    - python 3.8
# Preparation
1. Prepare the dataset, we spyder the data from [ScienceDB.](http://dx.doi.org/10.11922/sciencedb.j00001.00097)
2. Prepare the training dataset, the data structure as follow.
        
        $Data/                           # RootPath  
        
        $Data/sub-class folder                 # include images.   
# Training
1. Train your model on PASCAL VOC Format.
                
        cd $Classification_Task
        python3 train.py
                
2. Train results, it will create '.pth' model, loss log file and evaluation log file.
                
        # It will create model definition files and save snapshot models in:
        #   - $Classification_Task/weights/'{}_{}.pth'
        # the loss log and evaluation log saved in:
        #   - $Classification_Task/log/'{}'/loss.txt'.format(lstrftime('%b%d-%H'))
        #   - $RDNet/log/'{}'/result.txt'.format(lstrftime('%b%d-%H'))

        
        
