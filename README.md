# EVA4S6
To run the project you have to keep all the files into tha same directory and then just run the epoch.py file and it will run all the other modules which will prduce the train and test accuracy with the following combination of the regularization techniques:

Train and Test accuracy with Batch Normalisation.

Train and Test accuracy with with Batch Normalisation and L1 regularization.

Train and Test accuracy with L1 regularization.

Train and Test accuracy with L2 regularization.

Train and Test accuracy with Batch Normalization and L2 regularization.

Train and Test accuracy with no Batch Normalization no L1 regularization and no L2 regularization.

Train and Test accuracy with Batch Normalisation and L1 regularization and L2 regularization.

You need to insatll the following dependencies:

Pytorch

Tqdm

Matplotlib

Torchsummary

TorchVision

File Name and what feature they contaion.

epoch.py - To run all the epoch.

l1_l2_reg = This contains l1 and l2 regularization

l1_reg = This contains l1 regularization 

l2_ reg = This contains l2 regularization 

load_data = This contains data loader and data transformation file 

model_arc_noBN = THis contains the model in which I have removed the batch normaliation

model arc = This contains the model in which batch normalisation is there 

