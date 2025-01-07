# Attention-based Multimodal Deep Learning for Interpretable and Generalizable Prediction of Pathological Complete Response in Breast Cancer


This is the official documentation of the code produced for the paper: "Attention-based Multimodal Deep Learning for Interpretable and Generalizable Prediction of Pathological Complete Response in Breast Cancer"


## Dataset 
The I-SPY 1 and I-SPY 2 data can be downloaded from The Cancer Imaging Archive's official website: 

- I-SPY 1: https://www.cancerimagingarchive.net/collection/ispy1/  
David Newitt, Nola Hylton, on behalf of the I-SPY 1 Network and ACRIN 6657 Trial Team. (2016). Multi-center breast DCE-MRI data and segmentations from patients in the I-SPY 1/ACRIN 6657 trials. The Cancer Imaging Archive. https://doi.org/10.7937/K9/TCIA.2016.HdHpgJLK

- I-SPY 2: https://www.cancerimagingarchive.net/collection/ispy2/  
Li, W., Newitt, D. C., Gibbs, J., Wilmes, L. J., Jones, E. F., Arasu, V. A., Strand, F., Onishi, N., Nguyen, A. A.-T., Kornak, J., Joe, B. N., Price, E. R., Ojeda-Fournier, H., Eghtedari, M., Zamora, K. W., Woodard, S. A., Umphrey, H., Bernreuter, W., Nelson, M., … Hylton, N. M. (2022). I-SPY 2 Breast Dynamic Contrast Enhanced MRI Trial (ISPY2)  (Version 1) [Data set]. The Cancer Imaging Archive. https://doi.org/10.7937/TCIA.D8Z0-9T85

## Preprocessing
Preprocessing of the images followed the same format for both dataset. First, we identified the "uni-lateral original cropped DCE-MRI" folder, from which the first post-contrast image slices were extracted. We use the SER/PE2/PE6 folder sizes for dynamically referencing the slice indices of the folder. Next, the DCE-MRI were loaded using SimpleITK so that their spacing could be normalized to [0.7, 0.7, 2.0] for the depth, height, and width dimensions, respectively. The images are ultimately saved as npy files, and the images are cropped to a set size of [80, 80, 256] and randomly flipped during the creation of the data_loader. 

Preprocessing of the clinical features was done by loading the appropriate excel file using the Pandas library. The clinical feature values of age, race, HR, HER2, and menopausal status were available for the I-SPY 2 dataset, but I-SPY 1 did not have menopausal status but had additional information of ER, PR, and Ki-67. For training and testing of the model, age, race, HR, HER2, and menopausal status were used, where the menopausal status was set to N/A for I-SPY 1 external testing.

## Training and testing the model 
The main.py file can be used for training and testing the model.
