# Multi-Objective Optimization of Hyperparameters of Convolutional Neural Network and Explainable Model-Based miRNA Selection in Pan-Cancer


## Datasets
The following datasets are derived from the original pan-cancer data. These are as follows:  
- <a href = "https://github.com/joginder12/MOHCNN-STEAAS/blob/main/Breast_subtype.csv">Breast Dataset </a> : This data consists of 4 different subtypes of breast cancer (BASAL, HER2, LUM-A, LUM-B). It contains 792 cancer samples and 25 normal samples.  
- <a href = "https://github.com/joginder12/MOHCNN-STEAAS/blob/main/Lung_Subtype.csv">Lung Dataset </a> : This data consists of 2 different subtypes of lung cancer (LUAD, LUSC). It consists of 996 cancer samples and 91 normal samples.  
- <a href = "https://github.com/joginder12/MOHCNN-STEAAS/blob/main/Kidney_subtype.csv">Kidney Dataset </a> : This data consists of 3 different subtypes of kidney cancer (KICH, KIRC, KIRP). It contains 879 cancer samples and 130 normal samples.  
- <a href = "https://github.com/joginder12/MOHCNN-STEAAS/blob/main/CPCNS.csv">Classified Pan-cancer and normal samples (CPCNS) Dataset </a> : This classified pan-cancer data consists of 11019 samples derived from 33 different types of cancer and one (670) normal class.
- <a href = "https://drive.google.com/file/d/1wBcCt_yIXQxxC3Cah_u-5jMOO71pk_r2/view?usp=sharing">Classified Pan-cancer samples (CPCS) Dataset </a> : This classified pan-cancer data consists of 10349 samples derived from 33 different types of cancer.
- <a href = "https://drive.google.com/file/d/1yE-UjqNkCaEI7DxVFH7kznULTWW7Fn2M/view?usp=sharing">Class-wise Pan-cancer and Class-wise Normal samples (CPCCNS) Dataset </a> : This data consists of 653 samples from 16 normal classes and 10349 from 33 different types of cancer classes.
- <a href = "https://drive.google.com/file/d/1x9n_-XXkAw120uY1i0DsE6DZK-u8ERUp/view?usp=sharing">Classified Normal samples (CNS) Dataset </a> : This data consists of 653 samples from 16 normal classes.

## Steps to Run MOHCNN-STEAAS
1. Open Python and install the packages **numpy**, **math**, **csv**, **pandas**, **sklearn**, **matplotlib**, **time**, **scipy**, **tensorflow**, .  
(Use command `pip install package_name` e.g., `pip install pandas`.  
In higher versions of Python, use `pip3` in place of `pip`.)  
* In Windows environment, if **Spyder** is used for Python, then one has to install the **pip** package first using the command  
  `"python get-pip.py"`  

2. Download the code for **MOHCNN** from <a href = "https://github.com/joginder12/MOHCNN-STEAAS/blob/main/MOHCNN.py"> `MOHCNN.py` </a>

3. Download the code for **STEAAS** from <a href = "https://github.com/joginder12/MOHCNN-STEAAS/blob/main/MOHCNN_STEAAS.py">`STEAAS.py` </a>    

4. Keep the code and the datasets in the same folder, otherwise change the folder path along with the name of the dataset in the code (**Line number 25**).  

5. Run `MOHCNN.py` to produce `STEAAS_Result.csv` and `STEAAS_performance.csv`.  
   - `STEAAS_Result.csv` contains the **miRNA names**.  
