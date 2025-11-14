# Set-theoretic Explainable AI-based Attribution Score for Selecting miRNAs in Pan-cancer Data (MOHCNN-STEAAS)


## Datasets
The following datasets are derived from the original pan-cancer data. These are as follows:  
- <a href = "https://drive.google.com/file/d/1JEuFi3w0DlIRmUft3fjdHh8asn8hIu8J/view">Breast Dataset </a> : This data consists of 4 different subtypes of breast cancer (BASAL, HER2, LUM-A, LUM-B). It contains 792 cancer samples and 25 normal samples.  
- <a href = "https://drive.google.com/file/d/1DMWWqt4dqb4Ixq57QAyMiVdhdI1zJ1Wx/view">Lung Dataset </a> : This data consists of 2 different subtypes of lung cancer (LUAD, LUSC). It consists of 996 cancer samples and 91 normal samples.  
- <a href = "https://drive.google.com/file/d/1CR6wCbfdfqR3Dg7oZDtAy8kJPxD09RG4/view">Kidney Dataset </a> : This data consists of 3 different subtypes of kidney cancer (KICH, KIRC, KIRP). It contains 879 cancer samples and 130 normal samples.  
- <a href = "https://drive.google.com/file/d/1JaNfq2m87z1KtuFZrNWRvS-AbkmXK_49/view">Classified Pan-cancer (CPS) Dataset </a> : This classified pan-cancer data consists of 10349 samples derived from 33 different types of cancer.

## Steps to Run ICNNM
1. Open Python and install the packages **numpy**, **math**, **csv**, **pandas**, **sklearn**, **matplotlib**, **time**, **scipy**, **tensorflow**, .  
(Use command `pip install package_name` e.g., `pip install pandas`.  
In higher versions of Python, use `pip3` in place of `pip`.)  
* In Windows environment, if **Spyder** is used for Python, then one has to install the **pip** package first using the command  
  `"python get-pip.py"`  

2. Download the code for **MOHCNN** from: <a href = "https://drive.google.com/file/d/1d7UlNHs6cq7xx-Gf5wCY70f3xR6PwYhv/view?usp=drive_link">`MOHCNN.py` </a>

3. Download the code for **STEAAS** from: <a href = "https://drive.google.com/file/d/1Lj_acADTTKgjee9F45XMHz1I3UeNQruF/view?usp=drive_link">`STEAAS.py` </a>    

4. Keep the code and the datasets in the same folder, otherwise change the folder path along with the name of the dataset in the code (**Line number 25**).  

5. Run `MOHCNN.py` to produce `STEAAS_Result.csv` and `STEAAS_performance.csv`.  
   - `STEAAS_Result.csv` contains the **miRNA names**.  
