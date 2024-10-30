from src.data.source_codes.source_code_builder import SVMSourceCode


####################################################################
#  Structured source code generated for sk-learn datasets          #
####################################################################
# Source codes with iris dataset  - sklearn
SVM_iris_dataset_with_rbf_kernel_auto_gamma = SVMSourceCode.builder().buildDataSet(library="sklearn", dataset_name='iris').buildKernel('rbf').buildC(0.5).buildGamma("auto").build()  

SVM_iris_dataset_with_linear_kernel = SVMSourceCode.builder().buildDataSet(library="sklearn", dataset_name='iris').buildKernel('linear').buildC(0.5).build()    

SVM_iris_dataset_with_poly_kernel_auto_gamma = SVMSourceCode.builder().buildDataSet(library="sklearn", dataset_name='iris').buildKernel('poly').buildC(0.5).buildGamma("auto").buildCoef0("0.1").build()    

SVM_iris_dataset_with_sigmoid_kernel_auto_gamma = SVMSourceCode.builder().buildDataSet(library="sklearn", dataset_name='iris').buildKernel('sigmoid').buildC(0.5).buildGamma("auto").buildCoef0("0.1").build()    

SVM_iris_dataset_with_rbf_kernel_scale_gamma = SVMSourceCode.builder().buildDataSet(library="sklearn", dataset_name='iris').buildKernel('rbf').buildC(0.5).buildGamma("scale").build()  

SVM_iris_dataset_with_poly_kernel_scale_gamma = SVMSourceCode.builder().buildDataSet(library="sklearn", dataset_name='iris').buildKernel('poly').buildC(0.5).buildGamma("scale").buildCoef0("0.1").build()    

SVM_iris_dataset_with_sigmoid_kernel_scale_gamma = SVMSourceCode.builder().buildDataSet(library="sklearn", dataset_name='iris').buildKernel('sigmoid').buildC(0.5).buildGamma("scale").buildCoef0("0.1").build() 


# Source codes with digits dataset - sklearn
SVM_digits_dataset_with_linear_kernel = SVMSourceCode.builder().buildDataSet(library="sklearn", dataset_name='digits').buildKernel('linear').buildC(0.5).build()    

# Source codes with wine dataset - sklearn
SVM_wine_dataset_with_rbf_kernel_auto_gamma = SVMSourceCode.builder().buildDataSet(library="sklearn", dataset_name='wine').buildKernel('rbf').buildC(0.5).buildGamma("auto").build()  

# Source codes with diabetes dataset - sklearn
SVM_diabetes_dataset_with_rbf_kernel_auto_gamma = SVMSourceCode.builder().buildDataSet(library="sklearn", dataset_name='diabetes').buildKernel('rbf').buildC(0.5).buildGamma("auto").build()  

# Source codes with breast_cancer dataset - sklearn
SVM_breast_cancer_dataset_with_rbf_kernel_auto_gamma = SVMSourceCode.builder().buildDataSet(library="sklearn", dataset_name='breast_cancer').buildKernel('rbf').buildC(0.5).buildGamma("auto").build()  

# Source codes with svmlight_files dataset - sklearn
SVM_svmlight_files_dataset_with_rbf_kernel_auto_gamma = SVMSourceCode.builder().buildDataSet(library="sklearn", dataset_name='svmlight_files').buildKernel('rbf').buildC(0.5).buildGamma("auto").build()  

# Source codes with files dataset - sklearn
SVM_files_dataset_with_rbf_kernel_auto_gamma = SVMSourceCode.builder().buildDataSet(library="sklearn", dataset_name='files').buildKernel('rbf').buildC(0.5).buildGamma("auto").build()  

# Source codes with linnerud dataset - sklearn
SVM_linnerud_dataset_with_rbf_kernel_auto_gamma = SVMSourceCode.builder().buildDataSet(library="sklearn", dataset_name='linnerud').buildKernel('rbf').buildC(0.5).buildGamma("auto").build()  

# Source codes with sample_images dataset - sklearn
SVM_sample_images_dataset_with_rbf_kernel_auto_gamma = SVMSourceCode.builder().buildDataSet(library="sklearn", dataset_name='sample_images').buildKernel('rbf').buildC(0.5).buildGamma("auto").build()  


####################################################################
#  Structured source code generated for sk-learn datasets          #
####################################################################

# Source codes titanic dataset - openml
SVM_titanic_dataset = SVMSourceCode.builder().buildDataSet(library="openml", dataset_id=40945).buildKernel('rbf').buildC(0.5).buildGamma("auto").build()  

# Source codes MNIST dataset - openml
SVM_MNIST_dataset = SVMSourceCode.builder().buildDataSet(library="openml", dataset_id=554).buildKernel('rbf').buildC(0.5).buildGamma("auto").build()  

# Source codes Fashion-MNIST dataset - openml
SVM_Fashion_MNIST_dataset = SVMSourceCode.builder().buildDataSet(library="openml", dataset_id=40996).buildKernel('rbf').buildC(0.5).buildGamma("auto").build()  

# Source codes Adult-Income dataset - openml
SVM_Fashion_Adult_Income_dataset = SVMSourceCode.builder().buildDataSet(library="openml", dataset_id=1590).buildKernel('rbf').buildC(0.5).buildGamma("auto").build()  

# Source code Covertype dataset - openml
SVM_Covertype_dataset = SVMSourceCode.builder().buildDataSet(library="openml", dataset_id=180).buildKernel('rbf').buildC(0.5).buildGamma("auto").build()  

# Source code California Housing dataset - openml
SVM_California_Housing_dataset = SVMSourceCode.builder().buildDataSet(library="openml", dataset_id=537).buildKernel('rbf').buildC(0.5).buildGamma("auto").build()  

# Source code Bike Sharing dataset - openml
SVM_Bike_Sharing_dataset = SVMSourceCode.builder().buildDataSet(library="openml", dataset_id=42803).buildKernel('rbf').buildC(0.5).buildGamma("auto").build()  

# Source code Diabetes dataset - openml
SVM_Diabetes_dataset = SVMSourceCode.builder().buildDataSet(library="openml", dataset_id=37).buildKernel('rbf').buildC(0.5).buildGamma("auto").build()  

# Source code Ames House Prices dataset - openml
SVM_Ames_House_Prices_dataset = SVMSourceCode.builder().buildDataSet(library="openml", dataset_id=42165).buildKernel('rbf').buildC(0.5).buildGamma("auto").build()  

# Source code Pen-Based Handwritten Digits dataset - openml
SVM_Pen_Based_Handwritten_Digits_dataset = SVMSourceCode.builder().buildDataSet(library="openml", dataset_id=1508).buildKernel('rbf').buildC(0.5).buildGamma("auto").build()  

# Source code Wholesale Customers dataset - openml
SVM_Wholesale_Customers_dataset = SVMSourceCode.builder().buildDataSet(library="openml", dataset_id=41187).buildKernel('rbf').buildC(0.5).buildGamma("auto").build()  

# Source code Bank Marketing dataset - openml
SVM_Bank_Marketing_dataset = SVMSourceCode.builder().buildDataSet(library="openml", dataset_id=1461).buildKernel('rbf').buildC(0.5).buildGamma("auto").build()  

# Source code Electricity Load Diagrams dataset - openml
SVM_Electricity_Load_Diagrams_dataset = SVMSourceCode.builder().buildDataSet(library="openml", dataset_id=151).buildKernel('rbf').buildC(0.5).buildGamma("auto").build()  

# Source code Airline Delays dataset - openml
SVM_Airline_Delays_dataset = SVMSourceCode.builder().buildDataSet(library="openml", dataset_id=40984).buildKernel('rbf').buildC(0.5).buildGamma("auto").build()  

# Source code Credit Card Fraud dataset - openml
SVM_Credit_Card_Fraud_dataset = SVMSourceCode.builder().buildDataSet(library="openml", dataset_id=1597).buildKernel('rbf').buildC(0.5).buildGamma("auto").build()  

# Source code KDD Cup 1999 dataset - openml
SVM_KDD_Cup_1999_dataset = SVMSourceCode.builder().buildDataSet(library="openml", dataset_id=1113).buildKernel('rbf').buildC(0.5).buildGamma("auto").build()  

# Source code SMS Spam Collection dataset - openml
SVM_SMS_Spam_Collection_dataset = SVMSourceCode.builder().buildDataSet(library="openml", dataset_id=42195).buildKernel('rbf').buildC(0.5).buildGamma("auto").build()  

# Source code Magic Gamma Telescope dataset - openml
SVM_Magic_Gamma_Telescope_dataset = SVMSourceCode.builder().buildDataSet(library="openml", dataset_id=1120).buildKernel('rbf').buildC(0.5).buildGamma("auto").build()  

# Source code Letter Recognition dataset - openml
SVM_Letter_Recognition_dataset = SVMSourceCode.builder().buildDataSet(library="openml", dataset_id=6).buildKernel('rbf').buildC(0.5).buildGamma("auto").build()  

# Source code Phoneme dataset - openml
SVM_Phoneme_dataset = SVMSourceCode.builder().buildDataSet(library="openml", dataset_id=1489).buildKernel('rbf').buildC(0.5).buildGamma("auto").build()  

# Source code Connect-4 dataset - openml
SVM_Connect_4_dataset = SVMSourceCode.builder().buildDataSet(library="openml", dataset_id=40685).buildKernel('rbf').buildC(0.5).buildGamma("auto").build()  

# Source code Spambase dataset - openml
SVM_Spambase_dataset = SVMSourceCode.builder().buildDataSet(library="openml", dataset_id=44).buildKernel('rbf').buildC(0.5).buildGamma("auto").build()  

# Source code CIFAR-10 dataset - openml
SVM_CIFAR_10_dataset = SVMSourceCode.builder().buildDataSet(library="openml", dataset_id=40927).buildKernel('rbf').buildC(0.5).buildGamma("auto").build()  

# Source code QSAR Biodegradation dataset - openml
SVM_QSAR_Biodegradation_dataset = SVMSourceCode.builder().buildDataSet(library="openml", dataset_id=1494).buildKernel('rbf').buildC(0.5).buildGamma("auto").build()  

# Source code Yeast dataset - openml
SVM_Yeast_dataset = SVMSourceCode.builder().buildDataSet(library="openml", dataset_id=181).buildKernel('rbf').buildC(0.5).buildGamma("auto").build()  

# Source code Mice Protein Expression dataset - openml
SVM_Mice_Protein_Expression_dataset = SVMSourceCode.builder().buildDataSet(library="openml", dataset_id=40966).buildKernel('rbf').buildC(0.5).buildGamma("auto").build()  

# Source code Occupancy Detection dataset - openml
SVM_Occupancy_Detection_dataset = SVMSourceCode.builder().buildDataSet(library="openml", dataset_id=1590).buildKernel('rbf').buildC(0.5).buildGamma("auto").build()  

# Source code Heart Disease dataset - openml
SVM_Heart_Disease_dataset = SVMSourceCode.builder().buildDataSet(library="openml", dataset_id=53).buildKernel('rbf').buildC(0.5).buildGamma("auto").build()  

# Source code Wine Quality dataset - openml
SVM_Wine_Quality_dataset = SVMSourceCode.builder().buildDataSet(library="openml", dataset_id=187).buildKernel('rbf').buildC(0.5).buildGamma("auto").build()  

# Source code Optical Recognition of Handwritten Digits dataset - openml
SVM_Optical_Recognition_dataset = SVMSourceCode.builder().buildDataSet(library="openml", dataset_id=28).buildKernel('rbf').buildC(0.5).buildGamma("auto").build()  

# Source code Weather Prediction dataset - openml
SVM_Weather_Prediction_dataset = SVMSourceCode.builder().buildDataSet(library="openml", dataset_id=1515).buildKernel('rbf').buildC(0.5).buildGamma("auto").build()  

# Source code Steel Plates Fault dataset - openml
SVM_Steel_Plates_Fault_dataset = SVMSourceCode.builder().buildDataSet(library="openml", dataset_id=40982).buildKernel('rbf').buildC(0.5).buildGamma("auto").build()  

# Source code Gesture Phase Segmentation dataset - openml
SVM_Gesture_Phase_Segmentation_dataset = SVMSourceCode.builder().buildDataSet(library="openml", dataset_id=4534).buildKernel('rbf').buildC(0.5).buildGamma("auto").build()  

# Source code Banknote Authentication dataset - openml
SVM_Banknote_Authentication_dataset = SVMSourceCode.builder().buildDataSet(library="openml", dataset_id=1462).buildKernel('rbf').buildC(0.5).buildGamma("auto").build()  

# Source code Wilt dataset - openml
SVM_Wilt_dataset = SVMSourceCode.builder().buildDataSet(library="openml", dataset_id=469).buildKernel('rbf').buildC(0.5).buildGamma("auto").build()  

# Source code Cars Evaluation dataset - openml
SVM_Cars_Evaluation_dataset = SVMSourceCode.builder().buildDataSet(library="openml", dataset_id=21).buildKernel('rbf').buildC(0.5).buildGamma("auto").build()  

# Source code Poker Hand dataset - openml
SVM_Poker_Hand_dataset = SVMSourceCode.builder().buildDataSet(library="openml", dataset_id=354).buildKernel('rbf').buildC(0.5).buildGamma("auto").build()  

# Source code Indian Liver Patient dataset - openml
SVM_Indian_Liver_Patient_dataset = SVMSourceCode.builder().buildDataSet(library="openml", dataset_id=1480).buildKernel('rbf').buildC(0.5).buildGamma("auto").build()  

# Source code Forest Fires dataset - openml
SVM_Forest_Fires_dataset = SVMSourceCode.builder().buildDataSet(library="openml", dataset_id=1124).buildKernel('rbf').buildC(0.5).buildGamma("auto").build()  

# Source code Molecular Biology (Promoters) dataset - openml
SVM_Molecular_Biology_Promoters_dataset = SVMSourceCode.builder().buildDataSet(library="openml", dataset_id=1063).buildKernel('rbf').buildC(0.5).buildGamma("auto").build()  

# Source code Thyroid Disease dataset - openml
SVM_Thyroid_Disease_dataset = SVMSourceCode.builder().buildDataSet(library="openml", dataset_id=40474).buildKernel('rbf').buildC(0.5).buildGamma("auto").build()  

# Source code EEG Eye State dataset - openml
SVM_EEG_Eye_State_dataset = SVMSourceCode.builder().buildDataSet(library="openml", dataset_id=1471).buildKernel('rbf').buildC(0.5).buildGamma("auto").build()  

# Source code Page Blocks Classification dataset - openml
SVM_Page_Blocks_dataset = SVMSourceCode.builder().buildDataSet(library="openml", dataset_id=30).buildKernel('rbf').buildC(0.5).buildGamma("auto").build()  

# Source code Wilt (Binary Class) dataset - openml
SVM_Wilt_Binary_dataset = SVMSourceCode.builder().buildDataSet(library="openml", dataset_id=469).buildKernel('rbf').buildC(0.5).buildGamma("auto").build()  

# Source code Gas Sensor Array Drift dataset - openml
SVM_Gas_Sensor_Array_Drift_dataset = SVMSourceCode.builder().buildDataSet(library="openml", dataset_id=1478).buildKernel('rbf').buildC(0.5).buildGamma("auto").build()  

# Source code Blood Transfusion Service Center dataset - openml
SVM_Blood_Transfusion_dataset = SVMSourceCode.builder().buildDataSet(library="openml", dataset_id=1464).buildKernel('rbf').buildC(0.5).buildGamma("auto").build()  

# Source code Pen-Based Recognition of Handwritten Digits (Binary) dataset - openml
SVM_Pen_Based_Binary_dataset = SVMSourceCode.builder().buildDataSet(library="openml", dataset_id=1485).buildKernel('rbf').buildC(0.5).buildGamma("auto").build()  

# Source code Breast Cancer Wisconsin (Diagnostic) dataset - openml
SVM_Breast_Cancer_Wisconsin_dataset = SVMSourceCode.builder().buildDataSet(library="openml", dataset_id=15).buildKernel('rbf').buildC(0.5).buildGamma("auto").build()  

# Source code Seismic-Bumps dataset - openml
SVM_Seismic_Bumps_dataset = SVMSourceCode.builder().buildDataSet(library="openml", dataset_id=1512).buildKernel('rbf').buildC(0.5).buildGamma("auto").build()  

# Source code Ecoli dataset - openml
SVM_Ecoli_dataset = SVMSourceCode.builder().buildDataSet(library="openml", dataset_id=617).buildKernel('rbf').buildC(0.5).buildGamma("auto").build()  

# Source code Internet Advertisements dataset - openml
SVM_Internet_Ads_dataset = SVMSourceCode.builder().buildDataSet(library="openml", dataset_id=3).buildKernel('rbf').buildC(0.5).buildGamma("auto").build()  

# Source code Waveform Database Generator (v2) dataset - openml
SVM_Waveform_v2_dataset = SVMSourceCode.builder().buildDataSet(library="openml", dataset_id=60).buildKernel('rbf').buildC(0.5).buildGamma("auto").build()  

# Source code Magic Gamma Telescope (Imbalanced) dataset - openml
SVM_Magic_Gamma_Imbalanced_dataset = SVMSourceCode.builder().buildDataSet(library="openml", dataset_id=1460).buildKernel('rbf').buildC(0.5).buildGamma("auto").build()  

# Source code Chess (King-Rook vs. King-Pawn) dataset - openml
SVM_Chess_dataset = SVMSourceCode.builder().buildDataSet(library="openml", dataset_id=3).buildKernel('rbf').buildC(0.5).buildGamma("auto").build()  

# Source code Lymphography dataset - openml
SVM_Lymphography_dataset = SVMSourceCode.builder().buildDataSet(library="openml", dataset_id=1480).buildKernel('rbf').buildC(0.5).buildGamma("auto").build()  

# Source code Heart Disease (Cleveland) dataset - openml
SVM_Heart_Disease_Cleveland_dataset = SVMSourceCode.builder().buildDataSet(library="openml", dataset_id=901).buildKernel('rbf').buildC(0.5).buildGamma("auto").build()  

# Source code Dermatology dataset - openml
SVM_Dermatology_dataset = SVMSourceCode.builder().buildDataSet(library="openml", dataset_id=35).buildKernel('rbf').buildC(0.5).buildGamma("auto").build()  
