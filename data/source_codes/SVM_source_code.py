from source_code import SVMSourceCode


# Source codes with iris dataset  - sklearn
SVM_iris_dataset_with_rbf_kernel_auto_gamma = SVMSourceCode.builder().buildDataSet(library="sklearn", dataset_name='iris').buildKernel('rbf').buildC(0.5).buildGamma("auto").build()  

SVM_iris_dataset_with_linear_kernel = SVMSourceCode.builder().buildDataSet(library="sklearn", dataset_name='iris').buildKernel('linear').buildC(0.5).build()    

SVM_iris_dataset_with_poly_kernel_auto_gamma = SVMSourceCode.builder().buildDataSet(library="sklearn", dataset_name='iris').buildKernel('poly').buildC(0.5).buildGamma("auto").buildCoef0("0.1").build()    

SVM_iris_dataset_with_sigmoid_kernel_auto_gamma = SVMSourceCode.builder().buildDataSet(library="sklearn", dataset_name='iris').buildKernel('sigmoid').buildC(0.5).buildGamma("auto").buildCoef0("0.1").build()    

SVM_iris_dataset_with_rbf_kernel_scale_gamma = SVMSourceCode.builder().buildDataSet(library="sklearn", dataset_name='iris').buildKernel('rbf').buildC(0.5).buildGamma("scale").build()  

SVM_iris_dataset_with_poly_kernel_scale_gamma = SVMSourceCode.builder().buildDataSet(library="sklearn", dataset_name='iris').buildKernel('poly').buildC(0.5).buildGamma("scale").buildCoef0("0.1").build()    

SVM_iris_dataset_with_sigmoid_kernel_scale_gamma = SVMSourceCode.builder().buildDataSet(library="sklearn", dataset_name='iris').buildKernel('sigmoid').buildC(0.5).buildGamma("scale").buildCoef0("0.1").build() 


# Source codes with digits dataset - sklearn
SVM_digits_dataset_with_rbf_kernel_auto_gamma = SVMSourceCode.builder().buildDataSet(library="sklearn", dataset_name='digits').buildKernel('rbf').buildC(0.5).buildGamma("auto").build()  

SVM_digits_dataset_with_linear_kernel = SVMSourceCode.builder().buildDataSet(library="sklearn", dataset_name='digits').buildKernel('linear').buildC(0.5).build()    

SVM_digits_dataset_with_poly_kernel_auto_gamma = SVMSourceCode.builder().buildDataSet(library="sklearn", dataset_name='digits').buildKernel('poly').buildC(0.5).buildGamma("auto").buildCoef0("0.1").build()    

SVM_digits_dataset_with_sigmoid_kernel_auto_gamma = SVMSourceCode.builder().buildDataSet(library="sklearn", dataset_name='digits').buildKernel('sigmoid').buildC(0.5).buildGamma("auto").buildCoef0("0.1").build()   

SVM_digits_dataset_with_rbf_kernel_scale_gamma = SVMSourceCode.builder().buildDataSet(library="sklearn", dataset_name='digits').buildKernel('rbf').buildC(0.5).buildGamma("auto").build()  

SVM_digits_dataset_with_poly_kernel_scale_gamma = SVMSourceCode.builder().buildDataSet(library="sklearn", dataset_name='digits').buildKernel('poly').buildC(0.5).buildGamma("scale").buildCoef0("0.1").build()    

SVM_digits_dataset_with_sigmoid_kernel_scale_gamma = SVMSourceCode.builder().buildDataSet(library="sklearn", dataset_name='digits').buildKernel('sigmoid').buildC(0.5).buildGamma("scale").buildCoef0("0.1").build()   


# Source codes with wine dataset - sklearn
SVM_wine_dataset_with_rbf_kernel_auto_gamma = SVMSourceCode.builder().buildDataSet(library="sklearn", dataset_name='wine').buildKernel('rbf').buildC(0.5).buildGamma("auto").build()  

SVM_wine_dataset_with_linear_kernel = SVMSourceCode.builder().buildDataSet(library="sklearn", dataset_name='wine').buildKernel('linear').buildC(0.5).build()    

SVM_wine_dataset_with_poly_kernel_auto_gamma = SVMSourceCode.builder().buildDataSet(library="sklearn", dataset_name='wine').buildKernel('poly').buildC(0.5).buildGamma("auto").buildCoef0("0.1").build()    

SVM_wine_dataset_with_sigmoid_kernel_auto_gamma = SVMSourceCode.builder().buildDataSet(library="sklearn", dataset_name='wine').buildKernel('sigmoid').buildC(0.5).buildGamma("auto").buildCoef0("0.1").build()   

SVM_wine_dataset_with_rbf_kernel_scale_gamma = SVMSourceCode.builder().buildDataSet(library="sklearn", dataset_name='wine').buildKernel('rbf').buildC(0.5).buildGamma("scale").build()  

SVM_wine_dataset_with_poly_kernel_scale_gamma = SVMSourceCode.builder().buildDataSet(library="sklearn", dataset_name='wine').buildKernel('poly').buildC(0.5).buildGamma("scale").buildCoef0("0.1").build()    

SVM_wine_dataset_with_sigmoid_kernel_scale_gamma = SVMSourceCode.builder().buildDataSet(library="sklearn", dataset_name='wine').buildKernel('sigmoid').buildC(0.5).buildGamma("scale").buildCoef0("0.1").build() 


# Source codes with diabetes dataset - sklearn
SVM_diabetes_dataset_with_rbf_kernel_auto_gamma = SVMSourceCode.builder().buildDataSet(library="sklearn", dataset_name='diabetes').buildKernel('rbf').buildC(0.5).buildGamma("auto").build()  

SVM_diabetes_dataset_with_linear_kernel = SVMSourceCode.builder().buildDataSet(library="sklearn", dataset_name='diabetes').buildKernel('linear').buildC(0.5).build()

SVM_diabetes_dataset_with_poly_kernel_auto_gamma = SVMSourceCode.builder().buildDataSet(library="sklearn", dataset_name='diabetes').buildKernel('poly').buildC(0.5).buildGamma("auto").buildCoef0("0.1").build()

SVM_diabetes_dataset_with_sigmoid_kernel_auto_gamma = SVMSourceCode.builder().buildDataSet(library="sklearn", dataset_name='diabetes').buildKernel('sigmoid').buildC(0.5).buildGamma("auto").buildCoef0("0.1").build()

SVM_diabetes_dataset_with_rbf_kernel_scale_gamma = SVMSourceCode.builder().buildDataSet(library="sklearn", dataset_name='diabetes').buildKernel('rbf').buildC(0.5).buildGamma("scale").build()  

SVM_diabetes_dataset_with_poly_kernel_scale_gamma = SVMSourceCode.builder().buildDataSet(library="sklearn", dataset_name='diabetes').buildKernel('poly').buildC(0.5).buildGamma("scale").buildCoef0("0.1").build()

SVM_diabetes_dataset_with_sigmoid_kernel_scale_gamma = SVMSourceCode.builder().buildDataSet(library="sklearn", dataset_name='diabetes').buildKernel('sigmoid').buildC(0.5).buildGamma("scale").buildCoef0("0.1").build()


# Source codes with breast_cancer dataset - sklearn
SVM_breast_cancer_dataset_with_rbf_kernel_auto_gamma = SVMSourceCode.builder().buildDataSet(library="sklearn", dataset_name='breast_cancer').buildKernel('rbf').buildC(0.5).buildGamma("auto").build()  

SVM_breast_cancer_dataset_with_linear_kernel = SVMSourceCode.builder().buildDataSet(library="sklearn", dataset_name='breast_cancer').buildKernel('linear').buildC(0.5).build()

SVM_breast_cancer_dataset_with_poly_kernel_auto_gamma = SVMSourceCode.builder().buildDataSet(library="sklearn", dataset_name='breast_cancer').buildKernel('poly').buildC(0.5).buildGamma("auto").buildCoef0("0.1").build()

SVM_breast_cancer_dataset_with_sigmoid_kernel_auto_gamma = SVMSourceCode.builder().buildDataSet(library="sklearn", dataset_name='breast_cancer').buildKernel('sigmoid').buildC(0.5).buildGamma("auto").buildCoef0("0.1").build()

SVM_breast_cancer_dataset_with_rbf_kernel_scale_gamma = SVMSourceCode.builder().buildDataSet(library="sklearn", dataset_name='breast_cancer').buildKernel('rbf').buildC(0.5).buildGamma("scale").build()  

SVM_breast_cancer_dataset_with_poly_kernel_scale_gamma = SVMSourceCode.builder().buildDataSet(library="sklearn", dataset_name='breast_cancer').buildKernel('poly').buildC(0.5).buildGamma("scale").buildCoef0("0.1").build()

SVM_breast_cancer_dataset_with_sigmoid_kernel_scale_gamma = SVMSourceCode.builder().buildDataSet(library="sklearn", dataset_name='breast_cancer').buildKernel('sigmoid').buildC(0.5).buildGamma("scale").buildCoef0("0.1").build()


# Source codes with svmlight_files dataset - sklearn
SVM_svmlight_files_dataset_with_rbf_kernel_auto_gamma = SVMSourceCode.builder().buildDataSet(library="sklearn", dataset_name='svmlight_files').buildKernel('rbf').buildC(0.5).buildGamma("auto").build()  

SVM_svmlight_files_dataset_with_linear_kernel = SVMSourceCode.builder().buildDataSet(library="sklearn", dataset_name='svmlight_files').buildKernel('linear').buildC(0.5).build()

SVM_svmlight_files_dataset_with_poly_kernel_auto_gamma = SVMSourceCode.builder().buildDataSet(library="sklearn", dataset_name='svmlight_files').buildKernel('poly').buildC(0.5).buildGamma("auto").buildCoef0("0.1").build()

SVM_svmlight_files_dataset_with_sigmoid_kernel_auto_gamma = SVMSourceCode.builder().buildDataSet(library="sklearn", dataset_name='svmlight_files').buildKernel('sigmoid').buildC(0.5).buildGamma("auto").buildCoef0("0.1").build()

SVM_svmlight_files_dataset_with_rbf_kernel_scale_gamma = SVMSourceCode.builder().buildDataSet(library="sklearn", dataset_name='svmlight_files').buildKernel('rbf').buildC(0.5).buildGamma("scale").build()  

SVM_svmlight_files_dataset_with_poly_kernel_scale_gamma = SVMSourceCode.builder().buildDataSet(library="sklearn", dataset_name='svmlight_files').buildKernel('poly').buildC(0.5).buildGamma("scale").buildCoef0("0.1").build()

SVM_svmlight_files_dataset_with_sigmoid_kernel_scale_gamma = SVMSourceCode.builder().buildDataSet(library="sklearn", dataset_name='svmlight_files').buildKernel('sigmoid').buildC(0.5).buildGamma("scale").buildCoef0("0.1").build()


# Source codes with files dataset - sklearn
SVM_files_dataset_with_rbf_kernel_auto_gamma = SVMSourceCode.builder().buildDataSet(library="sklearn", dataset_name='files').buildKernel('rbf').buildC(0.5).buildGamma("auto").build()  

SVM_files_dataset_with_linear_kernel = SVMSourceCode.builder().buildDataSet(library="sklearn", dataset_name='files').buildKernel('linear').buildC(0.5).build()

SVM_files_dataset_with_poly_kernel_auto_gamma = SVMSourceCode.builder().buildDataSet(library="sklearn", dataset_name='files').buildKernel('poly').buildC(0.5).buildGamma("auto").buildCoef0("0.1").build()

SVM_files_dataset_with_sigmoid_kernel_auto_gamma = SVMSourceCode.builder().buildDataSet(library="sklearn", dataset_name='files').buildKernel('sigmoid').buildC(0.5).buildGamma("auto").buildCoef0("0.1").build()

SVM_files_dataset_with_rbf_kernel_scale_gamma = SVMSourceCode.builder().buildDataSet(library="sklearn", dataset_name='files').buildKernel('rbf').buildC(0.5).buildGamma("scale").build()  

SVM_files_dataset_with_poly_kernel_scale_gamma = SVMSourceCode.builder().buildDataSet(library="sklearn", dataset_name='files').buildKernel('poly').buildC(0.5).buildGamma("scale").buildCoef0("0.1").build()

SVM_files_dataset_with_sigmoid_kernel_scale_gamma = SVMSourceCode.builder().buildDataSet(library="sklearn", dataset_name='files').buildKernel('sigmoid').buildC(0.5).buildGamma("scale").buildCoef0("0.1").build()


# Source codes with linnerud dataset - sklearn
SVM_linnerud_dataset_with_rbf_kernel_auto_gamma = SVMSourceCode.builder().buildDataSet(library="sklearn", dataset_name='linnerud').buildKernel('rbf').buildC(0.5).buildGamma("auto").build()  

SVM_linnerud_dataset_with_linear_kernel = SVMSourceCode.builder().buildDataSet(library="sklearn", dataset_name='linnerud').buildKernel('linear').buildC(0.5).build()

SVM_linnerud_dataset_with_poly_kernel_auto_gamma = SVMSourceCode.builder().buildDataSet(library="sklearn", dataset_name='linnerud').buildKernel('poly').buildC(0.5).buildGamma("auto").buildCoef0("0.1").build()

SVM_linnerud_dataset_with_sigmoid_kernel_auto_gamma = SVMSourceCode.builder().buildDataSet(library="sklearn", dataset_name='linnerud').buildKernel('sigmoid').buildC(0.5).buildGamma("auto").buildCoef0("0.1").build()

SVM_linnerud_dataset_with_rbf_kernel_scale_gamma = SVMSourceCode.builder().buildDataSet(library="sklearn", dataset_name='linnerud').buildKernel('rbf').buildC(0.5).buildGamma("scale").build()  

SVM_linnerud_dataset_with_poly_kernel_scale_gamma = SVMSourceCode.builder().buildDataSet(library="sklearn", dataset_name='linnerud').buildKernel('poly').buildC(0.5).buildGamma("scale").buildCoef0("0.1").build()

SVM_linnerud_dataset_with_sigmoid_kernel_scale_gamma = SVMSourceCode.builder().buildDataSet(library="sklearn", dataset_name='linnerud').buildKernel('sigmoid').buildC(0.5).buildGamma("scale").buildCoef0("0.1").build()


# Source codes with sample_images dataset - sklearn
SVM_sample_images_dataset_with_rbf_kernel_auto_gamma = SVMSourceCode.builder().buildDataSet(library="sklearn", dataset_name='sample_images').buildKernel('rbf').buildC(0.5).buildGamma("auto").build()  

SVM_sample_images_dataset_with_linear_kernel = SVMSourceCode.builder().buildDataSet(library="sklearn", dataset_name='sample_images').buildKernel('linear').buildC(0.5).build()

SVM_sample_images_dataset_with_poly_kernel_auto_gamma = SVMSourceCode.builder().buildDataSet(library="sklearn", dataset_name='sample_images').buildKernel('poly').buildC(0.5).buildGamma("auto").buildCoef0("0.1").build()

SVM_sample_images_dataset_with_sigmoid_kernel_auto_gamma = SVMSourceCode.builder().buildDataSet(library="sklearn", dataset_name='sample_images').buildKernel('sigmoid').buildC(0.5).buildGamma("auto").buildCoef0("0.1").build()

SVM_sample_images_dataset_with_rbf_kernel_scale_gamma = SVMSourceCode.builder().buildDataSet(library="sklearn", dataset_name='sample_images').buildKernel('rbf').buildC(0.5).buildGamma("scale").build()  

SVM_sample_images_dataset_with_poly_kernel_scale_gamma = SVMSourceCode.builder().buildDataSet(library="sklearn", dataset_name='sample_images').buildKernel('poly').buildC(0.5).buildGamma("scale").buildCoef0("0.1").build()

SVM_sample_images_dataset_with_sigmoid_kernel_scale_gamma = SVMSourceCode.builder().buildDataSet(library="sklearn", dataset_name='sample_images').buildKernel('sigmoid').buildC(0.5).buildGamma("scale").buildCoef0("0.1").build()


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
