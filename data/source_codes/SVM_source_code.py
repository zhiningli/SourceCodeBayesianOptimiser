from source_code import SVMSourceCode


# Source codes with iris dataset
SVM_iris_dataset_with_rbf_kernel_auto_gamma = SVMSourceCode.builder().buildDataSet('iris').buildKernel('rbf').buildC(0.5).buildGamma("auto").build()  

SVM_iris_dataset_with_linear_kernel = SVMSourceCode.builder().buildDataSet('iris').buildKernel('linear').buildC(0.5).build()    

SVM_iris_dataset_with_poly_kernel_auto_gamma = SVMSourceCode.builder().buildDataSet('iris').buildKernel('poly').buildC(0.5).buildGamma("auto").buildCoef0("0.1").build()    

SVM_iris_dataset_with_sigmoid_kernel_auto_gamma = SVMSourceCode.builder().buildDataSet('iris').buildKernel('sigmoid').buildC(0.5).buildGamma("auto").buildCoef0("0.1").build()    

SVM_iris_dataset_with_rbf_kernel_scale_gamma = SVMSourceCode.builder().buildDataSet('iris').buildKernel('rbf').buildC(0.5).buildGamma("scale").build()  

SVM_iris_dataset_with_poly_kernel_scale_gamma = SVMSourceCode.builder().buildDataSet('iris').buildKernel('poly').buildC(0.5).buildGamma("scale").buildCoef0("0.1").build()    

SVM_iris_dataset_with_sigmoid_kernel_scale_gamma = SVMSourceCode.builder().buildDataSet('iris').buildKernel('sigmoid').buildC(0.5).buildGamma("scale").buildCoef0("0.1").build() 


# Source codes with digits dataset
SVM_digits_dataset_with_rbf_kernel_auto_gamma = SVMSourceCode.builder().buildDataSet('digits').buildKernel('rbf').buildC(0.5).buildGamma("auto").build()  

SVM_digits_dataset_with_linear_kernel = SVMSourceCode.builder().buildDataSet('digits').buildKernel('linear').buildC(0.5).build()    

SVM_digits_dataset_with_poly_kernel_auto_gamma = SVMSourceCode.builder().buildDataSet('digits').buildKernel('poly').buildC(0.5).buildGamma("auto").buildCoef0("0.1").build()    

SVM_digits_dataset_with_sigmoid_kernel_auto_gamma = SVMSourceCode.builder().buildDataSet('digits').buildKernel('sigmoid').buildC(0.5).buildGamma("auto").buildCoef0("0.1").build()   

SVM_digits_dataset_with_rbf_kernel_scale_gamma = SVMSourceCode.builder().buildDataSet('digits').buildKernel('rbf').buildC(0.5).buildGamma("auto").build()  

SVM_digits_dataset_with_poly_kernel_scale_gamma = SVMSourceCode.builder().buildDataSet('digits').buildKernel('poly').buildC(0.5).buildGamma("scale").buildCoef0("0.1").build()    

SVM_digits_dataset_with_sigmoid_kernel_scale_gamma = SVMSourceCode.builder().buildDataSet('digits').buildKernel('sigmoid').buildC(0.5).buildGamma("scale").buildCoef0("0.1").build()   


# Source codes with wine dataset
SVM_wine_dataset_with_rbf_kernel_auto_gamma = SVMSourceCode.builder().buildDataSet('wine').buildKernel('rbf').buildC(0.5).buildGamma("auto").build()  

SVM_wine_dataset_with_linear_kernel = SVMSourceCode.builder().buildDataSet('wine').buildKernel('linear').buildC(0.5).build()    

SVM_wine_dataset_with_poly_kernel_auto_gamma = SVMSourceCode.builder().buildDataSet('wine').buildKernel('poly').buildC(0.5).buildGamma("auto").buildCoef0("0.1").build()    

SVM_wine_dataset_with_sigmoid_kernel_auto_gamma = SVMSourceCode.builder().buildDataSet('wine').buildKernel('sigmoid').buildC(0.5).buildGamma("auto").buildCoef0("0.1").build()   

SVM_wine_dataset_with_rbf_kernel_scale_gamma = SVMSourceCode.builder().buildDataSet('wine').buildKernel('rbf').buildC(0.5).buildGamma("scale").build()  

SVM_wine_dataset_with_poly_kernel_scale_gamma = SVMSourceCode.builder().buildDataSet('wine').buildKernel('poly').buildC(0.5).buildGamma("scale").buildCoef0("0.1").build()    

SVM_wine_dataset_with_sigmoid_kernel_scale_gamma = SVMSourceCode.builder().buildDataSet('wine').buildKernel('sigmoid').buildC(0.5).buildGamma("scale").buildCoef0("0.1").build() 


# Source codes with diabetes dataset
SVM_diabetes_dataset_with_rbf_kernel_auto_gamma = SVMSourceCode.builder().buildDataSet('diabetes').buildKernel('rbf').buildC(0.5).buildGamma("auto").build()  

SVM_diabetes_dataset_with_linear_kernel = SVMSourceCode.builder().buildDataSet('diabetes').buildKernel('linear').buildC(0.5).build()

SVM_diabetes_dataset_with_poly_kernel_auto_gamma = SVMSourceCode.builder().buildDataSet('diabetes').buildKernel('poly').buildC(0.5).buildGamma("auto").buildCoef0("0.1").build()

SVM_diabetes_dataset_with_sigmoid_kernel_auto_gamma = SVMSourceCode.builder().buildDataSet('diabetes').buildKernel('sigmoid').buildC(0.5).buildGamma("auto").buildCoef0("0.1").build()

SVM_diabetes_dataset_with_rbf_kernel_scale_gamma = SVMSourceCode.builder().buildDataSet('diabetes').buildKernel('rbf').buildC(0.5).buildGamma("scale").build()  

SVM_diabetes_dataset_with_poly_kernel_scale_gamma = SVMSourceCode.builder().buildDataSet('diabetes').buildKernel('poly').buildC(0.5).buildGamma("scale").buildCoef0("0.1").build()

SVM_diabetes_dataset_with_sigmoid_kernel_scale_gamma = SVMSourceCode.builder().buildDataSet('diabetes').buildKernel('sigmoid').buildC(0.5).buildGamma("scale").buildCoef0("0.1").build()


# Source codes with breast_cancer dataset
SVM_breast_cancer_dataset_with_rbf_kernel_auto_gamma = SVMSourceCode.builder().buildDataSet('breast_cancer').buildKernel('rbf').buildC(0.5).buildGamma("auto").build()  

SVM_breast_cancer_dataset_with_linear_kernel = SVMSourceCode.builder().buildDataSet('breast_cancer').buildKernel('linear').buildC(0.5).build()

SVM_breast_cancer_dataset_with_poly_kernel_auto_gamma = SVMSourceCode.builder().buildDataSet('breast_cancer').buildKernel('poly').buildC(0.5).buildGamma("auto").buildCoef0("0.1").build()

SVM_breast_cancer_dataset_with_sigmoid_kernel_auto_gamma = SVMSourceCode.builder().buildDataSet('breast_cancer').buildKernel('sigmoid').buildC(0.5).buildGamma("auto").buildCoef0("0.1").build()

SVM_breast_cancer_dataset_with_rbf_kernel_scale_gamma = SVMSourceCode.builder().buildDataSet('breast_cancer').buildKernel('rbf').buildC(0.5).buildGamma("scale").build()  

SVM_breast_cancer_dataset_with_poly_kernel_scale_gamma = SVMSourceCode.builder().buildDataSet('breast_cancer').buildKernel('poly').buildC(0.5).buildGamma("scale").buildCoef0("0.1").build()

SVM_breast_cancer_dataset_with_sigmoid_kernel_scale_gamma = SVMSourceCode.builder().buildDataSet('breast_cancer').buildKernel('sigmoid').buildC(0.5).buildGamma("scale").buildCoef0("0.1").build()


# Source codes with svmlight_files dataset
SVM_svmlight_files_dataset_with_rbf_kernel_auto_gamma = SVMSourceCode.builder().buildDataSet('svmlight_files').buildKernel('rbf').buildC(0.5).buildGamma("auto").build()  

SVM_svmlight_files_dataset_with_linear_kernel = SVMSourceCode.builder().buildDataSet('svmlight_files').buildKernel('linear').buildC(0.5).build()

SVM_svmlight_files_dataset_with_poly_kernel_auto_gamma = SVMSourceCode.builder().buildDataSet('svmlight_files').buildKernel('poly').buildC(0.5).buildGamma("auto").buildCoef0("0.1").build()

SVM_svmlight_files_dataset_with_sigmoid_kernel_auto_gamma = SVMSourceCode.builder().buildDataSet('svmlight_files').buildKernel('sigmoid').buildC(0.5).buildGamma("auto").buildCoef0("0.1").build()

SVM_svmlight_files_dataset_with_rbf_kernel_scale_gamma = SVMSourceCode.builder().buildDataSet('svmlight_files').buildKernel('rbf').buildC(0.5).buildGamma("scale").build()  

SVM_svmlight_files_dataset_with_poly_kernel_scale_gamma = SVMSourceCode.builder().buildDataSet('svmlight_files').buildKernel('poly').buildC(0.5).buildGamma("scale").buildCoef0("0.1").build()

SVM_svmlight_files_dataset_with_sigmoid_kernel_scale_gamma = SVMSourceCode.builder().buildDataSet('svmlight_files').buildKernel('sigmoid').buildC(0.5).buildGamma("scale").buildCoef0("0.1").build()


# Source codes with files dataset
SVM_files_dataset_with_rbf_kernel_auto_gamma = SVMSourceCode.builder().buildDataSet('files').buildKernel('rbf').buildC(0.5).buildGamma("auto").build()  

SVM_files_dataset_with_linear_kernel = SVMSourceCode.builder().buildDataSet('files').buildKernel('linear').buildC(0.5).build()

SVM_files_dataset_with_poly_kernel_auto_gamma = SVMSourceCode.builder().buildDataSet('files').buildKernel('poly').buildC(0.5).buildGamma("auto").buildCoef0("0.1").build()

SVM_files_dataset_with_sigmoid_kernel_auto_gamma = SVMSourceCode.builder().buildDataSet('files').buildKernel('sigmoid').buildC(0.5).buildGamma("auto").buildCoef0("0.1").build()

SVM_files_dataset_with_rbf_kernel_scale_gamma = SVMSourceCode.builder().buildDataSet('files').buildKernel('rbf').buildC(0.5).buildGamma("scale").build()  

SVM_files_dataset_with_poly_kernel_scale_gamma = SVMSourceCode.builder().buildDataSet('files').buildKernel('poly').buildC(0.5).buildGamma("scale").buildCoef0("0.1").build()

SVM_files_dataset_with_sigmoid_kernel_scale_gamma = SVMSourceCode.builder().buildDataSet('files').buildKernel('sigmoid').buildC(0.5).buildGamma("scale").buildCoef0("0.1").build()


# Source codes with linnerud dataset
SVM_linnerud_dataset_with_rbf_kernel_auto_gamma = SVMSourceCode.builder().buildDataSet('linnerud').buildKernel('rbf').buildC(0.5).buildGamma("auto").build()  

SVM_linnerud_dataset_with_linear_kernel = SVMSourceCode.builder().buildDataSet('linnerud').buildKernel('linear').buildC(0.5).build()

SVM_linnerud_dataset_with_poly_kernel_auto_gamma = SVMSourceCode.builder().buildDataSet('linnerud').buildKernel('poly').buildC(0.5).buildGamma("auto").buildCoef0("0.1").build()

SVM_linnerud_dataset_with_sigmoid_kernel_auto_gamma = SVMSourceCode.builder().buildDataSet('linnerud').buildKernel('sigmoid').buildC(0.5).buildGamma("auto").buildCoef0("0.1").build()

SVM_linnerud_dataset_with_rbf_kernel_scale_gamma = SVMSourceCode.builder().buildDataSet('linnerud').buildKernel('rbf').buildC(0.5).buildGamma("scale").build()  

SVM_linnerud_dataset_with_poly_kernel_scale_gamma = SVMSourceCode.builder().buildDataSet('linnerud').buildKernel('poly').buildC(0.5).buildGamma("scale").buildCoef0("0.1").build()

SVM_linnerud_dataset_with_sigmoid_kernel_scale_gamma = SVMSourceCode.builder().buildDataSet('linnerud').buildKernel('sigmoid').buildC(0.5).buildGamma("scale").buildCoef0("0.1").build()


# Source codes with sample_images dataset
SVM_sample_images_dataset_with_rbf_kernel_auto_gamma = SVMSourceCode.builder().buildDataSet('sample_images').buildKernel('rbf').buildC(0.5).buildGamma("auto").build()  

SVM_sample_images_dataset_with_linear_kernel = SVMSourceCode.builder().buildDataSet('sample_images').buildKernel('linear').buildC(0.5).build()

SVM_sample_images_dataset_with_poly_kernel_auto_gamma = SVMSourceCode.builder().buildDataSet('sample_images').buildKernel('poly').buildC(0.5).buildGamma("auto").buildCoef0("0.1").build()

SVM_sample_images_dataset_with_sigmoid_kernel_auto_gamma = SVMSourceCode.builder().buildDataSet('sample_images').buildKernel('sigmoid').buildC(0.5).buildGamma("auto").buildCoef0("0.1").build()

SVM_sample_images_dataset_with_rbf_kernel_scale_gamma = SVMSourceCode.builder().buildDataSet('sample_images').buildKernel('rbf').buildC(0.5).buildGamma("scale").build()  

SVM_sample_images_dataset_with_poly_kernel_scale_gamma = SVMSourceCode.builder().buildDataSet('sample_images').buildKernel('poly').buildC(0.5).buildGamma("scale").buildCoef0("0.1").build()

SVM_sample_images_dataset_with_sigmoid_kernel_scale_gamma = SVMSourceCode.builder().buildDataSet('sample_images').buildKernel('sigmoid').buildC(0.5).buildGamma("scale").buildCoef0("0.1").build()




