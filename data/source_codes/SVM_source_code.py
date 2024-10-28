from source_code import SKLearnSVMSourceCode


# Source codes with iris dataset  - sklearn
SVM_iris_dataset_with_rbf_kernel_auto_gamma = SKLearnSVMSourceCode.builder().buildDataSet('iris').buildKernel('rbf').buildC(0.5).buildGamma("auto").build()  

SVM_iris_dataset_with_linear_kernel = SKLearnSVMSourceCode.builder().buildDataSet('iris').buildKernel('linear').buildC(0.5).build()    

SVM_iris_dataset_with_poly_kernel_auto_gamma = SKLearnSVMSourceCode.builder().buildDataSet('iris').buildKernel('poly').buildC(0.5).buildGamma("auto").buildCoef0("0.1").build()    

SVM_iris_dataset_with_sigmoid_kernel_auto_gamma = SKLearnSVMSourceCode.builder().buildDataSet('iris').buildKernel('sigmoid').buildC(0.5).buildGamma("auto").buildCoef0("0.1").build()    

SVM_iris_dataset_with_rbf_kernel_scale_gamma = SKLearnSVMSourceCode.builder().buildDataSet('iris').buildKernel('rbf').buildC(0.5).buildGamma("scale").build()  

SVM_iris_dataset_with_poly_kernel_scale_gamma = SKLearnSVMSourceCode.builder().buildDataSet('iris').buildKernel('poly').buildC(0.5).buildGamma("scale").buildCoef0("0.1").build()    

SVM_iris_dataset_with_sigmoid_kernel_scale_gamma = SKLearnSVMSourceCode.builder().buildDataSet('iris').buildKernel('sigmoid').buildC(0.5).buildGamma("scale").buildCoef0("0.1").build() 


# Source codes with digits dataset - sklearn
SVM_digits_dataset_with_rbf_kernel_auto_gamma = SKLearnSVMSourceCode.builder().buildDataSet('digits').buildKernel('rbf').buildC(0.5).buildGamma("auto").build()  

SVM_digits_dataset_with_linear_kernel = SKLearnSVMSourceCode.builder().buildDataSet('digits').buildKernel('linear').buildC(0.5).build()    

SVM_digits_dataset_with_poly_kernel_auto_gamma = SKLearnSVMSourceCode.builder().buildDataSet('digits').buildKernel('poly').buildC(0.5).buildGamma("auto").buildCoef0("0.1").build()    

SVM_digits_dataset_with_sigmoid_kernel_auto_gamma = SKLearnSVMSourceCode.builder().buildDataSet('digits').buildKernel('sigmoid').buildC(0.5).buildGamma("auto").buildCoef0("0.1").build()   

SVM_digits_dataset_with_rbf_kernel_scale_gamma = SKLearnSVMSourceCode.builder().buildDataSet('digits').buildKernel('rbf').buildC(0.5).buildGamma("auto").build()  

SVM_digits_dataset_with_poly_kernel_scale_gamma = SKLearnSVMSourceCode.builder().buildDataSet('digits').buildKernel('poly').buildC(0.5).buildGamma("scale").buildCoef0("0.1").build()    

SVM_digits_dataset_with_sigmoid_kernel_scale_gamma = SKLearnSVMSourceCode.builder().buildDataSet('digits').buildKernel('sigmoid').buildC(0.5).buildGamma("scale").buildCoef0("0.1").build()   


# Source codes with wine dataset - sklearn
SVM_wine_dataset_with_rbf_kernel_auto_gamma = SKLearnSVMSourceCode.builder().buildDataSet('wine').buildKernel('rbf').buildC(0.5).buildGamma("auto").build()  

SVM_wine_dataset_with_linear_kernel = SKLearnSVMSourceCode.builder().buildDataSet('wine').buildKernel('linear').buildC(0.5).build()    

SVM_wine_dataset_with_poly_kernel_auto_gamma = SKLearnSVMSourceCode.builder().buildDataSet('wine').buildKernel('poly').buildC(0.5).buildGamma("auto").buildCoef0("0.1").build()    

SVM_wine_dataset_with_sigmoid_kernel_auto_gamma = SKLearnSVMSourceCode.builder().buildDataSet('wine').buildKernel('sigmoid').buildC(0.5).buildGamma("auto").buildCoef0("0.1").build()   

SVM_wine_dataset_with_rbf_kernel_scale_gamma = SKLearnSVMSourceCode.builder().buildDataSet('wine').buildKernel('rbf').buildC(0.5).buildGamma("scale").build()  

SVM_wine_dataset_with_poly_kernel_scale_gamma = SKLearnSVMSourceCode.builder().buildDataSet('wine').buildKernel('poly').buildC(0.5).buildGamma("scale").buildCoef0("0.1").build()    

SVM_wine_dataset_with_sigmoid_kernel_scale_gamma = SKLearnSVMSourceCode.builder().buildDataSet('wine').buildKernel('sigmoid').buildC(0.5).buildGamma("scale").buildCoef0("0.1").build() 


# Source codes with diabetes dataset - sklearn
SVM_diabetes_dataset_with_rbf_kernel_auto_gamma = SKLearnSVMSourceCode.builder().buildDataSet('diabetes').buildKernel('rbf').buildC(0.5).buildGamma("auto").build()  

SVM_diabetes_dataset_with_linear_kernel = SKLearnSVMSourceCode.builder().buildDataSet('diabetes').buildKernel('linear').buildC(0.5).build()

SVM_diabetes_dataset_with_poly_kernel_auto_gamma = SKLearnSVMSourceCode.builder().buildDataSet('diabetes').buildKernel('poly').buildC(0.5).buildGamma("auto").buildCoef0("0.1").build()

SVM_diabetes_dataset_with_sigmoid_kernel_auto_gamma = SKLearnSVMSourceCode.builder().buildDataSet('diabetes').buildKernel('sigmoid').buildC(0.5).buildGamma("auto").buildCoef0("0.1").build()

SVM_diabetes_dataset_with_rbf_kernel_scale_gamma = SKLearnSVMSourceCode.builder().buildDataSet('diabetes').buildKernel('rbf').buildC(0.5).buildGamma("scale").build()  

SVM_diabetes_dataset_with_poly_kernel_scale_gamma = SKLearnSVMSourceCode.builder().buildDataSet('diabetes').buildKernel('poly').buildC(0.5).buildGamma("scale").buildCoef0("0.1").build()

SVM_diabetes_dataset_with_sigmoid_kernel_scale_gamma = SKLearnSVMSourceCode.builder().buildDataSet('diabetes').buildKernel('sigmoid').buildC(0.5).buildGamma("scale").buildCoef0("0.1").build()


# Source codes with breast_cancer dataset - sklearn
SVM_breast_cancer_dataset_with_rbf_kernel_auto_gamma = SKLearnSVMSourceCode.builder().buildDataSet('breast_cancer').buildKernel('rbf').buildC(0.5).buildGamma("auto").build()  

SVM_breast_cancer_dataset_with_linear_kernel = SKLearnSVMSourceCode.builder().buildDataSet('breast_cancer').buildKernel('linear').buildC(0.5).build()

SVM_breast_cancer_dataset_with_poly_kernel_auto_gamma = SKLearnSVMSourceCode.builder().buildDataSet('breast_cancer').buildKernel('poly').buildC(0.5).buildGamma("auto").buildCoef0("0.1").build()

SVM_breast_cancer_dataset_with_sigmoid_kernel_auto_gamma = SKLearnSVMSourceCode.builder().buildDataSet('breast_cancer').buildKernel('sigmoid').buildC(0.5).buildGamma("auto").buildCoef0("0.1").build()

SVM_breast_cancer_dataset_with_rbf_kernel_scale_gamma = SKLearnSVMSourceCode.builder().buildDataSet('breast_cancer').buildKernel('rbf').buildC(0.5).buildGamma("scale").build()  

SVM_breast_cancer_dataset_with_poly_kernel_scale_gamma = SKLearnSVMSourceCode.builder().buildDataSet('breast_cancer').buildKernel('poly').buildC(0.5).buildGamma("scale").buildCoef0("0.1").build()

SVM_breast_cancer_dataset_with_sigmoid_kernel_scale_gamma = SKLearnSVMSourceCode.builder().buildDataSet('breast_cancer').buildKernel('sigmoid').buildC(0.5).buildGamma("scale").buildCoef0("0.1").build()


# Source codes with svmlight_files dataset - sklearn
SVM_svmlight_files_dataset_with_rbf_kernel_auto_gamma = SKLearnSVMSourceCode.builder().buildDataSet('svmlight_files').buildKernel('rbf').buildC(0.5).buildGamma("auto").build()  

SVM_svmlight_files_dataset_with_linear_kernel = SKLearnSVMSourceCode.builder().buildDataSet('svmlight_files').buildKernel('linear').buildC(0.5).build()

SVM_svmlight_files_dataset_with_poly_kernel_auto_gamma = SKLearnSVMSourceCode.builder().buildDataSet('svmlight_files').buildKernel('poly').buildC(0.5).buildGamma("auto").buildCoef0("0.1").build()

SVM_svmlight_files_dataset_with_sigmoid_kernel_auto_gamma = SKLearnSVMSourceCode.builder().buildDataSet('svmlight_files').buildKernel('sigmoid').buildC(0.5).buildGamma("auto").buildCoef0("0.1").build()

SVM_svmlight_files_dataset_with_rbf_kernel_scale_gamma = SKLearnSVMSourceCode.builder().buildDataSet('svmlight_files').buildKernel('rbf').buildC(0.5).buildGamma("scale").build()  

SVM_svmlight_files_dataset_with_poly_kernel_scale_gamma = SKLearnSVMSourceCode.builder().buildDataSet('svmlight_files').buildKernel('poly').buildC(0.5).buildGamma("scale").buildCoef0("0.1").build()

SVM_svmlight_files_dataset_with_sigmoid_kernel_scale_gamma = SKLearnSVMSourceCode.builder().buildDataSet('svmlight_files').buildKernel('sigmoid').buildC(0.5).buildGamma("scale").buildCoef0("0.1").build()


# Source codes with files dataset - sklearn
SVM_files_dataset_with_rbf_kernel_auto_gamma = SKLearnSVMSourceCode.builder().buildDataSet('files').buildKernel('rbf').buildC(0.5).buildGamma("auto").build()  

SVM_files_dataset_with_linear_kernel = SKLearnSVMSourceCode.builder().buildDataSet('files').buildKernel('linear').buildC(0.5).build()

SVM_files_dataset_with_poly_kernel_auto_gamma = SKLearnSVMSourceCode.builder().buildDataSet('files').buildKernel('poly').buildC(0.5).buildGamma("auto").buildCoef0("0.1").build()

SVM_files_dataset_with_sigmoid_kernel_auto_gamma = SKLearnSVMSourceCode.builder().buildDataSet('files').buildKernel('sigmoid').buildC(0.5).buildGamma("auto").buildCoef0("0.1").build()

SVM_files_dataset_with_rbf_kernel_scale_gamma = SKLearnSVMSourceCode.builder().buildDataSet('files').buildKernel('rbf').buildC(0.5).buildGamma("scale").build()  

SVM_files_dataset_with_poly_kernel_scale_gamma = SKLearnSVMSourceCode.builder().buildDataSet('files').buildKernel('poly').buildC(0.5).buildGamma("scale").buildCoef0("0.1").build()

SVM_files_dataset_with_sigmoid_kernel_scale_gamma = SKLearnSVMSourceCode.builder().buildDataSet('files').buildKernel('sigmoid').buildC(0.5).buildGamma("scale").buildCoef0("0.1").build()


# Source codes with linnerud dataset - sklearn
SVM_linnerud_dataset_with_rbf_kernel_auto_gamma = SKLearnSVMSourceCode.builder().buildDataSet('linnerud').buildKernel('rbf').buildC(0.5).buildGamma("auto").build()  

SVM_linnerud_dataset_with_linear_kernel = SKLearnSVMSourceCode.builder().buildDataSet('linnerud').buildKernel('linear').buildC(0.5).build()

SVM_linnerud_dataset_with_poly_kernel_auto_gamma = SKLearnSVMSourceCode.builder().buildDataSet('linnerud').buildKernel('poly').buildC(0.5).buildGamma("auto").buildCoef0("0.1").build()

SVM_linnerud_dataset_with_sigmoid_kernel_auto_gamma = SKLearnSVMSourceCode.builder().buildDataSet('linnerud').buildKernel('sigmoid').buildC(0.5).buildGamma("auto").buildCoef0("0.1").build()

SVM_linnerud_dataset_with_rbf_kernel_scale_gamma = SKLearnSVMSourceCode.builder().buildDataSet('linnerud').buildKernel('rbf').buildC(0.5).buildGamma("scale").build()  

SVM_linnerud_dataset_with_poly_kernel_scale_gamma = SKLearnSVMSourceCode.builder().buildDataSet('linnerud').buildKernel('poly').buildC(0.5).buildGamma("scale").buildCoef0("0.1").build()

SVM_linnerud_dataset_with_sigmoid_kernel_scale_gamma = SKLearnSVMSourceCode.builder().buildDataSet('linnerud').buildKernel('sigmoid').buildC(0.5).buildGamma("scale").buildCoef0("0.1").build()


# Source codes with sample_images dataset - sklearn
SVM_sample_images_dataset_with_rbf_kernel_auto_gamma = SKLearnSVMSourceCode.builder().buildDataSet('sample_images').buildKernel('rbf').buildC(0.5).buildGamma("auto").build()  

SVM_sample_images_dataset_with_linear_kernel = SKLearnSVMSourceCode.builder().buildDataSet('sample_images').buildKernel('linear').buildC(0.5).build()

SVM_sample_images_dataset_with_poly_kernel_auto_gamma = SKLearnSVMSourceCode.builder().buildDataSet('sample_images').buildKernel('poly').buildC(0.5).buildGamma("auto").buildCoef0("0.1").build()

SVM_sample_images_dataset_with_sigmoid_kernel_auto_gamma = SKLearnSVMSourceCode.builder().buildDataSet('sample_images').buildKernel('sigmoid').buildC(0.5).buildGamma("auto").buildCoef0("0.1").build()

SVM_sample_images_dataset_with_rbf_kernel_scale_gamma = SKLearnSVMSourceCode.builder().buildDataSet('sample_images').buildKernel('rbf').buildC(0.5).buildGamma("scale").build()  

SVM_sample_images_dataset_with_poly_kernel_scale_gamma = SKLearnSVMSourceCode.builder().buildDataSet('sample_images').buildKernel('poly').buildC(0.5).buildGamma("scale").buildCoef0("0.1").build()

SVM_sample_images_dataset_with_sigmoid_kernel_scale_gamma = SKLearnSVMSourceCode.builder().buildDataSet('sample_images').buildKernel('sigmoid').buildC(0.5).buildGamma("scale").buildCoef0("0.1").build()




