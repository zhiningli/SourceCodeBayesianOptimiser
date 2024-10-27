from source_code import SVMSourceCode


# Source codes with iris dataset
SVM_iris_dataset_with_rbf_kernel_auto_gamma = SVMSourceCode.builder().buildDataSet('iris').buildKernel('rbf').buildC(0.5).buildGamma("auto").build()  

SVM_iris_dataset_with_linear_kernel_auto_gamma = SVMSourceCode.builder().buildDataSet('iris').buildKernel('linear').buildC(0.5).build()    

SVM_iris_dataset_with_poly_kernel_auto_gamma = SVMSourceCode.builder().buildDataSet('iris').buildKernel('poly').buildC(0.5).buildGamma("auto").buildCoef0("0.1").build()    

SVM_iris_dataset_with_sigmoid_kernel_auto_gamma = SVMSourceCode.builder().buildDataSet('iris').buildKernel('sigmoid').buildC(0.5).buildGamma("auto").buildCoef0("0.1").build()    

