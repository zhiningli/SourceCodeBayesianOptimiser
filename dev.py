from data.source_codes.source_code import SVMSourceCode


SVM_iris_dataset = SVMSourceCode.builder().buildDataSet('iris').buildKernel('rbf').buildC(0.5).buildGamma("auto").buildCoef0("0.5").build()  


print(SVM_iris_dataset.get_source_code)
print(SVM_iris_dataset.hyperparameters)
print(SVM_iris_dataset.get_optimal_BO_hyperParameters())

    