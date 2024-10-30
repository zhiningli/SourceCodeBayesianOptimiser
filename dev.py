from data.source_codes.source_code import SVMSourceCode


SVM_iris_dataset = SVMSourceCode.builder().buildDataSet(library="openml", dataset_id=1597).buildKernel('rbf').buildC(0.5).buildGamma("scale").buildCoef0("0.5").build()  

print(SVM_iris_dataset.name)

print(SVM_iris_dataset.get_source_code)
print(SVM_iris_dataset.source_code_hyperparameters)
print(SVM_iris_dataset.optimalSVMHyperparameter)
print(SVM_iris_dataset.BOHyperparameters_searchSpace)
print(SVM_iris_dataset.optimalBOHyperParameters)



    