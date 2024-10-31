from src.data.source_code_generator import SVMSourceCode

example = SVMSourceCode.builder().buildDataSet(library="sklearn", dataset_name="iris").buildKernel("linear").buildC(0.5).build()

print(example.get_source_code)