# teehee-might-delete

Go to Settings > Build, Execution, Deployment > CMake and set CMake Options to '-DICU_ROOT=/opt/homebrew/opt/icu4c' (or the win/linus equivalent)


to export to onnx:
pip install "optimum[exporters]"
pip install optimum[onnxruntime]
optimum-cli export onnx --model dbmdz/bert-large-cased-finetuned-conll03-english --task token-classification bert_ner_onnx/
