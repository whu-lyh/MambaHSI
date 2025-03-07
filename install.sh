pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

cd kernels/selective_scan && pip install .
cd ../../
cd kernels/dwconv2d && python3 setup.py install --user