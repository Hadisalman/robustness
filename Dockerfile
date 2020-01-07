FROM hadisalman/smoothing:latest

RUN pip install --upgrade pip

RUN git clone https://github.com/NVIDIA/apex \
 && cd apex \
 && pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

RUN pip install tables scikit-learn seaborn cox matplotlib GPUtil tensorboardX 
RUN conda install pandas numpy scipy dill 
RUN conda install pytorch torchvision cudatoolkit=9.0 -c pytorch
# RUN pip install --upgrade torch torchvision

