# FROM hadisalman/smoothing:latest

# RUN pip install --upgrade pip

# RUN conda install pytorch torchvision cudatoolkit=9.0 -c pytorch

# RUN git clone https://github.com/NVIDIA/apex \
#  && cd apex \
#  && pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

# RUN pip install tables scikit-learn seaborn cox matplotlib GPUtil tensorboardX 
# RUN conda install pandas numpy scipy dill 
# RUN pip install --upgrade torch torchvision
#-------------------------------------------------------------------------------------------------

FROM pytorch/pytorch:1.3-cuda10.1-cudnn7-devel

RUN pip install --upgrade pip

RUN export TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5"

RUN conda install pytorch torchvision cudatoolkit=10.1 -c pytorch

RUN export TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5"

RUN git clone https://github.com/NVIDIA/apex \
 && cd apex && pip uninstall apex && export TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5" \
 && pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

RUN pip install tables scikit-learn seaborn cox matplotlib GPUtil tensorboardX 
RUN conda install pandas numpy scipy dill 
