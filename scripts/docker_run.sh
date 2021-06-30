#docker run -it --shm-size=100g --name BasicSR_npy -v /lab/isys/t_izumi/workspace/BasicSR_npy:/workspace -w /workspace --runtime nvidia nvidia/cuda:10.2-devel-ubuntu18.04

docker run -it --name BasicSR_npy -v /lab/isys/t_izumi/workspace/BasicSR_npy:/workspace -w /workspace --runtime nvidia nvidia/cuda:10.2-devel-ubuntu18.04
