# CUDA_PATH=/usr/local/cuda
CUDA_PATH=/opt/cuda
CC = g++
# NVCC=${CUDA_PATH}/bin/nvcc -ccbin ${CC}
NVCC = $(shell which nvcc)

INCLUDES = -I$(CUDA_PATH)/include -Isrc -I/usr/include/opencv4
#INCLUDES = -I${CUDA_PATH}/samples/common/inc -I$(CUDA_PATH)/include
# NVCC_FLAGS= --resource-usage -Xcompiler -rdynamic -Xcompiler # -fopenmp #-rdc=true -lnvToolsExt
NVCC_FLAGS= --gpu-architecture=sm_86 --resource-usage -Xcompiler -rdynamic -Xcompiler -fopenmp #-rdc=true -lnvToolsExt


# Gencode argumentes
# SMS = #80 #70 # 30 35 37 50 52 60 61 70
# $(foreach sm, ${SMS}, $(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

# LIBRARIES += -L/usr/local/cuda/lib64 -lcublas -lcudnn  -lcurand #-lgomp -lnvToolsExt
LIBRARIES += -L$(CUDA_PATH)/lib64 -lcublas -lcudnn  -lcurand #-lgomp -lnvToolsExt
LIBRARIES += -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs
LIBRARIES += -DZDEBUG
ALL_CCFLAGS += -m64 -O2 -std=c++17 $(NVCC_FLAGS) $(INCLUDES) $(LIBRARIES)

SRC_DIR = src
OBJ_DIR = obj

TARGET = xcnn_cuda

all : ${TARGET}


${TARGET}: $(OBJ_DIR)/drv_cnn_cuda.o $(OBJ_DIR)/layer.o\
	$(OBJ_DIR)/softmax_layer.o\
	$(OBJ_DIR)/convolutional_layer.o\
	$(OBJ_DIR)/fully_connected_layer.o\
	$(OBJ_DIR)/pooling_layer.o\
	$(OBJ_DIR)/activation_layer.o\
	$(OBJ_DIR)/batchnorm_layer.o\
	$(OBJ_DIR)/residual_layer.o\
	$(OBJ_DIR)/resnet.o\
	$(OBJ_DIR)/network.o
	@echo "------------- Buliding ${TARGET} ------------"
	$(NVCC)  $(ALL_CCFLAGS) $(GENCODE_FLAGS) $^ -o $@
	@echo

$(OBJ_DIR)/%.o : ${SRC_DIR}/%.cc
	$(NVCC) $(ALL_CCFLAGS) $(GENCODE_FLAGS)  -c $< -o $@


$(OBJ_DIR)/%.o : ${SRC_DIR}/%.cu
	$(NVCC) $(ALL_CCFLAGS) $(GENCODE_FLAGS)  -c $< -o $@

.PHONY: clean
clean:
	rm -f x* ${OBJ_DIR}/*.o

