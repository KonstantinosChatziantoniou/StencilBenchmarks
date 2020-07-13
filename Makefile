CC = nvcc
CFLAGS = -O3
CUFLAGS = -O3 -lcuda -lcudart # -Xptxas -v#,-abi=no #Will print the number of lmembytes for each kernel (only if kernel uses LMEM)
#-std=c99 --default-stream per-thread
LIBS = -lm #-Wall -Wextra
OBJDIR = ./bin
HDRDIR = ./headers
SRCDIR = ./src

_OBJ =  main.o kernel_nvidia.o kernel_nvidia_32.o kernel_nvidia_4.o my_kernel_regs.o my_kernel_regs_32.o my_kernel_regs_4.o
OBJ = $(patsubst %, $(OBJDIR)/%, $(_OBJ))

_DEPS = kernel_nvidia.h kernel_nvidia_32.h my_kernel_regs.h my_kernel_regs_32.h kernel_nvidia_4.h my_kernel_regs_4.h
DEPS = $(patsubst %, $(HDRDIR)/%, $(_DEPS))


mainProgram: $(OBJ)
	nvcc -o $@ $^ $(CUFLAGS) $(LIBS)


$(OBJDIR)/%.o: $(SRCDIR)/%.cu $(DEPS)  #--maxrregcount 32 
	$(CC) -c -o $@ $<  $(CUFLAGS)

$(OBJDIR)/%.o: $(SRCDIR)/%.c $(DEPS)
	gcc -c -o $@ $<  $(CFLAGS)

clean:
	rm -rf ./*.csv; rm -rf ./bin/*.o; rm -rf mainProgram

clear: 
	rm -rf ./*.csv; 