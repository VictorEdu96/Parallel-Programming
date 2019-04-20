Para graficar:
    octave --persist plotg.txt

Para correr dentro de SSH:
    export LD_LIBRARY_PATH=/usr/local/cuda/lib
    export PATH=$PATH:/usr/local/cuda/bin
    nvcc -o matrixTrans matrixTrans.cu -O2 -lc -lm