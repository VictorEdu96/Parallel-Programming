Para correr dentro de SSH:

    $ export LD_LIBRARY_PATH=/usr/local/cuda/lib
    $ export PATH=$PATH:/usr/local/cuda/bin
    $ nvcc -o matrixTrans matrixTrans.cu -O2 -lc -lm


Para graficar:

    $ octave --persist plotg.txt

Gráficas:
![alt text](https://github.com/VictorEdu96/Parallel-Programming/blob/master/Matrix%20Transposition%20CUDA/time%20plots/cudaTimes100.jpg?raw=true)

![alt text](https://github.com/VictorEdu96/Parallel-Programming/blob/master/Matrix%20Transposition%20CUDA/time%20plots/cudaTimes1000.jpg?raw=true)

![alt text](https://github.com/VictorEdu96/Parallel-Programming/blob/master/Matrix%20Transposition%20CUDA/time%20plots/cudaTimes45804.jpg?raw=true)
