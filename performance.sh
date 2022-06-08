#!/bin/bash

max_gen=1000;

ml CUDA

export LANG=en_US.utf8
export LC_ALL=en_US.utf8
nvcc streams.cu -o streamed

for tpb in 64 128 512 1024
	do
	echo Threads per block : $tpb
	echo --------------------------------
	for N in 10000 25000 35000
		do
			echo Grid Dimensions : $N x $N
			echo "Streams"
			./streamed -n $N -m $N -max $max_gen -tpb $tpb
		done
		echo --------------------------------
	done
