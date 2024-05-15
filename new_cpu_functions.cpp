#include "mgpu.h"
float shannon_entropy_pheromone(float *PHEROMONE_MATRIX,float *PROB_MATRIX,float *ENTROPY_VECTOR,int flag){
	int i,j;
	float prom=0.0;float max=0.0;
	float shannon_entropy;
	for (i=0;i<N;i++){
		float sum_prob=0.0;
		float test=0.0;
		for (j=0;j<cl;j++){
			sum_prob+= PHEROMONE_MATRIX[i*cl+j];
		}
		for (j=0;j<cl;j++){
			PROB_MATRIX[i*cl+j]= PHEROMONE_MATRIX[i*cl+j]/sum_prob;
		}
	}
	for (i=0;i<N;i++){
		float entropy_node=0.0;
		for (j=0;j<cl;j++){
			entropy_node-= PROB_MATRIX[i*cl+j]*log2(PROB_MATRIX[i*cl+j]);
		}
		if (max <= entropy_node){
			max=entropy_node;
		}
		prom+=entropy_node;
		ENTROPY_VECTOR[i]=entropy_node;
	}
	prom/=N;
	if (flag == 0){
		shannon_entropy=prom;
	}
	if (flag == 1){
		shannon_entropy=max;
	}
	if (flag != 1 && flag != 2){
		printf("\n not suported, mean entropy is given");
		shannon_entropy=prom;
	}
	return shannon_entropy;	
}

