#include <iostream>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>
#include <curand_kernel.h>
#include <float.h>
#include <thrust/device_ptr.h>
#include <omp.h>
#include <thrust/sort.h>
#include <thrust/fill.h>
#include "mgpu.h"
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
int main(){
    
    cudaSetDevice(i_GPU);
    float *NODE_COORDINATE_2D;NODE_COORDINATE_2D=(float*)calloc(N*2,sizeof(float));
    int *DISTANCE_NODE;DISTANCE_NODE=(int*)calloc(N,sizeof(int));
    int *d_DISTANCE_NODE;cudaMalloc( (void **) &d_DISTANCE_NODE, (N)*sizeof( int ));
    int *d_NN_LIST_aux;cudaMalloc( (void **) &d_NN_LIST_aux, (N)*sizeof( int ));
    int *d_NN_LIST_cl;cudaMalloc( (void **) &d_NN_LIST_cl, (N*cl)*sizeof( int ));
    int *NN_LIST_cl;NN_LIST_cl=(int*)calloc(N*c_l,sizeof(int));
    lectura_2(NODE_COORDINATE_2D); //this  could be optimized
    int i,j;
    int *RANDOM_DEBUG;
    RANDOM_DEBUG=(int*)malloc(N_GPU*M*N*sizeof(int));
    printf("\n termine de leer");
    make_candidate_list(d_NN_LIST_aux,d_DISTANCE_NODE,DISTANCE_NODE,NODE_COORDINATE_2D,NN_LIST_cl); //this could be optimized. 
    printf("\n termine de ordenar");
    cudaMemcpy(d_NN_LIST_cl,NN_LIST_cl,N*cl*sizeof(int),cudaMemcpyHostToDevice);
    // ACO Parameters
    float alpha=1;
    float beta=2;
    float e=0.5;
    // STAD VECTORS 
    int *vec_solution;vec_solution= (int* )malloc(N_e*sizeof(int));
    int *vec_iter;vec_iter= (int* )malloc(N_e*sizeof(int)); //vectores para estadistica
    float *vec_warm_up_time;vec_warm_up_time= (float* )malloc(N_e*sizeof(float));
    float prom_time_2=0.0;
    float *vec_iteration_time;vec_iteration_time=(float*)malloc(N_e*sizeof(float));
    memset(vec_solution,0,N_e*sizeof(int));
    float *vec_ant_iteration_time_series;vec_ant_iteration_time_series=(float*)malloc(ITERACION*sizeof(float));

    for (i=0;i<ITERACION;i++)vec_ant_iteration_time_series[i]=0.0;
    int x;
    float elapsed_for_gpu_ant[N_GPU];
    /*
    Run N_e experiments of ant colony
    */
    for (x=0;x<N_e;x++){
        cudaEvent_t start_events[N_GPU];
        //int *HORMIGAS_COSTO;HORMIGAS_COSTO=(int*)malloc(ITERACION*M*N_GPU*sizeof(int));
        cudaEvent_t end_events[N_GPU];
        float prom_time=0.0;
        int *d_RANDOM_DEBUBG[N_GPU];
	int *d_LOCAL_SEARCH_LIST_MGPU[N_GPU];
	float tau_min,tau_max;
        float P_best=0.001,avg=(float)N/2.0;
        int *d_COST_MGPU[N_GPU];int *d_NN_LIST_CL_MGPU[N_GPU]; //4
        float *d_NODE_COORDINATE_MGPU[N_GPU];  // 5
        int *d_OPTIMAL_ROUTE_MGPU[N_GPU];
	//9
        int *d_NEW_LIST[N_GPU],*d_NEW_LIST_INDX[N_GPU],*d_POS_IN_ROUTE_MGPU[N_GPU],*d_ROUTE_AUX[N_GPU];
        float *d_HEURISTIC_PHEROMONE_MGPU[N_GPU]; //10
	int *d_POS_IN_ROUTE_OP[N_GPU];
	/* 10 MULTI GPU VARIABLES */
        int *NEW_LIST_GLOBAL,*NEW_LIST_INDX_GLOBAL;
        NEW_LIST_GLOBAL=(int*)malloc(M*N_GPU*(N+1)*sizeof(int));
        NEW_LIST_INDX_GLOBAL=(int*)malloc(M*N_GPU*(N+1)*sizeof(int));
        int *d_BEST_ANT,*d_OPTIMAL_ROUTE,*d_GLOBAL_COST,*d_GLOBAL_NEW_LIST; //4  
        cudaSetDevice(i_GPU);
        float *d_HEURISTIC_PHEROMONE,*d_PHEROMONE_MATRIX; //6
        /*6 MASTER GPU VARIABLES*/
        float *HEURISTIC_PHEROMONE;
        bool *VISITED_LIST;
        int *OPTIMAL_ROUTE,*GLOBAL_COST,*BEST_ANT;
        float *d_NODE_COORDINATE_2D;
        curandState *d_state[N_GPU];
        printf("\n fijando memoria en gpu \n");
        cudaMalloc( (void **) &d_BEST_ANT, N_GPU*M*sizeof(int) );//1
        cudaMalloc( (void **) &d_OPTIMAL_ROUTE, (N+1)*sizeof(int) );//2
        cudaMalloc( (void **) &d_HEURISTIC_PHEROMONE, N*cl*sizeof(float));//3
        cudaMalloc( (void **) &d_PHEROMONE_MATRIX, N*cl*sizeof(float) );//4
        cudaMalloc( (void **) &d_GLOBAL_COST, N_GPU*M*sizeof( int ) );//5
        cudaMalloc( (void **) &d_GLOBAL_NEW_LIST, N_GPU*(N+1)*M*sizeof( int ) );//6
        cudaMalloc( (void **) &d_NODE_COORDINATE_2D, N*2*sizeof(float ) );//6
        cudaMemcpy(d_NODE_COORDINATE_2D,NODE_COORDINATE_2D,2*N*sizeof( float ),cudaMemcpyHostToDevice);
        printf("\n fijando memoria en cpu \n");
	//ENTROPY CALCULATION
	float *PROB_PHERO;PROB_PHERO=(float*)malloc(N_GPU*M*sizeof(float));
	float *ENTROPY_ITERATION;ENTROPY_ITERATION=(float*)malloc(ITERACION);
	float *PROB_MATRIX;PROB_MATRIX=(float*)malloc(N*cl*sizeof(float));
	float *ENTROPY_VECTOR_PHEROMONE;ENTROPY_VECTOR_PHEROMONE=(float*)malloc(N*sizeof(float));
	float *ENTROPY_VECTOR_PHEROMONE_H;ENTROPY_VECTOR_PHEROMONE_H=(float*)malloc(N*sizeof(float));
	float *ENTROPY_VECTOR_mean;ENTROPY_VECTOR_mean=(float*)malloc(ITERACION*sizeof(float));
	float *ENTROPY_VECTOR_mean_h;ENTROPY_VECTOR_mean_h=(float*)malloc(ITERACION*sizeof(float));
	//ENTROPY CALCULATION
        int *POS_IN_ROUTE_OP;POS_IN_ROUTE_OP=(int*)malloc(N*sizeof(int));
	OPTIMAL_ROUTE=(int*)malloc((N+1)*sizeof(int));
        GLOBAL_COST=(int*)malloc(N_GPU*M*sizeof(int));
        HEURISTIC_PHEROMONE=(float*)malloc(N*cl*sizeof(float));
        //LOCAL_SEARCH_LIST_MGPU=(int*)malloc(N_GPU*M*N*sizeof(int));
        VISITED_LIST=(bool*)malloc(N*sizeof(bool));
        BEST_ANT=(int*)malloc(N_GPU*M*sizeof(int));
        for (i=0;i<N;i++)VISITED_LIST[i]=false; 
        //12
        printf("\n fijando memoria en mgpu \n");
        for (i=0;i<N_GPU;i++){
            cudaSetDevice(i_GPU+i);
            cudaEventCreate(&start_events[i]);
            cudaEventCreate(&end_events[i]);
            cudaMalloc( (void **) &d_NODE_COORDINATE_MGPU[i], 2*N*sizeof( float ) );//1
            cudaMemcpy(d_NODE_COORDINATE_MGPU[i],NODE_COORDINATE_2D,2*N*sizeof( float ),cudaMemcpyHostToDevice);
            cudaMalloc( (void **) &d_NEW_LIST[i], M*(N+1)*sizeof( int ) );//2
            cudaMalloc( (void **) &d_POS_IN_ROUTE_MGPU[i], M*N*sizeof( int ) );//2
            cudaMalloc( (void **) &d_ROUTE_AUX[i], M*(N+1)*sizeof( int ) );//2
            cudaMalloc( (void **) &d_NEW_LIST_INDX[i], M*(N+1)*sizeof( int ) );
    	    cudaMalloc( (void **) &d_RANDOM_DEBUBG[i], M*N*sizeof(int) );//1
            cudaMalloc( (void **) &d_COST_MGPU[i], M*sizeof( int ) );//6
            cudaMalloc( (void **) &d_LOCAL_SEARCH_LIST_MGPU[i], M*(N+1)*sizeof( int ) );
            cudaMalloc( (void **) &d_NN_LIST_CL_MGPU[i], cl*N*sizeof( int ) ); //7
            cudaMemcpy(d_NN_LIST_CL_MGPU[i],NN_LIST_cl,(c_l*N)*sizeof(int),cudaMemcpyHostToDevice);
            cudaMalloc( (void **) &d_HEURISTIC_PHEROMONE_MGPU[i], N*cl*sizeof( float ) );//10
            cudaMalloc( (void **) &d_POS_IN_ROUTE_OP[i], N*sizeof( int ) );//11
            cudaMalloc( (void **) &d_OPTIMAL_ROUTE_MGPU[i], N*sizeof( int ) );//11

        }
        cudaSetDevice(i_GPU);
        printf("\n greedy \n");
        int BEST_GLOBAL_SOLUTION=rutainicial(OPTIMAL_ROUTE,NODE_COORDINATE_2D,NEW_LIST_GLOBAL,
			NEW_LIST_INDX_GLOBAL,NN_LIST_cl,POS_IN_ROUTE_OP,1000);
	printf("solucion inicial %d ",BEST_GLOBAL_SOLUTION);
        OPT_2_nn(OPTIMAL_ROUTE, POS_IN_ROUTE_OP, &BEST_GLOBAL_SOLUTION,NN_LIST_cl,NODE_COORDINATE_2D,0);
	printf("solucion inicial %d ",BEST_GLOBAL_SOLUTION);
	cudaMemcpy(d_OPTIMAL_ROUTE,OPTIMAL_ROUTE, (N+1)*sizeof(int),cudaMemcpyHostToDevice);
	float ini_pheromone;
        float p=pow(P_best,1/(float)N);
        if (ACO_ALG == 1){
		ini_pheromone= 0.01;
       		tau_max=(float)e*((float)1/(float)BEST_GLOBAL_SOLUTION);
        	tau_min=tau_max*((1-p)/((avg-1)*p));
	}
	else{
		ini_pheromone=(float)BEST_GLOBAL_SOLUTION;
	}
		
        printf("\n fijando feromona %f \n",(float)1/ini_pheromone);
        fijar_pheromone<<<(N*cl+32-(N*cl%32)),32>>>(d_PHEROMONE_MATRIX,(float)1/ini_pheromone);
	if ( 1 == ACO_ALG)PHEROMONE_CHECK_MMAS<<<((N*cl+32-(N*cl%32)))/32,32>>>(d_PHEROMONE_MATRIX, tau_max, tau_min);;
        cudaMemcpy(HEURISTIC_PHEROMONE,d_PHEROMONE_MATRIX,(c_l*N)*sizeof(float),cudaMemcpyDeviceToHost);
        HEURISTIC_PHEROMONE_CALCULATION<<<N,cl>>>(d_NODE_COORDINATE_2D,d_PHEROMONE_MATRIX,
        d_HEURISTIC_PHEROMONE,d_NN_LIST_cl,alpha,beta);

        thrust::device_ptr<int> dev_inx = thrust::device_pointer_cast(d_BEST_ANT); //utilziar thrust para sorting
        thrust::device_ptr<int> dev_ptr = thrust::device_pointer_cast(d_GLOBAL_COST);      
        
        cudaMemcpy(HEURISTIC_PHEROMONE,d_HEURISTIC_PHEROMONE,(N*c_l)*sizeof(float),cudaMemcpyDeviceToHost);
        printf("\n copiando predecesor\n");

        for (i=0;i<N_GPU;i++){
            cudaSetDevice(i_GPU+i);
            cudaMemcpy(d_POS_IN_ROUTE_OP[i],POS_IN_ROUTE_OP, N*sizeof(int),cudaMemcpyHostToDevice);
            cudaMemcpy(d_OPTIMAL_ROUTE_MGPU[i],OPTIMAL_ROUTE, N*sizeof(int),cudaMemcpyHostToDevice);
            cudaMalloc((void **) &d_state[i], M*sizeof(curandState));iniciar_kernel<<<32,M/32>>>(d_state[i],i,(unsigned long long)x);
            cudaMemcpy(d_HEURISTIC_PHEROMONE_MGPU[i],HEURISTIC_PHEROMONE,N*cl*sizeof(float),cudaMemcpyHostToDevice);
            cudaDeviceSynchronize();
        }
        int mejor,it,LAST_IMPROVE_ITERATION=0;
        // ITERACIONES 

        int num_gpus;
        cudaGetDeviceCount( &num_gpus );
        cudaSetDevice(i_GPU);
        printf("\n INICIANDO ITERACIONES\n");
	float mean_entropy_Heuristic=0.0;
	float mean_entropy_Pheromone=0.0;
        for (it=0;it<ITERACION;it++){
            double begin_1 =omp_get_wtime();
            #pragma omp parallel for num_threads(N_GPU)
            for (i=0;i<N_GPU;i++){
                cudaSetDevice(i_GPU+i);
                cudaEventRecord(start_events[i]);
                LIST_INIT<<<N,min(M,1024)>>>(d_NEW_LIST[i],d_NEW_LIST_INDX[i]);
                ANT_SOLUTION_CONSTRUCT<<<M/4,4>>>(d_HEURISTIC_PHEROMONE_MGPU[i],d_NODE_COORDINATE_MGPU[i],i,d_POS_IN_ROUTE_OP[i],
				d_OPTIMAL_ROUTE_MGPU[i],d_POS_IN_ROUTE_MGPU[i],8,d_state[i],d_NN_LIST_CL_MGPU[i],
				d_NEW_LIST[i],d_NEW_LIST_INDX[i],d_RANDOM_DEBUBG[i],d_LOCAL_SEARCH_LIST_MGPU[i],s_s_flag);
                /*------------------------ SOBRE ANT_SOLUTION_CONSTRUCT--------------------*/            
                //aumentar el numero de thread ahora parece mejorar el rendimiento
                //en el alg anterior no ocurria eso, cuidado con la memoria compartida
                //por ahora con 4 threads y 4 bloques es lo optimo
                /*------------------------ SOBRE ANT_COST_CALCULATION_LS --------------------*/
                // aumentar el numero de threads mejora el rendimiento  hasta 32 thread
                //ANT_COST_CALCULATION_LS<<<M/32,32>>>(d_NEW_LIST[i],d_COST_MGPU[i],d_NODE_COORDINATE_MGPU[i],d_ROUTE_AUX[i],d_state[i]);
                ANT_COST_CALCULATION_FACO<<<M/32,32>>>(d_NEW_LIST[i],d_COST_MGPU[i],d_NODE_COORDINATE_MGPU[i],d_ROUTE_AUX[i],d_POS_IN_ROUTE_MGPU[i],
				d_LOCAL_SEARCH_LIST_MGPU[i],d_NN_LIST_CL_MGPU[i],d_state[i]);
                gpuErrchk(cudaMemcpyAsync(GLOBAL_COST+i*M,d_COST_MGPU[i],M*sizeof(int),cudaMemcpyDeviceToHost));   //esto ahorra 6 ms en 4000 nodos 
                gpuErrchk(cudaMemcpyAsync(NEW_LIST_GLOBAL+i*M*(N+1),d_NEW_LIST[i],(N+1)*M*sizeof(int),cudaMemcpyDeviceToHost));
                gpuErrchk(cudaMemcpyAsync(NEW_LIST_INDX_GLOBAL+i*M*(N+1),d_NEW_LIST_INDX[i],(N+1)*M*sizeof(int),cudaMemcpyDeviceToHost));   //esto ahorra 6 ms en 4000 nodos 
                gpuErrchk(cudaMemcpyAsync(RANDOM_DEBUG+i*M*N,d_RANDOM_DEBUBG[i],N*M*sizeof(int),cudaMemcpyDeviceToHost));   //esto ahorra 6 ms en 4000 nodos 
                cudaDeviceSynchronize();
                cudaEventRecord(end_events[i]);

            }
            cudaDeviceSynchronize();/*
	    for (j=0;j<N;j++){
 		printf("%d ",RANDOM_DEBUG[64*N+j]);		}
	    printf("\n");*/
	    double end_1 =omp_get_wtime();
            //cudaMemcpy(HORMIGAS_COSTO+it*N_GPU*M,GLOBAL_COST,N_GPU*M*sizeof(int),cudaMemcpyHostToHost);
            if(it==0)vec_warm_up_time[x]=(end_1-begin_1)*1000;
	    //printf("time it %lf \n",(end_1-begin_1)*1000);
            vec_ant_iteration_time_series[it]+=((end_1-begin_1)*1000.0)/((float)N_e);
            /*
            for(i = 0; i < 4; i++)
                {
                    cudaEventElapsedTime(&elapsed_for_gpu_ant[i], start_events[i], end_events[i]);
                    printf("Elapsed time on device %d: %f ms\n", i, elapsed_for_gpu_ant[i]);
                }
            */
            cudaSetDevice(i_GPU);
            gpuErrchk(cudaMemcpy(d_GLOBAL_COST,GLOBAL_COST,N_GPU*M*sizeof(int),cudaMemcpyHostToDevice));
            gpuErrchk(cudaMemcpy(d_GLOBAL_NEW_LIST,NEW_LIST_GLOBAL,N_GPU*(N+1)*M*sizeof(int),cudaMemcpyHostToDevice));
            thrust::sequence(thrust::device,dev_inx, dev_inx+N_GPU*M);
            thrust::sort_by_key(thrust::device,dev_ptr, dev_ptr + N_GPU*M, dev_inx,thrust::less<int>());
            cudaMemcpy(GLOBAL_COST,d_GLOBAL_COST,N_GPU*M*sizeof(int),cudaMemcpyDeviceToHost);
            cudaMemcpy(BEST_ANT,d_BEST_ANT,N_GPU*M*sizeof(int),cudaMemcpyDeviceToHost);
            mejor=BEST_ANT[0];  
            cudaSetDevice(i_GPU);
            if (it==0){
                GLOBAL_COST[0]=BEST_GLOBAL_SOLUTION;
                tau_max=(float)e*((float)1/(float)BEST_GLOBAL_SOLUTION);
                tau_min=tau_max*((1-p)/((avg-1)*p));
            }
            if (it%100==0)printf("\n %d it= %d\n",GLOBAL_COST[0],it);
            if (GLOBAL_COST[0]<BEST_GLOBAL_SOLUTION){
		    BEST_GLOBAL_SOLUTION=GLOBAL_COST[0];
                    LAST_IMPROVE_ITERATION=it;
                    tau_max=(float)e*((float)1/(float)BEST_GLOBAL_SOLUTION);
                    tau_min=tau_max*((1-p)/((avg-1)*p));
                    if (tau_max<tau_min)tau_min=tau_max;
                    printf("\n mejor global = %d en iter= %d en experimento %d con alpha= %f y beta=%f\n",BEST_GLOBAL_SOLUTION,LAST_IMPROVE_ITERATION,x,alpha,beta);
                    SAVE_LAST_IMRPOVED(LAST_IMPROVE_ITERATION,BEST_GLOBAL_SOLUTION,x,alpha,beta); 
		    //printf("\n tau max =%.16lf \n tau min =%.16lf \n",tau_max,tau_min);
                    for (i=0;i<N+1;i++){
                        OPTIMAL_ROUTE[i]=NEW_LIST_GLOBAL[mejor*(N+1)+i%N];
                        POS_IN_ROUTE_OP[OPTIMAL_ROUTE[i]]=i;
			//printf("%d ",OPTIMAL_ROUTE[i]);
                    }
                    cudaMemcpy(d_OPTIMAL_ROUTE,OPTIMAL_ROUTE, (N+1)*sizeof(int),cudaMemcpyHostToDevice);
                    for (i=0;i<N_GPU;i++){
                        cudaSetDevice(i_GPU+i);
                        cudaMemcpy(d_POS_IN_ROUTE_OP[i],POS_IN_ROUTE_OP, N*sizeof(int),cudaMemcpyHostToDevice);
                        cudaMemcpy(d_OPTIMAL_ROUTE_MGPU[i],OPTIMAL_ROUTE, N*sizeof(int),cudaMemcpyHostToDevice);
                    }
                    cudaSetDevice(i_GPU);
            }
            cudaSetDevice(i_GPU);
	    UPGRADE_PHEROMONE(d_GLOBAL_NEW_LIST,d_BEST_ANT,d_PHEROMONE_MATRIX, 
            d_NN_LIST_cl,d_GLOBAL_COST,d_OPTIMAL_ROUTE,BEST_GLOBAL_SOLUTION,tau_max,tau_min,e,ACO_ALG);	    
            cudaMemcpy(HEURISTIC_PHEROMONE,d_PHEROMONE_MATRIX,cl*N*sizeof(float),cudaMemcpyDeviceToHost);        
	    mean_entropy_Pheromone=shannon_entropy_pheromone(HEURISTIC_PHEROMONE,PROB_MATRIX,ENTROPY_VECTOR_PHEROMONE,0);           
	    ENTROPY_VECTOR_mean[it]=mean_entropy_Pheromone;
            HEURISTIC_PHEROMONE_CALCULATION<<<N,cl>>>(d_NODE_COORDINATE_2D,d_PHEROMONE_MATRIX,
            d_HEURISTIC_PHEROMONE,d_NN_LIST_cl,alpha,beta);
            cudaMemcpy(HEURISTIC_PHEROMONE,d_HEURISTIC_PHEROMONE,(N*c_l)*sizeof(float),cudaMemcpyDeviceToHost);
	    mean_entropy_Heuristic=shannon_entropy_pheromone(HEURISTIC_PHEROMONE,PROB_MATRIX,ENTROPY_VECTOR_PHEROMONE_H,0);           
	    ENTROPY_VECTOR_mean_h[it]=mean_entropy_Heuristic;
	    //guardar_entropias_pheromone(ENTROPY_VECTOR_PHEROMONE,ENTROPY_VECTOR_PHEROMONE_H,alpha,beta,e,it,x);
            for (i=0;i<N_GPU;i++){
                cudaSetDevice(i_GPU+i);
                cudaMemcpy(d_HEURISTIC_PHEROMONE_MGPU[i],HEURISTIC_PHEROMONE,(N*c_l)*sizeof(float),cudaMemcpyHostToDevice);
            }
            cudaDeviceSynchronize();
            double end_2 =omp_get_wtime(); 
            prom_time+=(end_2-begin_1)*1000;
	    //printf("time end %lf \n",(end_2-begin_1)*1000);
        }
	guardar_entropia_promedio(ENTROPY_VECTOR_mean,ENTROPY_VECTOR_mean_h,x,alpha,beta);
        vec_solution[x]=BEST_GLOBAL_SOLUTION;
        printf("\n -------------------------------\n");
        //for (i=0;i<N+1;i++)printf("%d ", OPTIMAL_ROUTE[i]);
        printf("\n -------------------------------\n");
        cudaSetDevice(i_GPU);
        free(BEST_ANT);cudaFree(d_GLOBAL_NEW_LIST);
        cudaFree(d_NODE_COORDINATE_2D);
        cudaFree(d_BEST_ANT);cudaFree(d_OPTIMAL_ROUTE);cudaFree(d_GLOBAL_COST);
        cudaFree(d_HEURISTIC_PHEROMONE);cudaFree(d_PHEROMONE_MATRIX);
	for (i=0;i<N_GPU;i++){
            cudaSetDevice(i_GPU+i);cudaFree(d_NODE_COORDINATE_MGPU[i]);
            cudaFree(d_ROUTE_AUX[i]);
            cudaFree(d_COST_MGPU[i]);
            cudaFree(d_state[i]);
            cudaFree(d_NEW_LIST[i]);
            cudaFree(d_NEW_LIST_INDX[i]);
            cudaFree(d_POS_IN_ROUTE_MGPU[i]);
            cudaFree(d_LOCAL_SEARCH_LIST_MGPU[i]);
            cudaFree(d_NN_LIST_CL_MGPU[i]);
   	    cudaFree(d_RANDOM_DEBUBG[i]);
	    cudaFree(d_POS_IN_ROUTE_OP[i]);
	    cudaFree(d_OPTIMAL_ROUTE_MGPU[i]);
            cudaFree(d_HEURISTIC_PHEROMONE_MGPU[i]);
            cudaEventDestroy(start_events[i]);
            cudaEventDestroy(end_events[i]);
        }
        //free(LOCAL_SEARCH_LIST_MGPU);
        //escribir_costo(HORMIGAS_COSTO,x); //segmentation fail
        //free(HORMIGAS_COSTO);
	free(PROB_PHERO);
        free(ENTROPY_ITERATION);
	free(PROB_MATRIX);
	free(ENTROPY_VECTOR_PHEROMONE);
	free(ENTROPY_VECTOR_PHEROMONE_H);
	free(ENTROPY_VECTOR_mean_h);
	free(ENTROPY_VECTOR_mean);
	free(OPTIMAL_ROUTE);free(VISITED_LIST);
	free(POS_IN_ROUTE_OP);
	free(GLOBAL_COST);
        free(HEURISTIC_PHEROMONE);free(NEW_LIST_GLOBAL);free(NEW_LIST_INDX_GLOBAL);
        prom_time/=(ITERACION);
        prom_time_2+=prom_time;
        vec_iteration_time[x]=prom_time;
        
    }
    prom_time_2/=(N_e); 
    free(RANDOM_DEBUG);
    guardar_resultados(vec_warm_up_time,vec_solution,vec_ant_iteration_time_series,vec_iteration_time,alpha,beta,e);
    printf("\n el tiempo promedio es de %f\n ",prom_time_2);
    free(NODE_COORDINATE_2D);free(DISTANCE_NODE);
    cudaFree(d_DISTANCE_NODE);free(NN_LIST_cl);cudaFree(d_NN_LIST_aux);cudaFree(d_NN_LIST_cl);
    return 0;
}
