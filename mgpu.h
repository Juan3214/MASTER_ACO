/*
* Author: Juan Manuel Aedo Sepulveda
* This is an Ant Colony Algorithm
*/
#ifndef MGPU_H_
#define MGPU_H_
#include <iostream>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <float.h>
#include <time.h>
#include <cuda.h>
#include <curand_kernel.h>
#include <float.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/fill.h>
#define N_GPU 1 
#define i_GPU 2 
#define N_e 20 
#define M 128//a partir de 512 si duplico 
#define n_new_edge 8
#define Q 1 // 
#define n_best 10
#define N 100 
#define ITERACION 5000 
#define LS_ITERATION 0 
#define c_l 99 
#define cl 99
#define s_s_flag 0 
#define ACO_ALG 1 // 0 RBAS 1 MMAS 2 AS
#define problem "kroA100" //solo hay que cambiar este
#define name_e "problems/"
#define alg_name "MMAS_Eq"
#define name_test_1 "iteration_time/iteration_time_"
#define name_test_2 "iteration_time_series/interation_time_"
#define name_test_3 "warm_up_time/warm_up_time_"
#define name_test_4 "soluciones/soluciones_"
#define name_test_5 "hormigas/recorridos_"
#define name_test_6 "hormigas/metricas_"
#define name_test_7 "matrices/data/ENTROPY_"
#define name_test_8 "matrices/data/ENTROPY_PHERO_"
#define name_test_9 "matrices/data/LAST_IMPROVE/LAST_IMPROVE_"
#define name_test_10 "matrices/data/ENTROPY_MEAN/ENTROPY_MEAN_"
#define solucion 426 //eil51
//mona-lisa100K
//#define solucion 7542.0 //berlin52
//#define solucion 675.0 //st70
//#define solucion 538.0 //eil76
//#define solucion 1211.0 //rat99
//#define solucion 21282.0 //kroA100
//#define solucion 629.0 //eil101
//#define solucion 6942.0 //gr120
//#define solucion 6528.0 //ch150
//#define solucion 29368.0 //kroA200
//#define solucion 50778.0 //pcb442
//#define solucion 2763.0 //pa561
// fnl4461
void escribir_costo(int *HORMIGAS_COSTOS,int x);
void SAVE_LAST_IMRPOVED(int LAST_IMRPOVE_IT,int BEST_SOLUTION,int experiment,float alpha,float beta);
void guardar_entropias_pheromone(float *ENTROPY_VECTOR_PHEROMONE,float *ENTROPY_VECTOR_PHEROMONE_H,float alpha,float beta, float e,int it,int x);
void guardar_resultados(float *vec_warm_up_time,int *vec_solution,float *vec_ant_iteration_time_series,float *vec_iteration_time,float alpha,float beta,float e );
float shannon_entropy_pheromone(float *PHEROMONE_MATRIX,float *PROB_MATRIX,float *ENTROPY_VECTOR,int flag);
float shannon_entropy_p_r(float *PHEROMONE_MATRIX,int *ROUTE,int *NN_LIST,float *PROB_ROUTE,float last_entropy,float *ENTROPY_ITERATION,int it);
void lectura_2(float *dis);
void guardar_entropia_promedio(float *entropia_shannon,float *entropia_shannon_h,int x,float alpha,float beta);
void SAVE_PHEROMONE_MATRIX(float *PHEROMONE_MATRIX,int it, int expriment,float alpha,float beta,float e);
//-------------------------------------------------------------------------------------
int rutainicial_2(int *rute_op,float *d,bool *lista_vis);
void guardar_warm_up(float *time,float alpha,float beta,float e);
void guardar_iteration_time(float *time,float alpha,float beta,float e);
void guardar_soluciones(int *soluciones,float alpha,float beta,float e);
void guardar_iteration_time_series(float *time,float alpha,float beta,float e);
float std_vec(float *vec, float prom);
float std_vec_it(int *vec,float prom);
float minimovec(float *vec);

int EUC_2D_C(float *d_d,int p1,int p2);

void C100HECK_VISITED_CPU(int *NEW_LIST,int *NEW_LIST_INDX,int ant,int CHOSEN_NODE);
bool IS_VISITED_CPU(int *NEW_LIST,int *NEW_LIST_INDX,int ant,int j);
int GET_CANDIDATE_CPU(int *NEW_LIST,int *NEW_LIST_INDX,int ant,int j);
int rutainicial(int *rute_op,float *d,int *NEW_LIST_GLOBAL,int *NEW_LIST_INDX_GLOBAL,int *NN_LIST,int *POS_IN_ROUTE,int seed);
void UPGRADE_PHEROMONE(int *ROUTE,int *BEST_ANT,float *PHEROMONE_MATRIX,int *NN_LIST,int *COST,int *OPTIMAL_ROUTE,int BEST_GLOBAL_SOLUTION,float tau_max,float tau_min,float e,int ACO_flag);
/*
*This function free the memory for the given vectores
*/

/*
* This function join the 4 gpu cost vectors
*/
__global__ void GPU_shannon_entropy_p_r(float *PHEROMONE_MATRIX,int *ROUTE,int *NN_LIST,float *PROB_ROUTE,float last_entropy);
void make_candidate_list(int *d_NN_LIST_aux,int *d_DISTANCE_NODE,int *DISTANCE_NODE,float *NODE_COORDINATE_2D,int *NN_LIST_cl);
__global__ void LIST_INIT(int *NEW_LIST,int *d_NEW_LIST_INDX);
__global__ void PHEROMONE_UPDATE_AS(int *ROUTE,int *BEST_ANT,float *PHEROMONE_MATRIX,int *NN_LIST,int *COST,int *OPTIMAL_ROUTE,int BEST_GLOBAL_SOLUTION);
__global__ void PHEROMONE_UPDATE_MMAS(int *ROUTE,int *BEST_ANT,
float *PHEROMONE_MATRIX,int *NN_LIST,int *COST,int *OPTIMAL_ROUTE,
int BEST_GLOBAL_SOLUTION,int update_flag);
__global__ void PHEROMONE_CHECK_MMAS(float *PHEROMONE_MATRIX,float tau_max,float tau_mim);
/*
* This function calculate the combinated factor of pheromone and visibility
*/
/*
* This function reset the rutes to 0
*/
__global__ void RESET_ROUTE(int *ROUTE);
/*
* This function join the 4 gpu rute vectors
*/
/*
* This function set the pheromone to a value 
*/
__global__ void fijar_pheromone(float *d_pheromone,float ini_pheromone);
/*
* This function set the visibility to a value 
*/
/*
* This function set the cost to a value 
*/
/*
* This function reset the visited_list of the ant tour
*/
__global__ void HEURISTIC_PHEROMONE_CALCULATION(float *NODE_COORDINATE,float *PHEROMONE_MATRIX,float *HEURISTIC_PHEROMONE,int *NN_LIST,float alpha,float beta);
__global__ void RESET_VISITED_LIST(bool *VISITED_LIST);
__global__ void EVAPORATION(float *PHEROMONE_MATRIX,float e);
__global__ void ANT_SOLUTION_CONSTRUCT(float *HEURISTIC_PHEROMONE,float *NODE_COORDINATE_2D,int di,
int *POS_IN_ROUTE,int *ROUTE_OP,int *POS_IN_ROUTE_ANT,
int max_new_edges,curandState *state,int *NN_LIST,int *NEW_LIST,int *NEW_LIST_INDX,int *RANDOM_DEBUG,int flag_source_solution);
__global__ void ANT_COST_CALCULATION_LS(int *ROUTE,int *COST,float *NODE_COORDINATE_2D,int *ROUTE_AUX,curandState *state);
__global__ void PHEROMONE_UPDATE(int *ROUTE,int *BEST_ANT,float *PHEROMONE_MATRIX,int *NN_LIST,int *COST,int *OPTIMAL_ROUTE,int BEST_GLOBAL_SOLUTION);
__global__ void iniciar_kernel(curandState *state,int di,unsigned long long seed=1000);
__device__ void L_S(float *d_d,int *d_rute,float *d_global_cost,int *local_search_list,
int *d_rute_succ_all,int *d_rute_pred_all,int *d_orden_indice,int i);
__device__ void opt3(int *ROUTE,float *NODE_COORDINATE_2D,int *COST,int k,int *ROUTE_AUX,curandState *state);
__device__ int EUC_2D(float *d_d,int x,int y);
__device__ float EUC_2D_f(float *d_d,int p1,int p2);
__device__ void CHECK_VISITED(int *NEW_LIST,int *NEW_LIST_INDX,int ant,int CHOSEN_NODE);
__device__ bool IS_VISITED(int *NEW_LIST,int *NEW_LIST_INDX,int ant,int j);
__device__ int GET_CANDIDATE(int *NEW_LIST,int *NEW_LIST_INDX,int ant,int j);
#endif
