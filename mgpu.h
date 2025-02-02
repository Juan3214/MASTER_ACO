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
#define i_GPU 0 
#define N_e 15
#define M 512
#define n_new_edge 8
#define Q 1  
#define n_best 6
#define ITERACION 50000
#define LS_ITERATION 0 
#define c_l 32 
#define cl 32
#define s_s_flag 1
#define ACO_ALG  0// 0 RBAS 1 MMAS 2 AS
#define problem "d18512"
#define N 18512
#define name_e "problems/"
#define alg_name "RBAS_ssaaaa"
#define name_test_1 "iteration_time/iteration_time_"
#define name_test_2 "iteration_time_series/interation_time_"
#define name_test_3 "warm_up_time/warm_up_time_"
#define name_test_4 "soluciones/soluciones_F"
#define name_test_5 "hormigas/recorridos_"
#define name_test_6 "hormigas/metricas_"
#define name_test_7 "matrices/data/ENTROPY_"
#define name_test_8 "matrices/data/ENTROPY_PHERO_"
#define name_test_9 "matrices/data/LAST_IMPROVE_TEST_ss/LAST_IMPROVE"
#define name_test_10 "matrices/data/ENTROPY_MEAN_TEST_ss/ENTROPY_MEAN"
#define solucion 426 //eil51
//mona-lisa100K
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
void OPT_2_nn(int *ROUTE, int *POS_IN_ROUTE_ANT, int *COST,int *NN_LIST,float *NODE_COORDINATE_2D,int ANT);
__global__ void GPU_shannon_entropy_p_r(float *PHEROMONE_MATRIX,int *ROUTE,int *NN_LIST,float *PROB_ROUTE,float last_entropy);
void make_candidate_list(int *d_NN_LIST_aux,int *d_DISTANCE_NODE,int *DISTANCE_NODE,float *NODE_COORDINATE_2D,int *NN_LIST_cl);
__global__ void LIST_INIT(int *NEW_LIST,int *d_NEW_LIST_INDX);
__global__ void PHEROMONE_UPDATE_AS(int *ROUTE,int *BEST_ANT,float *PHEROMONE_MATRIX,int *NN_LIST,int *COST,int *OPTIMAL_ROUTE,int BEST_GLOBAL_SOLUTION);
__global__ void PHEROMONE_UPDATE_MMAS(int *ROUTE,int *BEST_ANT,
float *PHEROMONE_MATRIX,int *NN_LIST,int *COST,int *OPTIMAL_ROUTE,
int BEST_GLOBAL_SOLUTION,int update_flag);
__global__ void PHEROMONE_CHECK_MMAS(float *PHEROMONE_MATRIX,float tau_max,float tau_mim);
__global__ void RESET_ROUTE(int *ROUTE);
__global__ void fijar_pheromone(float *d_pheromone,float ini_pheromone);
__global__ void HEURISTIC_PHEROMONE_CALCULATION(float *NODE_COORDINATE,float *PHEROMONE_MATRIX,float *HEURISTIC_PHEROMONE,int *NN_LIST,float alpha,float beta);
__global__ void RESET_VISITED_LIST(bool *VISITED_LIST);
__global__ void EVAPORATION(float *PHEROMONE_MATRIX,float e);
__global__ void ANT_SOLUTION_CONSTRUCT(float *HEURISTIC_PHEROMONE,float *NODE_COORDINATE_2D,int di,
int *POS_IN_ROUTE,int *ROUTE_OP,int *POS_IN_ROUTE_ANT,
int max_new_edges,curandState *state,int *NN_LIST,int *NEW_LIST,int *NEW_LIST_INDX,int *RANDOM_DEBUG,int *LS_CHECKLIST,int flag_source_solution);
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
__device__ void OPT_2_FACO(int *ROUTE, int *POS_IN_ROUTE_ANT, int *COST, int *LS_CHECKLIST, int *NN_LIST,float *NODE_COORDINATE_2D,int ANT);
__device__ void make_swap_move_route(int current_change_x1,int current_change_x2,int *ROUTE,int *POS_IN_ROUTE,
		int ROUTE_OFFSET,int POS_IN_R_OFFSET);
__global__ void ANT_COST_CALCULATION_FACO(int *ROUTE,int *COST,float *NODE_COORDINATE_2D,int *ROUTE_AUX,int *POS_IN_ROUTE,int *LS_CHECKLIST
		,int *NN_LIST,curandState *state);
#endif
