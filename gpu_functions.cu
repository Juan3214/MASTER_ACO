#include "mgpu.h"
__global__ void GPU_shannon_entropy_p_r(float *PHEROMONE_MATRIX,int *ROUTE,int *NN_LIST,float *PROB_ROUTE,float last_entropy){
	
    	int i=threadIdx.x+ (blockIdx.x * blockDim.x);
	int j,k;
	float global_sum=0.0;
	float sum_phero;
	int NN_neighbor;
	sum_phero=0;
	for (j=0;j<N;j++){
		for (k=0;k<cl;k++){
			NN_neighbor=NN_LIST[ROUTE[i*(N+1)+j]*cl+k];
			if (ROUTE[i*(N+1)+(j+1)%N]==NN_neighbor){
				sum_phero+=PHEROMONE_MATRIX[ROUTE[i*(N+1)+j]*cl+k];
			}
		} 	
	}
	PROB_ROUTE[i]=sum_phero;
}
__global__ void iniciar_kernel(curandState *state,int di,unsigned long long seed){
    int i=threadIdx.x+ (blockIdx.x * blockDim.x);
    if (seed==1000){
    	curand_init((unsigned long long)clock() + (unsigned long long)(i+M*di), 0, 0, &state[i]);
    }
    else{
    	curand_init(seed + (unsigned long long)(i+M*di), 0, 0, &state[i]);
    }
} 
__global__ void ANT_SOLUTION_CONSTRUCT(float *HEURISTIC_PHEROMONE,float *NODE_COORDINATE_2D,
int di, int *POS_IN_ROUTE,int *ROUTE_OP,int *POS_IN_ROUTE_ANT,
int max_new_edges,curandState *state,int *NN_LIST,int *NEW_LIST,int *NEW_LIST_INDX,int *RANDOM_DEBUG,int *LS_CHECKLIST,int flag_source_solution){
    int i=threadIdx.x+ (blockIdx.x * blockDim.x);
    int id=threadIdx.x;
    if (i<M){
        int loc_act;int j=0;int new_edges=0;
        curandState localstate=state[i];
        NEW_LIST[i*(N+1)+N]=N;
	float myrandf = curand_uniform(&localstate);
        myrandf *= (N - 2 + 0.999999);
        myrandf += 0;
        int random = (int)truncf(myrandf);
	RANDOM_DEBUG[i*N]=random;
        CHECK_VISITED(NEW_LIST,NEW_LIST_INDX,i,random);
	POS_IN_ROUTE_ANT[i*N+random]=N-1;
        int pos;  

        __shared__ float prob[(c_l)*4];
        while(j<N-1){
            loc_act=NEW_LIST[i*(N+1)+NEW_LIST[i*(N+1)+N]]; // ACTUAL LOCATION
             // CHECK AS VISITED
            prob[id*cl]=HEURISTIC_PHEROMONE[loc_act*cl]*(float)(1-(int)(IS_VISITED(NEW_LIST,NEW_LIST_INDX,i,NN_LIST[loc_act*cl]))); // IF IT WAS VISTED, THE PROBABILITY IS 0
            for (int k=1;k<cl;k++){
                    pos=NN_LIST[loc_act*cl+k];
                    prob[id*cl+k]=prob[id*cl+k-1]+HEURISTIC_PHEROMONE[loc_act*cl+k]*(float)(1-(int)(IS_VISITED(NEW_LIST,NEW_LIST_INDX,i,pos)));
            }
            float val =  curand_uniform(&localstate);
            float ranval=val*prob[id*cl+cl-1];
            int CHOSEN_NODE=-9;
            for (int ran=0;ran<cl;ran++){
                if (ranval<=prob[id*cl+ran]){
		    if (IS_VISITED(NEW_LIST,NEW_LIST_INDX,i,NN_LIST[loc_act*cl+ran])==false)
                    	CHOSEN_NODE=NN_LIST[loc_act*cl+ran]; 
                    	break;
                }
            }
            if (ranval==0.0){
                int min=INT_MAX;int dist; 
                for (int k=0;k<NEW_LIST[i*(N+1)+N];k++){
                    int candidate=GET_CANDIDATE(NEW_LIST,NEW_LIST_INDX,i,k);
                    dist=EUC_2D(NODE_COORDINATE_2D,candidate,loc_act);
                    if (IS_VISITED(NEW_LIST,NEW_LIST_INDX,i,candidate)==false && dist<min ){
                        CHOSEN_NODE=candidate; 
                        min=dist;
                    }
                }
            }
            CHECK_VISITED(NEW_LIST,NEW_LIST_INDX,i,CHOSEN_NODE);
            j++;
	    POS_IN_ROUTE_ANT[i*N+CHOSEN_NODE]=N - 1 - j;
	    if (flag_source_solution==1){
		    int u;
		    int pos_in_route;
		    pos_in_route = POS_IN_ROUTE[loc_act];
		    u = (pos_in_route == 0) ? ROUTE_OP[N-1] : ROUTE_OP[pos_in_route-1];
		    if (u != CHOSEN_NODE){
			LS_CHECKLIST[i*(N+1)+new_edges]=CHOSEN_NODE; 
			new_edges+=1;
			LS_CHECKLIST[i*(N+1)+N]=new_edges; 
		    }
		    if (new_edges>=max_new_edges){
			pos_in_route=POS_IN_ROUTE[CHOSEN_NODE];
			u = (pos_in_route == 0) ? ROUTE_OP[N-1] : ROUTE_OP[pos_in_route-1]; // THIS IS GOING BACKWARDS BECAUSE THE CHANGE ON THE ROUTE DATA STRUCTURE 
			while (IS_VISITED(NEW_LIST,NEW_LIST_INDX,i,u)==false && j<N-1){
			    CHECK_VISITED(NEW_LIST,NEW_LIST_INDX,i,u);
			    loc_act=u;
			    pos_in_route = POS_IN_ROUTE[u];
			    j++; 
			    POS_IN_ROUTE_ANT[i*N+u] = N - 1  - j;
			    u = (pos_in_route == 0) ? ROUTE_OP[N-1] : ROUTE_OP[pos_in_route-1];

			}
			pos_in_route=POS_IN_ROUTE[u];
			u = (pos_in_route == N-1) ? ROUTE_OP[0] : ROUTE_OP[pos_in_route+1];
			while (IS_VISITED(NEW_LIST,NEW_LIST_INDX,i,u)==false && j<N-1){ 
			    CHECK_VISITED(NEW_LIST,NEW_LIST_INDX,i,u);
			    loc_act=u;
			    pos_in_route = POS_IN_ROUTE[u];
			    j++;
	    		    POS_IN_ROUTE_ANT[i*N+u] = N - 1 - j;
			    u = (pos_in_route == N-1) ? ROUTE_OP[0] : ROUTE_OP[pos_in_route+1];
			}    
		    }
       		}
	}
        state[i]=localstate;
    }
}
__device__ void CHECK_VISITED(int *NEW_LIST,int *NEW_LIST_INDX,int ant,int CHOSEN_NODE){
    int ant_offset = ant*(N+1);
    int length=NEW_LIST[ant*(N+1)+N]-1;
    int CHOSEN_INDX=NEW_LIST_INDX[ant*(N+1)+CHOSEN_NODE];
    int temp_1=NEW_LIST[ant*(N+1)+length];
    int temp_2=NEW_LIST_INDX[ant*(N+1)+temp_1]  ;
    NEW_LIST[ant_offset+length]=CHOSEN_NODE;
    NEW_LIST_INDX[ant_offset+temp_1]=CHOSEN_INDX;
    NEW_LIST[ant_offset+CHOSEN_INDX]=temp_1;
    NEW_LIST_INDX[ant_offset+CHOSEN_NODE]=temp_2;
    NEW_LIST[ant_offset+N]-=1;
}

__device__ bool IS_VISITED(int *NEW_LIST,int *NEW_LIST_INDX,int ant,int j){
    if (NEW_LIST_INDX[ant*(N+1)+j]<NEW_LIST[ant*(N+1)+N]){
        return false;
    }
    else{
        return true;
    }
}
__device__ int GET_CANDIDATE(int *NEW_LIST,int *NEW_LIST_INDX,int ant,int j){
    return NEW_LIST[ant*(N+1)+j];
}
__global__ void LIST_INIT(int *d_NEW_LIST,int *d_NEW_LIST_INDX){
    int j=blockIdx.x; 
    if (j<N){
        for (int i=threadIdx.x;i<M;i+=blockDim.x){
	if (i<M){
        if (j==0){
            d_NEW_LIST[i*(N+1)+N]=N;
            d_NEW_LIST_INDX[i*(N+1)+N]=N;
        }
        d_NEW_LIST[i*(N+1)+j]=j;
        d_NEW_LIST_INDX[i*(N+1)+j]=j;
	}
    }
    }
}
__global__ void ANT_COST_CALCULATION_LS(int *ROUTE,int *COST,float *NODE_COORDINATE_2D,int *ROUTE_AUX,curandState *state){
    int i=threadIdx.x+ (blockIdx.x * blockDim.x);
    if( i<M){
        int j;
        int SUMA=0;
        for (j=0;j<N;j++){
            SUMA+=EUC_2D(NODE_COORDINATE_2D,ROUTE[i*(N+1)+j],ROUTE[i*(N+1)+(j+1)%N]);
        }
        COST[i]=SUMA;
    }
}
__global__ void ANT_COST_CALCULATION_FACO(int *ROUTE,int *COST,float *NODE_COORDINATE_2D,int *ROUTE_AUX,int *POS_IN_ROUTE,int *LS_CHECKLIST
		,int *NN_LIST,curandState *state){
    int i=threadIdx.x+ (blockIdx.x * blockDim.x);
    if( i<M){
        int j;
        int SUMA=0;
        for (j=0;j<N;j++){
            SUMA+=EUC_2D(NODE_COORDINATE_2D,ROUTE[i*(N+1)+j],ROUTE[i*(N+1)+(j+1)%N]);
        }
        COST[i]=SUMA;
	//OPT_2_FACO(ROUTE, POS_IN_ROUTE, COST, LS_CHECKLIST, NN_LIST,NODE_COORDINATE_2D,i);

    }
}
__global__ void PHEROMONE_UPDATE(int *ROUTE,int *BEST_ANT,float *PHEROMONE_MATRIX,int *NN_LIST,int *COST,int *OPTIMAL_ROUTE,int BEST_GLOBAL_SOLUTION){
    int j=threadIdx.x+ (blockIdx.x * blockDim.x);
    if (j<N){
    int ant,k;
    for (ant=0;ant<n_best-1;ant++)
        for (k=0;k<cl;k++){
            if (ROUTE[BEST_ANT[ant]*(N+1)+(j+1)%N]==NN_LIST[ROUTE[BEST_ANT[ant]*(N+1)+j%N]*cl+k]){
                PHEROMONE_MATRIX[ROUTE[BEST_ANT[ant]*(N+1)+j%N]*cl+k]+=(1.0*(n_best-1-ant))/((float)COST[ant]);
            }
        }
    for (k=0;k<cl;k++){
        if (OPTIMAL_ROUTE[j+1]==NN_LIST[OPTIMAL_ROUTE[j]*cl+k]){
            PHEROMONE_MATRIX[OPTIMAL_ROUTE[j]*cl+k]+=(1.0*(n_best))/((float)BEST_GLOBAL_SOLUTION);
        }
    }
    }
}
__global__ void PHEROMONE_UPDATE_AS(int *ROUTE,int *BEST_ANT,float *PHEROMONE_MATRIX,int *NN_LIST,int *COST,int *OPTIMAL_ROUTE,int BEST_GLOBAL_SOLUTION){
    int j=threadIdx.x+ (blockIdx.x * blockDim.x);
    if (j<N){
    int ant,k;
    for (ant=0;ant<M;ant++)
        for (k=0;k<cl;k++){
            if (ROUTE[BEST_ANT[ant]*(N+1)+(j+1)%N]==NN_LIST[ROUTE[BEST_ANT[ant]*(N+1)+j%N]*cl+k]){
                PHEROMONE_MATRIX[ROUTE[BEST_ANT[ant]*(N+1)+j%N]*cl+k]+=(1.0)/((float)COST[ant]);
            }
        }
    }
}
__global__ void PHEROMONE_UPDATE_MMAS(int *ROUTE,int *BEST_ANT,
float *PHEROMONE_MATRIX,int *NN_LIST,int *COST,int *OPTIMAL_ROUTE,
int BEST_GLOBAL_SOLUTION,int update_flag){
    int j=threadIdx.x+ (blockIdx.x * blockDim.x);
    if (j<N){
	int k;
	if ( 0 == update_flag){
	    for (k=0;k<cl;k++){
		if (OPTIMAL_ROUTE[j+1]==NN_LIST[OPTIMAL_ROUTE[j]*cl+k]){
		    PHEROMONE_MATRIX[OPTIMAL_ROUTE[j]*cl+k]+=1.0/((float)BEST_GLOBAL_SOLUTION);
		}
	    }
	}
	else if ( 1 == update_flag ){
	    int actual_node = ROUTE[BEST_ANT[0]*(N+1)+j%N];
	    int next_node = ROUTE[BEST_ANT[0]*(N+1)+(j+1)%N];
            for (k=0;k<cl;k++){
            	if (next_node == NN_LIST[actual_node*cl+k]){
                	PHEROMONE_MATRIX[actual_node*cl+k]+=1.0/((float)COST[0]);

            	}

            }
	    	
	}
    }
}
__global__ void PHEROMONE_CHECK_MMAS(float *PHEROMONE_MATRIX,float tau_max,float tau_min){
    int i=threadIdx.x+ (blockIdx.x * blockDim.x);
    if (i<N*cl){
        if (PHEROMONE_MATRIX[i]>tau_max)PHEROMONE_MATRIX[i]=tau_max;
        if (PHEROMONE_MATRIX[i]<tau_min)PHEROMONE_MATRIX[i]=tau_min;
    }
}
__global__ void EVAPORATION(float *PHEROMONE_MATRIX,float e){
    int i=threadIdx.x+ (blockIdx.x * blockDim.x);
    if (i<N*cl){
        if (PHEROMONE_MATRIX[i]*(1-e)<FLT_MIN){
        }
        else{
             PHEROMONE_MATRIX[i]=PHEROMONE_MATRIX[i]*(1-e);
        }
    }
}
__global__ void HEURISTIC_PHEROMONE_CALCULATION(float *NODE_COORDINATE,float *PHEROMONE_MATRIX,float *HEURISTIC_PHEROMONE
,int *NN_LIST,float alpha,float beta){
    int i=blockIdx.x;
    int j=threadIdx.x; 
    if ((i<N) && (j<cl)){    
        float H=1.0/(float)EUC_2D(NODE_COORDINATE,i,NN_LIST[i*cl+j]);
        HEURISTIC_PHEROMONE[i*cl+j]=powf(H,beta)*powf(PHEROMONE_MATRIX[i*cl+j],alpha);
    }
}
__global__ void RESET_ROUTE(int *ROUTE){
    int i=threadIdx.x+ (blockIdx.x * blockDim.x);

    if (i<(N+1)*M){
        ROUTE[i]=0;
    }
}
__global__ void RESET_VISITED_LIST(bool *VISITED_LIST){
    int i=threadIdx.x+ (blockIdx.x * blockDim.x);
    if (i<N*M){
        VISITED_LIST[i]=false;
    }
}
__global__ void fijar_pheromone(float *d_pheromone,float ini_pheromone){
    int i=threadIdx.x+ (blockIdx.x * blockDim.x); //cl
    if (i<cl*N){
        d_pheromone[i]=((float) 1)/ini_pheromone;
    }
}
__device__ int EUC_2D(float *d_d,int p1,int p2){
    float dx,dy;
    dx=(d_d[p1*2+0]-d_d[p2*2+0]);
    dy=(d_d[p1*2+1]-d_d[p2*2+1]);
    return (int)(sqrt(dx*dx+dy*dy)+0.5);                
}
void make_candidate_list(int *d_NN_LIST_aux,int *d_DISTANCE_NODE,int *DISTANCE_NODE,float *NODE_COORDINATE_2D,int *NN_LIST_cl){
    int i,j;
    thrust::device_ptr<int> dev_inx_d = thrust::device_pointer_cast(d_NN_LIST_aux);
 
    thrust::device_ptr<int> dev_d = thrust::device_pointer_cast(d_DISTANCE_NODE);
    for (i=0;i<N;i++){
        if (i%1000==0){
            printf("%d \n",i);
        }
        for (j=0;j<N;j++){
            DISTANCE_NODE[j]= EUC_2D_C(NODE_COORDINATE_2D,i,j); 
        }
        cudaMemcpy(d_DISTANCE_NODE,DISTANCE_NODE,N*sizeof(int),cudaMemcpyHostToDevice);
        thrust::sequence(thrust::device,dev_inx_d, dev_inx_d+N); //  HERE, WE SORT THE DISTANCES FOR THE NN_LIST
        thrust::sort_by_key(thrust::device,dev_d, dev_d +N, dev_inx_d,thrust::less<int>());
        cudaDeviceSynchronize();
        cudaMemcpy(NN_LIST_cl+i*cl,d_NN_LIST_aux+1,cl*sizeof(int),cudaMemcpyDeviceToHost);
        
        //cudaMemcpy(NN_LIST_cl,d_NN_LIST_aux+1,cl*N*sizeof(int),cudaMemcpyDeviceToHost);
        
    }
}

void UPGRADE_PHEROMONE(int *ROUTE,int *BEST_ANT,float *PHEROMONE_MATRIX,int *NN_LIST,int *COST,int *OPTIMAL_ROUTE,int BEST_GLOBAL_SOLUTION,
		float tau_max,float tau_min,float e,int ACO_flag){
	EVAPORATION<<<((N*cl+32-(N*cl%32)))/32,32>>>(PHEROMONE_MATRIX,e);
	if (ACO_flag == 0){
	    // RBAS
            PHEROMONE_UPDATE<<<((N+32-(N%32)))/32,32>>>(ROUTE,BEST_ANT,PHEROMONE_MATRIX,NN_LIST,COST,OPTIMAL_ROUTE,BEST_GLOBAL_SOLUTION); 
	}
	else if (ACO_flag == 1){	
            // MMAS
	    srand(time(NULL));
	    float randu = (float)rand()/(float)RAND_MAX; 
            int MMAS_flag = (randu<0.1) ? 1 : 0 ;
	    PHEROMONE_CHECK_MMAS<<<((N*cl+32-(N*cl%32)))/32,32>>>(PHEROMONE_MATRIX, tau_max, tau_min);
            
	    PHEROMONE_UPDATE_MMAS<<<((N+32-(N%32)))/32,32>>>(ROUTE,BEST_ANT,PHEROMONE_MATRIX,NN_LIST,COST,OPTIMAL_ROUTE,BEST_GLOBAL_SOLUTION,MMAS_flag); 

	    PHEROMONE_CHECK_MMAS<<<((N*cl+32-(N*cl%32)))/32,32>>>(PHEROMONE_MATRIX, tau_max, tau_min);
	}
	else if (ACO_flag == 2){
	    // AS
	    PHEROMONE_UPDATE_AS<<<((N+32-(N%32)))/32,32>>>(ROUTE,BEST_ANT,PHEROMONE_MATRIX,NN_LIST,COST,OPTIMAL_ROUTE,BEST_GLOBAL_SOLUTION); 

	}
	else{
		printf("\n Not suported ACO \n");
		exit(1);
	}
}
__device__ void swap(int *node_1,int *node_2){
	int temp = *node_1;
	*node_1 = *node_2;
	*node_2 = temp;
}
__device__ void make_swap_move_route(int current_change_x1,int current_change_x2,int *ROUTE,int *POS_IN_ROUTE,
		int ROUTE_OFFSET,int POS_IN_R_OFFSET){
	int current_node_x1 = ROUTE[ROUTE_OFFSET+current_change_x1];
	int current_node_x2 = ROUTE[ROUTE_OFFSET+current_change_x2];
	swap(&ROUTE[ROUTE_OFFSET+current_change_x1],&ROUTE[ROUTE_OFFSET+current_change_x2]);
	swap(&POS_IN_ROUTE[POS_IN_R_OFFSET+current_node_x1],&POS_IN_ROUTE[POS_IN_R_OFFSET+current_node_x2]);
}
__device__ void OPT_2_FACO(int *ROUTE, int *POS_IN_ROUTE_ANT, int *COST, int *LS_CHECKLIST, int *NN_LIST,float *NODE_COORDINATE_2D,int ANT){
	/*make a local seach using the LS_CHECKLIST where the new edges are stored,
	in this case using 2-OPT
	 */		
	int i,j;
	int LS_OFFSET = ANT*(N+1);
	int ROUTE_OFFSET = ANT*(N+1);
	int POS_IN_R_OFFSET = ANT*N;
	int pos_in_route_n_move_1,pos_in_route_n_move_2;
	int node_X1, node_X2,pos_in_route_x1,pos_in_route_x2;
	int d0,d1,gain,x1_succ_distance,x1_x2_distance,x1_pred_distance;
	int pos_x1_succ,pos_x1_pred,pos_x2_succ,pos_x2_pred;
	for (i = 0; i < LS_CHECKLIST[N]; i++){
		int move[2] = {-1,-1};
		int flag_2opt = -1; // = -1 no opt move, = 0 succ move, = 1 pred move 
		gain = 0;
		node_X1 = LS_CHECKLIST[LS_OFFSET+i];
		pos_in_route_x1 = POS_IN_ROUTE_ANT[POS_IN_R_OFFSET+node_X1];
		pos_x1_pred = (pos_in_route_x1+1)%N;
		pos_x1_succ = (pos_in_route_x1 == 0) ? N-1 : pos_in_route_x1-1;
		for (j = 0; j < cl; j++){
			node_X2 = NN_LIST[node_X1*cl+j]; 
			pos_in_route_x2 = POS_IN_ROUTE_ANT[POS_IN_R_OFFSET+node_X2];
			pos_x2_succ = (pos_in_route_x2 == 0) ?  N-1 : pos_in_route_x2-1;
			x1_succ_distance=EUC_2D(NODE_COORDINATE_2D,ROUTE[ROUTE_OFFSET+pos_in_route_x1],ROUTE[ROUTE_OFFSET+pos_x1_succ]);
			x1_x2_distance=EUC_2D(NODE_COORDINATE_2D,ROUTE[ROUTE_OFFSET+pos_in_route_x1],ROUTE[ROUTE_OFFSET+pos_in_route_x2]);
			if (x1_succ_distance > x1_x2_distance){
				d0 = EUC_2D(NODE_COORDINATE_2D,ROUTE[ROUTE_OFFSET+pos_in_route_x1],ROUTE[ROUTE_OFFSET+pos_x1_succ])+
				EUC_2D(NODE_COORDINATE_2D,ROUTE[ROUTE_OFFSET+pos_in_route_x2],ROUTE[ROUTE_OFFSET+pos_x2_succ]);
				d1 = EUC_2D(NODE_COORDINATE_2D,ROUTE[ROUTE_OFFSET+pos_in_route_x1],ROUTE[ROUTE_OFFSET+pos_in_route_x2])+
				EUC_2D(NODE_COORDINATE_2D,ROUTE[ROUTE_OFFSET+pos_x1_succ],ROUTE[ROUTE_OFFSET+pos_x2_succ]);
				if (d0 - d1 > gain){
					gain = d0 - d1;
					move[0] = node_X1;move[1] = node_X2;
					flag_2opt = 0;
				}
			} 
		}
		for (j = 0; j < cl; j++){
			node_X2 = NN_LIST[node_X1*cl+j]; 
			pos_in_route_x2 = POS_IN_ROUTE_ANT[POS_IN_R_OFFSET+node_X2];
			pos_x2_pred = (pos_in_route_x2+1)%N;
			x1_pred_distance=EUC_2D(NODE_COORDINATE_2D,ROUTE[ROUTE_OFFSET+pos_in_route_x1],ROUTE[ROUTE_OFFSET+pos_x1_pred]);
			x1_x2_distance=EUC_2D(NODE_COORDINATE_2D,ROUTE[ROUTE_OFFSET+pos_in_route_x1],ROUTE[ROUTE_OFFSET+pos_in_route_x2]);
			if (x1_pred_distance > x1_x2_distance){
				d0 = EUC_2D(NODE_COORDINATE_2D,ROUTE[ROUTE_OFFSET+pos_in_route_x1],ROUTE[ROUTE_OFFSET+pos_x1_pred])+
				EUC_2D(NODE_COORDINATE_2D,ROUTE[ROUTE_OFFSET+pos_in_route_x2],ROUTE[ROUTE_OFFSET+pos_x2_pred]);
				d1 = EUC_2D(NODE_COORDINATE_2D,ROUTE[ROUTE_OFFSET+pos_in_route_x1],ROUTE[ROUTE_OFFSET+pos_in_route_x2])+
				EUC_2D(NODE_COORDINATE_2D,ROUTE[ROUTE_OFFSET+pos_x1_pred],ROUTE[ROUTE_OFFSET+pos_x2_pred]);
				if (d0 - d1 > gain){
					gain = d0 - d1;
					move[0] = node_X1;move[1] = node_X2;
					flag_2opt = 1;
				}
			} 
		}
		if (flag_2opt != -1){
			int k;
			pos_in_route_n_move_1=POS_IN_ROUTE_ANT[POS_IN_R_OFFSET+move[0]];
			pos_in_route_n_move_2=POS_IN_ROUTE_ANT[POS_IN_R_OFFSET+move[1]];
			if (pos_in_route_n_move_1 > pos_in_route_n_move_2) swap(&pos_in_route_n_move_1,&pos_in_route_n_move_2);
			int delta = floor((float) (pos_in_route_n_move_2-pos_in_route_n_move_1)/2.0);
			if (flag_2opt == 0){
				for (k = 0; k < delta; k++){
					int current_change_x2 = pos_in_route_n_move_2-k-1; 
					int current_change_x1 = pos_in_route_n_move_1+k; 
					make_swap_move_route(current_change_x1,current_change_x2,ROUTE,POS_IN_ROUTE_ANT,
					ROUTE_OFFSET,POS_IN_R_OFFSET);	
				}
				COST[ANT]-=gain;
			}
			if (flag_2opt == 1){
				for (k = 0; k < delta; k++){
					int current_change_x2 = pos_in_route_n_move_2-k; 
					int current_change_x1 = pos_in_route_n_move_1+k+1; 
					make_swap_move_route(current_change_x1,current_change_x2,ROUTE,POS_IN_ROUTE_ANT,
					ROUTE_OFFSET,POS_IN_R_OFFSET);	
				}
				COST[ANT]-=gain;
				
			}
		}
	}
}
void swap_c(int *node_1,int *node_2){
	int temp = *node_1;
	*node_1 = *node_2;
	*node_2 = temp;
}
void make_swap_move_route_c(int current_change_x1,int current_change_x2,int *ROUTE,int *POS_IN_ROUTE,
		int ROUTE_OFFSET,int POS_IN_R_OFFSET){
	int current_node_x1 = ROUTE[ROUTE_OFFSET+current_change_x1];
	int current_node_x2 = ROUTE[ROUTE_OFFSET+current_change_x2];
	swap_c(&ROUTE[ROUTE_OFFSET+current_change_x1],&ROUTE[ROUTE_OFFSET+current_change_x2]);
	swap_c(&POS_IN_ROUTE[POS_IN_R_OFFSET+current_node_x1],&POS_IN_ROUTE[POS_IN_R_OFFSET+current_node_x2]);
}
void OPT_2_nn(int *ROUTE, int *POS_IN_ROUTE_ANT, int *COST,int *NN_LIST,float *NODE_COORDINATE_2D,int ANT){
	/*make a local seach using the LS_CHECKLIST where the new edges are stored,
	in this case using 2-OPT
	 */		
	int i,j;
	int ROUTE_OFFSET = 0;
	int POS_IN_R_OFFSET = 0;
	int pos_in_route_n_move_1,pos_in_route_n_move_2;
	int node_X1, node_X2,pos_in_route_x1,pos_in_route_x2;
	int d0,d1,gain,x1_succ_distance,x1_x2_distance,x1_pred_distance;
	int pos_x1_succ,pos_x1_pred,pos_x2_succ,pos_x2_pred;
	for (i = 0; i < N; i++){
		int move[2] = {-1,-1};
		int flag_2opt = -1; // = -1 no opt move, = 0 succ move, = 1 pred move 
		gain = 0;
		node_X1 = i;
		pos_in_route_x1 = POS_IN_ROUTE_ANT[POS_IN_R_OFFSET+node_X1];
		pos_x1_pred = (pos_in_route_x1+1)%N;
		pos_x1_succ = (pos_in_route_x1 == 0) ? N-1 : pos_in_route_x1-1;
		for (j = 0; j < cl; j++){
			node_X2 = NN_LIST[node_X1*cl+j]; 
			pos_in_route_x2 = POS_IN_ROUTE_ANT[POS_IN_R_OFFSET+node_X2];
			pos_x2_succ = (pos_in_route_x2 == 0) ?  N-1 : pos_in_route_x2-1;
			x1_succ_distance=EUC_2D_C(NODE_COORDINATE_2D,ROUTE[ROUTE_OFFSET+pos_in_route_x1],ROUTE[ROUTE_OFFSET+pos_x1_succ]);
			x1_x2_distance=EUC_2D_C(NODE_COORDINATE_2D,ROUTE[ROUTE_OFFSET+pos_in_route_x1],ROUTE[ROUTE_OFFSET+pos_in_route_x2]);
			if (x1_succ_distance > x1_x2_distance){
				d0 = EUC_2D_C(NODE_COORDINATE_2D,ROUTE[ROUTE_OFFSET+pos_in_route_x1],ROUTE[ROUTE_OFFSET+pos_x1_succ])+
				EUC_2D_C(NODE_COORDINATE_2D,ROUTE[ROUTE_OFFSET+pos_in_route_x2],ROUTE[ROUTE_OFFSET+pos_x2_succ]);
				d1 = EUC_2D_C(NODE_COORDINATE_2D,ROUTE[ROUTE_OFFSET+pos_in_route_x1],ROUTE[ROUTE_OFFSET+pos_in_route_x2])+
				EUC_2D_C(NODE_COORDINATE_2D,ROUTE[ROUTE_OFFSET+pos_x1_succ],ROUTE[ROUTE_OFFSET+pos_x2_succ]);
				if (d0 - d1 > gain){
					gain = d0 - d1;
					move[0] = node_X1;move[1] = node_X2;
					flag_2opt = 0;
				}
			} 
		}
		for (j = 0; j < cl; j++){
			node_X2 = NN_LIST[node_X1*cl+j]; 
			pos_in_route_x2 = POS_IN_ROUTE_ANT[POS_IN_R_OFFSET+node_X2];
			pos_x2_pred = (pos_in_route_x2+1)%N;
			x1_pred_distance=EUC_2D_C(NODE_COORDINATE_2D,ROUTE[ROUTE_OFFSET+pos_in_route_x1],ROUTE[ROUTE_OFFSET+pos_x1_pred]);
			x1_x2_distance=EUC_2D_C(NODE_COORDINATE_2D,ROUTE[ROUTE_OFFSET+pos_in_route_x1],ROUTE[ROUTE_OFFSET+pos_in_route_x2]);
			if (x1_pred_distance > x1_x2_distance){
				d0 = EUC_2D_C(NODE_COORDINATE_2D,ROUTE[ROUTE_OFFSET+pos_in_route_x1],ROUTE[ROUTE_OFFSET+pos_x1_pred])+
				EUC_2D_C(NODE_COORDINATE_2D,ROUTE[ROUTE_OFFSET+pos_in_route_x2],ROUTE[ROUTE_OFFSET+pos_x2_pred]);
				d1 = EUC_2D_C(NODE_COORDINATE_2D,ROUTE[ROUTE_OFFSET+pos_in_route_x1],ROUTE[ROUTE_OFFSET+pos_in_route_x2])+
				EUC_2D_C(NODE_COORDINATE_2D,ROUTE[ROUTE_OFFSET+pos_x1_pred],ROUTE[ROUTE_OFFSET+pos_x2_pred]);
				if (d0 - d1 > gain){
					gain = d0 - d1;
					move[0] = node_X1;move[1] = node_X2;
					flag_2opt = 1;
				}
			} 
		}
		if (flag_2opt != -1){
			int k;
			pos_in_route_n_move_1=POS_IN_ROUTE_ANT[POS_IN_R_OFFSET+move[0]];
			pos_in_route_n_move_2=POS_IN_ROUTE_ANT[POS_IN_R_OFFSET+move[1]];
			if (pos_in_route_n_move_1 > pos_in_route_n_move_2) swap_c(&pos_in_route_n_move_1,&pos_in_route_n_move_2);
			int delta = floor((float) (pos_in_route_n_move_2-pos_in_route_n_move_1)/2.0);
			if (flag_2opt == 0){
				for (k = 0; k < delta; k++){
					int current_change_x2 = pos_in_route_n_move_2-k-1; 
					int current_change_x1 = pos_in_route_n_move_1+k; 
					make_swap_move_route_c(current_change_x1,current_change_x2,ROUTE,POS_IN_ROUTE_ANT,
					ROUTE_OFFSET,POS_IN_R_OFFSET);	
				}
				COST[ANT]-=gain;
			}
			if (flag_2opt == 1){
				for (k = 0; k < delta; k++){
					int current_change_x2 = pos_in_route_n_move_2-k; 
					int current_change_x1 = pos_in_route_n_move_1+k+1; 
					make_swap_move_route_c(current_change_x1,current_change_x2,ROUTE,POS_IN_ROUTE_ANT,
					ROUTE_OFFSET,POS_IN_R_OFFSET);	
				}
				COST[ANT]-=gain;
				
			}
		}
	}
}
