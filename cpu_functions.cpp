#include "mgpu.h"

float shannon_entropy_p_r(float *PHEROMONE_MATRIX,int *ROUTE,int *NN_LIST,float *PROB_ROUTE,float last_entropy,float *ENTROPY_ITERATION,int it){
	int i,j,k;
	float global_sum=0.0;
	float sum_phero;
	int NN_neighbor;
	for (i=0;i<N_GPU*M;i++){
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
		global_sum+=sum_phero;
	}
	float aux_cumulative=0.0;
	float entropy_shannon=0.0;
	for (i=0;i<N_GPU*M;i++){
		PROB_ROUTE[i]/=global_sum;
		entropy_shannon+=PROB_ROUTE[i]*log2(PROB_ROUTE[i]);
	}
	entropy_shannon*=-1.0;
	float H=abs((entropy_shannon-last_entropy)/last_entropy);
	ENTROPY_ITERATION[it]=H;
	printf("%0.9f \n",H);
	return entropy_shannon;
}


int rutainicial_2(int *rute_op,float *d,bool *lista_vis){
    rute_op[0]= rand()%(N);
    
    printf("\n el random es %d \n",rute_op[0]);
    int cost=0;
    for (int i=0;i<N-1;i++){
        lista_vis[rute_op[i]]=true;
        int min=100000000;
        int mink=0;
        if (i%1000==0)printf("\n voy en el nodo %d \n",i);
        for (int j=0;j<N;j++){
            if (rute_op[i]!=j){
                if (lista_vis[j]==false){
		if (EUC_2D_C(d,rute_op[i],j)<min){
                    min=EUC_2D_C(d,rute_op[i],j);
                
		    mink=j;
                }
            }
            }
        }
        rute_op[i+1]=mink;
        cost+=EUC_2D_C(d,rute_op[i],mink);
    }
    cost+=EUC_2D_C(d,rute_op[N-1],rute_op[0]);
    printf("\n");
    rute_op[N]=rute_op[0];
    for (int i=0;i<N+1;i++)printf("%d ",rute_op[i%N]);
    printf("\n el coste incial es %d \n",cost);
    return cost;
}
void lectura_2(float *dis){
    FILE* f;
    int height, width, ii, jj;
    char buff[100]; 
    std::string file_name = name_e;
    file_name+=problem;
    file_name+=".tsp";
    if((f = fopen(file_name.c_str(), "r")) == NULL){
        exit(1);    
    }

    for (ii=0;ii<6;ii++){
        fscanf(f,"%[^\n]\n",&buff);
        printf("%s\n" ,buff);
    } 
    for(ii=0; ii<N; ii++)
        if(fscanf(f, "%*i %f %f",&dis[ii*2+0],&dis[ii*2+1]) != 2)
            exit(1);
    fclose(f);
}
void guardar_resultados(float *vec_warm_up_time,int *vec_solution,float *vec_ant_iteration_time_series,float *vec_iteration_time,float alpha,float beta,float e ){
    guardar_warm_up(vec_warm_up_time,alpha,beta,e);guardar_soluciones(vec_solution,alpha,beta,e);
    guardar_iteration_time_series(vec_ant_iteration_time_series,alpha,beta,e);
    guardar_iteration_time(vec_iteration_time,alpha,beta,e);

}
void guardar_entropias_pheromone(float *ENTROPY_VECTOR_PHEROMONE,float *ENTROPY_VECTOR_PHEROMONE_H,float alpha,float beta, float e,int it,int x){
    FILE *file1,*file2;
    std::string file_name_1 = name_test_8;
    std::string file_name_2 = name_test_8;
    file_name_1+=problem;
    file_name_2+=problem;
    file_name_2+="_H";
    int i;
    file_name_1+=".csv";
    file_name_2+=".csv";
    file1 = fopen(file_name_1.c_str(), "a");
    file2 = fopen(file_name_2.c_str(), "a");
    if(file1 == NULL)
    {
        printf("Error opening file, to write to it."); //archivo para guardar las iteraciones 
    }
    else
    {
        fprintf(file1,"%d,%d,%f,%f,%f,%d,%d",N,N_GPU*M,alpha,beta,e,it,x);
        fprintf(file2,"%d,%d,%f,%f,%f,%d,%d",N,N_GPU*M,alpha,beta,e,it,x);
	for (i=0;i<N;i++){
        	fprintf(file1,",%f",ENTROPY_VECTOR_PHEROMONE[i]);
        	fprintf(file2,",%f",ENTROPY_VECTOR_PHEROMONE_H[i]);
	} 
        fprintf(file1,"\n");
        fprintf(file2,"\n");
    }
    fclose(file1);
    fclose(file2);
	
}
void guardar_iteration_time(float *time,float alpha,float beta,float e){
    FILE *file1;
    std::string file_name = name_test_1;
    file_name+=problem;
    file_name+=".csv";
    file1 = fopen(file_name.c_str(), "a");
    if(file1 == NULL)
    {
        printf("Error opening file, to write to it."); //archivo para guardar las iteraciones 
    }
    else
    {
        fprintf(file1,"%d,%d,%f,%f,%f,",N,N_GPU*M,alpha,beta,e);
        for (int i=0;i<N_e;i++){
            if (i==0){
                fprintf(file1,"%f",time[i]);
            }
            else{
                fprintf(file1,",%f",time[i]);
            }
        }
        fprintf(file1,"\n");
    }
    fclose(file1);
}
void guardar_iteration_time_series(float *time,float alpha,float beta,float e){
    FILE *file1; 
    std::string file_name = name_test_2;
    file_name+=problem;
    file_name+=".csv";
    file1 = fopen(file_name.c_str(), "a");
    int i;
    if(file1 == NULL)
    {
        printf("Error opening file, to write to it."); //archivo para guardar las iteraciones 
    }
    else {
        fprintf(file1,"%d,%d,%f,%f,%f,",N,N_GPU*M,alpha,beta,e);
        for (i=1;i<ITERACION;i++){
            if (i==0){
                fprintf(file1,"%f",time[i]);
            }
            else{
                fprintf(file1,",%f",time[i]);
            }
        }
        fprintf(file1,"\n"); 
    }

}
void guardar_warm_up(float *time,float alpha,float beta,float e){
    FILE *file1;
    std::string file_name = name_test_3;
    file_name+=problem;
    file_name+=".csv";
    file1 = fopen(file_name.c_str(), "a");
    if(file1 == NULL)
    {
        printf("Error opening file, to write to it."); //archivo para guardar las iteraciones 
    }
    else
    {
        fprintf(file1,"%d,%d,%f,%f,%f,",N,N_GPU*M,alpha,beta,e);
        for (int i=0;i<N_e;i++){
            if (i==0){
                fprintf(file1,"%f",time[i]);
            }
            else{
                fprintf(file1,",%f",time[i]);
            }
        }
        fprintf(file1,"\n");
    }
    fclose(file1);
}
void guardar_soluciones(int *soluciones,float alpha,float beta,float e){
    FILE *file1;
    std::string file_name = name_test_4;
    file_name+=problem;
    file_name+=".csv";
    file1 = fopen(file_name.c_str(), "a");
    if(file1 == NULL)
    {
        printf("Error opening file, to write to it."); //archivo para guardar las iteraciones 
    }
    else
    {
        fprintf(file1,"%d,%d,%f,%f,%f,",N,N_GPU*M,alpha,beta,e);
        for (int i=0;i<N_e;i++){
            if (i==0){
                fprintf(file1,"%d",soluciones[i]);
            }
            else{
                fprintf(file1,",%d",soluciones[i]);
            }
        }
        fprintf(file1,"\n");
    }
    fclose(file1);
}
float promediar_int(int *vec){
    float sumaprom=0;
    for (int i=0;i<N_e;i++){
        sumaprom+=(float)vec[i];
    }
    return sumaprom/N_e;
}
float std_vec(float *vec, float prom){
    float suma_std=0;
    for (int i=0;i<N_e;i++){
        suma_std+=powf(vec[i]-prom,2);
    }
    suma_std=suma_std/N_e;
    suma_std=powf(suma_std,0.5);
    return suma_std;
}
float std_vec_it(int *vec, float prom){
    float suma_std=0;
    for (int i=0;i<N_e;i++){
        suma_std+=powf((float)vec[i]-prom,2);
    }
    suma_std=suma_std/N_e;
    suma_std=powf(suma_std,0.5);
    return suma_std;
}
int EUC_2D_C(float *d_d,int p1,int p2){
    float dx,dy;
    dx=(d_d[p1*2+0]-d_d[p2*2+0]);
    dy=(d_d[p1*2+1]-d_d[p2*2+1]);
    return (int)(sqrt(dx*dx+dy*dy)+0.5);                
}

void CHECK_VISITED_CPU(int *NEW_LIST,int *NEW_LIST_INDX,int ant,int CHOSEN_NODE){
    int length=NEW_LIST[ant*(N+1)+N]-1;
    int CHOSEN_INDX=NEW_LIST_INDX[ant*(N+1)+CHOSEN_NODE];
    int temp_1=NEW_LIST[ant*(N+1)+length];
    int temp_2=NEW_LIST_INDX[ant*(N+1)+temp_1]  ;
    NEW_LIST[ant*(N+1)+length]=CHOSEN_NODE;
    NEW_LIST_INDX[ant*(N+1)+temp_1]=CHOSEN_INDX;
    NEW_LIST[ant*(N+1)+CHOSEN_INDX]=temp_1;
    NEW_LIST_INDX[ant*(N+1)+CHOSEN_NODE]=temp_2;
    NEW_LIST[ant*(N+1)+N]-=1;
}

bool IS_VISITED_CPU(int *NEW_LIST,int *NEW_LIST_INDX,int ant,int j){
    if (NEW_LIST_INDX[ant*(N+1)+j]<NEW_LIST[ant*(N+1)+N]){
        return false;
    }
    else{
        return true;
    }
}
int GET_CANDIDATE_CPU(int *NEW_LIST,int *NEW_LIST_INDX,int ant,int j){
    return NEW_LIST[ant*(N+1)+j];
}
int rutainicial(int *rute_op,float *d,int *NEW_LIST_GLOBAL,int *NEW_LIST_INDX_GLOBAL,int *NN_LIST,int *POS_IN_ROUTE){
    int i,j;
    for (i=0;i<N+1;i++){
	NEW_LIST_GLOBAL[i]=i;
	NEW_LIST_INDX_GLOBAL[i]=i;
    }
    srand(0);
    int initial_node = rand()%(N);
    CHECK_VISITED_CPU(NEW_LIST_GLOBAL,NEW_LIST_INDX_GLOBAL,0,initial_node); 
    int cost=0;
    int candidate,candidate_distance,next_node,current_node;
    current_node=initial_node;
    for (i=0;i<N-1;i++){
	next_node=N;
	if (i%1000==0)printf("\n voy en el nodo %d \n",i); 	
	for (j=0;j<cl;j++){
		candidate=NN_LIST[current_node*cl+j];
		if (!IS_VISITED_CPU(NEW_LIST_GLOBAL,NEW_LIST_INDX_GLOBAL,0,candidate)){
			next_node=candidate;
			candidate_distance=EUC_2D_C(d,current_node,candidate);
			break;
		}
	}
	if (next_node==N){
		next_node=GET_CANDIDATE_CPU(NEW_LIST_GLOBAL,NEW_LIST_INDX_GLOBAL,0,0);
		int min_distance=EUC_2D_C(d,current_node,next_node);
		for (j<0;j<NEW_LIST_GLOBAL[N];j++){
			candidate=GET_CANDIDATE_CPU(NEW_LIST_GLOBAL,NEW_LIST_INDX_GLOBAL,0,j);
			candidate_distance=EUC_2D_C(d,candidate,current_node);
			if (candidate_distance<min_distance){
				next_node=candidate;
				min_distance=candidate_distance;
			}
		}
	}
	CHECK_VISITED_CPU(NEW_LIST_GLOBAL,NEW_LIST_INDX_GLOBAL,0,next_node);
        cost+=EUC_2D_C(d,current_node,next_node);
	current_node=next_node;
    }
    cost+=EUC_2D_C(d,next_node,initial_node);
    for (i=0;i<N+1;i++){
	    rute_op[i]=NEW_LIST_GLOBAL[i%N];
	    POS_IN_ROUTE[rute_op[i]]=i;
    }
    printf("\n el coste incial es %d \n",cost);
    for (i=0;i<N;i++)printf("%d ",rute_op[i]);
    return cost;
}

int mejorhormiga(float *cost,int *best_horm,float *best_cost){
    float min=cost[0];
    int horm_op;
    for (int i=0;i<M;i++){
        best_horm[i]=i;
        best_cost[i]=cost[i];
        if (cost[i]<=min){
            min=cost[i];
            horm_op=i;
        }
    }
    float temp;
    int temp1;
    for(int i = 0; i<M; i++) {
        for(int j = i+1; j<M; j++)
        {
            if(best_cost[j] < best_cost[i]) {
                temp1=best_horm[i];
            best_horm[i]=best_horm[j];
            best_horm[j]=temp1; 
            temp = best_cost[i];
                best_cost[i] = best_cost[j];
                best_cost[j] = temp;
            }
        }
    }
    return horm_op;
}
float minimovec(float *vec){
    float min=vec[0];
    for (int i=0;i<N_e;i++){
        if (min>=vec[i]){
            min=vec[i];
        }
    }
    return min;
}
float promediar(float *vec){
    float sumaprom=0;
    for (int i=0;i<N_e;i++){
        sumaprom+=vec[i];
    }
    return sumaprom/N_e;
}
float opt33(int *rute_op,int *rute_op_aux,float *d,float global_sol){
    
    int pos[3]={};
    for (int i=0;i<3;i++){
        int aux =0;
        while (aux==0){
            aux=1;
            int random = (int) rand()%(N-2)+1;
            for (int j=0;j<3;j++){
                if (random==pos[j]){
                    aux=0;
                }
            }
            if (aux==1){
                pos[i]=random;
            }
        }
    }
    int temp;
    if (pos[0]>pos[1]){
        temp=pos[1];
        pos[1]=pos[0];
        pos[0]=temp;
    } 
    if (pos[0]>pos[2]){
        temp=pos[2];
        pos[2]=pos[0];
        pos[0]=temp;
    }
    if (pos[1]>pos[2]){
        temp=pos[2];
        pos[2]=pos[1];
        pos[1]=temp;
    }  
    //printf("\n array \n %d %d %d",pos[0],pos[1],pos[2]);
    int delta0=pos[1]-pos[0];int delta1=pos[2]-pos[1];int aux=0;
    
    float d0,d1,d2,d3,d4,d5,d6,d7,change1,change2,change3,change4,change5,change6,change7;
    d0=d[rute_op[pos[0]-1]*N+rute_op[pos[0]]]   +      d[rute_op[pos[1]-1]*N+rute_op[pos[1]]]   +    d[rute_op[pos[2]-1]*N+rute_op[pos[2]]];
    d1=d[rute_op[pos[0]-1]*N+rute_op[pos[1]-1]]   +      d[rute_op[pos[0]]*N+rute_op[pos[1]]]   +    d[rute_op[pos[2]-1]*N+rute_op[pos[2]]];change1=d1-d0;
    d2=d[rute_op[pos[2]-1]*N+rute_op[pos[0]-1]]   +      d[rute_op[pos[1]-1]*N+rute_op[pos[1]]]   +    d[rute_op[pos[2]]*N+rute_op[pos[0]]];change2=d2-d0;
    d3=d[rute_op[pos[0]-1]*N+rute_op[pos[0]]]   +      d[rute_op[pos[1]-1]*N+rute_op[pos[2]-1]]   +    d[rute_op[pos[1]]*N+rute_op[pos[2]]];change3=d3-d0;
    d4=d[rute_op[pos[0]-1]*N+rute_op[pos[1]]]   +      d[rute_op[pos[2]-1]*N+rute_op[pos[1]-1]]   +    d[rute_op[pos[0]]*N+rute_op[pos[2]]];change4=d4-d0;
    d5=d[rute_op[pos[0]-1]*N+rute_op[pos[1]-1]]   +      d[rute_op[pos[0]]*N+rute_op[pos[2]-1]]   +    d[rute_op[pos[1]]*N+rute_op[pos[2]]];change5=d5-d0;
    d6=d[rute_op[pos[2]-1]*N+rute_op[pos[0]-1]]   +      d[rute_op[pos[2]]*N+rute_op[pos[1]-1]]   +    d[rute_op[pos[0]]*N+rute_op[pos[1]]];change6=d6-d0;
    d7=d[rute_op[pos[1]]*N+rute_op[pos[0]-1]]   +      d[rute_op[pos[2]-1]*N+rute_op[pos[0]]]   +    d[rute_op[pos[1]-1]*N+rute_op[pos[2]]];change7=d7-d0;
    if (d0>d1){d0=d1;aux=1;}if (d0>d2){d0=d2;aux=2;}if (d0>d3){d0=d3;aux=3;}
    if (d0>d4){d0=d4;aux=4;}if (d0>d5){d0=d5;aux=5;}if (d0>d6){d0=d6;aux=6;}
    if (d0>d7){d0=d7;aux=7;}
    if (aux==1){
    for (int i=0;i<pos[0];i++)rute_op_aux[i]=rute_op[i]; 
    for (int i=0;i<delta0;i++)rute_op_aux[i+pos[0]]=rute_op[pos[1]-i-1];
    for (int i=0;i<delta1;i++)rute_op_aux[i+pos[0]+delta0]=rute_op[pos[1]+i];
    for (int i=pos[2];i<N+1;i++)rute_op_aux[i]=rute_op[i];
    //printf(" caso 1 a' b c\n");for (int i=0;i<N+1;i++)printf("%d ",rute_op_aux[i]+1);
    for (int i=0;i<N+1;i++)rute_op[i]=rute_op_aux[i];printf(" \n%f \n",global_sol+change1);
    return global_sol+change1;
    }if (aux==2){
    for (int i=0;i<N;i++)rute_op_aux[i]=1;
    for (int i=0;i<pos[0];i++)rute_op_aux[i]=rute_op[i]; 
    for (int i=0;i<delta1;i++)rute_op_aux[i+pos[0]]=rute_op[pos[2]-i-1];
    for (int i=0;i<delta0;i++)rute_op_aux[i+delta1+pos[0]]=rute_op[pos[1]-i-1];
    for (int i=pos[2];i<N+1;i++)rute_op_aux[i]=rute_op[i];
    //printf("\n caso 2 a b c' \n");for (int i=0;i<N+1;i++)printf("%d ",rute_op_aux[i]+1);
    for (int i=0;i<N+1;i++)rute_op[i]=rute_op_aux[i];printf(" \n%f \n",global_sol+change2);
    return global_sol+change2;
    }if (aux==3){
    for (int i=0;i<N+1;i++)rute_op_aux[i]=1;
    for (int i=0;i<pos[0];i++)rute_op_aux[i]=rute_op[i]; 
    for (int i=0;i<delta0;i++)rute_op_aux[i+pos[0]]=rute_op[pos[0]+i];
    for (int i=0;i<delta1;i++)rute_op_aux[i+pos[0]+delta0]=rute_op[pos[2]-i-1];
    for (int i=pos[2];i<N+1;i++)rute_op_aux[i]=rute_op[i];
    //printf("\n caso 3 a b' c \n");for (int i=0;i<N+1;i++)printf("%d ",rute_op_aux[i]+1);
    for (int i=0;i<N+1;i++)rute_op[i]=rute_op_aux[i];printf(" \n%f \n",global_sol+change3);
    return global_sol+change3;
    }if (aux==4){
    for (int i=0;i<N+1;i++)rute_op_aux[i]=1;
    for (int i=0;i<pos[0];i++)rute_op_aux[i]=rute_op[i]; 
    for (int i=0;i<delta1;i++)rute_op_aux[i+pos[0]]=rute_op[pos[1]+i];
    for (int i=0;i<delta0;i++)rute_op_aux[i+pos[0]+delta1]=rute_op[pos[1]-i-1];
    for (int i=pos[2];i<N+1;i++)rute_op_aux[i]=rute_op[i];
    //printf("\n caso 4 a' c b \n");for (int i=0;i<N+1;i++)printf("%d ",rute_op_aux[i]+1);
    for (int i=0;i<N+1;i++)rute_op[i]=rute_op_aux[i];printf(" \n%f \n",global_sol+change4);
    return global_sol+change4;
    }if (aux==5){
    for (int i=0;i<N+1;i++)rute_op_aux[i]=1;
    for (int i=0;i<pos[0];i++)rute_op_aux[i]=rute_op[i]; 
    for (int i=0;i<delta0;i++)rute_op_aux[i+pos[0]]=rute_op[pos[1]-i-1];
    for (int i=0;i<delta1;i++)rute_op_aux[i+pos[0]+delta0]=rute_op[pos[2]-i-1];
    for (int i=pos[2];i<N+1;i++)rute_op_aux[i]=rute_op[i];
    //printf("\n caso 5 a' b' c \n");for (int i=0;i<N+1;i++)printf("%d ",rute_op_aux[i]+1);
    for (int i=0;i<N+1;i++)rute_op[i]=rute_op_aux[i];printf(" \n%f \n",global_sol+change5);
    return global_sol+change5;
    }if (aux==6){
    for (int i=0;i<N+1;i++)rute_op_aux[i]=1;
    for (int i=0;i<pos[0];i++)rute_op_aux[i]=rute_op[i]; 
    for (int i=0;i<delta1;i++)rute_op_aux[i+pos[0]]=rute_op[pos[2]-i-1];
    for (int i=0;i<delta0;i++)rute_op_aux[i+pos[0]+delta1]=rute_op[pos[0]+i];
    for (int i=pos[2];i<N+1;i++)rute_op_aux[i]=rute_op[i];
    //printf("\n caso 6 a' b c \n");for (int i=0;i<N+1;i++)printf("%d ",rute_op_aux[i]+1);
    for (int i=0;i<N+1;i++)rute_op[i]=rute_op_aux[i];printf(" \n%f \n",global_sol+change6);
    return global_sol+change6;
    }if (aux==7){
    for (int i=0;i<N+1;i++)rute_op_aux[i]=1;
    for (int i=0;i<pos[0];i++)rute_op_aux[i]=rute_op[i]; 
    for (int i=0;i<delta1;i++)rute_op_aux[i+pos[0]]=rute_op[pos[1]+i];
    for (int i=0;i<delta0;i++)rute_op_aux[i+pos[0]+delta1]=rute_op[pos[0]+i];
    for (int i=pos[2];i<N+1;i++)rute_op_aux[i]=rute_op[i];
    //printf("\n caso 7 a c b \n");for (int i=0;i<N+1;i++)printf("%d ",rute_op_aux[i]+1);
    for (int i=0;i<N+1;i++)rute_op[i]=rute_op_aux[i];printf(" \n%f \n",global_sol+change7);
    return global_sol+change7;
    }
    if (aux==0){
        //printf("no ha pasao na\n");
        //printf("d0 %f d1 %f d2 %f d3 %f d4 %f d5 %f d6 %f d7 %f",d0,d1,d2,d3,d4,d5,d6,d7);
        return global_sol;
    }
    //printf("\n");
    return global_sol;
}
void SAVE_PHEROMONE_MATRIX(float *PHEROMONE_MATRIX,int it, int experiment,float alpha,float beta,float e){
    FILE *file1;
    std::string file_name = name_test_7;
    file_name+=problem;
    file_name+="_"+std::to_string(alpha)+"_";
    file_name+=std::to_string(beta)+"_";
    file_name+=".csv";
    file1 = fopen(file_name.c_str(), "a");
    if(file1 == NULL)
    {
        printf("Error opening file, to write to it."); //archivo para guardar las iteraciones 
    }
    else
    {
	int i,j;
        for (i=0;i<N;i++){
		fprintf(file1,"%d,%d,%d",experiment,it,i);
		for (j=0;j<cl;j++){
			fprintf(file1,",%.9f",PHEROMONE_MATRIX[i*cl+j]);
		}
        	fprintf(file1,"\n");
	}
        fprintf(file1,"\n");
    }
    fclose(file1);
}
void SAVE_LAST_IMRPOVED(int LAST_IMRPOVE_IT,int BEST_SOLUTION,int experiment){
    FILE *file1;
    std::string file_name = name_test_9;
    file_name+=problem;
    file_name+=".csv";
    file1 = fopen(file_name.c_str(), "a");
    if(file1 == NULL)
    {
        printf("Error opening file, to write to it."); //archivo para guardar las iteraciones 
    }
    else
    {
	fprintf(file1,"%d,%d,%d",experiment,BEST_SOLUTION,LAST_IMRPOVE_IT);
        fprintf(file1,"\n");
    }	
}
void escribir_costo(int *HORMIGAS_COSTOS,int x){
	FILE *file1;
        std::string file_name = name_test_5;
        file_name+=problem;
        file_name+=".csv";
	file1 = fopen(file_name.c_str(), "a");
	if(file1 == NULL){
        	printf("Error opening file, to write to it."); //archivo para guardar las iteraciones 
	}
	 else{
		 for (int it=0;it<ITERACION;it++){
     	            fprintf(file1,"%d,",x);
                    for (int i=0;i<4*M;i++){
	                if (i==0){
		          fprintf(file1,"%d,%d",it,HORMIGAS_COSTOS[i+N_GPU*M*it]);
                     	}
		     	else{
			   fprintf(file1,",%d",HORMIGAS_COSTOS[i+N_GPU*M*it]);
			}
	             }
		     fprintf(file1,"\n");
	        }
		         
	 }
	fclose(file1);
}
