/*** Includes ***/
  #include <stdio.h>
  #include <stdlib.h>
  #include <string.h>
  #include <stdarg.h>
  #include <stdbool.h>
  #include <math.h>
  #include <time.h>
  #include <sys/timeb.h>
  #include "mpi.h"
  #include <semaphore.h>


/*** Macros ***/
  /* Versao Linux */
  #ifdef __linux__
    #define strcmpi(a,b)      strcasecmp((a),(b))
    #define strncmpi(a,b,c)   strncasecmp((a),(b),(c))
  #endif
  
  #define root 0
  
  enum VERBOSE {NO_VERB=0, CHKPOINT=0x01, RESULTS=0x02, PARMS=0x04, WARM=0x08, ERRS=0x10, PARMSG=0x20, ALL=0xFF};
  enum SEND_TAG {HTag, SxTag, SfxTag};

  #define VERSAOPROG "v 1.0 (2014) DOS/Windows"
  #define mprintf(...) printf(__VA_ARGS__)


/*** Typedefs ***/
typedef struct timeb timeb_t;


/*** Globais ***/
  struct F {
    unsigned int raizes;    // Número de raízes da função objetivo
    unsigned int nvars;     // Número de variáveis da função objetivo
    unsigned int fn;        // Número da função objetivo
    float *low,             // Boundary lower
          *high;            // Boundary upper
    float (*fObj)(float *); // Ponteiro para a função objetivo
  } funcao;


  int *H;             // Matriz de histograma
  int nh=0;           // Número de entradas (classes) na matriz H
  
  float *Sx;          // Matriz de variáveis obtidas pelo programa
  float *Sfx;         // Matriz de valor da função objetivo obtidas pelo programa
  
  float tol_accept=1e-3,  // Tolerância mínima para aceite de solução
        tol_same=0.001;   // Tolerância mínima para detrminação de mesma solução

  /* MPI */
  int    my_rank,     // Número de rank do processo
         np;          // Número de processos
  int n_aval=100;     // Número de execuções do algoritmo híbrido por nó


  /* Thread */
  sem_t lj_sem,       // Semáforo para Luus-Jacoola
        hj_sem;       // Semáforo para Hooke-Jeeves


  /* Parâmetros Luus-Jacoola */
  int n_out=5,        // Número de loops externos do LJ
      n_in=10;        // Número de loop internos do LJ
  float cc=0.95;      // Coeficiente de contração ]0,1[. Padrão=5%


  /* Parâmetros Hooke-jeeves */
  float csi=0.01,         // CSI ??????
        delta_value=0.2,  // Valor de delta>csi
        alfa=0.5;         // Fator de aceleração (>0)

/*** Prototipos ***/
  void *luusjacoola(void *unused);
  void *hookjeeves(void *unused);
  void *categorize(void *unused);
  void classify(int l, int ns);
  void randomize(void);
  float rRand(float, float);
  void Help();
  float f0(float *);
  float f1(float *);
  float f2(float *);
  float f3(float *);
  float f4(float *);
  float f5(float *);
  float f6(float *);
  void Start(unsigned int);
  void Finnish(void);
  void abortexecMPI(int rank, char *msg, int ret);
  float getMarks(const timeb_t t0, timeb_t *t1);
  double fdifftime(const timeb_t, const timeb_t);


/*** Função principal ***/
int main(int argc, char *argv[]){
  int fn=0;        // Número da função
  int total_aval;  // Número de execuções total do algoritmo híbrido
  
  int i,j,k,l;     // Contadores de iteração
  int n_maxsolutions; // Número máximo de soluções
  bool  timer=false; // Indica a contagem de tempos de processamento
  enum VERBOSE
         verbose=NO_VERB; // Indica a exibicao de detalhes de processamento
  timeb_t t0,       // Hora inicial
          t1;       // Hora atual


  /* Thread */
  pthread_t plj_id, phj_id, pcat_id;

  // Inicializa os semáforos
  sem_init (&lj_sem, 0, 0);
  sem_init (&hj_sem, 0, 0);
  
  
  /* MPI  */
  MPI_Request req;        // Identificador de requisicao
  MPI_Status statusReq;   // Status de comunicacao MPI
  

  /* Processo-global */
  bool ImRoot=0;          // Indica se a unidade de processamento e´ o root
  char   msg[100]="";     // Mensagem
  int *nhBuf=NULL;
  
  /* Processo-Local */
  //
  
  /**************************************************
  //// Procedimentos iniciais
  **************************************************/
  /* Incializa hora inicial */
  ftime(&t0);
  

  /**************************************************
  //// Inicializacao do MPI
  ***************************************************/
  if (verbose&(CHKPOINT+PARMSG)){ printf("[%d]- Inicializando MPI ...\n", my_rank); }
  MPI_Init(&argc, &argv);

  /* Acha o rank do processador */
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  /* Acha o numero total de processadores */
  MPI_Comm_size(MPI_COMM_WORLD, &np);
  
  /* Atualiza o flag de "Eu sou o root!" */
  ImRoot = (my_rank==root);

  
  /**************************************************
  //// Verificacao de argumentos
  ***************************************************/
  // Pega argumentos da linha de comandos (se houver!)
  for(i=1; i<argc; i++){
  
    // Help
    if(strcmpi(argv[i], "-h")==0){
      if (ImRoot) Help(argv[0]);
      abortexecMPI(my_rank, "", 0);
      return 0;
    
    // Contabiliza tempos de execucao
    }else if(strcmpi(argv[i], "-t")==0){
     timer=true;
     
    // Verbose
    }else if(strcmpi(argv[i], "-v")==0){
      // Verbose //
      verbose=ALL;
      
    // Verbose especifico
    }else if(strncmpi(argv[i], "-v", 2)==0){
      verbose = (enum VERBOSE) atoi(argv[i]+2);
      
    // Function number select
    }else if(strncmpi(argv[i], "-fn", 3)==0){
      fn = atoi(argv[i]+3);
      if (fn<0 || fn >6) {
        mprintf("Argumento [%s] invalido:\n", argv[i]);
        exit(0);
      }
      
    // Número de avaliações por nó
    }else if(strncmpi(argv[i], "-na", 3)==0){
      n_aval = atoi(argv[i]+3);
      if (n_aval<0) {
        mprintf("Argumento [%s] invalido:\n", argv[i]);
        exit(0);
      }

    // Invalid argument
    }else{
      strcpy(msg, "Argumento invalido: ");
      strcat(msg, argv[i]);
      abortexecMPI(my_rank, msg, 1);
      mprintf("Argumento [%s] invalido:\n", argv[i]);
      exit(0);
    }
  }

  
  /**************************************************
  //// Inicialização de variáveis
  ***************************************************/
  // Calculo do número total de avaliações
  total_aval = n_aval*np;
  
  // Número máximo de soluções do root é o número total de soluções máximo obtido por todos os nós
  n_maxsolutions = (ImRoot) ?total_aval :n_aval;

  // Executa os procedimentos iniciais para a função-ojetivo especificada
  Start(fn);


  /**************************************************
  //// Alocações dinâmicas de memória
  ***************************************************/
  H = (int *) malloc(sizeof(int)*n_maxsolutions);
  if (H==NULL) {
    printf("Erro de alocacao de memoria-distancias\n");
    exit(0);
  }

  Sfx = (float *) malloc(sizeof(float)*n_maxsolutions);
  if (Sfx==NULL) {
    printf("Erro de alocacao de memoria-distancias\n");
    exit(0);
  }

  Sx = (float *) malloc(sizeof(float)*funcao.nvars*n_maxsolutions);
  if (Sx==NULL) {
    printf("Erro de alocacao de memoria-distancias\n");
    exit(0);
  }

  if (ImRoot){
    nhBuf = (int *) malloc(sizeof(int)*np);
    if (nhBuf==NULL) {
      printf("Erro de alocacao de memoria-distancias\n");
      exit(0);
    }
  }


  /**************************************************
  //// Loop principal do algoritmo híbrido
  ***************************************************/
  // Inicialização de variáveis
  randomize();
  nh = 0;  // Inicialização do contador de classes
  
  // Inicia a contagem do histograma
  for(l=0; l<n_maxsolutions; l++) {
    H[l] = 1; 
  }

  /**************************************************
  //// Threads que implementam o algoritmo híbrido
  ***************************************************/
  pthread_create(&plj_id,  NULL, (void *) luusjacoola, NULL);
  pthread_create(&phj_id,  NULL, (void *) hookjeeves,  NULL);
  pthread_create(&pcat_id, NULL, (void *) categorize,  NULL);
  
  pthread_join(plj_id,  NULL);
  pthread_join(phj_id,  NULL);
  pthread_join(pcat_id, NULL);


  /**************************************************
  //// Junta as soluções obtidas pelos outros nós
  ***************************************************/

  // Recupera o número de classes obtidas em cada nó
  MPI_Gather(&nh, 1, MPI_INT, nhBuf, 1, MPI_INT, root, MPI_COMM_WORLD);
  
  if (ImRoot) {
    int dh=0; // Deslocamento
    
    printf("\n\n================\n");
    for(l=0; l<np; l++){
      if (l!=my_rank){
        MPI_Recv(H+dh,   sizeof(int)*nhBuf[l],   MPI_INT,   l, HTag,   MPI_COMM_WORLD, &statusReq);
        MPI_Recv(Sfx+dh, sizeof(float)*nhBuf[l], MPI_FLOAT, l, SfxTag, MPI_COMM_WORLD, &statusReq);
        MPI_Recv(Sx+(dh*funcao.nvars),  sizeof(float)*nhBuf[l], MPI_FLOAT, l, SxTag,  MPI_COMM_WORLD, &statusReq);
      }

      printf("%d ==>\n", l);
      for(i=dh; i<(dh+nhBuf[l]); i++) {
        printf("  [%d] (%d) ", i+1, H[i]);
        for(k=0; k<funcao.nvars; k++){
          printf("%g ", Sx[i*funcao.nvars+k]);
        }
        printf("%g \n", Sfx[i]);
      }
      
      dh += nhBuf[l];
    }

    classify(nh, (dh-nh));

  }else{
    MPI_Isend(H,   sizeof(int)*nh,   MPI_INT,   root, HTag,   MPI_COMM_WORLD, &req);
    MPI_Isend(Sfx, sizeof(float)*nh, MPI_FLOAT, root, SfxTag, MPI_COMM_WORLD, &req);
    MPI_Isend(Sx,  sizeof(float)*nh, MPI_FLOAT, root, SxTag,  MPI_COMM_WORLD, &req);
  }
  
  
  /**************************************************
  //// Imprime o conjunto de soluções
  ***************************************************/
  if (ImRoot) {
    int aceitos=0;
    
    printf("\n\n");
    for(l=0; l<nh; l++) {
      aceitos += H[l];
      
      printf("[%d] (%d) ", l+1, H[l]);
      for(k=0; k<funcao.nvars; k++){
        printf("%g ", Sx[l*funcao.nvars+k]);
      }
      printf("%g \n", Sfx[l]);
    }

    printf("\nAceitos: %d\n", aceitos);
    printf("Classes: %d\n", nh);
    printf("Rejeitados: %d\n\n", total_aval-aceitos);
  }


  /**************************************************
  //// Encerramento
  ***************************************************/
  if (ImRoot) {
  }


  /**************************************************
  //// Finalizacoes
  ***************************************************/
  /* Libera espacos de memoria reservados */
  if (verbose&CHKPOINT){ printf("[%d]- Liberando alocacoes de memoria...\n", my_rank); }
  free(H);
  free(Sx);
  free(Sfx);
  free(nhBuf);

  Finnish();

  /* Finalizando o MPI */
  if (verbose&(CHKPOINT+PARMSG)){ printf("[%d]- Finalizando MPI ...\n", my_rank); }
  MPI_Finalize();

  /* Fim do programa */
  if (verbose&(CHKPOINT+WARM)){ printf("[%d]- TERMINO DO PROGRAMA ...\n", my_rank); }
  if (timer){ printf("[%d]- Tempo transcorrido= %f\n", my_rank, getMarks(t0, &t1)); }

  pthread_exit((void *) NULL);

  return 0;
}



/**************************************************
//// Declaração de funções
***************************************************/

/*
 * Algoritmo de Luus-Jacoola
 */

void *luusjacoola(void *unused){
  float *x;         // Vetor de parâmetros calculados
  float *r;         // Vetor de distâncias
  int i,j,k,l;      // Contadores de iteração
  float fxj;        // Valor da função objetivo na iteração j
  float *var=NULL;  // Ponteiro para o início do buffer de variáveis

  // Alocação de memória
  x = (float *) malloc(sizeof(float)*funcao.nvars);
  if (x==NULL) {
    printf("Erro de alocacao de memoria-variaveis\n");
    exit(0);
  }
  
  r = (float *) malloc(sizeof(float)*funcao.nvars);
  if (r==NULL) {
    printf("Erro de alocacao de memoria-distancias\n");
    exit(0);
  }


  /**************************************************
  //// Loop de avaliações do algoritmo Luus Jaakola
  ***************************************************/
  for(l=0; l<n_aval; l++){
    /* Gera uma solução inicial aleatória x* e */
    var = Sx+(l*funcao.nvars); // Aponta var para o início do buffer de variáveis
    
    // inicializa os tamanhos do vetor de busca r
    for(k=0; k<funcao.nvars; k++){
      var[k] = rRand(funcao.low[k], funcao.high[k]);
      r[k]   = funcao.high[k] - funcao.low[k];
    }
//var[0] = 0.70710678118661;
//var[1] = 0.70710678118648;

    // Assume que o melhor resultado é o valor inicial aleatório
    Sfx[l] = funcao.fObj(var);


    /**************************************************
    //// Loop principal do algoritmo Luus Jaakola
    ***************************************************/
    for(i=0; i<n_out; i++){  // Busca padrão
      for(j=0; j<n_in; j++){  // Busca exploratória

        // Busca exploratória ao longo do eixo das coordenadas
        for(k=0; k<funcao.nvars; k++){
          x[k] = var[k] + rRand(-0.5, 0.5) * r[k];
        }

        // Calcula a função objetivo com os novos valores
        fxj = funcao.fObj(x);

        // Verifica se as novas variáveis produz uma solução melhor que a melhor anterior
        if (fxj<Sfx[l]) {
          // Faz da nova solução a melhor
          Sfx[l] = fxj;

          // copia x[i] para var[i]
          for(k=0; k<funcao.nvars; k++){
            var[k] = x[k];
          }
        }
      }

      // Contrai o espaço de busca
      for(k=0; k<funcao.nvars; k++){
        r[k] *= cc;
      }
    } // Fim LJ

    // Libera a solução[l] para processamento do HJ
    sem_post (&lj_sem);
  } // Fim Loop

  free(x);
  free(r);
  pthread_exit((void *) NULL);
}


/*
 * Algoritmo de Hooke-Jeeves
 */
void *hookjeeves(void *unused){
  float delta,      // Variável de trabalho delta
        dif_xk,     // x(k+1)-x(k)
        fyi,        // Valor da função objeto usando os parâmetros yi
        fyi_m1,     // Valor da função objeto usando os parâmetros yi_m1
        *yi,        // Vetor de novos parâmetros HJ no passo i
        *yi_m1,     // Vetor de novos parâmetros HJ no passo i+1
        *ei;
  int i,l,k;      // Contadores de iteração
  float *var=NULL;  // Ponteiro para o início do buffer de variáveis

  // Alocação de memória
  yi = (float *) malloc(sizeof(float)*funcao.nvars);
  if (yi==NULL) {
    printf("Erro de alocacao de memoria-distancias\n");
    exit(0);
  }

  yi_m1 = (float *) malloc(sizeof(float)*funcao.nvars);
  if (yi_m1==NULL) {
    printf("Erro de alocacao de memoria-distancias\n");
    exit(0);
  }

  ei = (float *) malloc(sizeof(float)*funcao.nvars);
  if (ei==NULL) {
    printf("Erro de alocacao de memoria-distancias\n");
    exit(0);
  }

  /**************************************************
  //// Loop de avaliações do algoritmo Hooke-Jeeves
  ***************************************************/
  for(l=0; l<n_aval; l++){
    // Aguarda solução[l] de LJ
    sem_wait (&lj_sem);

    var = Sx+(l*funcao.nvars);

    /**************************************************
    //// Loop principal do algoritmo Hooke-Jeeves
    ***************************************************/
    // Inicialização de variáveis para o HJ
    for(k=0; k<funcao.nvars; k++){
      yi[k] = var[k];
      ei[k] = 0.001*var[k]; //??
    }
    fyi = Sfx[l];
    delta = delta_value;

    // Loop principal do algoritmo de Hooke-Jeeves
    while(delta>csi){
      // Faz y(i+1) = y(i)
      for(k=0; k<funcao.nvars; k++){
        yi_m1[k] = yi[k];
      }

      // Busca exploratória
      for(i=0; i<funcao.nvars; i++){
        yi_m1[i] = yi[i] + delta*ei[i];
        fyi_m1 = funcao.fObj(yi_m1);
        if (fyi_m1 < fyi) {
          fyi = fyi_m1;
          yi[i] = yi_m1[i];
        }else{
          yi_m1[i]  = yi[i] - delta*ei[i];
          fyi_m1 = funcao.fObj(yi_m1);
          if (fyi_m1 < fyi) {
             fyi = fyi_m1;
            yi[i] = yi_m1[i];
          }else{
            yi_m1[i] = yi[i];
          }
        }
      }

      if (fyi<Sfx[l]){
        // Aceleração em direção a x(k+1)-x(k)
        for(k=0; k<funcao.nvars; k++){
          dif_xk = yi[k] - var[k];
          var[k] = yi[k];
          yi[k] = yi[k] + alfa*dif_xk;
        }
        Sfx[l] = fyi;
      }else{
        // Redução do tamanho de delta
        delta /= 2.0;
        for(k=0; k<funcao.nvars; k++){
          yi[k]  = var[k];  // Faz y(0) = var[k]
        }
        fyi = Sfx[l];
      }
    } // Fim HJ
    
    // Libera a solução[l] para categorização
    sem_post (&hj_sem);
  }

  free(yi);
  free(yi_m1);
  free(ei);
  pthread_exit((void *) NULL);
}


/*
 * Categorização
 */

void *categorize(void *unused){
  int l;
  
  /**************************************************
  //// Loop de categorização das avaliações
  ***************************************************/
  for(l=0; l<n_aval; l++){
    // Aguarda solução[l] de HJ
    sem_wait (&hj_sem);
    
    // Rejeita soluções acima de uma tolerância de aceite
    if (Sfx[l] > tol_accept) {
      continue;
    }
    
    
    /**************************************************
    //// Classifica e contabiliza a solução dentro do próprio nó
    ***************************************************/
    classify(l, 1);
  }
  
  pthread_exit((void *) NULL);
}



/*
 * Classifica o conjunto de novas soluções S dentro da classe
 * de soluções classificadas H
 */
void classify(int l, int ns){
  bool novaclasse,     // Flag indicador de nova classe de solução
       same;           // Flag indicador de que dois argumentos são os mesmos
  int i,j,k;           // Contadores
  float erro;          // Erro
  float *var_l=NULL;   // Ponteiro para o início do buffer de variáveis nh
  float *var_i=NULL;   // Ponteiro para o início do buffer de variáveis i

  // Classifica os ns novos elementos de S
  for(j=0; j<ns; j++, l++){
    var_l = Sx+(l*funcao.nvars);
    
    /**************************************************
    //// Busca por novas classes
    ***************************************************/
    // Verifica se o conjunto de soluções representa uma nova classe de solução
    novaclasse = true; // Admite que o conjunto de variáveis representa uma nova classe de solução

    for(i=0; i<nh; i++){
      same = true; // Admite-se que o conjunto de variáveis é o mesmo
      var_i = Sx+(i*funcao.nvars);
      
      for(k=0; k<funcao.nvars; k++){
        // Cálcula a diferença entre os argumentos
        erro =  fabs(var_l[k] - var_i[k]);  // Erro absoluto

        // Considera-se um novo conjunto de soluções se,
        // pelo menos, uma variável estiver fora da tolerância
        if (erro>tol_same){
          same = false;
          break;
        }
      }
      
      // Se todos os argumentos são os mesmos, então o conjunto de solução é o mesma
      if (same){
        novaclasse = false;
        break;
      }
    }

    /**************************************************
    //// Verifica se o conjunto de soluções representa uma nova classe de solução
    ***************************************************/
    if (novaclasse){
      float *var_nh=Sx+(nh*funcao.nvars);
      
      // Copia os dados de S[l] para S[nh]
      Sfx[nh] = Sfx[l];
      for(k=0; k<funcao.nvars; k++){
        var_nh[k] = var_l[k];
      }
      // Cadastra a nova classe em H
      nh++;  // Adiciona mais uma classe de solução a H
    }else{ // Classe cadastrada em H
      // Verifica se o conjunto de variáveis é uma solução melhor que a classe cadastrada em H
      if (Sfx[l] < Sfx[i]) {
        // Copia os dados de S[l] para S[i]
        Sfx[i] = Sfx[l];
        for(k=0; k<funcao.nvars; k++){
          var_i[k] = var_l[k];
        }
      }
      H[i] += H[l]; // Adiciona o item a classe de solução
    }
  }

  return;
}


/* Inicialização de semente randômica
 */
void randomize() {
  srand((unsigned int) time(NULL)*(my_rank+1)); // Muda a semente aleatória em função do número de rank do nó
}

/* Geração dos números randômicos em PF dentro da faixa [low, high]
 */
float rRand(float low, float high) {
  float d;
  d = (float)rand()/(float)RAND_MAX;
  return (low + d * (high - low));
}

// Help
void Help(){
  mprintf("\nSintaxe: programa [Opcoes... ]\n");
  mprintf("\n");
  mprintf("Opcoes: \n");
  mprintf(" -h (Help) Exibe este help\n");
  mprintf(" -v (Verbose) Exibe todos os detalhes de processamento\n");
  mprintf("\n**** Copyright. Dalmo Stutz, %s ****\n\n", VERSAOPROG);
}

/* Função de Himmelblau (HIMM)
 */
float f0(float *x){
  float fx =
      pow(4*pow(x[0], 3) +4*x[0]*x[1] +2*pow(x[1], 2) -42*x[0] -14, 2)
    + pow(4*pow(x[1], 3) +2*pow(x[0], 2) +4*x[0]*x[1] -26*x[1] -22, 2);
  return fx;
}

/* Sistema trigonométrico - ST
 */
float f1(float *x){
  float fx =
      pow(-sin(x[0])*cos(x[1])-2*cos(x[0])*sin(x[1]), 2)
    + pow(cos(x[0])*sin(x[1])-2*sin(x[0])*cos(x[1]), 2);
  return fx;
}

/* Sistema polinomial de alto grau - SPAG
 */
float f2(float *x){
  float fx =
      pow(5*pow(x[0],9) -6*pow(x[0],5)*pow(x[1],2) +x[0]*pow(x[1],4) +2*x[0]*x[2], 2)
    + pow(-2*pow(x[0],6)*x[1] -2*pow(x[0],2)*pow(x[1],3) +2*x[1]*x[2], 2)
    + pow(pow(x[0],2) +pow(x[1],2) -0.265625, 2);
  return fx;
}

/* Sistema quase linear de Brown - PQL
 */
float f3(float *x){
  float fx =
      pow(2*x[0] +x[1] +x[2] +x[3] +x[4] -6, 2)
    + pow(x[0] +2*x[1] +x[2] +x[3] +x[4] -6, 2)
    + pow(x[0] +x[1] +2*x[2] +x[3] +x[4] -6, 2)
    + pow(x[0] +x[1] +x[2] +2*x[3] +x[4] -6, 2)
    + pow(x[0]*x[1]*x[2]*x[3]*x[4] -1, 2);
  return fx;
}


/* Bini e Mourrain - BM
 */
float f4(float *x){
  float fx =
      pow(-pow(x[1],2)*pow(x[2],2) -pow(x[1],2) +24*x[1]*x[2] -pow(x[2],2) -13, 2)
    + pow(-pow(x[0],2)*pow(x[2],2) -pow(x[0],2) +24*x[0]*x[2] -pow(x[2],2) -13, 2)
    + pow(-pow(x[0],2)*pow(x[1],2) -pow(x[0],2) +24*x[0]*x[1] -pow(x[1],2) -13, 2);
  return fx;
}


/* Sistema Não Linear - SNL
 */
float f5(float *x){
  float fx =
      pow(x[0] -x[1], 2)
    + pow(pow(x[0], 2) +pow(x[1], 2) -1, 2);
  return fx;
}

/* Sistema Não Linear - SNL
 */
float f6(float *x){
  float fx =
      pow(x[0] -2, 4)
    + pow(x[0] -2*x[1], 2);
  return fx;
}

/*
 */
void Start(unsigned int fn){
  int i;

  // Atualiza o número da função
  funcao.fn  = fn;

  // Seleciona a função objetivo
  switch (fn){
    case 0: // HIMM
      funcao.raizes = 9;
      funcao.nvars  = 2;
      funcao.fObj   = f0;
      break;

    case 1: // ST
      funcao.raizes = 13;
      funcao.nvars  = 2;
      funcao.fObj   = f1;
      break;

    case 2: // SPAG
      funcao.raizes = 12;
      funcao.nvars  = 3;
      funcao.fObj   = f2;
      break;

    case 3: // PQL
      funcao.raizes = 3;
      funcao.nvars  = 5;
      funcao.fObj   = f3;
      break;

    case 4: // SPAG
      funcao.raizes = 8;
      funcao.nvars  = 3;
      funcao.fObj   = f4;
      break;

    case 5: // SNL
      funcao.raizes = 2;
      funcao.nvars  = 2;
      funcao.fObj   = f5;
      break;

    case 6: // SNL
      funcao.raizes = 2;
      funcao.nvars  = 2;
      funcao.fObj   = f6;
      break;

    default:
      mprintf("Numero de funcao objetivo invalido [fn=%d]\n", fn);
      exit(0);
  }

  // Aloca memória para armazenamento das variáveis da função objetivo
  funcao.low = (float *) malloc(sizeof(float)*funcao.nvars);
  if (funcao.low==NULL) {
    mprintf("Erro de alocacao de memoria\n");
    exit(0);
  }

  funcao.high = (float *) malloc(sizeof(float)*funcao.nvars);
  if (funcao.high==NULL) {
    mprintf("Erro de alocacao de memoria\n");
    exit(0);
  }

  // Define o espaço de busca
  switch (fn){
    case 0: // HIMM
      for(i=0; i<funcao.nvars; i++) {
        funcao.low[i]  = -5;
        funcao.high[i] = 5;
      }
      break;

    case 1: // ST
      for(i=0; i<funcao.nvars; i++) {
        funcao.low[i]  = 0;
        funcao.high[i] = 2*M_PI;
      }
      break;

    case 2: // SPAG
      funcao.low[0]  = -0.6;
      funcao.high[0] = 6;

      funcao.low[1]  = -0.6;
      funcao.high[1] = 0.6;

      funcao.low[2]  = -5;
      funcao.high[2] = 5;
      break;

    case 3: // PQL
      for(i=0; i<funcao.nvars; i++) {
        funcao.low[i]  = -10;
        funcao.high[i] = 10;
      }
      break;

    case 4: // BM
      for(i=0; i<funcao.nvars; i++) {
        funcao.low[i]  = 0;
        funcao.high[i] = 20;
      }
      break;

    case 5: // SNL
      for(i=0; i<funcao.nvars; i++) {
        funcao.low[i]  = -1;
        funcao.high[i] = 1;
      }

    case 6: // SNL
      for(i=0; i<funcao.nvars; i++) {
        funcao.low[i]  = 1;
        funcao.high[i] = 4;
      }
      break;
  }
}

/*
 */
void Finnish(){
  free(funcao.low);
  free(funcao.high);
}

/*------------------------------------------------------------------*/
/* abortexecMPI - Aborta a execucao do programa paralelo,
       emitindo uma mensagem e retornado o codigo de erro
   @rank - Numero de rank
   @msg  - Mensagem de erro
   @ret  - Codigo de erro
*/
void abortexecMPI(int rank, char *msg, int ret){
  printf("[%d] - MSG_ABORT_ERROR(%d): %s\n", rank, ret, msg);
  MPI_Abort(MPI_COMM_WORLD, ret);
}


//------------------------------------------------------------------
/*  getMarks - Pega uma marcacao de tempo e retorna a diferenca

    @t0 (const) - Marcacao inicial
    @t1 (pont)  - Marcacao atual
*/
float getMarks(const timeb_t t0, timeb_t *t1) {
  ftime(t1);
  return (fdifftime(*t1, t0));
}

//------------------------------------------------------------------
/* fdifftime  - Calcula a diferenca entre dois tempos em segundos com milisegundos

   @timeb_t t1 - Tempo final
   @timeb_t t0 - Tempo inicial
   @return - (double) diferenca em segundos com milisegundos
*/
double fdifftime(const timeb_t t1, const timeb_t t0){
  double dt0, dt1;

  dt0 = t0.time+t0.millitm/1000.;
  dt1 = t1.time+t1.millitm/1000.;

  return (dt1-dt0);
}
