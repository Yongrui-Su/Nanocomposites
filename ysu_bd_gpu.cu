#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <malloc.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <limits>
#include <cuda.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <iomanip> 
#include <vector>
#include <numeric>
using namespace std;

#define PI      3.141592653589793238462643383
#define NSS     60000
#define SA 40000
#define BLOCK_SIZE 1024
#define NANO_THREADSHOLD 50
#define THRESHOLD 400000
#define CHECK_SS 200

const int Nint=14;
const int zntotal = 100; 
const int buffer_size = 100;
const int output_stress = 5;
bool LeesEd = false;
bool SSlink = true;
bool mayavi = false;
bool Tra = false;
bool RE_control = false;
bool RDF_control = false;
int Read_ss = 0;
bool MSD=true;
double hc_attraction = -1.0;
int ss_rdf_count = 0;
int start_from_zero=1;
vector<double> RWD;



struct vect{
  double x;
  double y;
  double z;
}*atom_v,*nano_v,*chain_mid,*chain_bead;


struct atoms 
{
  vect pos;
  int type;
  int ss_tag;
}*atom,*nano_atom;

struct nanos{
  vect pos;
}*nano;

struct tensor{
	tensor():xx(0),yy(0),zz(0),xy(0),yz(0),xz(0){}
  double xx;
  double yy;
  double zz;
  double xy;
  double yz;
  double xz;
}Rg;


struct Box_Data 
{
  double NP_vol;
  double bulk_vol;
  int natoms;
  int natoms0;
  int natoms1;
  double vol;
  double length1;
  double length2;
  double length3;
  int chains;
  int cell;                    //questions for its existance		
  double temp;
  double period;
  double ***phi0;
  double ***phi1;
  double ***phi2;
}box;

struct SimData
{
  double Re;
  double bb;
  double bsl;
  int N;
  int N0;
  int N1;
  double chi;
  double lambda0;
  double lambda1;
  double kappa;
  double core_rep;
  double in_pol;
  double dp;
  double dp2;
  double Nano_frac;
  int Nano_num;
  double Nano_R;
  double Nano_Rc;
  int Nano_bead;               //number of beads in one nanoparticle  
  double Nano_weight;
  double Nano_den;
  double frict;
  double v_d;
  double tim;
  double dt;
  double ddx;
  double NP_weight;
  double T;
  int gxz_tmax;       
  int gxz_it0;
  int gxz_t0max; 
  int tmax;       
  int it0;
  int t0max; 
  int nano_tmax;       
  int nano_it0;
  int nano_t0max; 
  double beta;
  int nss;
  double rss;
  double freq;
  double str_ss;
} sim;

struct energy 
{
  double nonbonded_E;
  double hardcore;
  double bonded_E;
  double total;
  double kinet;
  double sl_ener;						
} en;			

struct Force
{
  vect bond_force;				
  vect nonbond_force;
  vect hardcore_force;
  vect ss_f;	
  vect total_force;
}*force_bead;


struct NanoForce
{			
  vect nonbond_force;
  vect hardcore_force;
  vect total_force;
}*force_nano;

struct ss_t 
{
  int end1;
  int end2;
  double en;
  int begin;
} ss[NSS];


//__device__ double Cuda_HcForce(int,int,nanos *,atoms *,NanoForce *,Force *,double *,double *,double *,double *,double *,double *,double,double);

__device__ double Cuda_Nbforce(int,int,atoms *, Force *,double *,double *,double *,double *,double *,double *,double);
void initial_config(void);
void read_config(void);
void initial_all(void);		
void read_sim(void);
double ran_num_double(int,int);
void nano_pbc_one(int);
void pbc_one(int);
__device__ void Cuda_Pbc_One(double &,double &,double &,double,double,double);
void pbc_one(double &,double &,double &);
int Round(double);
__device__ int Cuda_Round(double);
__device__ double Cuda_modulo(double,double);
double modulo(double,double);
void xyz_iniconfig (void);
int ran_num_int(int,int);
void xyz_config (void);
void den_avex(void);
void den_avey(void);
void den_avez(void);
void construct_link(void);
//void nn_construct_link(void);                               
void stress_out(void);
void initial_nano(void);
int grid(double,int);
__device__ int Cuda_Grid(double,int);
//__device__ int NN_Cuda_Grid(double,int);
//int nn_grid(double,int);
void initial_rdf(void);                         //initialize calculation for g(r)  
void fin_rdf(void);                         //average g(r) over all atoms and certain number of loops and report fi
void save_check(void);
void read_check(void);
void den_pro(void);
void forces(void);
void nonbond_forces(void);
__global__ void Cuda_NONBOND_ForceCompute(double *,nanos *,atoms *,atoms *,NanoForce *, Force *,int *,int *,double *,double *,double *,double *,double *,double *,double ,int ,double,double);
void bond_forces(void);
void momentum(void);
double gauss(double);
__device__ double Cuda_Gauss(double,int);
void mayavi_density(int);
void ini_grid(void);
void hardcore_forces(void);
//void link_hardcore_forces(void);
void pair_hardcore_forces(void);
void isoverlap(vect);
__device__ double Cuda_HardForce(double,double,double);
__device__ double Cuda_HardEn(double,double,double);
__device__ double Cuda_Attract_Force(double);
__device__ double Cuda_Attract_En(double);
//double min(double,double);
__global__ void SS_update(Force *,atoms *,ss_t *,int ,double);
double distance(double &,double &,double &);
__device__ double atomicAdd(double*, double);
__device__ double Cuda_Distance(double &,double &,double &,double);
__global__ void Cuda_Chain_Center(vect *,vect *,int,int);
__global__ void BondForce_Compute(vect *,atoms *,double *,double *,double *,double *,double *,double *,double *,int,int,double);
__global__ void Cuda_RWDSS(atoms *,double *,int *,int ,int ,double ,double );
__global__ void Cuda_Bead_Brownian(Force *,atoms *,vect *,double,double,double,int,double,int);
__global__ void Cuda_Nano_Brownian(NanoForce *,nanos *,vect *,double,double,double,double,double ,int,double,int);
__global__ void SS_Force_Compute(vect *,atoms *,ss_t *,double *,double *,double *,double *,double *,double *,int,double);
//__global__ void Cuda_NanoNano_ForceCompute(double *,nanos *,NanoForce *,double *,double *,double *,double *,double *,double *,double,double,double);
__global__ void Cuda_BeadNano_ForceCompute(double *,atoms *,nanos *,Force *,NanoForce *,double *,double *,double *,double *,double *,double *,double,double);
__global__ void Cuda_Attractive_Hc_ForceCompute(double *,atoms *,nanos *,Force *,NanoForce *,double *,double *,double *,double *,double *,double *,double,double);
//__global__ void Cuda_Link_HardCoreForce(double *,nanos *,atoms *,NanoForce *, Force *,int *,int *,double *,double *,double *,double *,double *,double *,double,int,double,int,double);
void system_out(void);
inline double ss_en(double);
__device__ inline double Cuda_ss_en(double);
__global__ void Cuda_Reset_Force(Force *,NanoForce *);
void ss_forces(void);
//double compensate_forces(void);
void ini_ss(void);
void add_ss(void);
void del_ss(int,int,double &);
void ss_evolve(void);
void ss_displace(void);
double test_sum(void);
double mod(double,double);
__device__ inline int Cuda_Mod(int,int);
void chain_center(void);
void chain_msd(int);
void msd(int);
void nano_msd(int);
void ratio(void);
void output(int);
void ss_count(void);
void get_last_time(void);
void ss_update();
double Rosenbluth_Del(int,int);
std::istream& ignoreline(std::ifstream& in, std::ifstream::pos_type& pos)
{
  pos = in.tellg();
  return in.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
}

std::string getLastLine(std::ifstream& in)
{
  std::ifstream::pos_type pos = in.tellg();

  std::ifstream::pos_type lastPos;
  while (in >> std::ws && ignoreline(in, lastPos))
    pos = lastPos;

  in.clear();
  in.seekg(pos);

  std::string line;
  std::getline(in, line);
  return line;
}
//void end_add_ss(void);
//void end_del_ss(void);
tensor Rg_tensor(int);
void Rg_ave(void);
tensor diag_tensor(tensor);
void Re_time(void);
void ss_dis(int);
void nano_traj(void);
void poly_traj(void);
void Brownian(void);
void rdf(void);                         //calculate g(r) for all atoms in one loop 
void slip_spring_check(void);
void temp_stress_out(void);
double Lcor;
double delt_r; 
int overlap=0;
double phiAx[zntotal],phiAy[zntotal],phiAz[zntotal];
double phiBx[zntotal],phiBy[zntotal],phiBz[zntotal];
double phix_na[zntotal],phiy_na[zntotal],phiz_na[zntotal];
int exstep=0;
double del_L1;
double del_L2;
double del_L3;
double del_nnL1;
double del_nnL2;
double del_nnL3;
double density;
double acc_ratio;
int NStep;
double cur_t;
double del_checkx;
double del_checky;
double del_checkz;
int L_1,L_2,L_3,L;
//int nn_L_1,nn_L_2,nn_L_3,nn_L;
double R_cut;                      
double del_sphere;
double nor;                           
int *head,*lscl;  
//int *nn_head,*nn_lscl;                      
int *la_head,*la_lscl;                          
double delt_ss;
double delt_ki;
double *hist_p2p;
double *hist_n2p;
double *hist_n2s;
double *hist_n2n;
int samp;
int tra_samp;
int ex_time;
int ndfx,ndfy,ndfz;
double dLx_field, dLy_field, dLz_field,rho_field;
int Nbin=200;                               //number of interval for generating histogram of g(r)
double sqrdt;
double den;
double denPC;
int *ss_life;
int t0;
double *Re;
int ntel;
double rg_check = 0;

int msd_samp=500;
int gxz_samp=1;
int movie_counter=0;
double dx_LE = 0.0;
double SS[9];
double SSB[9];
double SSNB[9];
double SSHC[9];
double SS_f[9];
double *hist_ss,*hist_Q;
double *hist_velocity;
double wxx,wyy,wzz;
double wyx,wzy,wzx; 
double check1,check2;
int *dist_c;
int freq;
tensor temp_b,temp_nb,temp_hc,temp_ss;
int *Z_dump,*N_dist;
int Stress_Laststep;
int buffer_count = 0;
double buffer_stress[buffer_size][16];
double F_coef,SS_F_coef;

//msd data is defined here
double *ntime;
double *nano_ntime;
double *chain_ntime;
double *r2f;
double *r2_dx;
double *r2_dy;
double *r2_dz;
double *chain_r2f;
double *nano_r2f;
double *nano_dx;
double *nano_dy;
double *nano_dz;
double *time0;
double *nano_time0;
double *chain_time0;
double *X0,*Y0,*Z0;
double *nano_X0,*nano_Y0,*nano_Z0;
double *chain_X0,*chain_Y0,*chain_Z0;
int nano_t0;
int chain_t0;
int nano_ntel;
int chain_ntel;


/*Cuda Global variabls are defined below.
 * They should be initialized by CPU global variable in the Initialization function.
 */
//LeesEd condition switcher
__constant__ bool Cuda_LeesEd;
//boxsize define
__device__ double CudaLength1;  
__device__ double CudaLength2;
__device__ double CudaLength3;

__device__ int Cuda_L1;  
__device__ int Cuda_L2;
__device__ int Cuda_L3;


__device__ double Cuda_dL1;  
__device__ double Cuda_dL2;
__device__ double Cuda_dL3;



__device__ int CudaNum_Bead;
__device__ int CudaNum_Nano;

__device__ double Cuda_Rcut;
__device__ double Cuda_Chi;
__device__ double Cuda_lambda0;
__device__ double Cuda_lambda1;
__device__ double Cuda_Kappa;
__device__ double Cuda_Density;
__device__ double Cuda_Nanoden;
__device__ double Cuda_Nor;


//Bond Force coefficient
__constant__ double f_coeff;
__constant__ double SS_f_coeff;
__constant__ double Hc_Magnitude;
__constant__ double Hc_Attraction;
__constant__ double Cuda_NanoRc;
__constant__ double Cuda_NanoR;
__constant__ double A;
__constant__ double B;
__constant__ double C;
__constant__ double D;



ss_t *c_ss; //slip springs on gpu
atoms *c_atom; //atom on gpu
vect *c_atom_v; //virtual atom on gpu
Force *c_atomf; //atom force on gpu

NanoForce *c_nanof; //nanoparticles force on gpu
nanos *c_nano; //nanoparticles on gpu
vect *c_nano_v; //virtual nanoparticles on gpu
atoms *c_Nanobead;

//float t1=0,t2=0,t3=0,t4=0,t5=0;
//float c1,c2,c3,c4,c5;


int main()
{
  srand(0);//srand(int(time(NULL)));
  
  sim.tmax = 2000000;           //msd(t) the maximum number of times for
  sim.it0 = 10;           //msd(t) the interval of sample times between initial conditions
  sim.t0max = 80;           //msd(t) the maximum number of initial conditions
  
  sim.nano_tmax = 2000000;     
  sim.nano_it0 = 1;          
  sim.nano_t0max = 100000;       
  read_sim();
  initial_all();
  system_out();
  cout << "NP volume fraction:" << sim.Nano_frac << endl;
  if(MSD== true)chain_center();
  xyz_iniconfig();		 	//write the initial configuration
  if(mayavi == true)ini_grid();
  construct_link();


  if(mayavi == true)mayavi_density(0);
  if(SSlink == true&&Read_ss == 0){
	  /*sim.nss = 0;
	  int nss_prev=0;
	  int equi_c = 0;
    while(sim.nss==0||equi_c <50){
		nss_prev = sim.nss;
      ss_evolve();
      if(nss_prev > sim.nss)equi_c++;
    }*/
    double RWD_guess = 40;
    double beta = sim.beta/RWD_guess;
    double ss_ave_guess = (sim.N+beta)/(1+beta)-1;
    while(2.0*sim.nss<ss_ave_guess*box.chains){
		ini_ss();
    }
  }
  forces(); 	
  if(SSlink == true)ss_dis(0);

  en.total = en.bonded_E+en.nonbonded_E+en.hardcore+en.sl_ener; 
  if(exstep == 0)output(0);
  //output(0);
    //c5 = clock();
  if(Read_ss==1&&start_from_zero==1){
	  for(int i=0;i < sim.nss;i++)ss[i].begin = 0;
   }

  for (cur_t=++exstep*sim.dt; cur_t<= sim.tim; cur_t=cur_t+sim.dt,exstep++)
    {

		//t5 += (c1-c5);
      construct_link();

      //c2 = clock();
      //t1 += (c2-c1);
      forces(); 
      //c3 = clock();
      //t2 += (c3-c2);

	//cout<<"time1 "<<(t2-t1)/(double) CLOCKS_PER_SEC<<"sec"<<endl;
      Brownian();

      //forces();
      //c4 = clock();
      //t3 += (c4-c3);

	//cout<<"time2 "<<(t3-t2)/(double) CLOCKS_PER_SEC<<"sec"<<endl;
      if(exstep>Stress_Laststep&&exstep % gxz_samp == 0)temp_stress_out();
      //if(mayavi == true && exstep%2 == 0)mayavi_density(1);
      if(RDF_control == true)rdf();

      if(SSlink == true && exstep % freq == 0 &&exstep>=1){

	ss_evolve();
	ss_count();
      }
      
      	//c5 = clock();
	//t4 += (c5-c4);
	
      //den_avex();
      //den_avey();
      //den_avez();
      en.total = en.bonded_E+en.nonbonded_E+en.hardcore+en.sl_ener; 
      if(SSlink == true)ss_dis(1);
      if(RE_control == true)Rg_ave();
      if(RE_control == true)Re_time();
      if (LeesEd == true){
	dx_LE += sim.v_d*sim.dt;
	if (dx_LE >= box.length1)  dx_LE -= box.length1;
      }else dx_LE = 0;

      
      if(MSD == true){
	if(exstep % msd_samp == 0){
	  //c1 = clock();
	  msd(1);
	  chain_msd(1);
	  if(sim.Nano_num>0){
	    nano_msd(1);
	  }
	  //c2 = clock();
	  //t5+=(c2-c1);
	  
	}
      }

      if(MSD == true && Tra == true && exstep % tra_samp == 0){
	if(sim.Nano_num>0)nano_traj();
	poly_traj();
      }




      if(exstep% samp ==0 && exstep>0){
		  //cout<<"time1 force section "<<t1/(double) CLOCKS_PER_SEC<<"sec"<<endl;
		  //cout<<"time2 brownian section "<<t2/(double) CLOCKS_PER_SEC<<"sec"<<endl;
		  //cout<<"time3 Re section "<<t3/(double) CLOCKS_PER_SEC<<"sec"<<endl;
		  //cout<<"time4 diffusion section "<<t4/(double) CLOCKS_PER_SEC<<"sec"<<endl;
		  //cout<<"time5 total "<<t5/(double) CLOCKS_PER_SEC<<"sec"<<endl;
		  //ct_1 = clock();
		  //t1 = 0;
		  //t2 = 0;
		  //t3 = 0;
		  //t4 = 0;
		  //t5 = 0;
		  
	if(SSlink == true)slip_spring_check();
	output(0);
	save_check();
	xyz_config();
	if(mayavi == true)mayavi_density(2);
	if(RDF_control == true)fin_rdf();
	if(SSlink == true)ss_dis(2);

	//den_pro();

	if(MSD == true){
		//c1 = clock();
	  msd(2);
	  chain_msd(2);
	  if(sim.Nano_num>0)nano_msd(2);
	  //c2 = clock();
	  //t5+=(c2-c1);
	}
      }
      if ((exstep% output_stress)==0 && exstep>0){
	stress_out();

      }
    }

  cudaFree(c_atom);
  cudaFree(c_atom_v);
  cudaFree(c_ss);
  cudaFree(c_atomf);
  cudaFree(c_nanof);
  cudaFree(c_nano);
  cudaFree(c_nano_v);
  cudaFree(c_Nanobead);

}



void read_sim (void)
{
  char tt[80];
  FILE *das;

  if( NULL==(das=fopen("./simul_MD.input","r")) )
    {
      fprintf(stdout,"input file simul.input does not exist!!!\n");
      exit(1);
    }
  //das = fopen("./simul_MD.input","r");
  fscanf(das,"%lf",&box.length1);  fgets(tt,200,das);
  fscanf(das,"%lf",&box.length2);  fgets(tt,200,das);
  fscanf(das,"%lf",&box.length3);  fgets(tt,200,das);
  fscanf(das,"%d",&sim.N0);  fgets(tt,200,das);
  fscanf(das,"%d",&sim.N1);  fgets(tt,200,das); 
  fscanf(das,"%lf",&sim.chi); fgets(tt,200,das);                          
  fscanf(das,"%lf",&sim.lambda0); fgets(tt,200,das);
  fscanf(das,"%lf",&sim.lambda1); fgets(tt,200,das);
  fscanf(das,"%lf",&density); fgets(tt,200,das);
  fscanf(das,"%lf",&sim.kappa); fgets(tt,200,das);    //kappaN         
  fscanf(das,"%lf",&sim.core_rep); fgets(tt,200,das);  
  fscanf(das,"%lf",&sim.dp); fgets(tt,200,das);  
  fscanf(das,"%lf",&sim.tim); fgets(tt,200,das);           //total amount of time to simulate
  fscanf(das,"%lf",&sim.dt); fgets(tt,200,das);    //one time step                    
  fscanf(das,"%lf",&sim.Nano_frac); fgets(tt,200,das);  
  fscanf(das,"%lf",&sim.Nano_Rc); fgets(tt,200,das);        //radius of nanoparticles core   
  fscanf(das,"%lf",&sim.Nano_R); fgets(tt,200,das);  //radius of nanoparticles    
  fscanf(das,"%lf",&sim.Nano_den); fgets(tt,200,das);  
  fscanf(das,"%lf",&sim.frict); fgets(tt,200,das);  //friction coefficient of NPs
  fscanf(das,"%lf",&sim.v_d); fgets(tt,200,das);  //speed of the laminar flow difference
  fscanf(das,"%d",&Nbin); fgets(tt,200,das);  //number of interval for generating histogram of g(r)
  fscanf(das,"%d",&samp); fgets(tt,200,das);      //number of interval for sampling
  fscanf(das,"%d",&tra_samp); fgets(tt,200,das);      //number of interval for trajectory sampling
  fscanf(das,"%lf",&sim.ddx); fgets(tt,200,das);   //Size of the grid for iso-surfaces
  fscanf(das,"%lf",&sim.beta); fgets(tt,200,das);   //fugacity
  fscanf(das,"%lf",&sim.str_ss); fgets(tt,200,das);   //strength of the slip-spring
  fscanf(das,"%lf",&sim.freq); fgets(tt,200,das);   //frequency to move Slip-link
  fgets(tt,200,das);
  fclose(das);
  sim.v_d *= box.length3;
}



void initial_all (void)
{
  double vol_R;
  buffer_count = 0;
  en.kinet = 0.0;
  freq = sim.freq/sim.dt;
  sim.bb=1.0/sqrt(31.0);
  F_coef = -3.0/(sim.bb*sim.bb);


  sim.dp2 = sim.dp*sim.dp;
  sqrdt = sqrt(2.0/sim.dt);
  sim.N = sim.N0 + sim.N1;
  sim.Re = sim.bb*sqrt(sim.N-1);
  sim.in_pol = density*sim.Re*sim.Re*sim.Re/sim.N;
  box.vol = box.length1 * box.length2 * box.length3;
  vol_R = 4.0/3.0*PI*sim.Nano_R*sim.Nano_R*sim.Nano_R;
  sim.Nano_num = (int)(box.vol * sim.Nano_frac / vol_R);
  box.NP_vol = sim.Nano_num * vol_R;
  sim.Nano_frac = box.NP_vol / box.vol;
  box.bulk_vol = box.vol - box.NP_vol;
  box.chains = int((box.bulk_vol*density)/sim.N);
  box.natoms0 = box.chains * sim.N0;
  box.natoms1 = box.chains * sim.N1;
  box.natoms = box.natoms0 + box.natoms1;
  density=box.natoms/box.bulk_vol;
  sim.in_pol = density*sim.Re*sim.Re*sim.Re/sim.N;
  R_cut = pow(double(3.0/(4.0*PI)*Nint/density),1.0/3.0);
  nor=8.0/Nint;
  L_1=int(box.length1/(1.0*R_cut));
  L_2=int(box.length2/(1.0*R_cut));
  L_3=int(box.length3/(1.0*R_cut));

  if(L_1==0)L_1=1;
  if(L_2==0)L_2=1;
  if(L_3==2)L_3=1;


  del_L1=box.length1/double(L_1);                             //construct linked list                               
  del_L2=box.length2/double(L_2);
  del_L3=box.length3/double(L_3);


  del_sphere=pow(1.0/(density*sim.Nano_den),1.0/3.0);
  double del_sph=pow(1.0/(density),1.0/3.0);
  sim.Nano_bead = 0;
  int Nbead = 0;
  for(double x=-sim.Nano_R;x<=sim.Nano_R;x+=del_sph)
    for(double y=-sim.Nano_R;y<=sim.Nano_R;y+=del_sph)
      for(double z=-sim.Nano_R;z<=sim.Nano_R;z+=del_sph)
	{
	  if((x*x+y*y+z*z)<=sim.Nano_R*sim.Nano_R)Nbead++;
	}

  for(double x=-sim.Nano_R;x<=sim.Nano_R;x+=del_sphere)
    for(double y=-sim.Nano_R;y<=sim.Nano_R;y+=del_sphere)
      for(double z=-sim.Nano_R;z<=sim.Nano_R;z+=del_sphere)
	{
	  if((x*x+y*y+z*z)<=sim.Nano_R*sim.Nano_R)sim.Nano_bead++;
	}

  sim.Nano_den = double(sim.Nano_bead)/double(Nbead);      
  sim.Nano_weight = 1.0*double(Nbead);

  L=L_1*L_2*L_3;                  //number of cells in the cell-list


  ndfx = int(box.length1/sim.ddx);
  ndfy = int(box.length2/sim.ddx);
  ndfz = int(box.length3/sim.ddx);

  dLx_field = box.length1/double(ndfx);
  dLy_field = box.length2/double(ndfy);
  dLz_field = box.length3/double(ndfz);

  ss_life=(int *)calloc(CHECK_SS,sizeof(int));
  N_dist=(int *)calloc(sim.N-1,sizeof(int));
  Z_dump=(int *)calloc(sim.N+1,sizeof(int));
  atom=(atoms *)calloc(box.natoms,sizeof(atoms));
  nano_atom=(atoms *)calloc(sim.Nano_bead,sizeof(atoms));
  Re=(double *)calloc(box.chains,sizeof(double));
  if(SSlink==true)dist_c=(int *)calloc(sim.N,sizeof(int));

if(MSD == true){
    atom_v=(vect *)calloc(box.natoms,sizeof(vect));
    if(sim.Nano_num > 0)nano_v=(vect *)calloc(sim.Nano_num,sizeof(vect));
    chain_mid=(vect *)calloc(box.chains,sizeof(vect));
    ntime = (double *)calloc(sim.tmax,sizeof(double));
    if(sim.Nano_num > 0)nano_ntime = (double *)calloc(sim.nano_tmax,sizeof(double));
    chain_ntime = (double *)calloc(sim.tmax,sizeof(double));
    r2f = (double *)calloc(sim.tmax,sizeof(double));
    r2_dx = (double *)calloc(sim.tmax,sizeof(double));
    r2_dy = (double *)calloc(sim.tmax,sizeof(double));
    r2_dz = (double *)calloc(sim.tmax,sizeof(double));
    if(sim.Nano_num > 0){
      nano_r2f = (double *)calloc(sim.nano_tmax,sizeof(double));
      nano_dx = (double *)calloc(sim.nano_tmax,sizeof(double));
      nano_dy = (double *)calloc(sim.nano_tmax,sizeof(double));
      nano_dz = (double *)calloc(sim.nano_tmax,sizeof(double));
    }
    chain_r2f = (double *)calloc(sim.tmax,sizeof(double));

    time0 = (double *)calloc(sim.t0max,sizeof(double));
    if(sim.Nano_num > 0)nano_time0 = (double *)calloc(sim.nano_t0max,sizeof(double));
    chain_time0 = (double *)calloc(sim.t0max,sizeof(double));
    X0=(double *)calloc(box.natoms*sim.t0max,sizeof(double));
    Y0=(double *)calloc(box.natoms*sim.t0max,sizeof(double));
    Z0=(double *)calloc(box.natoms*sim.t0max,sizeof(double));
    
      
    chain_X0=(double *)calloc(box.chains*sim.t0max,sizeof(double));
    chain_Y0=(double *)calloc(box.chains*sim.t0max,sizeof(double));
    chain_Z0=(double *)calloc(box.chains*sim.t0max,sizeof(double));
    
    if(sim.Nano_num > 0){
      nano_X0=(double *)calloc(sim.Nano_num*sim.nano_t0max,sizeof(double));
      nano_Y0=(double *)calloc(sim.Nano_num*sim.nano_t0max,sizeof(double));
      nano_Z0=(double *)calloc(sim.Nano_num*sim.nano_t0max,sizeof(double));
    }
  }


  SS[0]=0;SS[1]=0;SS[2]=0;
  SS[3]=0;SS[4]=0;SS[5]=0;
  SS[6]=0;SS[7]=0;SS[8]=0;

  SSB[0]=0;SSB[1]=0;SSB[2]=0;
  SSB[3]=0;SSB[4]=0;SSB[5]=0;
  SSB[6]=0;SSB[7]=0;SSB[8]=0;

  SSNB[0]=0;SSNB[1]=0;SSNB[2]=0;
  SSNB[3]=0;SSNB[4]=0;SSNB[5]=0;
  SSNB[6]=0;SSNB[7]=0;SSNB[8]=0;

  SS_f[0]=0;SS_f[1]=0;SS_f[2]=0;
  SS_f[3]=0;SS_f[4]=0;SS_f[5]=0;
  SS_f[6]=0;SS_f[7]=0;SS_f[8]=0;

  SSHC[0]=0;SSHC[1]=0;SSHC[2]=0;
  SSHC[3]=0;SSHC[4]=0;SSHC[5]=0;
  SSHC[6]=0;SSHC[7]=0;SSHC[8]=0;





  //Re=(double *)calloc(box.chains+1,sizeof(double));
  nano=(nanos *)calloc(sim.Nano_num,sizeof(nanos));
  head=(int *)calloc(L,sizeof(int));
  lscl=(int *)calloc(box.natoms,sizeof(int));
  hist_n2p=(double *)calloc(Nbin,sizeof(double));
  hist_n2s=(double *)calloc(Nbin,sizeof(double));
  hist_n2n=(double *)calloc(Nbin,sizeof(double));
  hist_p2p=(double *)calloc(Nbin,sizeof(double));
  hist_ss=(double *)calloc(Nbin,sizeof(double));
  hist_Q=(double *)calloc(Nbin,sizeof(double));
  hist_velocity=(double *)calloc(Nbin,sizeof(double));
  force_bead = (Force *)calloc(box.natoms,sizeof(Force));
  if(sim.Nano_num >=1)force_nano = (NanoForce *)calloc(sim.Nano_num,sizeof(NanoForce));


  if(SSlink == true){
    sim.bsl = sim.bb;
    sim.rss = 2.5*sim.bsl;
  }

  SS_F_coef = -3.0/(sim.bsl*sim.bsl)*sim.str_ss;
  den=(double)box.natoms/(double)(zntotal);
  denPC=(double)(sim.Nano_num)/(double)zntotal;


  del_checkx=box.length1/zntotal;
  del_checky=box.length2/zntotal;
  del_checkz=box.length3/zntotal;

  read_config();

  read_check();
  if(box.length3<=box.length2&&box.length3<=box.length1)Lcor=box.length3;
  else if(box.length1<=box.length2&&box.length1<=box.length3)Lcor=box.length1;
  else if(box.length2<=box.length1&&box.length2<=box.length3)Lcor=box.length2;
  delt_r=Lcor/(2*Nbin);


  get_last_time();

  /*
   * initiate global variable on gpu
   */
   double a,b,c,d,param;
   a = 4.0/3.0*(R_cut/2.0)*(R_cut/2.0)*(R_cut/2.0);
   b = (R_cut/2.0)*(R_cut/2.0);
   param = nor*nor*density;
   c = -1.0*param*sim.kappa*PI;
   d = param*sim.kappa*PI;
   
   cudaMemcpyToSymbol(A,&a, sizeof(double));
   cudaMemcpyToSymbol(B,&b, sizeof(double));
   cudaMemcpyToSymbol(C,&c, sizeof(double));
   cudaMemcpyToSymbol(D,&d, sizeof(double));
   
  cudaMemcpyToSymbol(Cuda_LeesEd,&LeesEd, sizeof(bool));
  cudaMemcpyToSymbol(CudaLength1,&box.length1, sizeof(double));
  cudaMemcpyToSymbol(CudaLength2,&box.length2, sizeof(double));
  cudaMemcpyToSymbol(CudaLength3,&box.length3, sizeof(double));
  cudaMemcpyToSymbol(f_coeff,&F_coef, sizeof(double));
  cudaMemcpyToSymbol(SS_f_coeff,&SS_F_coef, sizeof(double));
  cudaMemcpyToSymbol(Hc_Magnitude,&sim.core_rep, sizeof(double));  
  cudaMemcpyToSymbol(Hc_Attraction,&hc_attraction, sizeof(double));  
  cudaMemcpyToSymbol(Cuda_NanoRc,&sim.Nano_Rc, sizeof(double));  
  cudaMemcpyToSymbol(Cuda_NanoR,&sim.Nano_R, sizeof(double));    
  cudaMemcpyToSymbol(CudaNum_Bead,&box.natoms, sizeof(int));
  cudaMemcpyToSymbol(CudaNum_Nano,&sim.Nano_num, sizeof(int));
  cudaMemcpyToSymbol(Cuda_Rcut,&R_cut, sizeof(double));
  cudaMemcpyToSymbol(Cuda_Chi,&sim.chi, sizeof(double));
  cudaMemcpyToSymbol(Cuda_lambda0,&sim.lambda0, sizeof(double));
  cudaMemcpyToSymbol(Cuda_lambda1,&sim.lambda1, sizeof(double));
  cudaMemcpyToSymbol(Cuda_Kappa,&sim.kappa, sizeof(double));
  cudaMemcpyToSymbol(Cuda_Density,&density, sizeof(double));
  cudaMemcpyToSymbol(Cuda_Nanoden,&sim.Nano_den, sizeof(double));
  cudaMemcpyToSymbol(Cuda_Nor,&nor, sizeof(double));

  cudaMemcpyToSymbol(Cuda_L1,&L_1, sizeof(int));
  cudaMemcpyToSymbol(Cuda_L2,&L_2, sizeof(int));
  cudaMemcpyToSymbol(Cuda_L3,&L_3, sizeof(int));
  

  cudaMemcpyToSymbol(Cuda_dL1,&del_L1, sizeof(double));
  cudaMemcpyToSymbol(Cuda_dL2,&del_L2, sizeof(double));
  cudaMemcpyToSymbol(Cuda_dL3,&del_L3, sizeof(double));
  
  cudaMalloc((void **)&c_atom, box.natoms*sizeof(atoms));
  cudaMalloc((void **)&c_atom_v, box.natoms*sizeof(vect));
  cudaMalloc((void **)&c_ss, NSS*sizeof(ss_t));
  cudaMalloc((void **)&c_atomf, box.natoms*sizeof(Force));
  
  cudaMalloc((void **)&c_nanof, sim.Nano_num*sizeof(NanoForce));
  cudaMalloc((void **)&c_nano, sim.Nano_num*sizeof(nanos));
  cudaMalloc((void **)&c_nano_v,sim.Nano_num*sizeof(vect));
  
  cudaMalloc((void **)&c_Nanobead,sim.Nano_bead*sizeof(atoms));
    
  cudaMemcpy(c_atom,atom,box.natoms*sizeof(atoms), cudaMemcpyHostToDevice);
  cudaMemcpy(c_atom_v,atom_v,box.natoms*sizeof(vect), cudaMemcpyHostToDevice);
  cudaMemcpy(c_nano,nano,sim.Nano_num*sizeof(nanos), cudaMemcpyHostToDevice);
  cudaMemcpy(c_nano_v,nano_v,sim.Nano_num*sizeof(vect), cudaMemcpyHostToDevice);
}


void get_last_time(void){
  ifstream ifs("Gxz_stress.dat",ifstream::in);
  if(!ifs.is_open())
    {
      Stress_Laststep = 0;
      return;
    }else{
    string line = getLastLine(ifs);
    Stress_Laststep = (atof(line.c_str())/sim.dt)+gxz_samp;
  }
}
/*void read_config()
  {
  int bead_num,type,nano_num;
  FILE *config;
  config=fopen("./config.dat","r");
  char junk[80];
  double x,y,z;

  if (config==NULL)
  {
  initial_nano();
  initial_config();
  initial_vel();
  dx_LE = 0;
  return;
  }

  fscanf(config,"%d",&bead_num);

  if (bead_num!=box.natoms+sim.Nano_bead)
  {
  fprintf(stdout,"\n\nBEAD NUMBER IN CONFIG FILE DOES NOT MATCH WITH INPUT FILE\n");
  exit(EXIT_FAILURE);
  }

  fgets(junk,80,config);

  for(int i=0;i < box.natoms;i++)
  {
  fscanf(config,"%d%lf%lf%lf",&type,&x,&y,&z);
  atom[i].pos.x = x;
  atom[i].pos.y = y;
  atom[i].pos.z = z;
  atom[i].type=type;
  fgets(junk,80,config);
  }

  for(int i=0;i < sim.Nano_bead;i++)
  {
  fscanf(config,"%d%lf%lf%lf",&type,&x,&y,&z);
  nano_atom[i].pos.x = x;
  nano_atom[i].pos.y = y;
  nano_atom[i].pos.z = z;
  nano_atom[i].type=type;
  fgets(junk,80,config);
  }

  fclose(config);

  config=fopen("./nano.dat","r");

  if (config == NULL)
  {   
  fprintf(stdout,"\n\nNANO FILE NOT FOUND\n");
  exit(EXIT_FAILURE);
  }

  fscanf(config,"%d",&nano_num);

  if (nano_num!=sim.Nano_num)
  {
  fprintf(stdout,"\n\nNANO NUMBER IN CONFIG FILE DOESNOT MATCH WITH INPUT FILE\n");
  exit(EXIT_FAILURE);
  }

  fgets(junk,80,config);


  for(int i=0;i<sim.Nano_num;i++)
  {
  fscanf(config,"%lf%lf%lf",&x,&y,&z);
  nano[i].pos.x=x;
  nano[i].pos.y=y;
  nano[i].pos.z=z;
  fgets(junk,80,config);
  }
  fclose(config);

  config=fopen("./vel_check.dat","r");
  if (config == NULL)
  {
  initial_vel();
  return;
  }
  for(int i=0;i<box.natoms;i++)fread(&atom[i].vel,sizeof(vect),1,config);
  for(int i=0;i<sim.Nano_num;i++)fread(&nano[i].vel,sizeof(vect),1,config);
  fclose(config);

  if(LeesEd == true){
  config=fopen("./dx_LE.dat","r");
  if(config == NULL){
  dx_LE == 0;
  }else{
  fread(&dx_LE,sizeof(double),1,config);
  fclose(config);
  }
  }
  }*/

void read_config()
{
  FILE *config;
  config=fopen("./config.dat","r");
  int bead_num,nano_num;
  if (config==NULL)
    {
      initial_nano();
      initial_config();
      dx_LE = 0;
      return;
    }
  fread(&bead_num,sizeof(int),1,config);  
  if (bead_num!=box.natoms+sim.Nano_bead)
    {
      fprintf(stdout,"\n\nBEAD NUMBER IN CONFIG FILE DOES NOT MATCH WITH INPUT FILE\n");
      exit(EXIT_FAILURE);
    }
  fread(&atom[0],sizeof(atoms),box.natoms,config);
  fread(&nano_atom[0],sizeof(atoms),sim.Nano_bead,config);
  fread(&dx_LE,sizeof(double),1,config);  
  fclose(config);

  config=fopen("./nano.dat","r");
  if (config == NULL)
    {
      fprintf(stdout,"\n\nNANO FILE NOT FOUND\n");
      exit(EXIT_FAILURE);
    }
  fread(&nano_num,sizeof(int),1,config);  
  if (nano_num!=sim.Nano_num)
    {
      fprintf(stdout,"\n\nNANO NUMBER IN CONFIG FILE DOESNOT MATCH WITH INPUT FILE\n");
      exit(EXIT_FAILURE);
    }  

  fread(&nano[0],sizeof(nanos),sim.Nano_num,config);
  fclose(config);
  
  if(SSlink == true){
      config = fopen("./sslink.dat", "r");
      if(config != NULL){
      fread(&ss[0],sizeof(ss_t),NSS,config);
      fread(&sim.nss,sizeof(int),1,config);
      fclose(config);
       Read_ss = 1;
      }else {
		  fclose(config);
		  Read_ss = 0;
		  for(int i=0;i<box.natoms;i++)atom[i].ss_tag=0;
	  }

  }

}



void initial_nano()
{
  double dx,dy,dz;
  double dr2;
  double dmin = (sim.Nano_R+sim.Nano_Rc)*(sim.Nano_R+sim.Nano_Rc);
  for(int i=0;i<sim.Nano_num;)
    {
      nano[i].pos.z=(ran_num_double(0,1)-0.5)*box.length3;
      nano[i].pos.x=(ran_num_double(0,1)-0.5)*box.length1;
      nano[i].pos.y=(ran_num_double(0,1)-0.5)*box.length2;
      overlap=0;
      for(int k=0;k<i;k++)
	{
	  dx=nano[i].pos.x-nano[k].pos.x;
	  dy=nano[i].pos.y-nano[k].pos.y;
	  dz=nano[i].pos.z-nano[k].pos.z;
	  dx = modulo(dx,box.length1);
	  dy = modulo(dy,box.length2);
	  dz = modulo(dz,box.length3);
	  dr2=dx*dx+dy*dy+dz*dz;
	  if(dr2<=dmin)
	    {
	      overlap=1;
	      break;
	    }
	}
      if(overlap==1)continue;       
      i++;
    }
  int j=0;
  for(double x=-sim.Nano_R;x<=sim.Nano_R;x+=del_sphere)
    for(double y=-sim.Nano_R;y<=sim.Nano_R;y+=del_sphere)
      for(double z=-sim.Nano_R;z<=sim.Nano_R;z+=del_sphere)
	{
	  if((x*x+y*y+z*z)<=sim.Nano_R*sim.Nano_R){
	    nano_atom[j].pos.x=x;
	    nano_atom[j].pos.y=y;
	    nano_atom[j].pos.z=z;
	    nano_atom[j].type=2;
	    j++;
	  }
	}
}

void isoverlap(vect pos)
{
  double dx,dy,dz;
  double dr2;
  overlap = 0;
  for(int i = 0;i < sim.Nano_num;i++)
    {
      dx = nano[i].pos.x - pos.x;
      dy = nano[i].pos.y - pos.y;
      dz = nano[i].pos.z - pos.z;

      dx = modulo(dx,box.length1);
      dy = modulo(dy,box.length2);
      dz = modulo(dz,box.length3);
      dr2 = dx * dx + dy * dy + dz * dz;
      if(dr2 < (sim.Nano_Rc * sim.Nano_Rc)){
	overlap = 1;
	return;
      }
    }
}

void initial_config(void)
{
  double p1,p2,s;
  double x,y,z;
  vect temp;
  for(int i=0;i<box.chains;)
    {
      x=ran_num_double(0,1);
      y=ran_num_double(0,1);
      z=ran_num_double(0,1);
      atom[i*sim.N].pos.x=(x-0.5)*box.length1;
      atom[i*sim.N].pos.y=(y-0.5)*box.length2;
      atom[i*sim.N].pos.z=(z-0.5)*box.length3;
      temp.x=atom[i*sim.N].pos.x;
      temp.y=atom[i*sim.N].pos.y;
      temp.z=atom[i*sim.N].pos.z;
      isoverlap(temp);
      if(overlap==1)continue;
      else {
	atom[i*sim.N].type=0;
	atom[i*sim.N].ss_tag = 0;
	pbc_one(i*sim.N);
      }

      for(int j=1;j<sim.N;)
	{
	  p1=2.0*ran_num_double(0,1)-1;
	  p2=2.0*ran_num_double(0,1)-1;
	  s=p1*p1+p2*p2;
	  if(s<1){
	    atom[i*sim.N+j].pos.x=atom[i*sim.N+j-1].pos.x+2.0*sqrt(1.0-s)*p1*sim.bb;
	    atom[i*sim.N+j].pos.y=atom[i*sim.N+j-1].pos.y+2.0*sqrt(1.0-s)*p2*sim.bb;
	    atom[i*sim.N+j].pos.z=atom[i*sim.N+j-1].pos.z+(1.0-s*2.0)*sim.bb;
	    temp.x=atom[i*sim.N+j].pos.x;
	    temp.y=atom[i*sim.N+j].pos.y;
	    temp.z=atom[i*sim.N+j].pos.z;
	    isoverlap(temp);
	    if(overlap==1)continue;
	    if(j<sim.N0)atom[i*sim.N+j].type=0;
	    else atom[i*sim.N+j].type=1;
	    atom[i*sim.N+j].ss_tag = 0;
	    pbc_one(i*sim.N+j);
	    j++;
	  }
	}
      i++;
    }
}

int Round(double x)
{
  if(x >= 0) return (int)(x+0.5);
  else return (int)(x-0.5);
}

__device__ int Cuda_Round(double x)
{
  if(x >= 0) return (int)(x+0.5);
  else return (int)(x-0.5);
}

double modulo(double x,double y)
{
  while(x >= 0.5*y)x-=y;
  while(x <= -0.5*y)x+=y;
  return x;
}


void msd(int sw){
  FILE *io1;
  double dx,dy,dz;
  int delt;
  switch(sw){
  case 0:
    for(int i = 0;i < sim.tmax; i++){
      ntime[i] = 0;
      r2f[i] = 0;
      r2_dx[i] = 0;
      r2_dy[i] = 0;
      r2_dz[i] = 0;
    }
    ntel = 0;
    t0 = 0;
    break;
  case 1:
    double r2f_sum;
    double r2x_sum;
    double r2y_sum;
    double r2z_sum;
    if(++ntel%sim.it0 ==0 && t0<sim.t0max)
      {
	time0[t0] = ntel;
	for(int i =0;i < box.natoms;i++){
	  X0[i*sim.t0max+t0] = atom_v[i].x;
	  Y0[i*sim.t0max+t0] = atom_v[i].y;
	  Z0[i*sim.t0max+t0] = atom_v[i].z;
	}
	
	t0++;
      }
    for(int i=0;i<t0;i++){
      delt = ntel-time0[i];
      if(delt<sim.tmax){
	ntime[delt]++;
	r2f_sum = 0.0;
	r2x_sum = 0.0;
	r2y_sum = 0.0;
	r2z_sum = 0.0;
	for(int j=0;j<box.natoms;j++){
	  dx = atom_v[j].x-X0[j*sim.t0max+i];
	  dy = atom_v[j].y-Y0[j*sim.t0max+i];
	  dz = atom_v[j].z-Z0[j*sim.t0max+i];
	  r2f_sum += dx*dx+dy*dy+dz*dz;
	  r2x_sum += dx*dx;
	  r2y_sum += dy*dy;
	  r2z_sum += dz*dz;
	}
	r2f[delt] += r2f_sum;
	r2_dx[delt] += r2x_sum;
	r2_dy[delt] += r2y_sum;
	r2_dz[delt] += r2z_sum;
      }
    }
    break;
  case 2:
    char buff[20];
    sprintf(buff,"./%d_polymer_msd.out",sim.N);
    io1 = fopen(buff,"w");
    for(int i=0;i<sim.tmax;i++)if(ntime[i] == 0)break;
      else fprintf(io1,"%lf\t%lf\t%lf\t%lf\t%lf\n",sim.dt*i*msd_samp,r2f[i]/(double)(box.natoms*ntime[i]),r2_dx[i]/(double)(box.natoms*ntime[i]),r2_dy[i]/(double)(box.natoms*ntime[i]),r2_dz[i]/(double)(box.natoms*ntime[i]));
    fclose(io1);
  }
}

void nano_msd(int sw){
  FILE *io1;
  double dx,dy,dz;
  int delt;
  switch(sw){
  case 0:
    for(int i = 0;i < sim.nano_tmax; i++){
      nano_ntime[i] = 0;
      nano_r2f[i] = 0;
      nano_dx[i] = 0;
      nano_dy[i] = 0;
      nano_dz[i] = 0;
    }
    nano_ntel = 0;
    nano_t0 = 0;
    break;
  case 1:
    if(++nano_ntel%sim.nano_it0 ==0 && nano_t0<sim.nano_t0max)
      {
	nano_time0[nano_t0] = nano_ntel;
	for(int i =0;i < sim.Nano_num;i++){
	  nano_X0[i*sim.nano_t0max+nano_t0] = nano_v[i].x;
	  nano_Y0[i*sim.nano_t0max+nano_t0] = nano_v[i].y;
	  nano_Z0[i*sim.nano_t0max+nano_t0] = nano_v[i].z;
	}
	nano_t0++;
      }
    for(int i=0;i<nano_t0;i++){
      delt = nano_ntel-nano_time0[i];
      if(delt<sim.nano_tmax){
	nano_ntime[delt]++;
	for(int j=0;j<sim.Nano_num;j++){
	  dx = nano_v[j].x-nano_X0[j*sim.nano_t0max+i];
	  dy = nano_v[j].y-nano_Y0[j*sim.nano_t0max+i];
	  dz = nano_v[j].z-nano_Z0[j*sim.nano_t0max+i];
	  nano_r2f[delt] += dx*dx+dy*dy+dz*dz;
	  nano_dx[delt] += dx*dx;
	  nano_dy[delt] += dy*dy;
	  nano_dz[delt] += dz*dz;
	}
      }
    }
    break;
  case 2:
    char buff[20];
    sprintf(buff,"./%d_nano_msd.out",sim.N);
    io1 = fopen(buff,"w");
    for(int i=0;i<sim.nano_tmax;i++)if(nano_ntime[i] == 0)break;
      else fprintf(io1,"%lf\t%lf\t%lf\t%lf\t%lf\n",sim.dt*i*msd_samp,nano_r2f[i]/(double)(sim.Nano_num*nano_ntime[i]),nano_dx[i]/(double)(sim.Nano_num*nano_ntime[i]),nano_dy[i]/(double)(sim.Nano_num*nano_ntime[i]),nano_dz[i]/(double)(sim.Nano_num*nano_ntime[i]));
    fclose(io1);
  }
}

void chain_msd(int sw){
  FILE *io1;
  double dx,dy,dz;
  int delt;
  switch(sw){
  case 0:
    for(int i = 0;i < sim.tmax; i++){
      chain_ntime[i] = 0;
      chain_r2f[i] = 0;
    }
    chain_ntel = 0;
    chain_t0 = 0;
    break;
  case 1:
    if(++chain_ntel%sim.it0 ==0 && chain_t0<sim.t0max)
      {
	chain_time0[chain_t0] = chain_ntel;
	for(int i =0;i < box.chains;i++){
	  chain_X0[i*sim.t0max+chain_t0] = chain_mid[i].x;
	  chain_Y0[i*sim.t0max+chain_t0] = chain_mid[i].y;
	  chain_Z0[i*sim.t0max+chain_t0] = chain_mid[i].z;
	}
	chain_t0++;
      }
    for(int i=0;i<chain_t0;i++){
      delt = chain_ntel-chain_time0[i];
      if(delt<sim.tmax){
	chain_ntime[delt]++;
	for(int j=0;j<box.chains;j++){
	  dx = chain_mid[j].x-chain_X0[j*sim.t0max+i];
	  dy = chain_mid[j].y-chain_Y0[j*sim.t0max+i];
	  dz = chain_mid[j].z-chain_Z0[j*sim.t0max+i];
	  chain_r2f[delt] += dx*dx+dy*dy+dz*dz;
	}
      }
    }
    break;
  case 2:
    char buff[20];
    sprintf(buff,"./%d_chain_msd.out",sim.N);
    io1 = fopen(buff,"w");
    for(int i=0;i<sim.tmax;i++)if(chain_ntime[i] == 0)break;
      else fprintf(io1,"%lf\t%lf\n",sim.dt*i*msd_samp,chain_r2f[i]/(double)(box.chains*chain_ntime[i]));;
    fclose(io1);
  }
}
__device__ double Cuda_modulo(double x,double y)
{
  while(x >= 0.5*y)x-=y;
  while(x <= -0.5*y)x+=y;
  return x;
}
void pbc_one (int i) {
  double dx,dy,dz;

  dx = atom[i].pos.x;
  dy = atom[i].pos.y;
  dz = atom[i].pos.z;

  while(dz >= box.length3/2.0) {

    dz -= box.length3; 

    if (LeesEd==true) {
      dx -= (dx_LE+sim.v_d*sim.dt);
    } 
  }

  while(dz <= -box.length3/2.0) {
    dz += box.length3; 
    if (LeesEd==true) {
      dx += (dx_LE+sim.v_d*sim.dt);
    }
  }

  dx = modulo(dx,box.length1);
  dy = modulo(dy,box.length2);

  atom[i].pos.x = dx;
  atom[i].pos.y = dy;
  atom[i].pos.z = dz;
}


__device__ void Cuda_Pbc_One(double &dx,double &dy,double &dz,double dx_le,double v_d,double dt){

  while(dz >= CudaLength3/2.0) {	
    dz -= CudaLength3; 
    if (Cuda_LeesEd==true) {
      dx -= (dx_le+v_d*dt);
    } 
  }

  while(dz <= -CudaLength3/2.0) {
    dz += CudaLength3; 
    if (Cuda_LeesEd==true) {
      dx += (dx_le+v_d*dt);
    }
  }

  dx = Cuda_modulo(dx,CudaLength1);
  dy = Cuda_modulo(dy,CudaLength2);

}


void pbc_one(double &dx,double &dy,double &dz){

  while(dz >= box.length3/2.0) {	
    dz -= box.length3; 
    if (LeesEd==true) {
      dx -= (dx_LE+sim.v_d*sim.dt);
    } 
  }

  while(dz <= -box.length3/2.0) {
    dz += box.length3; 
    if (LeesEd==true) {
      dx += (dx_LE+sim.v_d*sim.dt);
    }
  }

  dx = modulo(dx,box.length1);
  dy = modulo(dy,box.length2);

}

void nano_pbc_one (int i) {
  double dx,dy,dz;

  dx = nano[i].pos.x;
  dy = nano[i].pos.y;
  dz = nano[i].pos.z;

  if(dz >= box.length3/2.0) {

    dz -= box.length3; 

    if (LeesEd==true) {
      dx -= (dx_LE+sim.v_d*sim.dt);
    } 
  }

  if(dz <= -box.length3/2.0) {
    dz += box.length3; 
    if (LeesEd==true) {
      dx += (dx_LE+sim.v_d*sim.dt);
    }
  }

  dx = modulo(dx,box.length1);
  dy = modulo(dy,box.length2);

  nano[i].pos.x = dx;
  nano[i].pos.y = dy;
  nano[i].pos.z = dz;
}

void xyz_iniconfig(){
  FILE *io;
  double x,y,z;
  io=fopen("./config.xyz","w");
  fprintf(io,"\t%d\n",box.natoms+sim.Nano_num*sim.Nano_bead);
  fprintf(io,"\n");
  for(int i=0; i<box.natoms; i++){
    if(atom[i].type == 0){
      fprintf(io," A   ");
    }else if(atom[i].type ==1){
      fprintf(io," B   ");
    }
    fprintf(io," %15.6lf\t",atom[i].pos.x);
    fprintf(io," %15.6lf\t",atom[i].pos.y);
    fprintf(io," %15.6lf\n",atom[i].pos.z);
  }

  for(int i=0; i<sim.Nano_num; i++)
    for(int j=0;j<sim.Nano_bead;j++){
      x=nano_atom[j].pos.x+nano[i].pos.x;
      x = modulo(x,box.length1);
      y = nano_atom[j].pos.y+nano[i].pos.y;
      y = modulo(y,box.length2);
      z=nano_atom[j].pos.z+nano[i].pos.z;
      z = modulo(z,box.length3);
      fprintf(io," E   ");
      fprintf(io," %15.6lf\t",x);
      fprintf(io," %15.6lf\t",y);
      fprintf(io," %15.6lf\n",z);
    }

  fclose(io);
}


/*void xyz_config (){
  FILE *io;
  double x,y,z;
  io=fopen("./config.xyz","w");
  fprintf(io,"\t%d\n",box.natoms+sim.Nano_num*sim.Nano_bead);
  fprintf(io,"\n");
  for(int i=0; i<box.natoms; i++){
  if(atom[i].type == 0){
  fprintf(io," A   ");
  }else if(atom[i].type ==1){
  fprintf(io," B   ");
  }
  fprintf(io," %15.6lf\t",atom[i].pos.x);
  fprintf(io," %15.6lf\t",atom[i].pos.y);
  fprintf(io," %15.6lf\n",atom[i].pos.z);
  }

  for(int i=0; i<sim.Nano_num; i++)
  for(int j=0;j<sim.Nano_bead;j++){
  x=nano_atom[j].pos.x+nano[i].pos.x;
  x = modulo(x,box.length1);
  y = nano_atom[j].pos.y+nano[i].pos.y;
  y = modulo(y,box.length2);
  z=nano_atom[j].pos.z+nano[i].pos.z;
  z = modulo(z,box.length3);
  fprintf(io," E   ");
  fprintf(io," %15.6lf\t",x);
  fprintf(io," %15.6lf\t",y);
  fprintf(io," %15.6lf\n",z);
  }

  fclose(io);

  io=fopen("./config.dat","w");
  fprintf(io,"\t%d\n",box.natoms+sim.Nano_bead);
  fprintf(io,"\n");

  for(int i=0; i<box.natoms; i++){
  fprintf(io," %d   ",atom[i].type);
  fprintf(io," %15.6lf\t",atom[i].pos.x);
  fprintf(io," %15.6lf\t",atom[i].pos.y);
  fprintf(io," %15.6lf\n",atom[i].pos.z);
  }

  for(int i=0; i<sim.Nano_bead; i++){
  fprintf(io," %d   ",nano_atom[i].type);
  fprintf(io," %15.6lf\t",nano_atom[i].pos.x);
  fprintf(io," %15.6lf\t",nano_atom[i].pos.y);
  fprintf(io," %15.6lf\n",nano_atom[i].pos.z);
  }

  fclose(io);

  io=fopen("./nano.dat","w");
  fprintf(io,"\t%d\n",sim.Nano_num);
  fprintf(io,"\n");
  for(int i=0; i<sim.Nano_num; i++){
  fprintf(io," %15.6lf\t",nano[i].pos.x);
  fprintf(io," %15.6lf\t",nano[i].pos.y);
  fprintf(io," %15.6lf\n",nano[i].pos.z);
  }
  fclose(io);

  io=fopen("./vel_check.dat","w");
  for(int i=0;i<box.natoms;i++)fwrite(&atom[i].vel,sizeof(vect),1,io);
  for(int i=0;i<sim.Nano_num;i++)fwrite(&nano[i].vel,sizeof(vect),1,io);
  fclose(io);

  if(LeesEd == true){
  io=fopen("./dx_LE.dat","w");
  fwrite(&dx_LE,sizeof(double),1,io);
  fclose(io);
  }
  }*/

void xyz_config (){
  FILE *io;
  double x,y,z;
  io=fopen("./config.xyz","w");
  fprintf(io,"\t%d\n",box.natoms+sim.Nano_num*sim.Nano_bead);
  fprintf(io,"\n");
  for(int i=0; i<box.natoms; i++){
    if(atom[i].type == 0){
      fprintf(io," A   ");
    }else if(atom[i].type ==1){
      fprintf(io," B   ");
    }
    fprintf(io," %15.6lf\t",atom[i].pos.x);
    fprintf(io," %15.6lf\t",atom[i].pos.y);
    fprintf(io," %15.6lf\n",atom[i].pos.z);
  }

  for(int i=0; i<sim.Nano_num; i++)
    for(int j=0;j<sim.Nano_bead;j++){
      x=nano_atom[j].pos.x+nano[i].pos.x;
      x = modulo(x,box.length1);
      y = nano_atom[j].pos.y+nano[i].pos.y;
      y = modulo(y,box.length2);
      z=nano_atom[j].pos.z+nano[i].pos.z;
      z = modulo(z,box.length3);
      fprintf(io," E   ");
      fprintf(io," %15.6lf\t",x);
      fprintf(io," %15.6lf\t",y);
      fprintf(io," %15.6lf\n",z);
    }

  fclose(io);

  io=fopen("./config.dat","w");
  int bead_num = box.natoms+sim.Nano_bead;
  fwrite(&bead_num,sizeof(int),1,io);
  fwrite(&atom[0],sizeof(atoms),box.natoms,io);
  fwrite(&nano_atom[0],sizeof(atoms),sim.Nano_bead,io);
  fwrite(&dx_LE,sizeof(double),1,io);
  fclose(io);

  io=fopen("./nano.dat","w");
  int nano_num = sim.Nano_num;
  fwrite(&nano_num,sizeof(int),1,io);
  fwrite(&nano[0],sizeof(nanos),sim.Nano_num,io);
  fclose(io);
  
  
    if(SSlink == true){
    io=fopen("./sslink.dat","w");
    fwrite(&ss[0],sizeof(ss_t),NSS,io);
    fwrite(&sim.nss,sizeof(int),1,io);
    fclose(io);
  }

}

void output (int sw){
  FILE *sout1; 
  char name1[50];
  double x,y,z;
  sprintf(name1,"./unitchainquantities.out");
  sout1= fopen(name1,"a");
  fprintf(stdout,"%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\n",cur_t,en.bonded_E/box.chains,en.nonbonded_E/box.chains,en.hardcore/box.chains,en.sl_ener/box.chains,en.total/box.chains,2.0*sim.nss/box.chains);
  cout<<"slip_link: "<<sim.nss<<"  energy per ss: "<<en.sl_ener/sim.nss<<"  energy per bond: "<<en.bonded_E/(box.chains*(sim.N-1))<<endl;

  fprintf(sout1,"%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\n",cur_t,en.bonded_E/box.chains,en.nonbonded_E/box.chains,en.hardcore/box.chains,en.sl_ener/box.chains,en.total/box.chains,2.0*sim.nss/box.chains);

  fclose(sout1);
  double sum = std::accumulate(RWD.begin(), RWD.end(), 0.0);
  double mean = sum/RWD.size();
  sout1= fopen("slip-link.out","a");
  fprintf(sout1,"%lf\t%lf\t%lf\n",cur_t,(2.0*sim.nss/box.chains),mean);
  fclose(sout1);
  RWD.clear();


  //cout<<"1: "<<check1/(double)(check_cou)<<" 2: "<<check2/(double)(check_cou)<<endl;
  if(sw==1){
    FILE *io;
    io=fopen("./dynamic.xyz","a");
    fprintf(io,"   %d\n",box.natoms+sim.Nano_num*sim.Nano_bead);
    fprintf(io,"\n");
    for(int i=0; i<box.natoms; i++){
      if(atom[i].type==0)fprintf(io," %s\t","A");
      if(atom[i].type==1)fprintf(io," %s\t","B");
      fprintf(io," %15.6lf  ",atom[i].pos.x);
      fprintf(io," %15.6lf  ",atom[i].pos.y);
      fprintf(io," %15.6lf\n",atom[i].pos.z);
    }

    for(int i=0; i<sim.Nano_num; i++)
      for(int j=0;j<sim.Nano_bead;j++){
	x=nano_atom[j].pos.x+nano[i].pos.x;
	x=modulo(x,box.length1);
	y=nano_atom[j].pos.y+nano[i].pos.y;
	y=modulo(y,box.length2);
	z=nano_atom[j].pos.z+nano[i].pos.z;
	z=modulo(z,box.length3);
	fprintf(io," %s\t","E");
	fprintf(io," %15.6lf\t",x);
	fprintf(io," %15.6lf\t",y);
	fprintf(io," %15.6lf\n",z);
      }
    fclose(io);
  }
  
  	ofstream ofs ("ss_lifetime.out", ofstream::out);
  	int sum_SSlife = 0;
	for(int i=0;i < CHECK_SS;i++)sum_SSlife += ss_life[i];
	for(int i=0;i < CHECK_SS;i++){
		ofs <<i*freq*sim.dt<<" "<<(double)ss_life[i]/sum_SSlife/(freq*sim.dt)<<endl;
	}
	ofs.close();
  
}

void den_pro(){
  FILE *iophi;
  double N=exstep;

  iophi=fopen("./phix.out","w");
  for(int i=0; i<zntotal; i++){
    fprintf(iophi,"%lf\t%lf\t%lf\t%lf\n  ",i*del_checkx,phiAx[i]/N,phiBx[i]/N,phix_na[i]/N);
  }

  fclose(iophi);
  iophi=fopen("./phiy.out","w");
  for(int i=0; i<zntotal; i++){
    fprintf(iophi,"%lf\t%lf\t%lf\t%lf\n  ",i*del_checky,phiAy[i]/N,phiBy[i]/N,phiy_na[i]/N);
  }
  fclose(iophi);
  iophi=fopen("./phiz.out","w");
  for(int i=0; i<zntotal; i++){
    fprintf(iophi,"%lf\t%lf\t%lf\t%lf\n  ",i*del_checkz,phiAz[i]/N,phiBz[i]/N,phiz_na[i]/N);
  }  
  fclose(iophi);
}

void den_avex()
{
  int kx;
  double m;
  for(int i=0;i<box.natoms;i++)
    {
      if(atom[i].type==0)
	{
	  kx=(int)floor((atom[i].pos.x+box.length1/2.0)/del_checkx);
	  phiAx[kx]=phiAx[kx]+1.0/den;
	}
      if(atom[i].type==1)
	{
	  kx=(int)floor((atom[i].pos.x+box.length1/2.0)/del_checkx);
	  phiBx[kx]=phiBx[kx]+1.0/den;
	}
    }
  for(int j=0;j<sim.Nano_num;j++)
    {
      m=nano[j].pos.x;
      kx=(int)floor((m+box.length1/2.0)/del_checkx);
      phix_na[kx]=phix_na[kx]+1.0/(denPC);
    }  
}

void den_avey()
{
  int ky;
  double m;
  for(int i=0;i<box.natoms;i++)
    {
      if(atom[i].type==0)
	{
	  ky=(int)floor((atom[i].pos.y+box.length2/2.0)/del_checky);
	  phiAy[ky]=phiAy[ky]+1.0/den;
	}
      if(atom[i].type==1)
	{
	  ky=(int)floor((atom[i].pos.y+box.length2/2.0)/del_checky);
	  phiBy[ky]=phiBy[ky]+1.0/den;
	}
    }

  for(int j=0;j<sim.Nano_num;j++)
    {
      m=nano[j].pos.y;
      ky=(int)floor((m+box.length2/2.0)/del_checky);
      phiy_na[ky]=phiy_na[ky]+1.0/(denPC);
    }  

}


void den_avez()
{
  int kz;
  double m;
  for(int i=0;i<box.natoms;i++)
    {
      if(atom[i].type==0)
	{
	  kz=(int)floor((atom[i].pos.z+box.length3/2.0)/del_checkz);
	  phiAz[kz]=phiAz[kz]+1.0/den;
	}
      if(atom[i].type==1)
	{
	  kz=(int)floor((atom[i].pos.z+box.length3/2.0)/del_checkz);
	  phiBz[kz]=phiBz[kz]+1.0/den;
	}
    }
  for(int j=0;j<sim.Nano_num;j++)
    {
      m=nano[j].pos.z;
      kz=(int)floor((m+box.length3/2.0)/del_checkz);
      phiz_na[kz]=phiz_na[kz]+1.0/(denPC);
    }  
}


void save_check()
{
  FILE *io;
  int Nsave=exstep;
  system("mv rdf.dat rdf.datbackup");
  system("mv phix.dat phix.datbackup");
  system("mv phiy.dat phiy.datbackup");
  system("mv phiz.dat phiz.datbackup");
  system("mv msd.dat msd.datbackup");
  system("mv stress.dat stress.datbackup");
  system("mv config.dat config.datbackup");
  system("mv nano.dat nano.datbackup");
  system("mv sslink.dat sslink.datbackup");
  system("cp Gxy_stress.dat Gxy_stress.datbackup");
  system("cp Gxz_stress.dat Gxz_stress.datbackup");
  system("cp Gyz_stress.dat Gyz_stress.datbackup");
  io=fopen("./rdf.dat","w");
  fwrite(&Nsave,sizeof(int),1,io);
  fwrite(&hist_n2p[0],sizeof(double),Nbin,io);
  fwrite(&hist_n2n[0],sizeof(double),Nbin,io);
  fwrite(&hist_p2p[0],sizeof(double),Nbin,io);
  fclose(io);

  io=fopen("./phix.dat","w");

  for(int i=0; i<zntotal; i++)fprintf(io,"%d\t%lf\t%lf\t%lf\n  ",i,phiAx[i],phiBx[i],phix_na[i]);
  fclose(io);

  io=fopen("./phiy.dat","w");

  for(int i=0; i<zntotal; i++)fprintf(io,"%d\t%lf\t%lf\t%lf\n  ",i,phiAy[i],phiBy[i],phiy_na[i]);
  fclose(io);

  io=fopen("./phiz.dat","w");

  for(int i=0; i<zntotal; i++)fprintf(io,"%d\t%lf\t%lf\t%lf\n  ",i,phiAz[i],phiBz[i],phiz_na[i]);
  fclose(io);

 if(MSD == true){
    io=fopen("./msd.dat","w");
    fwrite(&atom_v[0],sizeof(vect),box.natoms,io);
    if(sim.Nano_num>0)fwrite(&nano_v[0],sizeof(vect),sim.Nano_num,io);
    fwrite(&chain_mid[0],sizeof(vect),box.chains,io);
    fwrite(&ntime[0],sizeof(double),sim.tmax,io);
    if(sim.Nano_num>0)fwrite(&nano_ntime[0],sizeof(double),sim.nano_tmax,io);
    fwrite(&chain_ntime[0],sizeof(double),sim.tmax,io);
    fwrite(&r2f[0],sizeof(double),sim.tmax,io);
    fwrite(&r2_dx[0],sizeof(double),sim.tmax,io);
    fwrite(&r2_dy[0],sizeof(double),sim.tmax,io);
    fwrite(&r2_dz[0],sizeof(double),sim.tmax,io);
    fwrite(&t0,sizeof(int),1,io);
    fwrite(&ntel,sizeof(double),1,io);
    if(sim.Nano_num>0){
      fwrite(&nano_r2f[0],sizeof(double),sim.nano_tmax,io);
      fwrite(&nano_dx[0],sizeof(double),sim.nano_tmax,io);
      fwrite(&nano_dy[0],sizeof(double),sim.nano_tmax,io);
      fwrite(&nano_dz[0],sizeof(double),sim.nano_tmax,io);
      fwrite(&nano_t0,sizeof(int),1,io);
      fwrite(&nano_ntel,sizeof(int),1,io);
    }
    fwrite(&chain_r2f[0],sizeof(double),sim.tmax,io);
    fwrite(&chain_t0,sizeof(int),1,io);
    fwrite(&chain_ntel,sizeof(int),1,io);
    fwrite(&time0[0],sizeof(double),sim.t0max,io);
    if(sim.Nano_num>0)fwrite(&nano_time0[0],sizeof(double),sim.nano_t0max,io);
    fwrite(&chain_time0[0],sizeof(double),sim.t0max,io);
    fwrite(&X0[0],sizeof(double),sim.t0max*box.natoms,io);
    fwrite(&Y0[0],sizeof(double),sim.t0max*box.natoms,io);
    fwrite(&Z0[0],sizeof(double),sim.t0max*box.natoms,io);
    fwrite(&chain_X0[0],sizeof(double),sim.t0max*box.chains,io);
    fwrite(&chain_Y0[0],sizeof(double),sim.t0max*box.chains,io);
    fwrite(&chain_Z0[0],sizeof(double),sim.t0max*box.chains,io);
    if(sim.Nano_num>0){
      fwrite(&nano_X0[0],sizeof(double),sim.nano_t0max*sim.Nano_num,io);
      fwrite(&nano_Y0[0],sizeof(double),sim.nano_t0max*sim.Nano_num,io);
      fwrite(&nano_Z0[0],sizeof(double),sim.nano_t0max*sim.Nano_num,io);
    }
    fclose(io);
  }


  io=fopen("./stress.dat","w");
  fwrite(&SS[0],sizeof(double),9,io);
  fwrite(&SSB[0],sizeof(double),9,io);
  fwrite(&SSNB[0],sizeof(double),9,io);
  fwrite(&SS_f[0],sizeof(double),9,io);
  fwrite(&SSHC[0],sizeof(double),9,io);
  fclose(io);

}

void read_check()
{
  FILE *io;
  io=fopen("./rdf.dat","r");
  char junk[80];
  double x,y,z;
  int k;

  if (io==NULL)
    {   
      exstep=0;
      for(int i=0;i<zntotal;i++)
	{
	  phiAx[i]=0.0;
	  phiAy[i]=0.0;
	  phiAz[i]=0.0;
	  phiBx[i]=0.0;
	  phiBy[i]=0.0;
	  phiBz[i]=0.0;
	  phix_na[i]=0.0;
	  phiy_na[i]=0.0;
	  phiz_na[i]=0.0;
	}

      initial_rdf();                                      //initialize calculation for g(r)		
      if(MSD==true){
	msd(0);
	chain_msd(0);
	if(sim.Nano_num > 0)nano_msd(0);
      }
      return;
    }else {
	start_from_zero = 0;

    fread(&exstep,sizeof(int),1,io);
    fread(&hist_n2p[0],sizeof(double),Nbin,io);
    fread(&hist_n2n[0],sizeof(double),Nbin,io);
    fread(&hist_p2p[0],sizeof(double),Nbin,io);

    fclose(io);

    io=fopen("./phix.dat","r");
    if (io==NULL)
      {   
	fprintf(stdout,"\n phix.dat does not exist\n");
	exit(EXIT_FAILURE);
      }
    for(int i=0;i<zntotal;i++)
      {
	fscanf(io,"%d%lf%lf%lf",&k,&x,&y,&z);
	phiAx[k]=x;
	phiBx[k]=y;
	phix_na[k]=z;
	fgets(junk,80,io);
      }
    fclose(io);

    io=fopen("./phiy.dat","r");
    if (io==NULL)
      {   
	fprintf(stdout,"\n phiy.dat does not exist\n");
	exit(EXIT_FAILURE);
      }
    for(int i=0;i<zntotal;i++)
      {
	fscanf(io,"%d%lf%lf%lf",&k,&x,&y,&z);
	phiAy[k]=x;
	phiBy[k]=y;
	phiy_na[k]=z;
	fgets(junk,80,io);
      }
    fclose(io);

    io=fopen("./phiz.dat","r");
    if (io==NULL)
      {   
	fprintf(stdout,"\n phiz.dat does not exist\n");
	exit(EXIT_FAILURE);
      }
    for(int i=0;i<zntotal;i++)
      {
	fscanf(io,"%d%lf%lf%lf",&k,&x,&y,&z);
	phiAz[k]=x;
	phiBz[k]=y;
	phiz_na[k]=z;
	fgets(junk,80,io);
      }
    fclose(io);






  
    if(MSD==true){
      io=fopen("./msd.dat","r");
      if(io==NULL){
	msd(0);
	chain_msd(0);
	if(sim.Nano_num > 0)nano_msd(0);
      }else{
	fread(&atom_v[0],sizeof(vect),box.natoms,io);
	if(sim.Nano_num>0)fread(&nano_v[0],sizeof(vect),sim.Nano_num,io);
	fread(&chain_mid[0],sizeof(vect),box.chains,io);
	fread(&ntime[0],sizeof(double),sim.tmax,io);
	if(sim.Nano_num>0)fread(&nano_ntime[0],sizeof(double),sim.nano_tmax,io);
	fread(&chain_ntime[0],sizeof(double),sim.tmax,io);
	fread(&r2f[0],sizeof(double),sim.tmax,io);
	fread(&r2_dx[0],sizeof(double),sim.tmax,io);
	fread(&r2_dy[0],sizeof(double),sim.tmax,io);
	fread(&r2_dz[0],sizeof(double),sim.tmax,io);
	fread(&t0,sizeof(int),1,io);
	fread(&ntel,sizeof(double),1,io);
	if(sim.Nano_num>0){
	  fread(&nano_r2f[0],sizeof(double),sim.nano_tmax,io);
	  fread(&nano_dx[0],sizeof(double),sim.nano_tmax,io);
	  fread(&nano_dy[0],sizeof(double),sim.nano_tmax,io);
	  fread(&nano_dz[0],sizeof(double),sim.nano_tmax,io);\
	  fread(&nano_t0,sizeof(int),1,io);
          fread(&nano_ntel,sizeof(int),1,io);
	}
	fread(&chain_r2f[0],sizeof(double),sim.tmax,io);
	fread(&chain_t0,sizeof(int),1,io);
	fread(&chain_ntel,sizeof(int),1,io);
	fread(&time0[0],sizeof(double),sim.t0max,io);
	if(sim.Nano_num>0)fread(&nano_time0[0],sizeof(double),sim.nano_t0max,io);
	fread(&chain_time0[0],sizeof(double),sim.t0max,io);
	fread(&X0[0],sizeof(double),sim.t0max*box.natoms,io);
	fread(&Y0[0],sizeof(double),sim.t0max*box.natoms,io);
	fread(&Z0[0],sizeof(double),sim.t0max*box.natoms,io);
	fread(&chain_X0[0],sizeof(double),sim.t0max*box.chains,io);
	fread(&chain_Y0[0],sizeof(double),sim.t0max*box.chains,io);
	fread(&chain_Z0[0],sizeof(double),sim.t0max*box.chains,io);
	if(sim.Nano_num>0){
	  fread(&nano_X0[0],sizeof(double),sim.nano_t0max*sim.Nano_num,io);
	  fread(&nano_Y0[0],sizeof(double),sim.nano_t0max*sim.Nano_num,io);
	  fread(&nano_Z0[0],sizeof(double),sim.nano_t0max*sim.Nano_num,io);
	}
	fclose(io);
      }
    }




    io=fopen("./stress.dat","r");
    if(io!=NULL){
      fread(&SS[0],sizeof(double),9,io);
      fread(&SSB[0],sizeof(double),9,io);
      fread(&SSNB[0],sizeof(double),9,io);
      fread(&SS_f[0],sizeof(double),9,io);
      fread(&SSHC[0],sizeof(double),9,io);

      fclose(io);


    }
    return;
  }
}


__global__ void Cuda_Reset_Force(Force *C_atomf,NanoForce *C_nanof){
	int index = blockIdx.x * blockDim.x +threadIdx.x;
	if(index < CudaNum_Bead){
	C_atomf[index].bond_force.x = 0.0;
    C_atomf[index].bond_force.y = 0.0;
    C_atomf[index].bond_force.z = 0.0;

    C_atomf[index].nonbond_force.x = 0.0;
    C_atomf[index].nonbond_force.y = 0.0;
    C_atomf[index].nonbond_force.z = 0.0;

    C_atomf[index].hardcore_force.x = 0.0;
    C_atomf[index].hardcore_force.y = 0.0;
    C_atomf[index].hardcore_force.z = 0.0;

    C_atomf[index].ss_f.x = 0.0;
    C_atomf[index].ss_f.y = 0.0;
    C_atomf[index].ss_f.z = 0.0;
    
    C_atomf[index].total_force.x = 0.0;
    C_atomf[index].total_force.y = 0.0;
    C_atomf[index].total_force.z = 0.0;
	}else if(index < (CudaNum_Nano+CudaNum_Bead)){
	C_nanof[index-CudaNum_Bead].nonbond_force.x = 0.0;
    C_nanof[index-CudaNum_Bead].nonbond_force.y = 0.0;
    C_nanof[index-CudaNum_Bead].nonbond_force.z = 0.0;

    C_nanof[index-CudaNum_Bead].hardcore_force.x = 0.0;
    C_nanof[index-CudaNum_Bead].hardcore_force.y = 0.0;
    C_nanof[index-CudaNum_Bead].hardcore_force.z = 0.0;
    
    C_nanof[index-CudaNum_Bead].total_force.x = 0.0;
    C_nanof[index-CudaNum_Bead].total_force.y = 0.0;
    C_nanof[index-CudaNum_Bead].total_force.z = 0.0;
	}
	

}



void forces(void){
  

  
  Cuda_Reset_Force<<<(box.natoms+sim.Nano_num)/BLOCK_SIZE+1,BLOCK_SIZE>>>(c_atomf,c_nanof);
  
  cudaMemcpy(c_ss, ss, NSS*sizeof(ss_t), cudaMemcpyHostToDevice);
  
  
  //cudaMemcpy(c_atomf,force_bead, box.natoms*sizeof(Force), cudaMemcpyHostToDevice);
  //cudaMemcpy(c_nanof,force_nano, sim.Nano_num*sizeof(NanoForce), cudaMemcpyHostToDevice);

  if(sim.Nano_num > 0)hardcore_forces();

  bond_forces();

  nonbond_forces();

  

    
  cudaMemcpy(force_bead,c_atomf, box.natoms*sizeof(Force), cudaMemcpyDeviceToHost);
  cudaMemcpy(force_nano,c_nanof, sim.Nano_num*sizeof(NanoForce), cudaMemcpyDeviceToHost);
   

if(SSlink == true)ss_forces();

}

__device__ double atomicAdd(double* address, double val)
{
  unsigned long long int* address_as_ull = (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed, 
		    __double_as_longlong(val + 
					 __longlong_as_double(assumed)));
  } while (assumed != old);
  return __longlong_as_double(old);
}

__global__ void BondForce_Compute(Force *cudaForce,atoms *cudaAtom,double *en,double *bxx,double *byy,double *bzz,double *bxy,double *byz,double *bxz,int num_chains,int N,double c_dxle){
  int index = blockIdx.x * blockDim.x +threadIdx.x;

  double dx,dy,dz,dr2;
  if(index < num_chains*N){
    if((index+1)%N != 0){

      dx = cudaAtom[index].pos.x-cudaAtom[index+1].pos.x;
      dy = cudaAtom[index].pos.y-cudaAtom[index+1].pos.y;
      dz = cudaAtom[index].pos.z-cudaAtom[index+1].pos.z;
      dr2 = Cuda_Distance(dx,dy,dz,c_dxle);



      en[index] = (-0.5*f_coeff)*dr2;

      bxx[index] = f_coeff * dx * dx;
      byy[index] = f_coeff * dy * dy;
      bzz[index] = f_coeff * dz * dz;
      bxy[index] = f_coeff * dx * dy;
      byz[index] = f_coeff * dy * dz;
      bxz[index] = f_coeff * dx * dz;

      /*atomicAdd(&cudaForce[index].bond_force.x,f_coeff*dx);
	atomicAdd(&cudaForce[index].bond_force.y,f_coeff*dy);
	atomicAdd(&cudaForce[index].bond_force.z,f_coeff*dz);
	atomicAdd(&cudaForce[index+1].bond_force.x,-1.0*f_coeff*dx);
	atomicAdd(&cudaForce[index+1].bond_force.y,-1.0*f_coeff*dy);
	atomicAdd(&cudaForce[index+1].bond_force.z,-1.0*f_coeff*dz);*/

      cudaForce[index].bond_force.x += f_coeff*dx;
      cudaForce[index].bond_force.y += f_coeff*dy;
      cudaForce[index].bond_force.z += f_coeff*dz;
      __syncthreads();
      cudaForce[index+1].bond_force.x -= f_coeff*dx;
      cudaForce[index+1].bond_force.y -= f_coeff*dy;
      cudaForce[index+1].bond_force.z -= f_coeff*dz;

    }
  }
}

void bond_forces()
{
  double sxx=0,syy=0,szz=0;
  double sxy=0,syz=0,sxz=0; 
  double *b_en,*b_sxx,*b_syy,*b_szz,*b_sxy,*b_syz,*b_sxz;
  double *b_En,*stxx,*styy,*stzz,*stxy,*styz,*stxz;
  en.bonded_E = 0;

  b_En = (double *)calloc(box.natoms,sizeof(double));
  stxx = (double *)calloc(box.natoms,sizeof(double));
  styy = (double *)calloc(box.natoms,sizeof(double));
  stzz = (double *)calloc(box.natoms,sizeof(double));
  stxy = (double *)calloc(box.natoms,sizeof(double));
  styz = (double *)calloc(box.natoms,sizeof(double));
  stxz = (double *)calloc(box.natoms,sizeof(double));

  cudaMalloc((void **)&b_en, box.natoms*sizeof(double));
  cudaMalloc((void **)&b_sxx, box.natoms*sizeof(double));
  cudaMalloc((void **)&b_syy, box.natoms*sizeof(double));
  cudaMalloc((void **)&b_szz, box.natoms*sizeof(double));
  cudaMalloc((void **)&b_sxy, box.natoms*sizeof(double));
  cudaMalloc((void **)&b_syz, box.natoms*sizeof(double));
  cudaMalloc((void **)&b_sxz, box.natoms*sizeof(double));

  cudaMemcpy(b_en, b_En, box.natoms*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(b_sxx, stxx, box.natoms*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(b_syy, styy, box.natoms*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(b_szz, stzz, box.natoms*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(b_sxy, stxy, box.natoms*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(b_syz, styz, box.natoms*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(b_sxz, stxz, box.natoms*sizeof(double), cudaMemcpyHostToDevice);
  // call kernel to calculate bonded energy
  BondForce_Compute<<<box.natoms/BLOCK_SIZE+1,BLOCK_SIZE>>>(c_atomf,c_atom,b_en,b_sxx,b_syy,b_szz,b_sxy,b_syz,b_sxz,box.chains,sim.N,dx_LE);

  cudaMemcpy(b_En, b_en, box.natoms*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(stxx, b_sxx, box.natoms*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(styy, b_syy, box.natoms*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(stzz, b_szz, box.natoms*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(stxy, b_sxy, box.natoms*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(styz, b_syz, box.natoms*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(stxz, b_sxz, box.natoms*sizeof(double), cudaMemcpyDeviceToHost);

  for(int i = 0;i<box.natoms;i++){
    en.bonded_E += b_En[i];
    sxx += stxx[i];
    syy += styy[i];
    szz += stzz[i];
    sxy += stxy[i];
    syz += styz[i];
    sxz += stxz[i];
  }



  cudaFree(b_en);
  cudaFree(b_sxx);
  cudaFree(b_syy);
  cudaFree(b_szz);
  cudaFree(b_sxy);
  cudaFree(b_syz);
  cudaFree(b_sxz);

  free(b_En);
  free(stxx);
  free(styy);
  free(stzz);
  free(stxy);
  free(styz);
  free(stxz);



  sxx=sxx/box.vol;
  syy=syy/box.vol;
  szz=szz/box.vol;

  sxy=sxy/box.vol;
  syz=syz/box.vol;
  sxz=sxz/box.vol;



  SS[0]+=sxx;SS[1]+=sxy;SS[2]+=sxz;
  SS[3]+=sxy;SS[4]+=syy;SS[5]+=syz;
  SS[6]+=sxz;SS[7]+=syz;SS[8]+=szz;

  SSB[0]+=sxx;SSB[1]+=sxy;SSB[2]+=sxz;
  SSB[3]+=sxy;SSB[4]+=syy;SSB[5]+=syz;
  SSB[6]+=sxz;SSB[7]+=syz;SSB[8]+=szz;

  temp_b.xx = sxx;
  temp_b.yy = syy;
  temp_b.zz = szz;
  temp_b.xy = sxy;
  temp_b.yz = syz;
  temp_b.xz = sxz;

}


/*void nonbond_forces(){
  double nonbonded_E=0;
  int vdxx = (int)(dx_LE/del_L1);
  double V=box.length1*box.length2*box.length3;
  wxx=0;
  wyy=0;
  wzz=0;
  wyx=0;
  wzy=0;
  wzx=0;
  int link_1,link_2;                    //the link_2 is for the neighboring cell
  int ind,ind_l;
  int m1,n1,l1;
  for(int i=0;i<box.natoms-1;i++)
  for(int j=i+1;j<box.natoms;j++){
  nonbonded_E += nbforce(i,j);
  }

  en.nonbonded_E = nonbonded_E;
  if(STRESS==true){
  wxx /= V;
  wyx /= V;
  wyy /= V;
  wzx /= V;
  wzy /= V;
  wzz /= V;

  SS[0]+=wxx;SS[1]+=wxy;SS[2]+=wxz;
  SS[3]+=wxy;SS[4]+=wyy;SS[5]+=wyz;
  SS[6]+=wxz;SS[7]+=wyz;SS[8]+=wzz;

  SSB[0]+=wxx;SSB[1]+=wxy;SSB[2]+=wxz;
  SSB[3]+=wxy;SSB[4]+=wyy;SSB[5]+=wyz;
  SSB[6]+=wxz;SSB[7]+=wyz;SSB[8]+=wzz;
  }
  }*/


__global__ void Cuda_NONBOND_ForceCompute(double *nb_En,nanos *C_nano,atoms *C_Nanobead,atoms *C_atom,NanoForce *C_nanof, Force *C_atomf,int *C_head,int *C_lscl,double *c_bxx,double *c_byy,double *c_bzz,double *c_bxy,double *c_byz,double *c_bxz,double v_d,int dt,double c_dxle,int c_vdxx){
  int index = blockIdx.x * blockDim.x +threadIdx.x;
  double dx,dy,dz;
  int kx,ky,kz,ind,m1,n1,l1,link;
  if(index < CudaNum_Bead ){

      dx = C_atom[index].pos.x;
      dy = C_atom[index].pos.y;
      dz = C_atom[index].pos.z;
      kx = Cuda_Grid(dx,1);
      ky = Cuda_Grid(dy,2);
      kz = Cuda_Grid(dz,3);

    
    if(Cuda_LeesEd == true){

		if(kz == 0){

	      for(int m = kx-1;m <= kx+1;m++)
		for(int n = ky-1;n <= ky+1;n++)
		  for(int l= kz;l <= kz+1;l++)
		    {
		      m1 = Cuda_Mod(m,Cuda_L1);
		      n1 = Cuda_Mod(n,Cuda_L2);
		      l1 = l;
		      ind = m1*Cuda_L2*Cuda_L3 + n1*Cuda_L3 + l1;
		      
			  link = C_head[ind];
			  while(link != -1){
			    if(index < link){
			      nb_En[index] += Cuda_Nbforce(index,link,C_atom,C_atomf,c_bxx,c_byy,c_bzz,c_bxy,c_byz,c_bxz,c_dxle);
			    }
			    link = C_lscl[link];
			  }

			
		    }
		    
		    
	      for(int m = -1;m <= 2;m++)
		for(int n = ky-1;n <= ky+1;n++)
		  {
		    m1 = Cuda_Mod(kx+c_vdxx+m,Cuda_L1);
		    n1 = Cuda_Mod(n,Cuda_L2);
		    l1 = Cuda_L3-1;
		    ind = m1*Cuda_L2*Cuda_L3 + n1*Cuda_L3 + l1;

			link = C_head[ind];
			while(link != -1){
			  if(index < link){
			      nb_En[index] += Cuda_Nbforce(index,link,C_atom,C_atomf,c_bxx,c_byy,c_bzz,c_bxy,c_byz,c_bxz,c_dxle);
			  }
			  link = C_lscl[link];
			}

	
		  }

	    }else if (kz == Cuda_L3-1){


	      for(int m = kx-1;m <= kx+1;m++)
		for(int n = ky-1;n <= ky+1;n++)
		  for(int l = kz-1;l <= kz;l++)
		    {
		      m1 = Cuda_Mod(m,Cuda_L1);
		      n1 = Cuda_Mod(n,Cuda_L2);
		      l1 = l;
		      ind = m1*Cuda_L2*Cuda_L3 + n1*Cuda_L3+l1;

			  link = C_head[ind];
			  while(link!=-1){
			    if(index < link){
			      nb_En[index] += Cuda_Nbforce(index,link,C_atom,C_atomf,c_bxx,c_byy,c_bzz,c_bxy,c_byz,c_bxz,c_dxle);
			    }
			    link = C_lscl[link];
			  }
		
		    }
		    
		    
	      for(int m = -2;m <= 1;m++)
		for(int n = ky-1;n <= ky+1;n++)
		  {
		    m1 = Cuda_Mod(kx-c_vdxx+m,Cuda_L1);
		    n1 = Cuda_Mod(n,Cuda_L2);
		    l1 = 0;
		    ind = m1*Cuda_L2*Cuda_L3 + n1*Cuda_L3 + l1;

			link = C_head[ind];
			while(link != -1){
			  if(index < link){
			      nb_En[index] += Cuda_Nbforce(index,link,C_atom,C_atomf,c_bxx,c_byy,c_bzz,c_bxy,c_byz,c_bxz,c_dxle);
			  }
			  link = C_lscl[link];
			}
		
		  }
	    }else{

	      for(int m = kx-2;m <= kx+2;m++)
		for(int n = ky-2;n <= ky+2;n++)
		  for(int l = kz-2;l <= kz+2;l++)
		    {
		      m1 = Cuda_Mod(m,Cuda_L1);
		      n1 = Cuda_Mod(n,Cuda_L2);
		      l1 = Cuda_Mod(l,Cuda_L3);
		      ind = m1*Cuda_L2*Cuda_L3 + n1*Cuda_L3 + l1;

			  link = C_head[ind];
			  while(link != -1){
			    if(index < link){
			      nb_En[index] += Cuda_Nbforce(index,link,C_atom,C_atomf,c_bxx,c_byy,c_bzz,c_bxy,c_byz,c_bxz,c_dxle);
			    }
			    link = C_lscl[link];
			  }
		
		    }
	    }
		  
		  
		  
      }else{

	for(int m=kx-1;m<=kx+1;m++)
	  for(int n=ky-1;n<=ky+1;n++)
	    for(int l=kz-1;l<=kz+1;l++){

	      m1 = Cuda_Mod(m,Cuda_L1);
	      n1 = Cuda_Mod(n,Cuda_L2);
	      l1 = Cuda_Mod(l,Cuda_L3);
	      ind=m1*Cuda_L2*Cuda_L3+n1*Cuda_L3+l1;


	      link = C_head[ind];
	      while(link != -1){
		if(index < link){
			      nb_En[index] += Cuda_Nbforce(index,link,C_atom,C_atomf,c_bxx,c_byy,c_bzz,c_bxy,c_byz,c_bxz,c_dxle);
		}
		link=C_lscl[link];
	      }

	    }
      }
  }
}

/*__global__ void Cuda_NONBOND_ForceCompute(double *nb_En,nanos *C_nano,atoms *C_Nanobead,atoms *C_atom,NanoForce *C_nanof, Force *C_atomf,int *C_head,int *C_lscl,double *c_bxx,double *c_byy,double *c_bzz,double *c_bxy,double *c_byz,double *c_bxz,double v_d,int dt,double c_dxle,int c_vdxx){
  int index = blockIdx.x * blockDim.x +threadIdx.x;

	if(index < CudaNum_Bead-1){
		for(int link=index+1;link < CudaNum_Bead;link++){
			nb_En[index] += Cuda_Nbforce(index,link,C_atom,C_atomf,c_bxx,c_byy,c_bzz,c_bxy,c_byz,c_bxz,c_dxle);

	  }
	}
}*/



void nonbond_forces(){
  
  double sxx=0,syy=0,szz=0;
  double sxy=0,syz=0,sxz=0; 
  double *c_nben,*c_sxx,*c_syy,*c_szz,*c_sxy,*c_syz,*c_sxz;
  double *nb_En,*Sxx,*Syy,*Szz,*Sxy,*Syz,*Sxz;
  int *c_head,*c_lscl;
  
  int vdxx = (int)(dx_LE/del_L1);
  en.nonbonded_E = 0;

  nb_En = (double *)calloc(box.natoms,sizeof(double));
  Sxx = (double *)calloc(box.natoms,sizeof(double));
  Syy = (double *)calloc(box.natoms,sizeof(double));
  Szz = (double *)calloc(box.natoms,sizeof(double));
  Sxy = (double *)calloc(box.natoms,sizeof(double));
  Syz = (double *)calloc(box.natoms,sizeof(double));
  Sxz = (double *)calloc(box.natoms,sizeof(double));
  

  cudaMalloc((void **)&c_head, L*sizeof(int));
  cudaMalloc((void **)&c_lscl, (box.natoms)*sizeof(int));
  
  cudaMalloc((void **)&c_nben, (box.natoms)*sizeof(double));
  cudaMalloc((void **)&c_sxx, (box.natoms)*sizeof(double));
  cudaMalloc((void **)&c_syy, (box.natoms)*sizeof(double));
  cudaMalloc((void **)&c_szz, (box.natoms)*sizeof(double));
  cudaMalloc((void **)&c_sxy, (box.natoms)*sizeof(double));
  cudaMalloc((void **)&c_syz, (box.natoms)*sizeof(double));
  cudaMalloc((void **)&c_sxz, (box.natoms)*sizeof(double));

  cudaMemcpy(c_head,head,L*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(c_lscl, lscl,(box.natoms)*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(c_nben, nb_En, (box.natoms)*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(c_sxx, Sxx, (box.natoms)*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(c_syy, Syy, (box.natoms)*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(c_szz, Szz, (box.natoms)*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(c_sxy, Sxy, (box.natoms)*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(c_syz, Syz, (box.natoms)*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(c_sxz, Sxz, (box.natoms)*sizeof(double), cudaMemcpyHostToDevice);

  Cuda_NONBOND_ForceCompute<<<box.natoms/BLOCK_SIZE+1,BLOCK_SIZE>>>(c_nben,c_nano,c_Nanobead,c_atom,c_nanof,c_atomf,c_head,c_lscl,c_sxx,c_syy,c_szz,c_sxy,c_syz,c_sxz,sim.v_d,sim.dt,dx_LE,vdxx);

  cudaMemcpy(nb_En, c_nben, (box.natoms)*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(Sxx, c_sxx, (box.natoms)*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(Syy, c_syy, (box.natoms)*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(Szz, c_szz, (box.natoms)*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(Sxy, c_sxy, (box.natoms)*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(Syz, c_syz, (box.natoms)*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(Sxz, c_sxz, (box.natoms)*sizeof(double), cudaMemcpyDeviceToHost);

  for(int i = 0;i<box.natoms;i++){
    en.nonbonded_E += nb_En[i];
    sxx += Sxx[i];
    syy += Syy[i];
    szz += Szz[i];
    sxy += Sxy[i];
    syz += Syz[i];
    sxz += Sxz[i];
  }

  cudaFree(c_head);
  cudaFree(c_lscl);
  cudaFree(c_nben);
  cudaFree(c_sxx);
  cudaFree(c_syy);
  cudaFree(c_szz);
  cudaFree(c_sxy);
  cudaFree(c_syz);
  cudaFree(c_sxz);

  free(nb_En);
  free(Sxx);
  free(Syy);
  free(Szz);
  free(Sxy);
  free(Syz);
  free(Sxz);

  sxx /= box.vol;
  syy /= box.vol;
  szz /= box.vol;
  sxy /= box.vol;
  syz /= box.vol;
  sxz /= box.vol;

  SS[0]+=sxx;SS[1]+=sxy;SS[2]+=sxz;
  SS[3]+=sxy;SS[4]+=syy;SS[5]+=syz;
  SS[6]+=sxz;SS[7]+=syz;SS[8]+=szz;

  SSNB[0]+=sxx;SSNB[1]+=sxy;SSNB[2]+=sxz;
  SSNB[3]+=sxy;SSNB[4]+=syy;SSNB[5]+=syz;
  SSNB[6]+=sxz;SSNB[7]+=syz;SSNB[8]+=szz;

  temp_nb.xx = sxx;
  temp_nb.yy = syy;
  temp_nb.zz = szz;
  temp_nb.xy = sxy;
  temp_nb.yz = syz;
  temp_nb.xz = sxz;
}



__device__ double Cuda_HardForce(double dr,double buf,double Rc){
  if(dr < (Rc-buf)){
    return (3.0-2.0*dr/(Rc-buf))*Hc_Magnitude;
  }else{
    double m=(dr-Rc+buf)/buf-1.0;
    return m*m*Hc_Magnitude;
  }
}

__device__ double Cuda_HardEn(double dr,double buf,double Rc){
  if(dr < (Rc-buf)){
    return (dr*dr/(Rc-buf)-3.0*dr+2.0*(Rc-buf)+buf/3.0)*Hc_Magnitude;
  }else{
    double m = 1.0-(dr-Rc+buf)/buf;
    return buf*m*m*m*Hc_Magnitude/3.0;
  }
}

__device__ double Cuda_Attract_En(double dr){
	double m = (dr-Cuda_NanoRc)/(Cuda_NanoR-Cuda_NanoRc);
	if(dr < Cuda_NanoRc&&dr > (2.0*Cuda_NanoRc-Cuda_NanoR))return Hc_Attraction*(-2.0*m*m*m-3.0*m*m+1.0);
	else if (dr < Cuda_NanoR&&dr > Cuda_NanoRc)return Hc_Attraction*(2.0*m*m*m-3.0*m*m+1.0);
	else return 0.0;
}

__device__ double Cuda_Attract_Force(double dr){
	double m = (dr-Cuda_NanoRc)/(Cuda_NanoR-Cuda_NanoRc);
	double param;
	if(dr < Cuda_NanoRc&&dr > (2.0*Cuda_NanoRc-Cuda_NanoR)){
	param = 6.0*Hc_Attraction/(Cuda_NanoR-Cuda_NanoRc);
    return param*(m*m+m);
    }else if(dr < Cuda_NanoR&&dr > Cuda_NanoRc){
	param = 6.0*Hc_Attraction/(Cuda_NanoR-Cuda_NanoRc);
    return -1.0*param*(m*m-m);
	}else return 0;
}
/*void link_hardcore_forces(){
  double sxx=0,syy=0,szz=0;
  double sxy=0,syz=0,sxz=0; 
  double *c_hcen,*c_sxx,*c_syy,*c_szz,*c_sxy,*c_syz,*c_sxz;
  double *Hc_En,*Sxx,*Syy,*Szz,*Sxy,*Syz,*Sxz;
  int *c_head,*c_lscl;
  
  int vdxx = (int)(dx_LE/del_nnL1);
  en.hardcore = 0;

  Hc_En = (double *)calloc(box.natoms+sim.Nano_num,sizeof(double));
  Sxx = (double *)calloc(box.natoms+sim.Nano_num,sizeof(double));
  Syy = (double *)calloc(box.natoms+sim.Nano_num,sizeof(double));
  Szz = (double *)calloc(box.natoms+sim.Nano_num,sizeof(double));
  Sxy = (double *)calloc(box.natoms+sim.Nano_num,sizeof(double));
  Syz = (double *)calloc(box.natoms+sim.Nano_num,sizeof(double));
  Sxz = (double *)calloc(box.natoms+sim.Nano_num,sizeof(double));
  

  cudaMalloc((void **)&c_head, nn_L*sizeof(int));
  cudaMalloc((void **)&c_lscl, (box.natoms+sim.Nano_num)*sizeof(int));
  
  cudaMalloc((void **)&c_hcen, (box.natoms+sim.Nano_num)*sizeof(double));
  cudaMalloc((void **)&c_sxx, (box.natoms+sim.Nano_num)*sizeof(double));
  cudaMalloc((void **)&c_syy, (box.natoms+sim.Nano_num)*sizeof(double));
  cudaMalloc((void **)&c_szz, (box.natoms+sim.Nano_num)*sizeof(double));
  cudaMalloc((void **)&c_sxy, (box.natoms+sim.Nano_num)*sizeof(double));
  cudaMalloc((void **)&c_syz, (box.natoms+sim.Nano_num)*sizeof(double));
  cudaMalloc((void **)&c_sxz, (box.natoms+sim.Nano_num)*sizeof(double));

  cudaMemcpy(c_head,nn_head,nn_L*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(c_lscl, nn_lscl,(box.natoms+sim.Nano_num)*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(c_hcen, Hc_En, (box.natoms+sim.Nano_num)*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(c_sxx, Sxx, (box.natoms+sim.Nano_num)*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(c_syy, Syy, (box.natoms+sim.Nano_num)*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(c_szz, Szz, (box.natoms+sim.Nano_num)*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(c_sxy, Sxy, (box.natoms+sim.Nano_num)*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(c_syz, Syz, (box.natoms+sim.Nano_num)*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(c_sxz, Sxz, (box.natoms+sim.Nano_num)*sizeof(double), cudaMemcpyHostToDevice);

  Cuda_Link_HardCoreForce<<<(box.natoms+sim.Nano_num)/BLOCK_SIZE+1,BLOCK_SIZE>>>(c_hcen,c_nano,c_atom,c_nanof, c_atomf,c_head,c_lscl,c_sxx,c_syy,c_szz,c_sxy,c_syz,c_sxz,sim.v_d,sim.dt,dx_LE,vdxx,sim.dp);

  cudaMemcpy(Hc_En, c_hcen, (box.natoms+sim.Nano_num)*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(Sxx, c_sxx, (box.natoms+sim.Nano_num)*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(Syy, c_syy, (box.natoms+sim.Nano_num)*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(Szz, c_szz, (box.natoms+sim.Nano_num)*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(Sxy, c_sxy, (box.natoms+sim.Nano_num)*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(Syz, c_syz, (box.natoms+sim.Nano_num)*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(Sxz, c_sxz, (box.natoms+sim.Nano_num)*sizeof(double), cudaMemcpyDeviceToHost);

  for(int i = 0;i<(box.natoms+sim.Nano_num);i++){
    en.hardcore += Hc_En[i];
    sxx += Sxx[i];
    syy += Syy[i];
    szz += Szz[i];
    sxy += Sxy[i];
    syz += Syz[i];
    sxz += Sxz[i];
  }

  cudaFree(c_head);
  cudaFree(c_lscl);
  cudaFree(c_hcen);
  cudaFree(c_sxx);
  cudaFree(c_syy);
  cudaFree(c_szz);
  cudaFree(c_sxy);
  cudaFree(c_syz);
  cudaFree(c_sxz);

  free(Hc_En);
  free(Sxx);
  free(Syy);
  free(Szz);
  free(Sxy);
  free(Syz);
  free(Sxz);


  sxx /= box.vol;
  sxy /= box.vol;
  syy /= box.vol;
  sxy /= box.vol;
  syz /= box.vol;
  sxz /= box.vol;

  SS[0]+=sxx;SS[1]+=sxy;SS[2]+=sxz;
  SS[3]+=sxy;SS[4]+=syy;SS[5]+=syz;
  SS[6]+=sxz;SS[7]+=syz;SS[8]+=szz;

  SSHC[0]+=sxx;SSHC[1]+=sxy;SSHC[2]+=sxz;
  SSHC[3]+=sxy;SSHC[4]+=syy;SSHC[5]+=syz;
  SSHC[6]+=sxz;SSHC[7]+=syz;SSHC[8]+=szz;

  temp_hcxz = sxz;
}*/

/*__device__ double Cuda_HcForce(int a,int b,nanos *C_nano,atoms *C_atom,NanoForce *C_nanof,Force *C_atomf,double *bxx,double *byy,double *bzz,double *bxy,double *byz,double *bxz,double c_dxle,double buf){
  double force;
  double dx,dy,dz,dr;
  double hc_en=0.0;
  if(a >= CudaNum_Bead && b >= CudaNum_Bead){
    int inano_a = (a-CudaNum_Bead);
    int inano_b = (b-CudaNum_Bead);

    dx = C_nano[inano_a].pos.x-C_nano[inano_b].pos.x;
    dy = C_nano[inano_a].pos.y-C_nano[inano_b].pos.y;
    dz = C_nano[inano_a].pos.z-C_nano[inano_b].pos.z;

    dr = sqrt(Cuda_Distance(dx,dy,dz,c_dxle));
    if(dr < (Cuda_NanoR+Cuda_NanoRc+buf)&&dr > 1.0E-9){
      hc_en = Cuda_HardEn(dr,buf,(Cuda_NanoR+Cuda_NanoRc+buf));
      force = Cuda_HardForce(dr,buf,(Cuda_NanoR+Cuda_NanoRc+buf))/dr;

      
      atomicAdd(&C_nanof[inano_a].hardcore_force.x,force*dx);
      atomicAdd(&C_nanof[inano_a].hardcore_force.y,force*dy);
      atomicAdd(&C_nanof[inano_a].hardcore_force.z,force*dz);
      
      atomicAdd(&C_nanof[inano_b].hardcore_force.x,-1.0*force*dx);
      atomicAdd(&C_nanof[inano_b].hardcore_force.y,-1.0*force*dy);
      atomicAdd(&C_nanof[inano_b].hardcore_force.z,-1.0*force*dz);
      	
      atomicAdd(&bxx[a],dx*dx*force);
	  atomicAdd(&byy[a],dy*dy*force);
	  atomicAdd(&bzz[a],dz*dz*force);
	  atomicAdd(&bxy[a],dx*dy*force);
	  atomicAdd(&byz[a],dy*dz*force);
	  atomicAdd(&bxz[a],dx*dz*force);
    }


  }else if ((a >= CudaNum_Bead && b < CudaNum_Bead)){
    int inano_a = (a-CudaNum_Bead);

    dx = C_nano[inano_a].pos.x-C_atom[b].pos.x;
    dy = C_nano[inano_a].pos.y-C_atom[b].pos.y;
    dz = C_nano[inano_a].pos.z-C_atom[b].pos.z;


    dr = sqrt(Cuda_Distance(dx,dy,dz,c_dxle));
    if(dr < (Cuda_NanoRc+buf) && dr > 1.0E-10){
      hc_en = Cuda_HardEn(dr,buf,(Cuda_NanoRc+buf));
      force = Cuda_HardForce(dr,buf,(Cuda_NanoRc+buf))/dr;
      
      
      atomicAdd(&C_nanof[inano_a].hardcore_force.x,force*dx);
      atomicAdd(&C_nanof[inano_a].hardcore_force.y,force*dy);
      atomicAdd(&C_nanof[inano_a].hardcore_force.z,force*dz);
      
      atomicAdd(&C_atomf[b].hardcore_force.x,-1.0*force*dx);
      atomicAdd(&C_atomf[b].hardcore_force.y,-1.0*force*dy);
      atomicAdd(&C_atomf[b].hardcore_force.z,-1.0*force*dz);
      	
      atomicAdd(&bxx[a],dx*dx*force);
	  atomicAdd(&byy[a],dy*dy*force);
	  atomicAdd(&bzz[a],dz*dz*force);
	  atomicAdd(&bxy[a],dx*dy*force);
	  atomicAdd(&byz[a],dy*dz*force);
	  atomicAdd(&bxz[a],dx*dz*force);
    }


  }else if ((b >= CudaNum_Bead && a < CudaNum_Bead)){
    int inano_b = (b-CudaNum_Bead);

    dx = C_atom[a].pos.x - C_nano[inano_b].pos.x;
    dy = C_atom[a].pos.y - C_nano[inano_b].pos.y;
    dz = C_atom[a].pos.z - C_nano[inano_b].pos.z;


    dr = sqrt(Cuda_Distance(dx,dy,dz,c_dxle));
    if(dr < (Cuda_NanoRc+buf) && dr > 1.0E-10){	
      hc_en = Cuda_HardEn(dr,buf,(Cuda_NanoRc+buf));
      force = Cuda_HardForce(dr,buf,(Cuda_NanoRc+buf))/dr;

      
      atomicAdd(&C_atomf[a].hardcore_force.x,force*dx);
      atomicAdd(&C_atomf[a].hardcore_force.y,force*dy);
      atomicAdd(&C_atomf[a].hardcore_force.z,force*dz);
      
      
      atomicAdd(&C_nanof[inano_b].hardcore_force.x,-1.0*force*dx);
      atomicAdd(&C_nanof[inano_b].hardcore_force.y,-1.0*force*dy);
      atomicAdd(&C_nanof[inano_b].hardcore_force.z,-1.0*force*dz);
      	
      atomicAdd(&bxx[a],dx*dx*force);
	  atomicAdd(&byy[a],dy*dy*force);
	  atomicAdd(&bzz[a],dz*dz*force);
	  atomicAdd(&bxy[a],dx*dy*force);
	  atomicAdd(&byz[a],dy*dz*force);
	  atomicAdd(&bxz[a],dx*dz*force);
    }



  }else hc_en = 0.0;

  return hc_en;
}*/



__global__ void Cuda_Attractive_Hc_ForceCompute(double *C_hcen,atoms *C_atom,nanos *C_nano,Force *C_atomf,NanoForce *C_nanof,double *bxx,double *byy,double *bzz,double *bxy,double *byz,double *bxz,double c_delx,double buffer){
	int index = blockIdx.x * blockDim.x +threadIdx.x;
	double dx,dy,dz,dr2,dr,force;
	if(index < CudaNum_Bead){
		for(int i=0;i < CudaNum_Nano;i++){
			
      dx = C_atom[index].pos.x - C_nano[i].pos.x;
      dy = C_atom[index].pos.y - C_nano[i].pos.y;
      dz = C_atom[index].pos.z - C_nano[i].pos.z;
      dr2 = Cuda_Distance(dx,dy,dz,c_delx);
      dr = sqrt(dr2);
      
      if(dr < Cuda_NanoR&&dr > 2.0*Cuda_NanoRc - Cuda_NanoR){
			C_hcen[index] += Cuda_Attract_En(dr);
	        force = Cuda_Attract_Force(dr)/ dr;
	    
	    C_atomf[index].hardcore_force.x += force*dx;
	    C_atomf[index].hardcore_force.y += force*dy;
	    C_atomf[index].hardcore_force.z += force*dz;
	    
	    atomicAdd(&C_nanof[i].hardcore_force.x,-1.0*force*dx);
	    atomicAdd(&C_nanof[i].hardcore_force.y,-1.0*force*dy);
	    atomicAdd(&C_nanof[i].hardcore_force.z,-1.0*force*dz);
	     
	    bxx[index] += dx * dx * force;
	    byy[index] += dy * dy * force;
	    bzz[index] += dz * dz * force;
	    bxy[index] += dx * dy * force;
	    byz[index] += dy * dz * force;
	    bxz[index] += dx * dz * force;
		}
		
		if(dr < (Cuda_NanoRc+buffer)&&dr > 1.0E-10){
      	C_hcen[index] += Cuda_HardEn(dr,buffer,Cuda_NanoRc+buffer);
	    force = Cuda_HardForce(dr,buffer,Cuda_NanoRc+buffer)/ dr;
	    
	    C_atomf[index].hardcore_force.x += force*dx;
	    C_atomf[index].hardcore_force.y += force*dy;
	    C_atomf[index].hardcore_force.z += force*dz;
	    
	    atomicAdd(&C_nanof[i].hardcore_force.x,-1.0*force*dx);
	    atomicAdd(&C_nanof[i].hardcore_force.y,-1.0*force*dy);
	    atomicAdd(&C_nanof[i].hardcore_force.z,-1.0*force*dz);
	     
	    bxx[index] += dx * dx * force;
	    byy[index] += dy * dy * force;
	    bzz[index] += dz * dz * force;
	    bxy[index] += dx * dy * force;
	    byz[index] += dy * dz * force;
	    bxz[index] += dx * dz * force;

		};
		
	  }
	}else if(index < (CudaNum_Bead+CudaNum_Nano)){
	  for(int i=index-CudaNum_Bead+1;i < CudaNum_Nano;i++){
      dx = C_nano[index-CudaNum_Bead].pos.x - C_nano[i].pos.x;
      dy = C_nano[index-CudaNum_Bead].pos.y - C_nano[i].pos.y;
      dz = C_nano[index-CudaNum_Bead].pos.z - C_nano[i].pos.z;
      dr2 = Cuda_Distance(dx,dy,dz,c_delx);
      dr = sqrt(dr2);
      if(dr < (Cuda_NanoR+Cuda_NanoRc+buffer)&&dr > 1.0E-10){
      	C_hcen[index] += Cuda_HardEn(dr,buffer,Cuda_NanoR+Cuda_NanoRc+buffer);
	    force = Cuda_HardForce(dr,buffer,Cuda_NanoR+Cuda_NanoRc+buffer)/ dr;
	    
	    atomicAdd(&C_nanof[index-CudaNum_Bead].hardcore_force.x,force*dx);
	    atomicAdd(&C_nanof[index-CudaNum_Bead].hardcore_force.y,force*dy);
	    atomicAdd(&C_nanof[index-CudaNum_Bead].hardcore_force.z,force*dz);
	    
	    atomicAdd(&C_nanof[i].hardcore_force.x,-1.0*force*dx);
	    atomicAdd(&C_nanof[i].hardcore_force.y,-1.0*force*dy);
	    atomicAdd(&C_nanof[i].hardcore_force.z,-1.0*force*dz);
	     
	    bxx[index] += dx * dx * force;
	    byy[index] += dy * dy * force;
	    bzz[index] += dz * dz * force;
	    bxy[index] += dx * dy * force;
	    byz[index] += dy * dz * force;
	    bxz[index] += dx * dz * force;

		};
		
	  }
	}
}
__global__ void Cuda_BeadNano_ForceCompute(double *C_hcen,atoms *C_atom,nanos *C_nano,Force *C_atomf,NanoForce *C_nanof,double *bxx,double *byy,double *bzz,double *bxy,double *byz,double *bxz,double c_delx,double buffer){
	int index = blockIdx.x * blockDim.x +threadIdx.x;
	double dx,dy,dz,dr2,dr,force;
	if(index < CudaNum_Bead){
		for(int i=0;i < CudaNum_Nano;i++){
			
      dx = C_atom[index].pos.x - C_nano[i].pos.x;
      dy = C_atom[index].pos.y - C_nano[i].pos.y;
      dz = C_atom[index].pos.z - C_nano[i].pos.z;
      dr2 = Cuda_Distance(dx,dy,dz,c_delx);
      dr = sqrt(dr2);
      if(dr < (Cuda_NanoRc+buffer)&&dr > 1.0E-10){
      	C_hcen[index] += Cuda_HardEn(dr,buffer,Cuda_NanoRc+buffer);
	    force = Cuda_HardForce(dr,buffer,Cuda_NanoRc+buffer)/ dr;
	    
	    C_atomf[index].hardcore_force.x += force*dx;
	    C_atomf[index].hardcore_force.y += force*dy;
	    C_atomf[index].hardcore_force.z += force*dz;
	    
	    atomicAdd(&C_nanof[i].hardcore_force.x,-1.0*force*dx);
	    atomicAdd(&C_nanof[i].hardcore_force.y,-1.0*force*dy);
	    atomicAdd(&C_nanof[i].hardcore_force.z,-1.0*force*dz);
	     
	    bxx[index] += dx * dx * force;
	    byy[index] += dy * dy * force;
	    bzz[index] += dz * dz * force;
	    bxy[index] += dx * dy * force;
	    byz[index] += dy * dz * force;
	    bxz[index] += dx * dz * force;

		};
		
	  }
	}else if(index < (CudaNum_Bead+CudaNum_Nano)){
	  for(int i=index-CudaNum_Bead+1;i < CudaNum_Nano;i++){
      dx = C_nano[index-CudaNum_Bead].pos.x - C_nano[i].pos.x;
      dy = C_nano[index-CudaNum_Bead].pos.y - C_nano[i].pos.y;
      dz = C_nano[index-CudaNum_Bead].pos.z - C_nano[i].pos.z;
      dr2 = Cuda_Distance(dx,dy,dz,c_delx);
      dr = sqrt(dr2);
      if(dr < (Cuda_NanoR+Cuda_NanoRc+buffer)&&dr > 1.0E-10){
      	C_hcen[index] += Cuda_HardEn(dr,buffer,Cuda_NanoR+Cuda_NanoRc+buffer);
	    force = Cuda_HardForce(dr,buffer,Cuda_NanoR+Cuda_NanoRc+buffer)/ dr;
	    
	    atomicAdd(&C_nanof[index-CudaNum_Bead].hardcore_force.x,force*dx);
	    atomicAdd(&C_nanof[index-CudaNum_Bead].hardcore_force.y,force*dy);
	    atomicAdd(&C_nanof[index-CudaNum_Bead].hardcore_force.z,force*dz);
	    
	    atomicAdd(&C_nanof[i].hardcore_force.x,-1.0*force*dx);
	    atomicAdd(&C_nanof[i].hardcore_force.y,-1.0*force*dy);
	    atomicAdd(&C_nanof[i].hardcore_force.z,-1.0*force*dz);
	     
	    bxx[index] += dx * dx * force;
	    byy[index] += dy * dy * force;
	    bzz[index] += dz * dz * force;
	    bxy[index] += dx * dy * force;
	    byz[index] += dy * dz * force;
	    bxz[index] += dx * dz * force;

		};
		
	  }
	}
}



void pair_hardcore_forces(){
  en.hardcore = 0;
  double sxx=0,syy=0,szz=0;
  double sxy=0,syz=0,sxz=0; 
  double *c_en,*c_sxx,*c_syy,*c_szz,*c_sxy,*c_syz,*c_sxz;
  double *En,*Sxx,*Syy,*Szz,*Sxy,*Syz,*Sxz;

  En = (double *)calloc(box.natoms+sim.Nano_num,sizeof(double));
  Sxx = (double *)calloc(box.natoms+sim.Nano_num,sizeof(double));
  Syy = (double *)calloc(box.natoms+sim.Nano_num,sizeof(double));
  Szz = (double *)calloc(box.natoms+sim.Nano_num,sizeof(double));
  Sxy = (double *)calloc(box.natoms+sim.Nano_num,sizeof(double));
  Syz = (double *)calloc(box.natoms+sim.Nano_num,sizeof(double));
  Sxz = (double *)calloc(box.natoms+sim.Nano_num,sizeof(double));

  cudaMalloc((void **)&c_en, (box.natoms+sim.Nano_num)*sizeof(double));
  cudaMalloc((void **)&c_sxx,(box.natoms+sim.Nano_num)*sizeof(double));
  cudaMalloc((void **)&c_syy, (box.natoms+sim.Nano_num)*sizeof(double));
  cudaMalloc((void **)&c_szz, (box.natoms+sim.Nano_num)*sizeof(double));
  cudaMalloc((void **)&c_sxy, (box.natoms+sim.Nano_num)*sizeof(double));
  cudaMalloc((void **)&c_syz, (box.natoms+sim.Nano_num)*sizeof(double));
  cudaMalloc((void **)&c_sxz, (box.natoms+sim.Nano_num)*sizeof(double));

  cudaMemcpy(c_en, En, (box.natoms+sim.Nano_num)*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(c_sxx, Sxx, (box.natoms+sim.Nano_num)*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(c_syy, Syy, (box.natoms+sim.Nano_num)*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(c_szz, Szz, (box.natoms+sim.Nano_num)*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(c_sxy, Sxy, (box.natoms+sim.Nano_num)*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(c_syz, Syz, (box.natoms+sim.Nano_num)*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(c_sxz, Sxz, (box.natoms+sim.Nano_num)*sizeof(double), cudaMemcpyHostToDevice);
  // call kernel to calculate bonded energy
  Cuda_Attractive_Hc_ForceCompute<<<(box.natoms+sim.Nano_num)/BLOCK_SIZE+1,BLOCK_SIZE>>>(c_en,c_atom,c_nano,c_atomf,c_nanof,c_sxx,c_syy,c_szz,c_sxy,c_syz,c_sxz,dx_LE,sim.dp);

  cudaMemcpy(En, c_en, (box.natoms+sim.Nano_num)*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(Sxx, c_sxx, (box.natoms+sim.Nano_num)*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(Syy, c_syy, (box.natoms+sim.Nano_num)*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(Szz, c_szz, (box.natoms+sim.Nano_num)*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(Sxy, c_sxy, (box.natoms+sim.Nano_num)*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(Syz, c_syz, (box.natoms+sim.Nano_num)*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(Sxz, c_sxz, (box.natoms+sim.Nano_num)*sizeof(double), cudaMemcpyDeviceToHost);

  for(int i = 0;i<(box.natoms+sim.Nano_num);i++){
    en.hardcore += En[i];
    sxx += Sxx[i];
    syy += Syy[i];
    szz += Szz[i];
    sxy += Sxy[i];
    syz += Syz[i];
    sxz += Sxz[i];
  }

  cudaFree(c_en);
  cudaFree(c_sxx);
  cudaFree(c_syy);
  cudaFree(c_szz);
  cudaFree(c_sxy);
  cudaFree(c_syz);
  cudaFree(c_sxz);

  free(En);
  free(Sxx);
  free(Syy);
  free(Szz);
  free(Sxy);
  free(Syz);
  free(Sxz);

  sxx /= box.vol;
  syy /= box.vol;
  szz /= box.vol;
  sxy /= box.vol;
  syz /= box.vol;
  sxz /= box.vol;

  SS[0]+=sxx;SS[1]+=sxy;SS[2]+=sxz;
  SS[3]+=sxy;SS[4]+=syy;SS[5]+=syz;
  SS[6]+=sxz;SS[7]+=syz;SS[8]+=szz;

  SSHC[0]+=sxx;SSHC[1]+=sxy;SSHC[2]+=sxz;
  SSHC[3]+=sxy;SSHC[4]+=syy;SSHC[5]+=syz;
  SSHC[6]+=sxz;SSHC[7]+=syz;SSHC[8]+=szz;


  temp_hc.xx = sxx;
  temp_hc.yy = syy;
  temp_hc.zz = szz;
  temp_hc.xy = sxy;
  temp_hc.yz = syz;
  temp_hc.xz = sxz;
}

void hardcore_forces(){
  pair_hardcore_forces();
}


__global__ void SS_Force_Compute(Force *cudaForce,atoms *cudaAtom,ss_t *cuda_ss,double *bxx,double *byy,double *bzz,double *bxy,double *byz,double *bxz,int n_ss,double c_dxle){
  int index = blockIdx.x * blockDim.x +threadIdx.x;
  double dx,dy,dz,dr2;
  int end1,end2;
  if(index < n_ss){
	  end1 = cuda_ss[index].end1;
	  end2 = cuda_ss[index].end2;
    dx = cudaAtom[end1].pos.x - cudaAtom[end2].pos.x;
    dy = cudaAtom[end1].pos.y - cudaAtom[end2].pos.y;
    dz = cudaAtom[end1].pos.z - cudaAtom[end2].pos.z;
    dr2 = Cuda_Distance(dx,dy,dz,c_dxle);
    cuda_ss[index].en = Cuda_ss_en(dr2);


    bxx[index] = SS_f_coeff * dx * dx;
    byy[index] = SS_f_coeff * dy * dy;
    bzz[index] = SS_f_coeff * dz * dz;
    bxy[index] = SS_f_coeff * dx * dy;
    byz[index] = SS_f_coeff * dy * dz;
    bxz[index] = SS_f_coeff * dx * dz;

    cudaForce[end1].ss_f.x = SS_f_coeff*dx;
    cudaForce[end1].ss_f.y = SS_f_coeff*dy;
    cudaForce[end1].ss_f.z = SS_f_coeff*dz;
    cudaForce[end2].ss_f.x = -1.0*SS_f_coeff*dx;
    cudaForce[end2].ss_f.y = -1.0*SS_f_coeff*dy;
    cudaForce[end2].ss_f.z = -1.0*SS_f_coeff*dz;

  }
}




/*void ss_forces(void){
  double sxx=0,syy=0,szz=0;
  double sxy=0,syz=0,sxz=0;

  en.sl_ener = 0;

  double *c_sxx,*c_syy,*c_szz,*c_sxy,*c_syz,*c_sxz;
  double *Sxx,*Syy,*Szz,*Sxy,*Syz,*Sxz;

  Sxx = (double *)calloc(sim.nss,sizeof(double));
  Syy = (double *)calloc(sim.nss,sizeof(double));
  Szz = (double *)calloc(sim.nss,sizeof(double));
  Sxy = (double *)calloc(sim.nss,sizeof(double));
  Syz = (double *)calloc(sim.nss,sizeof(double));
  Sxz = (double *)calloc(sim.nss,sizeof(double));


  cudaMalloc((void **)&c_sxx, sim.nss*sizeof(double));
  cudaMalloc((void **)&c_syy, sim.nss*sizeof(double));
  cudaMalloc((void **)&c_szz, sim.nss*sizeof(double));
  cudaMalloc((void **)&c_sxy, sim.nss*sizeof(double));
  cudaMalloc((void **)&c_syz, sim.nss*sizeof(double));
  cudaMalloc((void **)&c_sxz, sim.nss*sizeof(double));

  cudaMemcpy(c_sxx, Sxx, sim.nss*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(c_syy, Syy, sim.nss*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(c_szz, Szz, sim.nss*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(c_sxy, Sxy, sim.nss*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(c_syz, Syz, sim.nss*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(c_sxz, Sxz, sim.nss*sizeof(double), cudaMemcpyHostToDevice);
  // call kernel to calculate bonded energy
  //cudaMemcpy(c_ss,ss, NSS*sizeof(ss_t), cudaMemcpyHostToDevice);
  SS_Force_Compute<<<sim.nss/BLOCK_SIZE+1,BLOCK_SIZE>>>(c_atomf,c_atom,c_ss,c_sxx,c_syy,c_szz,c_sxy,c_syz,c_sxz,sim.nss,dx_LE);

  cudaMemcpy(ss, c_ss, NSS*sizeof(ss_t), cudaMemcpyDeviceToHost);

  cudaMemcpy(Sxx, c_sxx, sim.nss*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(Syy, c_syy, sim.nss*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(Szz, c_szz, sim.nss*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(Sxy, c_sxy, sim.nss*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(Syz, c_syz, sim.nss*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(Sxz, c_sxz, sim.nss*sizeof(double), cudaMemcpyDeviceToHost);

  for(int i = 0;i<sim.nss;i++){	  
    en.sl_ener += ss[i].en;
    sxx += Sxx[i];
    syy += Syy[i];
    szz += Szz[i];
    sxy += Sxy[i];
    syz += Syz[i];
    sxz += Sxz[i];
  }





  cudaFree(c_sxx);
  cudaFree(c_syy);
  cudaFree(c_szz);
  cudaFree(c_sxy);
  cudaFree(c_syz);
  cudaFree(c_sxz);


  free(Sxx);
  free(Syy);
  free(Szz);
  free(Sxy);
  free(Syz);
  free(Sxz);



  sxx /= box.vol;
  sxy /= box.vol;
  syy /= box.vol;
  sxy /= box.vol;
  syz /= box.vol;
  sxz /= box.vol;

  SS[0]+=sxx;SS[1]+=sxy;SS[2]+=sxz;
  SS[3]+=sxy;SS[4]+=syy;SS[5]+=syz;
  SS[6]+=sxz;SS[7]+=syz;SS[8]+=szz;

  SS_f[0]+=sxx;SS_f[1]+=sxy;SS_f[2]+=sxz;
  SS_f[3]+=sxy;SS_f[4]+=syy;SS_f[5]+=syz;
  SS_f[6]+=sxz;SS_f[7]+=syz;SS_f[8]+=szz;

  temp_ss.xx = sxx;
  temp_ss.yy = syy;
  temp_ss.zz = szz;
  temp_ss.xy = sxy;
  temp_ss.yz = syz;
  temp_ss.xz = sxz;

}*/


void ss_forces(void){
  double ener;
  double sl_ener = 0.0;
  double dx,dy,dz,dr;
  double sxx=0,syy=0,szz=0;
  double sxy=0,syz=0,sxz=0;

 for(int i=0;i < sim.nss;i++){
    dx = atom[ss[i].end1].pos.x - atom[ss[i].end2].pos.x;
    dy = atom[ss[i].end1].pos.y - atom[ss[i].end2].pos.y;
    dz = atom[ss[i].end1].pos.z - atom[ss[i].end2].pos.z;
    dr = distance(dx,dy,dz);
    ener = ss_en(dr);

    force_bead[ss[i].end1].ss_f.x = SS_F_coef * dx;
    force_bead[ss[i].end2].ss_f.x = -1.0 * SS_F_coef * dx; 
    force_bead[ss[i].end1].ss_f.y = SS_F_coef*dy;
    force_bead[ss[i].end2].ss_f.y = -1.0 * SS_F_coef * dy;
    force_bead[ss[i].end1].ss_f.z = SS_F_coef * dz;
    force_bead[ss[i].end2].ss_f.z = -1.0*SS_F_coef * dz;
    

      sxx += dx* dx * SS_F_coef;
      syy += dy* dy * SS_F_coef;
      szz += dz* dz * SS_F_coef;

      sxy += dy* dx * SS_F_coef;
      syz += dz* dy * SS_F_coef;
      sxz += dx* dz * SS_F_coef;
    
    


	
    sl_ener += ener;
    ss[i].en = ener;
  }
  
  sxx /= box.vol;
  syy /= box.vol;
  szz /= box.vol;
  sxy /= box.vol;
  syz /= box.vol;
  sxz /= box.vol;

  SS[0]+=sxx;SS[1]+=sxy;SS[2]+=sxz;
  SS[3]+=sxy;SS[4]+=syy;SS[5]+=syz;
  SS[6]+=sxz;SS[7]+=syz;SS[8]+=szz;

  SS_f[0]+=sxx;SS_f[1]+=sxy;SS_f[2]+=sxz;
  SS_f[3]+=sxy;SS_f[4]+=syy;SS_f[5]+=syz;
  SS_f[6]+=sxz;SS_f[7]+=syz;SS_f[8]+=szz;

  temp_ss.xx = sxx;
  temp_ss.yy = syy;
  temp_ss.zz = szz;
  temp_ss.xy = sxy;
  temp_ss.yz = syz;
  temp_ss.xz = sxz;
  en.sl_ener = sl_ener;
}


__device__ double Cuda_Nbforce(int a,int b,atoms *C_atom, Force *C_atomf,double *bxx,double *byy,double *bzz,double *bxy,double *byz,double *bxz,double delx){
  double force;
  double dx,dy,dz,d,d2,d3;
  double nb_en=0.0;
  if(a < CudaNum_Bead && b < CudaNum_Bead){
    dx = C_atom[a].pos.x-C_atom[b].pos.x;
    dy = C_atom[a].pos.y-C_atom[b].pos.y;
    dz = C_atom[a].pos.z-C_atom[b].pos.z;

    d = sqrt(Cuda_Distance(dx,dy,dz,delx));
    d2 = d*d;
    d3 = d2*d;
    if(d<Cuda_Rcut){
      nb_en = D*(A-B*d+d3*0.0833333333333);
      force = C*(0.25*d2-B);

	force/=d;

	atomicAdd(&C_atomf[a].nonbond_force.x,force *dx);
	atomicAdd(&C_atomf[a].nonbond_force.y,force *dy);
	atomicAdd(&C_atomf[a].nonbond_force.z,force *dz);
	
	atomicAdd(&C_atomf[b].nonbond_force.x,-1*force *dx);
	atomicAdd(&C_atomf[b].nonbond_force.y,-1*force *dy);
	atomicAdd(&C_atomf[b].nonbond_force.z,-1*force *dz);



	//bxx[a] += dx*dx*force;
	//byy[a] += dy*dy*force;
	//bzz[a] += dz*dz*force;
	bxy[a] += dx*dy*force;
	byz[a] += dy*dz*force;
	bxz[a] += dx*dz*force;


    }
  }

  return nb_en;
}

__global__ void Cuda_Bead_Brownian(Force *cudaForce,atoms *cudaAtom,vect* cuda_atom_v,double dt,double sqrdt,double v_d,int N_atom,double c_dxle,int sseed){
  int index = blockIdx.x * blockDim.x +threadIdx.x;
  double dis_x,dis_y,dis_z;
  if(index < N_atom){
    cudaForce[index].total_force.x = cudaForce[index].bond_force.x+cudaForce[index].nonbond_force.x+cudaForce[index].hardcore_force.x+cudaForce[index].ss_f.x;

    cudaForce[index].total_force.y = cudaForce[index].bond_force.y+cudaForce[index].nonbond_force.y+cudaForce[index].hardcore_force.y+cudaForce[index].ss_f.y;

    cudaForce[index].total_force.z = cudaForce[index].bond_force.z+cudaForce[index].nonbond_force.z+cudaForce[index].hardcore_force.z+cudaForce[index].ss_f.z;		

    curandState state;
    curand_init(sseed, index, 0, &state);
    
    dis_x = (cudaForce[index].total_force.x + sqrdt * curand_normal_double(&state)) * dt;
    dis_y = (cudaForce[index].total_force.y + sqrdt * curand_normal_double(&state) )* dt;
    dis_z = (cudaForce[index].total_force.z + sqrdt * curand_normal_double(&state) )* dt;

    cudaAtom[index].pos.x += dis_x;
    cudaAtom[index].pos.y += dis_y;
    cudaAtom[index].pos.z += dis_z;

    if(Cuda_LeesEd == true)cudaAtom[index].pos.x += v_d*(cudaAtom[index].pos.z/CudaLength3)*dt;

    Cuda_Pbc_One(cudaAtom[index].pos.x,cudaAtom[index].pos.y,cudaAtom[index].pos.z,c_dxle,v_d,dt);


    cuda_atom_v[index].x += dis_x;
    cuda_atom_v[index].y += dis_y;
    cuda_atom_v[index].z += dis_z;
  }
}


__global__ void Cuda_Nano_Brownian(NanoForce *cudaForce,nanos *cudaNano,vect *cuda_nano_v,double dt,double nano_sqrdt,double nano_mass,double nano_frict,double v_d,int N_nano,double c_dxle,int sseed){
  int index = blockIdx.x * blockDim.x +threadIdx.x;
  double dis_x,dis_y,dis_z;
  if(index < N_nano){
    cudaForce[index].total_force.x = cudaForce[index].nonbond_force.x+cudaForce[index].hardcore_force.x;

    cudaForce[index].total_force.y = cudaForce[index].nonbond_force.y+cudaForce[index].hardcore_force.y;

    cudaForce[index].total_force.z = cudaForce[index].nonbond_force.z+cudaForce[index].hardcore_force.z;		

    curandState state;
    curand_init(sseed, index, 0, &state);

    dis_x = (cudaForce[index].total_force.x/(nano_mass*nano_frict) + nano_sqrdt * curand_normal_double(&state)) * dt;
    dis_y = (cudaForce[index].total_force.y/(nano_mass*nano_frict) + nano_sqrdt * curand_normal_double(&state)) * dt;
    dis_z = (cudaForce[index].total_force.z/(nano_mass*nano_frict) + nano_sqrdt * curand_normal_double(&state)) * dt;

    cudaNano[index].pos.x += dis_x;
    cudaNano[index].pos.y += dis_y;
    cudaNano[index].pos.z += dis_z;

    if(Cuda_LeesEd == true)cudaNano[index].pos.x += v_d*(cudaNano[index].pos.z/CudaLength3)*dt;

    Cuda_Pbc_One(cudaNano[index].pos.x,cudaNano[index].pos.y,cudaNano[index].pos.z,c_dxle,v_d,dt);


    cuda_nano_v[index].x += dis_x;
    cuda_nano_v[index].y += dis_y;
    cuda_nano_v[index].z += dis_z;
  }
}


void Brownian()
{
  double sqrt_fric = sqrt(sim.Nano_R*sim.frict);
  double disx,disy,disz;

  for(int i=0; i<box.natoms; i++) 
    {

      force_bead[i].total_force.x = force_bead[i].bond_force.x+force_bead[i].nonbond_force.x+force_bead[i].hardcore_force.x+force_bead[i].ss_f.x;
      force_bead[i].total_force.y = force_bead[i].bond_force.y+force_bead[i].nonbond_force.y+force_bead[i].hardcore_force.y+force_bead[i].ss_f.y;
      force_bead[i].total_force.z = force_bead[i].bond_force.z+force_bead[i].nonbond_force.z+force_bead[i].hardcore_force.z+force_bead[i].ss_f.z;
      
      disx = (force_bead[i].total_force.x + sqrdt * gauss(1.0)) * sim.dt;
      disy = (force_bead[i].total_force.y + sqrdt * gauss(1.0)) * sim.dt;
      disz	= (force_bead[i].total_force.z + sqrdt * gauss(1.0)) * sim.dt;
      atom[i].pos.x += disx;
      atom[i].pos.y += disy;
      atom[i].pos.z += disz;
      
      if(LeesEd == true)atom[i].pos.x += sim.v_d*(atom[i].pos.z/box.length3)*sim.dt;

      pbc_one(i);
      
//For the self-diffusivities

      if(MSD == true){
	atom_v[i].x += disx;
	atom_v[i].y += disy;
	atom_v[i].z += disz;
      
    }
      

  }
  cudaMemcpy(c_atom, atom, box.natoms*sizeof(atoms), cudaMemcpyHostToDevice);
  cudaMemcpy(c_atom_v, atom_v, box.natoms*sizeof(vect), cudaMemcpyHostToDevice);
  ss_update();

  for (int i=0; i<sim.Nano_num;i++){
    force_nano[i].total_force.x = force_nano[i].nonbond_force.x+force_nano[i].hardcore_force.x;
    force_nano[i].total_force.y = force_nano[i].nonbond_force.y+force_nano[i].hardcore_force.y;
    force_nano[i].total_force.z = force_nano[i].nonbond_force.z+force_nano[i].hardcore_force.z;
      
   disx = (force_nano[i].total_force.x/(sim.Nano_R*sim.frict) + sqrt(2.0/sim.dt)*gauss(1.0)/sqrt_fric) * sim.dt;
   disy = (force_nano[i].total_force.y/(sim.Nano_R*sim.frict) + sqrt(2.0/sim.dt)*gauss(1.0)/sqrt_fric) * sim.dt;
   disz	= (force_nano[i].total_force.z/(sim.Nano_R*sim.frict) + sqrt(2.0/sim.dt)*gauss(1.0)/sqrt_fric) * sim.dt;
      
    nano[i].pos.x += disx;
    nano[i].pos.y += disy;
    nano[i].pos.z += disz;
    
    if(LeesEd == true)nano[i].pos.x += sim.v_d*(nano[i].pos.z/box.length3)*sim.dt;
    nano_pbc_one(i);

    if(MSD == true){
      nano_v[i].x += disx;
      nano_v[i].y += disy;
      nano_v[i].z += disz;
    }

    

  }
  
  cudaMemcpy(c_nano,nano,sim.Nano_num*sizeof(nanos), cudaMemcpyHostToDevice);
  //cudaMemcpy(c_nano_v,nano_v,sim.Nano_num*sizeof(vect), cudaMemcpyHostToDevice);
  
 if(MSD == true)chain_center();

}

__global__ void SS_update(Force *cudaForce,atoms *cudaAtom,ss_t *cuda_ss,int Nss,double c_dxle){
  int index = blockIdx.x * blockDim.x +threadIdx.x;
  double dx,dy,dz,dr2;
  if(index < Nss){

    dx = cudaAtom[cuda_ss[index].end1].pos.x - cudaAtom[cuda_ss[index].end2].pos.x;
    dy = cudaAtom[cuda_ss[index].end1].pos.y - cudaAtom[cuda_ss[index].end2].pos.y;
    dz = cudaAtom[cuda_ss[index].end1].pos.z - cudaAtom[cuda_ss[index].end2].pos.z;
    dr2 = Cuda_Distance(dx,dy,dz,c_dxle);
    cuda_ss[index].en = Cuda_ss_en(dr2);


    cudaForce[cuda_ss[index].end1].ss_f.x = SS_f_coeff*dx;
    cudaForce[cuda_ss[index].end1].ss_f.y = SS_f_coeff*dy;
    cudaForce[cuda_ss[index].end1].ss_f.z = SS_f_coeff*dz;
    //__syncthreads();
    cudaForce[cuda_ss[index].end2].ss_f.x = -SS_f_coeff*dx;
    cudaForce[cuda_ss[index].end2].ss_f.y = -SS_f_coeff*dy;
    cudaForce[cuda_ss[index].end2].ss_f.z = -SS_f_coeff*dz;

    /*atomicAdd(&cudaForce[cuda_ss[index].end1].ss_f.x,SS_f_coeff*dx);
      atomicAdd(&cudaForce[cuda_ss[index].end1].ss_f.y,SS_f_coeff*dy);
      atomicAdd(&cudaForce[cuda_ss[index].end1].ss_f.z,SS_f_coeff*dz);
      atomicAdd(&cudaForce[cuda_ss[index].end2].ss_f.x,-1.0*SS_f_coeff*dx);
      atomicAdd(&cudaForce[cuda_ss[index].end2].ss_f.y,-1.0*SS_f_coeff*dy);
      atomicAdd(&cudaForce[cuda_ss[index].end2].ss_f.z,-1.0*SS_f_coeff*dz);*/

  }
}




void ss_update(void){


  en.sl_ener = 0;

  // call kernel to calculate bonded energy
  SS_update<<<sim.nss/BLOCK_SIZE+1,BLOCK_SIZE>>>(c_atomf,c_atom,c_ss,sim.nss,dx_LE);

  cudaMemcpy(ss, c_ss, NSS*sizeof(ss_t), cudaMemcpyDeviceToHost);


  for(int i = 0;i<sim.nss;i++){	  
    en.sl_ener += ss[i].en;

  }


}
/*void Brownian()
{
  double sqrt_wei_fric = sqrt(sim.Nano_weight*sim.frict);
  double nano_sqrdt = sqrdt/sqrt_wei_fric;

  //cudaMemcpy(c_atomf,force_bead,box.natoms*sizeof(Force), cudaMemcpyHostToDevice);
  //cudaMemcpy(c_atom,atom,box.natoms*sizeof(atoms), cudaMemcpyHostToDevice);
  //cudaMemcpy(c_atom_v,atom_v,box.natoms*sizeof(vect), cudaMemcpyHostToDevice);

  Cuda_Bead_Brownian<<<box.natoms/BLOCK_SIZE+1,BLOCK_SIZE>>>(c_atomf,c_atom,c_atom_v,sim.dt,sqrdt,sim.v_d,box.natoms,dx_LE,rand());
  
  ss_update();
  //cudaMemcpy(force_bead,c_atomf,box.natoms*sizeof(Force), cudaMemcpyDeviceToHost);
  cudaMemcpy(atom,c_atom,box.natoms*sizeof(atoms), cudaMemcpyDeviceToHost);
  cudaMemcpy(atom_v,c_atom_v,box.natoms*sizeof(vect), cudaMemcpyDeviceToHost);
  

  if(sim.Nano_num > 0){
    //cudaMemcpy(c_nanof,force_nano,sim.Nano_num*sizeof(NanoForce), cudaMemcpyHostToDevice);
    //cudaMemcpy(c_nano,nano,sim.Nano_num*sizeof(nanos), cudaMemcpyHostToDevice);
    //cudaMemcpy(c_nano_v,nano_v,sim.Nano_num*sizeof(vect), cudaMemcpyHostToDevice);

    Cuda_Nano_Brownian<<<sim.Nano_num/BLOCK_SIZE+1,BLOCK_SIZE>>>(c_nanof,c_nano,c_nano_v,sim.dt,nano_sqrdt,sim.Nano_weight,sim.frict,sim.v_d,sim.Nano_num,dx_LE,rand());

    //cudaMemcpy(force_nano,c_nanof,sim.Nano_num*sizeof(NanoForce), cudaMemcpyDeviceToHost);
    cudaMemcpy(nano,c_nano,sim.Nano_num*sizeof(nanos), cudaMemcpyDeviceToHost);
    cudaMemcpy(nano_v,c_nano_v,sim.Nano_num*sizeof(vect), cudaMemcpyDeviceToHost);
  }

  chain_center();
}*/


void construct_link()
{
  int ind_x,ind_y,ind_z;
  int ind;

  /*reset the headers,head*/
  for(int i=0;i<L;i++){
    head[i]=-1;
  }
  /* Scan atoms to construct headers, head, & linked lists, lscl */

  for(int i=0;i<box.natoms;i++)
    {

	ind_x=grid(atom[i].pos.x,1);
	ind_y=grid(atom[i].pos.y,2);
	ind_z=grid(atom[i].pos.z,3);
      /* Translate the vector cell index, mc, to a scalar cell index */
      ind=ind_x*L_2*L_3+ind_y*L_3+ind_z;
      lscl[i]=head[ind];
      head[ind]=i;

    }
}

/*void nn_construct_link()
{
  int ind_x,ind_y,ind_z;
  int ind;

  for(int i=0;i<nn_L;i++){
    nn_head[i]=-1;
  }
  for(int i=0;i<box.natoms+sim.Nano_num;i++)
    {
      if(i<box.natoms){
	ind_x=nn_grid(atom[i].pos.x,1);
	ind_y=nn_grid(atom[i].pos.y,2);
	ind_z=nn_grid(atom[i].pos.z,3);
      }else{
	ind_x=nn_grid(nano[i-box.natoms].pos.x,1);
	ind_y=nn_grid(nano[i-box.natoms].pos.y,2);
	ind_z=nn_grid(nano[i-box.natoms].pos.z,3);
      }

      ind=ind_x*nn_L_2*nn_L_3+ind_y*nn_L_3+ind_z;
      nn_lscl[i]=nn_head[ind];
      nn_head[ind]=i;
    }
}*/


/*int nn_grid(double pos,int sw)
{
  int k;
  if(sw==1){
    pos = modulo(pos,box.length1);
    k=(int)((pos+box.length1/2.0)/del_nnL1);
  }else if(sw==2){
    pos = modulo(pos,box.length2);
    k=(int)((pos+box.length2/2.0)/del_nnL2);
  }else{
    pos = modulo(pos,box.length3);
    k=(int)((pos+box.length3/2.0)/del_nnL3);
  }
  return k;
}*/
int grid(double pos,int sw)
{
  int k;
  if(sw==1){
    pos = modulo(pos,box.length1);
    k=(int)((pos+box.length1/2.0)/del_L1);
  }else if(sw==2){
    pos = modulo(pos,box.length2);
    k=(int)((pos+box.length2/2.0)/del_L2);
  }else{
    pos = modulo(pos,box.length3);
    k=(int)((pos+box.length3/2.0)/del_L3);
  }
  return k;
}


__device__ int Cuda_Grid(double pos,int sw)
{
  int k;
  if(sw==1){
    pos = Cuda_modulo(pos,CudaLength1);
    k=(int)((pos+CudaLength1/2.0)/Cuda_dL1);
  }else if(sw==2){
    pos = Cuda_modulo(pos,CudaLength2);
    k=(int)((pos+CudaLength2/2.0)/Cuda_dL2);
  }else{
    pos = Cuda_modulo(pos,CudaLength3);
    k=(int)((pos+CudaLength3/2.0)/Cuda_dL3);
  }
  return k;
}



double gauss(double sigma)
{
  double ran1,ran2,ransq,v;
  do
    {
      ran1=2.0*ran_num_double(0,1)-1.0;
      ran2=2.0*ran_num_double(0,1)-1.0;
      ransq=ran1*ran1+ran2*ran2;
    }
  while(ransq >= 1.0);

  v=sqrt(sigma)*ran1*sqrt(-2.0*log(ransq)/ransq);
  return v;
}

__device__ double Cuda_Gauss(double sigma,int seed)
{
  int index = blockIdx.x * blockDim.x +threadIdx.x;
  curandState state;
  curand_init(seed, index, 0, &state);
  double ran1,ran2,ransq,v;
  do
    {
      ran1=2.0*curand_uniform(&state)-1.0;
      ran2=2.0*curand_uniform(&state)-1.0;
      ransq=ran1*ran1+ran2*ran2;
    }
  while(ransq >= 1.0);

  v=sqrt(sigma)*ran1*sqrt(-2.0*log(ransq)/ransq);
  return v;
}


void initial_rdf()
{
  for(int i=0;i<Nbin;i++)
    {
      hist_n2p[i]=0;
      hist_n2s[i]=0;
      hist_n2n[i]=0;
      hist_p2p[i]=0;
    }

}
void rdf()
{
  double dx,dy,dz;
  double dr2,dr;
  int ibins;
  double ax,ay,az;

  for( int i=0; i<sim.Nano_num; i++)
    {
    ax = nano[i].pos.x;
    ay = nano[i].pos.y;
    az = nano[i].pos.z;

    for(int j=0; j<box.natoms; j++)
    {
    dx = ax - atom[j].pos.x;
    dy = ay - atom[j].pos.y;
    dz = az - atom[j].pos.z;


    dr2 = distance(dx,dy,dz);
    dr=sqrt(dr2);
    ibins=(int)(dr/delt_r);
    if(ibins<Nbin)
    {
    hist_n2p[ibins]+=1;
    }
    }
    }
    
    
    for( int i=0; i<sim.Nano_num; i++)
    {
    ax = nano[i].pos.x;
    ay = nano[i].pos.y;
    az = nano[i].pos.z;

    for(int j=0; j<sim.nss; j++)
    {
    dx = ax - atom[ss[j].end1].pos.x;
    dy = ay - atom[ss[j].end1].pos.y;
    dz = az - atom[ss[j].end1].pos.z;


    dr2 = distance(dx,dy,dz);
    dr=sqrt(dr2);
    ibins=(int)(dr/delt_r);
    if(ibins<Nbin)
    {
    hist_n2s[ibins]+=1;
    }
    
    
    dx = ax - atom[ss[j].end2].pos.x;
    dy = ay - atom[ss[j].end2].pos.y;
    dz = az - atom[ss[j].end2].pos.z;


    dr2 = distance(dx,dy,dz);
    dr=sqrt(dr2);
    ibins=(int)(dr/delt_r);
    if(ibins<Nbin)
    {
    hist_n2s[ibins]+=1;
    }    
    }
    }  
    ss_rdf_count += 2.0*sim.nss;
    
  for( int i=0; i<sim.Nano_num-1; i++)
    {
      ax = nano[i].pos.x;
      ay = nano[i].pos.y;
      az = nano[i].pos.z;

      for(int j=i+1; j<sim.Nano_num; j++)
	{

	  dx = ax - nano[j].pos.x;
	  dy = ay - nano[j].pos.y;
	  dz = az - nano[j].pos.z;

	  dr2 = distance(dx,dy,dz);
	  dr=sqrt(dr2);
	  ibins=(int)(dr/delt_r);
	  if(ibins<Nbin)
	    {
	      hist_n2n[ibins]+=2;
	    }
	}
    }

  /*for( int i=0; i<box.natoms-1; i++)
    {
    ax = atom[i].pos.x;
    ay = atom[i].pos.y;
    az = atom[i].pos.z;

    for(int j=i+1; j<box.natoms; j++)
    {

    dx = ax - atom[j].pos.x;
    dy = ay - atom[j].pos.y;
    dz = az - atom[j].pos.z;

    dr2 = distance(dx,dy,dz);
    dr=sqrt(dr2);
    ibins=(int)(dr/delt_r);
    if(ibins<Nbin)
    {
    hist_p2p[ibins]+=1;
    }
    }
    }  */

}
void fin_rdf()
{
  FILE *io;
  double Ngr=exstep;
    
  io=fopen("./rdf_n2p.hist","w");
  for(int i=0; i<Nbin; i++)fprintf(io,"%lf\t%lf\n",(i+0.5)*delt_r,hist_n2p[i]/(4*PI*(i+0.5)*(i+0.5)*delt_r*delt_r*delt_r*box.natoms*Ngr*sim.Nano_num/box.vol));
  fclose(io);
  io=fopen("./rdf_n2s.hist","w");
  for(int i=0; i<Nbin; i++)fprintf(io,"%lf\t%lf\n",(i+0.5)*delt_r,hist_n2s[i]/(4*PI*(i+0.5)*(i+0.5)*delt_r*delt_r*delt_r*ss_rdf_count*sim.Nano_num/box.vol));
  fclose(io);
  ss_rdf_count = 0;
  io=fopen("./rdf_n2n.hist","w");
  for(int i=0; i<Nbin; i++)fprintf(io,"%lf\t%lf\n",(i+0.5)*delt_r,hist_n2n[i]/(4*PI*(i+0.5)*(i+0.5)*delt_r*delt_r*delt_r*sim.Nano_num*Ngr*sim.Nano_num/box.vol));
  fclose(io);
  io=fopen("./rdf_p2p.hist","w");
  for(int i=0; i<Nbin; i++)fprintf(io,"%lf\t%lf\n",(i+0.5)*delt_r,hist_p2p[i]/(4*PI*(i+0.5)*(i+0.5)*delt_r*delt_r*delt_r*box.natoms*Ngr*density));
  fclose(io);
}


double ran_num_double(int range1,int range2)
{
  //return range1+(range2-range1)*ran_num_double(0,1);
  return range1+(range2-range1)*(double)rand()/RAND_MAX;
}

// Function to generate integer type random numbers.

int ran_num_int(int range1,int range2)
{
  int re = (int)ran_num_double(range1,range2+1);
  if(re == range2+1)re = range2;
  return re;
}

/* #### END OF RANDOM NUMBER GENERATOR CODE ########## */

void ini_grid()
{
  box.phi0=(double ***)calloc(ndfx,sizeof(double **));
  box.phi1=(double ***)calloc(ndfx,sizeof(double **));
  box.phi2=(double ***)calloc(ndfx,sizeof(double **));
  for(int i=0;i < ndfx;i++)
    {
      box.phi0[i]=(double **)calloc(ndfy,sizeof(double *));
      box.phi1[i]=(double **)calloc(ndfy,sizeof(double *));
      box.phi2[i]=(double **)calloc(ndfy,sizeof(double *));
    }

  for(int i=0;i < ndfx;i++)
    for(int j=0;j < ndfy;j++)
      {
	box.phi0[i][j]=(double *)calloc(ndfz,sizeof(double));
	box.phi1[i][j]=(double *)calloc(ndfz,sizeof(double));
	box.phi2[i][j]=(double *)calloc(ndfz,sizeof(double));
      }
}

void mayavi_density(int sw){
  if(sw == 0){
    for(int i=0;i<ndfx;i++)
      for(int j=0;j<ndfy;j++)
	for(int z=0;z<ndfz;z++)
	  {
	    box.phi0[i][j][z]=0.0;
	    box.phi1[i][j][z]=0.0;
	    box.phi2[i][j][z]=0.0;
	  }
  }else if(sw ==1){

    int ind_x,ind_y,ind_z;

    for(int i=0;i<box.natoms+sim.Nano_num*sim.Nano_bead;i++)
      {
	if(i<box.natoms){
	  ind_x = int((atom[i].pos.x+box.length1/2.0)/dLx_field);
	  ind_y = int((atom[i].pos.y+box.length2/2.0)/dLy_field);
	  ind_z = int((atom[i].pos.z+box.length3/2.0)/dLz_field);
	  if(atom[i].type == 0){
	    box.phi0[ind_x][ind_y][ind_z] += 1.0;
	  }else{
	    box.phi1[ind_x][ind_y][ind_z] += 1.0;
	  }
	}else{
	  int inano = (i-box.natoms)/sim.Nano_bead;
	  int ibead = (i-box.natoms)%sim.Nano_bead;
	  double dx = (nano[inano].pos.x+nano_atom[ibead].pos.x);
	  double dy = (nano[inano].pos.y+nano_atom[ibead].pos.y);
	  double dz = (nano[inano].pos.z+nano_atom[ibead].pos.z);

	  if(dz >= box.length3/2.0) {
	    dz -= box.length3; 
	    if (LeesEd==true) {
	      dx -= dx_LE;
	    } 
	  }
	  if(dz <= -box.length3/2.0) {
	    dz += box.length3; 
	    if (LeesEd==true) {
	      dx += dx_LE;
	    }
	  }

	  dx = modulo(dx,box.length1);
	  dy = modulo(dy,box.length2);

	  ind_x = int((dx+box.length1/2.0)/dLx_field);
	  ind_y = int((dy+box.length2/2.0)/dLy_field);
	  ind_z = int((dz+box.length3/2.0)/dLz_field);
	  box.phi2[ind_x][ind_y][ind_z] += 1.0/sim.Nano_den;
	}
      }
  }else{

    FILE *io;
    char buff[20];
    sprintf(buff,"./field_%d.vtk",exstep/samp);
    io=fopen(buff,"w");
    fprintf(io,"# vtk DataFile Version 2.0\n");
    fprintf(io,"Density field\n");
    fprintf(io,"ASCII\n\n");
    fprintf(io,"DATASET RECTILINEAR_GRID\n");
    fprintf(io,"DIMENSIONS %d %d %d\n",ndfx,ndfy,ndfz);
    fprintf(io,"X_COORDINATES %d double\n",ndfx);
    for(int i=0;i < ndfx; i++)
      fprintf(io,"%lf ",(i+1)*dLx_field);
    fprintf(io,"\nY_COORDINATES %d double\n",ndfy);
    for(int i=0;i < ndfy; i++)
      fprintf(io,"%lf ",(i+1)*dLy_field);	
    fprintf(io,"\nZ_COORDINATES %d double\n",ndfz);
    for(int i=0;i < ndfz; i++)
      fprintf(io,"%lf ",(i+1)*dLz_field);
    fprintf(io,"\n\n\nPOINT_DATA %d\n",ndfx*ndfy*ndfz);
    fprintf(io,"SCALARS scalars double\n");
    fprintf(io,"LOOKUP_TABLE default\n\n");

    for(int z=0;z<ndfz;z++)
      for(int j=0;j<ndfy;j++)
	for(int i=0;i<ndfx;i++)
	  {
	    if((box.phi0[i][j][z]+box.phi1[i][j][z]) != 0.0)fprintf(io,"\t%lf\n",(box.phi0[i][j][z]-box.phi1[i][j][z])/(box.phi0[i][j][z]+box.phi1[i][j][z]));
	    else fprintf(io,"\t%lf\n",0.0);
	  }
    fclose(io);
    sprintf(buff,"./fieldNP_%d.vtk",exstep/samp);
    io=fopen(buff,"w");
    fprintf(io,"# vtk DataFile Version 2.0\n");
    fprintf(io,"Density field\n");
    fprintf(io,"ASCII\n\n");
    fprintf(io,"DATASET RECTILINEAR_GRID\n");
    fprintf(io,"DIMENSIONS %d %d %d\n",ndfx,ndfy,ndfz);
    fprintf(io,"X_COORDINATES %d double\n",ndfx);
    for(int i=0;i < ndfx; i++)
      fprintf(io,"%lf ",(i+1)*dLx_field);
    fprintf(io,"\nY_COORDINATES %d double\n",ndfy);
    for(int i=0;i < ndfy; i++)
      fprintf(io,"%lf ",(i+1)*dLy_field);	
    fprintf(io,"\nZ_COORDINATES %d double\n",ndfz);
    for(int i=0;i < ndfz; i++)
      fprintf(io,"%lf ",(i+1)*dLz_field);
    fprintf(io,"\n\n\nPOINT_DATA %d\n",ndfx*ndfy*ndfz);
    fprintf(io,"SCALARS scalars double\n");
    fprintf(io,"LOOKUP_TABLE default\n\n");

    for(int z=0;z<ndfz;z++)
      for(int j=0;j<ndfy;j++)
	for(int i=0;i<ndfx;i++)
	  {
	    if(box.phi2[i][j][z] >= 1.0/sim.Nano_den)fprintf(io,"\t%lf\n",box.phi2[i][j][z]);
	    else fprintf(io,"\t%lf\n",0.0);
	  }
    fclose(io);
  }
}




/*double min(double var1,double var2){
  return (var1>var2?var2:var1);
  }*/



__global__ void Cuda_Chain_Center(vect *c_mid,vect *C_atoms_v,int N,int Nc){
  int index = blockIdx.x * blockDim.x +threadIdx.x;
  vect mid,c_bead,tem;
  if(index < Nc){
	  mid.x = C_atoms_v[index*N].x;
	  mid.y = C_atoms_v[index*N].y;
	  mid.z = C_atoms_v[index*N].z;
	  tem.x = mid.x;
	  tem.y = mid.y;
	  tem.z = mid.z;
	  for(int j=1;j < N;j++){
		  c_bead.x = C_atoms_v[index*N+j].x-Cuda_Round((C_atoms_v[index*N+j].x-tem.x)/CudaLength1)*CudaLength1;
		  c_bead.y = C_atoms_v[index*N+j].y-Cuda_Round((C_atoms_v[index*N+j].y-tem.y)/CudaLength2)*CudaLength2;
		  c_bead.z = C_atoms_v[index*N+j].z-Cuda_Round((C_atoms_v[index*N+j].z-tem.z)/CudaLength3)*CudaLength3;
		  mid.x += c_bead.x;
		  mid.y += c_bead.y;
		  mid.z += c_bead.z;
		  tem.x = c_bead.x;
		  tem.y = c_bead.y;
		  tem.z = c_bead.z;
	  }
    c_mid[index].x = mid.x/N;
    c_mid[index].y = mid.y/N;
    c_mid[index].z = mid.z/N;
  }
}

void chain_center(){
	vect *c_chain_mid;
	cudaMalloc((void **)&c_chain_mid, box.chains*sizeof(vect));
	Cuda_Chain_Center<<<box.chains/BLOCK_SIZE,BLOCK_SIZE>>>(c_chain_mid,c_atom_v,sim.N,box.chains);
	cudaMemcpy(chain_mid, c_chain_mid, (box.chains)*sizeof(vect), cudaMemcpyDeviceToHost);
	cudaFree(c_chain_mid);
}
/*void chain_center(){
  double tem_x,tem_y,tem_z;
  vect mid;
  mid.x=0;
  mid.y=0;
  mid.z=0;
  vect chain_bead;

  for(int i=0;i<box.chains;i++){
    tem_x = atom_v[i*sim.N].x;
    tem_y = atom_v[i*sim.N].y;
    tem_z = atom_v[i*sim.N].z;

    mid.x = tem_x;
    mid.y = tem_y;
    mid.z = tem_z;
    for(int j=1;j<sim.N;j++){
      chain_bead.x=atom_v[i*sim.N+j].x-Round((atom_v[i*sim.N+j].x-tem_x)/box.length1)*box.length1;
      chain_bead.y=atom_v[i*sim.N+j].y-Round((atom_v[i*sim.N+j].y-tem_y)/box.length2)*box.length2;
      chain_bead.z=atom_v[i*sim.N+j].z-Round((atom_v[i*sim.N+j].z-tem_z)/box.length3)*box.length3;
      mid.x += chain_bead.x;
      mid.y += chain_bead.y;
      mid.z += chain_bead.z;
      tem_x = chain_bead.x;
      tem_y = chain_bead.y;
      tem_z = chain_bead.z;
    }
    chain_mid[i].x = mid.x/sim.N;
    chain_mid[i].y = mid.y/sim.N;
    chain_mid[i].z = mid.z/sim.N;
  }
}*/

double distance(double &dx,double &dy,double &dz){
  double dr2;
  if(LeesEd == true){
    dx-=Round(dz/box.length3)*dx_LE;
    dx = modulo(dx,box.length1);
  }else{
    dx = modulo(dx,box.length1);
  }

  dy = modulo(dy,box.length2);
  dz = modulo(dz,box.length3);

  dr2 = dx*dx + dy*dy + dz*dz;
  return dr2;
}


__device__ double Cuda_Distance(double &dx,double &dy,double &dz,double CudaDx_LE){
  double dr2;
  if(Cuda_LeesEd == true){
    dx-=Cuda_Round(dz/CudaLength3)*CudaDx_LE;
    dx = Cuda_modulo(dx,CudaLength1);
  }else{
    dx = Cuda_modulo(dx,CudaLength1);
  }

  dy = Cuda_modulo(dy,CudaLength2);
  dz = Cuda_modulo(dz,CudaLength3);

  dr2 = dx*dx + dy*dy + dz*dz;
  return dr2;
}

void temp_stress_out(void){
	buffer_stress[buffer_count][0] = cur_t-gxz_samp*sim.dt;
	buffer_stress[buffer_count][1] = (temp_b.xz+temp_ss.xz+temp_nb.xz+temp_hc.xz);
	buffer_stress[buffer_count][2] = (temp_b.xz);
	buffer_stress[buffer_count][3] = (temp_ss.xz);
	buffer_stress[buffer_count][4] = (temp_nb.xz);
	buffer_stress[buffer_count][5] = (temp_hc.xz);
	
	buffer_stress[buffer_count][6] = (temp_b.xy+temp_ss.xy+temp_nb.xy+temp_hc.xy);
	buffer_stress[buffer_count][7] = (temp_b.xy);
	buffer_stress[buffer_count][8] = (temp_ss.xy);
	buffer_stress[buffer_count][9] = (temp_nb.xy);
	buffer_stress[buffer_count][10] = (temp_hc.xy);
	
	
	buffer_stress[buffer_count][11] = (temp_b.yz+temp_ss.yz+temp_nb.yz+temp_hc.yz);
	buffer_stress[buffer_count][12] = (temp_b.yz);
	buffer_stress[buffer_count][13] = (temp_ss.yz);
	buffer_stress[buffer_count][14] = (temp_nb.yz);
	buffer_stress[buffer_count][15] = (temp_hc.yz);
		
	buffer_count++;
	if(exstep%samp==0||buffer_count==buffer_size){
  ofstream ofs("Gxz_stress.dat",ofstream::app);
  for(int i=0;i<buffer_count;i++)
  ofs <<std::setprecision(10)<<buffer_stress[i][0]<<" "<<buffer_stress[i][1]<<" "<<buffer_stress[i][2]<<" "<<buffer_stress[i][3]<<" "<<buffer_stress[i][4]<<" "<<buffer_stress[i][5]<<endl;
  ofs.close();
  
  ofs.open("Gxy_stress.dat",ofstream::app);
  for(int i=0;i<buffer_count;i++)
  ofs <<std::setprecision(10)<<buffer_stress[i][0]<<" "<<buffer_stress[i][6]<<" "<<buffer_stress[i][7]<<" "<<buffer_stress[i][8]<<" "<<buffer_stress[i][9]<<" "<<buffer_stress[i][10]<<endl;
  ofs.close();
  
  ofs.open("Gyz_stress.dat",ofstream::app);
  for(int i=0;i<buffer_count;i++)
  ofs <<std::setprecision(10)<<buffer_stress[i][0]<<" "<<buffer_stress[i][11]<<" "<<buffer_stress[i][12]<<" "<<buffer_stress[i][13]<<" "<<buffer_stress[i][14]<<" "<<buffer_stress[i][15]<<endl;
  ofs.close();
  
  buffer_count = 0;
  }
}



void stress_out (void){

  FILE *sout7;         
  sout7 = fopen("./stress_xz.out","a");

  /*FILE *sout6; char name6[50];
    sprintf(name6,"./stress.out");           
    sout6 = fopen(name6,"w");

    FILE *soutxx; char namexx[50];
    sprintf(namexx,"./stress_xx.out");           
    soutxx = fopen(namexx,"w");

    FILE *soutzz; char namezz[50];
    sprintf(namezz,"./stress_zz.out");           
    soutzz = fopen(namezz,"w");

    FILE *soutyy; char nameyy[50];
    sprintf(nameyy,"./stress_yy.out");           
    soutyy = fopen(nameyy,"w");

    FILE *soutxy; char namexy[50];
    sprintf(namexy,"./stress_xy.out");           
    soutxy = fopen(namexy,"w");

    FILE *soutyz; char nameyz[50];
    sprintf(nameyz,"./stress_yz.out");           
    soutyz = fopen(nameyz,"w");

    FILE *soutxz; char namexz[50];
    sprintf(namexz,"./stress_xz.out");           
    soutxz = fopen(namexz,"w");*/

  double ssave;

  SS[0]/=output_stress;SS[1]/=output_stress;SS[2]/=output_stress;
  SS[3]/=output_stress;SS[4]/=output_stress;SS[5]/=output_stress;
  SS[6]/=output_stress;SS[7]/=output_stress;SS[8]/=output_stress;


  ssave = (SS[0]+SS[4]+SS[8])/3.0;

  SS[0]-=ssave;
  SS[4]-=ssave;
  SS[8]-=ssave;


  SSB[0]/=output_stress;SSB[1]/=output_stress;SSB[2]/=output_stress;
  SSB[3]/=output_stress;SSB[4]/=output_stress;SSB[5]/=output_stress;
  SSB[6]/=output_stress;SSB[7]/=output_stress;SSB[8]/=output_stress;

  ssave = (SSB[0]+SSB[4]+SSB[8])/3.0;

  SSB[0]-=ssave;
  SSB[4]-=ssave;
  SSB[8]-=ssave;

  SSNB[0]/=output_stress;SSNB[1]/=output_stress;SSNB[2]/=output_stress;
  SSNB[3]/=output_stress;SSNB[4]/=output_stress;SSNB[5]/=output_stress;
  SSNB[6]/=output_stress;SSNB[7]/=output_stress;SSNB[8]/=output_stress;

  ssave = (SSNB[0]+SSNB[4]+SSNB[8])/3.0;

  SSNB[0]-=ssave;
  SSNB[4]-=ssave;
  SSNB[8]-=ssave;

  SS_f[0]/=output_stress;SS_f[1]/=output_stress;SS_f[2]/=output_stress;
  SS_f[3]/=output_stress;SS_f[4]/=output_stress;SS_f[5]/=output_stress;
  SS_f[6]/=output_stress;SS_f[7]/=output_stress;SS_f[8]/=output_stress;

  ssave = (SS_f[0]+SS_f[4]+SS_f[8])/3.0;

  SS_f[0]-=ssave;
  SS_f[4]-=ssave;
  SS_f[8]-=ssave;

  SSHC[0]/=output_stress;SSHC[1]/=output_stress;SSHC[2]/=output_stress;
  SSHC[3]/=output_stress;SSHC[4]/=output_stress;SSHC[5]/=output_stress;
  SSHC[6]/=output_stress;SSHC[7]/=output_stress;SSHC[8]/=output_stress;

  ssave = (SSHC[0]+SSHC[4]+SSHC[8])/3.0;

  SSHC[0]-=ssave;
  SSHC[4]-=ssave;
  SSHC[8]-=ssave;

  /*cout<<"Total Stress Tensor"<<endl; 
    cout<<SS[0][0]<<"  "<<SS[0][1]<<"   "<<SS[0][2]<<endl;
    cout<<SS[1][0]<<"  "<<SS[1][1]<<"   "<<SS[1][2]<<endl;
    cout<<SS[2][0]<<"  "<<SS[2][1]<<"   "<<SS[2][2]<<endl;
    cout<<"Bonded Stress Tensor"<<endl; 
    cout<<SSB[0][0]<<"  "<<SSB[0][1]<<"   "<<SSB[0][2]<<endl;
    cout<<SSB[1][0]<<"  "<<SSB[1][1]<<"   "<<SSB[1][2]<<endl;
    cout<<SSB[2][0]<<"  "<<SSB[2][1]<<"   "<<SSB[2][2]<<endl;
    cout<<"Nonbonded Stress Tensor"<<endl; 
    cout<<SSNB[0][0]<<"  "<<SSNB[0][1]<<"   "<<SSNB[0][2]<<endl;
    cout<<SSNB[1][0]<<"  "<<SSNB[1][1]<<"   "<<SSNB[1][2]<<endl;
    cout<<SSNB[2][0]<<"  "<<SSNB[2][1]<<"   "<<SSNB[2][2]<<endl;

    cout<<"Hardcore exclusion Stress Tensor"<<endl; 
    cout<<SSHC[0][0]<<"  "<<SSHC[0][1]<<"   "<<SSHC[0][2]<<endl;
    cout<<SSHC[1][0]<<"  "<<SSHC[1][1]<<"   "<<SSHC[1][2]<<endl;
    cout<<SSHC[2][0]<<"  "<<SSHC[2][1]<<"   "<<SSHC[2][2]<<endl;*/


  //fprintf(sout6,"%lf\t%lf\t%lf\n",SS[0],SS[1],SS[2]);	
  //fprintf(sout6,"%lf\t%lf\t%lf\n",SS[3],SS[4],SS[5]);	
  //fprintf(sout6,"%lf\t%lf\t%lf\n\n",SS[6],SS[7],SS[8]);

  fprintf(sout7,"%lf\t%lf\t%lf\t%lf\t%lf\t%lf\n",cur_t,-SSB[2]+SSNB[2]-SS_f[2]+SSHC[2],-SSB[2],SSNB[2],-SS_f[2],SSHC[2]);

  fclose(sout7);
  //double gridcell = box.length3/zntotal;


  //fclose(sout6);
  //fclose(soutzz);
  //fclose(soutyy);
  //fclose(soutxx);
  //fclose(soutxy);
  //fclose(soutxz);
  //fclose(soutyz);

  SS[0]=0.0;SS[1]=0.0;SS[2]=0.0;
  SS[3]=0.0;SS[4]=0.0;SS[5]=0.0;
  SS[6]=0.0;SS[7]=0.0;SS[8]=0.0;

  SSB[0]=0.0;SSB[1]=0.0;SSB[2]=0.0;
  SSB[3]=0.0;SSB[4]=0.0;SSB[5]=0.0;
  SSB[6]=0.0;SSB[7]=0.0;SSB[8]=0.0;

  SSNB[0]=0.0;SSNB[1]=0.0;SSNB[2]=0.0;
  SSNB[3]=0.0;SSNB[4]=0.0;SSNB[5]=0.0;
  SSNB[6]=0.0;SSNB[7]=0.0;SSNB[8]=0.0;

  SSHC[0]=0.0;SSHC[1]=0.0;SSHC[2]=0.0;
  SSHC[3]=0.0;SSHC[4]=0.0;SSHC[5]=0.0;
  SSHC[6]=0.0;SSHC[7]=0.0;SSHC[8]=0.0;

  SS_f[0]=0.0;SS_f[1]=0.0;SS_f[2]=0.0;
  SS_f[3]=0.0;SS_f[4]=0.0;SS_f[5]=0.0;
  SS_f[6]=0.0;SS_f[7]=0.0;SS_f[8]=0.0;
}





void system_out(void){
  cout<<"system info:"<<endl;
  cout<<box.chains<<" chains"<<endl;
  cout<<box.natoms<<" Total Beads"<<"  NP weight "<<sim.Nano_weight<<endl;
  cout<<sim.Nano_num<<" Total NPs"<<endl;
  cout<<sim.Nano_bead<<" Bead discretization"<<endl;
  cout<<R_cut<<" Cut of radius"<<endl;;
  cout<<density<<" Density"<<endl;
  cout<<sim.dt<<" dt"<<endl;
  cout<<nor<<" 1/C"<<endl;
  cout<<L_1<<" L1"<<endl;
  cout<<L_2<<" L2"<<endl;
  cout<<L_3<<" L3"<<endl;
  cout<<box.natoms/box.vol<<" G0"<<endl;
  //cout<<sim.Nss<<" Slip Springs Per Chain"<<endl;
  //cout<<sim.gamma_ss<<" gamma for springs"<<endl;
}

inline double ss_en(double r2){
  return 1.5*r2/(sim.bsl*sim.bsl)*sim.str_ss;
}

__device__ inline double Cuda_ss_en(double r2){
  return (-0.5*SS_f_coeff)*r2;
}




/*void add_ss(){
  int target_end = ran_num_int(0,box.natoms-1);
  int other_end;
  double trial_en;
  double dx,dy,dz,dr2,mindr2=box.length1*box.length1;
  for(int i = 0;i<box.natoms;i++){
  if(i == target_end)continue;
  dx = atom[target_end].pos.x - atom[i].pos.x;
  dy = atom[target_end].pos.y - atom[i].pos.y;
  dz = atom[target_end].pos.z - atom[i].pos.z;
  dr2 = distance(dx,dy,dz);
  if(dr2<mindr2){
  mindr2 = dr2;
  trial_en = ss_en(dr2);
  other_end = i;
  }
  }

  double rand = ran_num_double(0,1);

  if(rand < box.natoms*sim.z/(sim.nss+1)*exp(-trial_en)){
  ss[sim.nss].end1 = target_end;
  ss[sim.nss].end2 = other_end;

  ss[sim.nss].en = trial_en;
  en.sl_ener += trial_en;
  sim.nss++;
  add += 1.0;
  }

  }


  void del_ss(){
  int target = ran_num_int(0,sim.nss-1);
  double del_en = ss[target].en;
  double rand = ran_num_double(0,1);

  if(rand < sim.nss/(box.natoms*sim.z)*exp(del_en)){
  ss[target] =  ss[sim.nss-1];
  sim.nss--;
  del += 1.0;
  en.sl_ener -= del_en;
  }
  }*/

/*void add_ss(){
  int ran_t = ran_num_int(0,2*box.chains-1);
  int target_end = ran_t/2*sim.N+(ran_t%2)*(sim.N-1);
  double dx,dy,dz,dr2;
  int n_0 = 0;
  int sslink[SA];
  int prop;
  double enslink[SA];
  double sum_ssen = 0.0;
  double zeff=exp(-log(sim.beta));
  if(atom[target_end].ss_tag == 1)return;
  double target_chain = target_end/sim.N;
  for(int i = 0;i<box.natoms;i++){
    if(i/sim.N == target_chain)continue;
    if(atom[i].ss_tag == 1)continue;
    dx = atom[target_end].pos.x - atom[i].pos.x;
    dy = atom[target_end].pos.y - atom[i].pos.y;
    dz = atom[target_end].pos.z - atom[i].pos.z;
    dr2 = distance(dx,dy,dz);
    if(dr2 < sim.rss*sim.rss){
      enslink[n_0] = ss_en(dr2);
      sslink[n_0] = i;
      sum_ssen += exp(-(enslink[n_0]));
      n_0++;
    }
  }
  double rand = ran_num_double(0,1.0);
  for(int i=0;i<n_0;i++){
    rand -= exp(-(enslink[i]))/sum_ssen;
    if(rand <= 0.0){
      prop = i;
      break;
    }
  }

  rand = ran_num_double(0,1);
  double factor = sim.N/sim.beta;
  if(rand < (box.chains*factor/(4.0*(sim.nss+1)))*zeff){
    ss[sim.nss].end1 = target_end;
    ss[sim.nss].end2 = sslink[prop];
    ss[sim.nss].en = enslink[prop];
    en.sl_ener += enslink[prop];
    atom[ss[sim.nss].end1].ss_tag = 1;
    atom[ss[sim.nss].end2].ss_tag = 1;
    sim.nss++;

  }

}*/

__global__ void Cuda_RWDSS(atoms *C_atom,double *pr,int *count,int t_end,int N,double c_dxle,double rss){
	int index = blockIdx.x * blockDim.x +threadIdx.x;
	int ichain,target_chain;
	double dx,dy,dz,dr2,ss_e;
	if(index < CudaNum_Bead){
		ichain = index/N;
		target_chain = t_end/N;
		if(ichain != target_chain && C_atom[index].ss_tag != 1){
			
    dx = C_atom[t_end].pos.x - C_atom[index].pos.x;
    dy = C_atom[t_end].pos.y - C_atom[index].pos.y;
    dz = C_atom[t_end].pos.z - C_atom[index].pos.z;
    dr2 = Cuda_Distance(dx,dy,dz,c_dxle);
    if(dr2 < rss*rss){
		ss_e = Cuda_ss_en(dr2);
		pr[index] = exp(-ss_e);
		count[index] = 1;
	}
    
		}
	}
}

/*void add_ss_cpu(){
  int ran_t = ran_num_int(0,2*box.chains-1);
  int target_end = ran_t/2*sim.N+(ran_t%2)*(sim.N-1);
  double dx,dy,dz,dr2;
  int n_0 = 0;
  int sslink[SA];
  int prop;
  double enslink[SA];
  double sum_ssen = 0.0;
  double zeff=exp(-log(sim.beta));
  if(atom[target_end].ss_tag == 1)return;
  double target_chain = target_end/sim.N;
  for(int i = 0;i<box.natoms;i++){
    if(i/sim.N == target_chain)continue;
    if(atom[i].ss_tag == 1)continue;
    dx = atom[target_end].pos.x - atom[i].pos.x;
    dy = atom[target_end].pos.y - atom[i].pos.y;
    dz = atom[target_end].pos.z - atom[i].pos.z;
    dr2 = distance(dx,dy,dz);
    if(dr2 < sim.rss*sim.rss){
      enslink[n_0] = ss_en(dr2);
      sslink[n_0] = i;
      sum_ssen += exp(-(enslink[n_0]));
      n_0++;
    }
  }
  double rand = ran_num_float(0,1.0);
  for(int i=0;i<n_0;i++){
    rand -= exp(-(enslink[i]))/sum_ssen;
    if(rand <= 0.0){
      prop = i;
      break;
    }
  }

  rand = ran_num_float(0,1);
  double factor = sim.N/sim.beta;
  if(rand < (box.chains*factor/(4.0*(sim.nss+1)))*zeff){
    ss[sim.nss].end1 = target_end;
    ss[sim.nss].end2 = sslink[prop];
    ss[sim.nss].en = enslink[prop];
    en.sl_ener += enslink[prop];
    atom[ss[sim.nss].end1].ss_tag = 1;
    atom[ss[sim.nss].end2].ss_tag = 1;
    sim.nss++;
    
  }
	
}*/

void add_ss(){
  int ran_t = ran_num_int(0,2*box.chains-1);
  int target_end = ran_t/2*sim.N+(ran_t%2)*(sim.N-1);
  int *c_count,*ct;
  double *c_pr,*Pr;
  int prop = -1;
  double sum_ssen = 0.0,check_sum = 0.0;
  double zeff=1.0/sim.beta;//exp(-log(sim.beta));
  if(atom[target_end].ss_tag == 1)return;
  
  
  
  Pr = (double *)calloc(box.natoms,sizeof(double));
  ct = (int *)calloc(box.natoms,sizeof(int));

  cudaMalloc((void **)&c_pr, box.natoms*sizeof(double));
  cudaMalloc((void **)&c_count, box.natoms*sizeof(int));
  
  cudaMemcpy(c_pr, Pr, box.natoms*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(c_count, ct, box.natoms*sizeof(int), cudaMemcpyHostToDevice);
  
  Cuda_RWDSS<<<box.natoms/BLOCK_SIZE,BLOCK_SIZE>>>(c_atom,c_pr,c_count,target_end,sim.N,dx_LE,sim.rss);
  
  
  cudaMemcpy(Pr, c_pr, box.natoms*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(ct, c_count, box.natoms*sizeof(int), cudaMemcpyDeviceToHost);

  double rand = ran_num_double(0,1.0);
  for(int i=0;i<box.natoms;i++)sum_ssen += Pr[i];

  
  if(sum_ssen < 1.0E-10)return;
  for(int i=0;i<box.natoms;i++){
	  
	  if(ct[i] == 1){
    rand -= Pr[i]/sum_ssen;
    if(rand <= 0.0){
      prop = i;
      break;
    }
     }else check_sum+=Pr[i];
     if(check_sum > 1E-9)cout<<check_sum<<" something wrong!"<<endl;
  }

  

  rand = ran_num_double(0,1);
  double factor = sim.N/sim.beta;
  RWD.push_back(sum_ssen);
  if(rand < (box.chains*factor/(4.0*(sim.nss+1)))*zeff*sum_ssen){
    ss[sim.nss].end1 = target_end;
    ss[sim.nss].end2 = prop;
    ss[sim.nss].begin = exstep;
    ss[sim.nss].en = -log(Pr[prop]);
    en.sl_ener += ss[sim.nss].en;
    atom[ss[sim.nss].end1].ss_tag = 1;
    atom[ss[sim.nss].end2].ss_tag = 1;
    sim.nss++;
  }
  
  free(Pr);
  free(ct);
  cudaFree(c_pr);
  cudaFree(c_count);
}

void ini_ss(){
  int target_end = ran_num_int(0,box.natoms-1);
  double dx,dy,dz,dr2;
  int n_0 = 0;
  int sslink[SA];
  int prop;
  double enslink[SA];
  double sum_ssen = 0.0;
  double target_chain = target_end/sim.N;
  if(atom[target_end].ss_tag == 1)return;
  for(int i = 0;i<box.natoms;i++){
    if(i/sim.N == target_chain)continue;
    if(atom[i].ss_tag == 1)continue;
    dx = atom[target_end].pos.x - atom[i].pos.x;
    dy = atom[target_end].pos.y - atom[i].pos.y;
    dz = atom[target_end].pos.z - atom[i].pos.z;
    dr2 = distance(dx,dy,dz);
    if(dr2 < sim.rss*sim.rss){
      enslink[n_0] = ss_en(dr2);
      sslink[n_0] = i;
      sum_ssen += exp(-(enslink[n_0]));
      n_0++;
    }
  }
  double rand = ran_num_double(0,1);
  for(int i=0;i<n_0;i++){
    rand -= exp(-(enslink[i]))/sum_ssen;
    if(rand <= 0){
      prop = i;
      break;
    }
  }

  ss[sim.nss].end1 = target_end;
  ss[sim.nss].end2 = sslink[prop];
  ss[sim.nss].en = enslink[prop];
  ss[sim.nss].begin = exstep;
  en.sl_ener += enslink[prop];
  atom[ss[sim.nss].end1].ss_tag = 1;
  atom[ss[sim.nss].end2].ss_tag = 1;
  sim.nss++;

}

double Rosenbluth_Del(int target,int jend){
  int *c_count,*ct;
  double *c_pr,*Pr;
  double sum_ssen = 0.0;
  int target_end;
  if(jend == 1)target_end = ss[target].end1;
  else target_end = ss[target].end2;
  
  Pr = (double *)calloc(box.natoms,sizeof(double));
  ct = (int *)calloc(box.natoms,sizeof(int));

  cudaMalloc((void **)&c_pr, box.natoms*sizeof(double));
  cudaMalloc((void **)&c_count, box.natoms*sizeof(int));
  
  cudaMemcpy(c_pr, Pr, box.natoms*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(c_count, ct, box.natoms*sizeof(int), cudaMemcpyHostToDevice);
  
  Cuda_RWDSS<<<box.natoms/BLOCK_SIZE,BLOCK_SIZE>>>(c_atom,c_pr,c_count,target_end,sim.N,dx_LE,sim.rss);
  
  
  cudaMemcpy(Pr, c_pr, box.natoms*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(ct, c_count, box.natoms*sizeof(int), cudaMemcpyDeviceToHost);

  double rand = ran_num_double(0,1.0);
  for(int i=0;i<box.natoms;i++)sum_ssen += Pr[i];
  sum_ssen += exp(-ss[target].en);
  free(Pr);
  free(ct);
  cudaFree(c_pr);
  cudaFree(c_count);
  return sum_ssen;
}
void del_ss(int target,int jend,double &en_change){
  double pp;
  double e;
  double factor = sim.N/sim.beta;
  int ibin;
  double zeff=1.0/sim.beta;
  double dx,dy,dz,dr2;
  dx = atom[ss[target].end1].pos.x - atom[ss[target].end2].pos.x;
  dy = atom[ss[target].end1].pos.y - atom[ss[target].end2].pos.y;
  dz = atom[ss[target].end1].pos.z - atom[ss[target].end2].pos.z;
  dr2 = distance(dx,dy,dz);

  if(dr2 < sim.rss*sim.rss){

  double Rwd = Rosenbluth_Del(target,jend);
  RWD.push_back(Rwd);
  
  pp = (4.0*sim.nss/(box.chains*factor))/(zeff*Rwd); 	
  e = ran_num_double(0,1);
  if(e < pp){
	ibin = (int)((exstep - ss[target].begin)/freq);
	if(ibin < CHECK_SS)ss_life[ibin]++;
    en_change = -1.0*ss[target].en;
    atom[ss[target].end1].ss_tag = 0;
    atom[ss[target].end2].ss_tag = 0;
    ss[target] =  ss[sim.nss-1];
    sim.nss--;
  }else en_change = 0.0;
  }else en_change = 0.0;
}

void ss_displace(){
  double change_sl_ener = 0;
  int target = ran_num_int(0,sim.nss-1);
  double rand_1 = ran_num_double(0,1);
  if(rand_1 < 0.5){                //each end has an equal possibility of being chosen
    double rand_2 = ran_num_double(0,1);
    if(rand_2 < 0.5){             //go left 
      if((ss[target].end1 %sim.N) == 0){
	del_ss(target,1,change_sl_ener);
      }else{
	if(atom[ss[target].end1-1].ss_tag == 0){		  
	  double dx = atom[ss[target].end2].pos.x - atom[ss[target].end1-1].pos.x;
	  double dy = atom[ss[target].end2].pos.y - atom[ss[target].end1-1].pos.y;
	  double dz = atom[ss[target].end2].pos.z - atom[ss[target].end1-1].pos.z;
	  double dr2 = distance(dx,dy,dz);
	  double old_en = ss[target].en;
	  double new_en = ss_en(dr2);
	  double rand_3 = ran_num_double(0,1); //accept or reject
	  if(rand_3<exp(-1.0*(new_en-old_en))){
	    atom[ss[target].end1-1].ss_tag = 1;
	    atom[ss[target].end1].ss_tag = 0;
	    ss[target].end1 = ss[target].end1-1;
	    ss[target].en = new_en;
	    change_sl_ener += (new_en-old_en); 
	  }
	}
      }
    }else{                //go right
      if((ss[target].end1+1) %sim.N == 0){
	del_ss(target,1,change_sl_ener);
      }else{
	if(atom[ss[target].end1+1].ss_tag == 0){
	  double dx = atom[ss[target].end2].pos.x - atom[ss[target].end1+1].pos.x;
	  double dy = atom[ss[target].end2].pos.y - atom[ss[target].end1+1].pos.y;
	  double dz = atom[ss[target].end2].pos.z - atom[ss[target].end1+1].pos.z;
	  double dr2 = distance(dx,dy,dz);
	  double old_en = ss[target].en;
	  double new_en = ss_en(dr2);
	  double rand_3 = ran_num_double(0,1); //accept or reject
	  if(rand_3<exp(-1.0*(new_en-old_en))){
	    atom[ss[target].end1+1].ss_tag = 1;
	    atom[ss[target].end1].ss_tag = 0;
	    ss[target].end1 = ss[target].end1+1;
	    ss[target].en = new_en;
	    change_sl_ener += (new_en-old_en);
	  }
	}
      }
    }
  }else{
    double rand_2 = ran_num_double(0,1);
    if(rand_2 < 0.5){                //go left 
      if((ss[target].end2 %sim.N) == 0){
	del_ss(target,2,change_sl_ener);
      }else{
	if(atom[ss[target].end2-1].ss_tag == 0){
	  double dx = atom[ss[target].end1].pos.x - atom[ss[target].end2-1].pos.x;
	  double dy = atom[ss[target].end1].pos.y - atom[ss[target].end2-1].pos.y;
	  double dz = atom[ss[target].end1].pos.z - atom[ss[target].end2-1].pos.z;
	  double dr2 = distance(dx,dy,dz);
	  double old_en = ss[target].en;
	  double new_en = ss_en(dr2);
	  double rand_3 = ran_num_double(0,1); //accept or reject
	  if(rand_3<exp(-1.0*(new_en-old_en))){
	    atom[ss[target].end2-1].ss_tag = 1;
	    atom[ss[target].end2].ss_tag = 0;
	    ss[target].end2 = ss[target].end2-1;
	    ss[target].en = new_en;
	    change_sl_ener += (new_en-old_en);
	  }
	}
      }

    }else{                //go right
      if((ss[target].end2+1) %sim.N == 0){
	del_ss(target,2,change_sl_ener);
      }else{
	if(atom[ss[target].end2+1].ss_tag == 0){
	  double dx = atom[ss[target].end1].pos.x - atom[ss[target].end2+1].pos.x;
	  double dy = atom[ss[target].end1].pos.y - atom[ss[target].end2+1].pos.y;
	  double dz = atom[ss[target].end1].pos.z - atom[ss[target].end2+1].pos.z;
	  double dr2 = distance(dx,dy,dz);
	  double old_en = ss[target].en;
	  double new_en = ss_en(dr2);
	  double rand_3 = ran_num_double(0,1); //accept or reject
	  if(rand_3<exp(-1.0*(new_en-old_en))){
	    atom[ss[target].end2+1].ss_tag = 1;
	    atom[ss[target].end2].ss_tag = 0;
	    ss[target].end2 = ss[target].end2+1;
	    ss[target].en = new_en;
	    change_sl_ener += (new_en-old_en);
	  }
	}
      }
    }
  }




  en.sl_ener += change_sl_ener;
  en.total += change_sl_ener;
}


void ss_evolve(){
	//double c1,c2,c3,c4;
	//double t1=0.0;
	//double t2=0.0;
  double factor = sim.N/sim.beta;
  double Pss = factor/(2.0 + factor) ;
  int ss_step = (sim.nss+2.0*box.chains);
  if(SSlink == true){
    for(int i=0;i<ss_step;i++){
      double ran=ran_num_double(0,1);
      if(ran<Pss){
		  //c1 = clock();

		  if(sim.nss > 0)ss_displace();
		  	
		  //c2 = clock();
		  //t1+=c2-c1;
      }else {
		  //c3 = clock();
	  add_ss();
	  //c4 = clock();
	  //t2+=c4-c3;
	  
	}
      }
    }
    
    //cout<<"displace time"<<t1/(double) CLOCKS_PER_SEC<<endl;
    //cout<<"add ss time"<<t2/(double) CLOCKS_PER_SEC<<endl;
}




double mod(double div,double base){
  while(div<0){
    div += base;
  }

  while(div>=base){
    div -= base;
  }
  return div;
}

__device__ inline int Cuda_Mod(int div,int base){
  while(div<0){
    div += base;
  }

  while(div>=base){
    div -= base;
  }
  return div;
}

double test_sum(){
  double sum = 0.0;
  for(int i =0;i<sim.nss;i++)sum+=ss[i].en;
  return sum;
}


void ss_count(){
  for(int i = 0;i<sim.nss;i++){
    dist_c[ss[i].end1%sim.N]++;
    dist_c[ss[i].end2%sim.N]++;
  }
  if(exstep % (100*freq) == 0){
    double sum_distc = 0;
    for(int i=0; i<sim.N; i++){
      sum_distc += dist_c[i];
    }
    FILE * io;
    io=fopen("./dist_ss.out","w");
    for(int i=0; i<sim.N; i++){
      fprintf(io,"%lf\t%lf\n  ",(double)i/(double)(sim.N),(double)dist_c[i]/(double)sum_distc*sim.N);
    }
    fclose(io);
  }
}


/*void end_add_ss(){
  int target_end = ran_num_int(0,box.chains-1)*sim.N;
  double rrand = ran_num_double(0,1);
  if(rrand < 0.5)target_end += (sim.N-1);
  int other_end;
  double trial_en;
  double dx,dy,dz,dr2,mindr2=box.length1*box.length1;
  for(int i = 0;i<box.natoms;i++){
  if(i == target_end)continue;
  dx = atom[target_end].pos.x - atom[i].pos.x;
  dy = atom[target_end].pos.y - atom[i].pos.y;
  dz = atom[target_end].pos.z - atom[i].pos.z;
  dr2 = distance(dx,dy,dz);
  if(dr2<mindr2){
  mindr2 = dr2;
  trial_en = ss_en(dr2);
  other_end = i;
  }
  }

  double rand = ran_num_double(0,1);

  if(rand < box.natoms*sim.z/(sim.nss+1)*exp(-trial_en)){
  ss[sim.nss].end1 = target_end;
  ss[sim.nss].end2 = other_end;

  ss[sim.nss].en = trial_en;
  en.sl_ener += trial_en;
  sim.nss++;
  add += 1.0;
  }

  }


  void end_del_ss(){

  int target = ran_num_int(0,sim.nss-1);
  double del_en = ss[target].en;
  double rand = ran_num_double(0,1);

  if(rand < sim.nss/(box.natoms*sim.z)*exp(del_en)){
  ss[target] =  ss[sim.nss-1];
  sim.nss--;
  del += 1.0;
  en.sl_ener -= del_en;
  }
  }*/
  
void Rg_ave(){
  if(exstep > 0){
	  tensor Rg_temp;
	  for(int i=0;i<box.chains;i++){
		  Rg_temp = Rg_tensor(i);
		  Rg.xx += Rg_temp.xx;
		  Rg.yy += Rg_temp.yy;
		  Rg.zz += Rg_temp.zz;
	  }
  if(exstep>0 && exstep % samp == 0){
	ofstream ofs ("Rg.out", ofstream::app);
	ofs <<cur_t<<" "<<Rg.xx/(box.chains*samp)<<" "<<Rg.yy/(box.chains*samp)<<" "<<Rg.zz/(box.chains*samp)<<" "<<(Rg.xx+Rg.yy+Rg.zz)/(box.chains*samp)<<endl;
	ofs.close();
	Rg.xx = 0;
	Rg.yy = 0;
	Rg.zz = 0;
  }
}
}

tensor Rg_tensor(int ichain){
	double dx,dy,dz;
	tensor rg;
	tensor dia_rg;
	for(int i = 0;i < sim.N;i++)
	for(int j = 0;j < sim.N;j++){
		dx=atom[ichain*sim.N+i].pos.x-atom[ichain*sim.N+j].pos.x;
	    dy=atom[ichain*sim.N+i].pos.y-atom[ichain*sim.N+j].pos.y;
	    dz=atom[ichain*sim.N+i].pos.z-atom[ichain*sim.N+j].pos.z;
	    distance(dx,dy,dz);
	    rg.xx += dx * dx ;
	    rg.yy += dy * dy ;
	    rg.zz += dz * dz ;
	    rg.xy += dx * dy ;
	    rg.yz += dy * dz ;
	    rg.xz += dx * dz ;
	}
	rg.xx /= (2.0*sim.N*sim.N);
	rg.yy /= (2.0*sim.N*sim.N);
	rg.zz /= (2.0*sim.N*sim.N);
	rg.xy /= (2.0*sim.N*sim.N);
	rg.yz /= (2.0*sim.N*sim.N);
	rg.xz /= (2.0*sim.N*sim.N);
	dia_rg = diag_tensor(rg);
	

	return dia_rg;
}

void inline swap(double &a,double &b){
	double temp;
	temp = b;
	b = a;
	a = temp;
}


tensor diag_tensor(tensor rg){
	double p,q,d;
	double b;
	tensor dia_rg;
	double e1,e2,e3;
	b = rg.xx + rg.yy + rg.zz;
	p = 0.5 *((rg.xx - rg.yy)*(rg.xx - rg.yy) + (rg.xx - rg.zz)*(rg.xx - rg.zz) + (rg.yy - rg.zz)*(rg.yy - rg.zz)) + 3.0 *(rg.xy * rg.xy + rg.xz * rg.xz + rg.yz * rg.yz);
	q = 18.0 * (rg.xx * rg.yy * rg.zz + 3.0 * rg.xy * rg.yz * rg.xz) + 2.0 *(rg.xx * rg.xx * rg.xx + rg.yy * rg.yy * rg.yy + rg.zz* rg.zz * rg.zz) + 9.0 * (rg.xx + rg.yy + rg.zz) * (rg.xy*rg.xy + rg.xz*rg.xz + rg.yz*rg.yz) - 3.0 * (rg.xx + rg.yy)*(rg.xx + rg.zz)*(rg.yy + rg.zz) - 27.0*(rg.xx * rg.yz * rg.yz + rg.yy *rg.xz *rg.xz + rg.zz*rg.xy*rg.xy);
	d = acos(q/(2.0*sqrt(p*p*p)));
	e1 = 1/3.0*(b + 2.0*sqrt(p)*cos(d/3.0));
	e2 = 1/3.0*(b + 2.0*sqrt(p)*cos((d + 2*PI)/3.0));
	e3 = 1/3.0*(b + 2.0*sqrt(p)*cos((d - 2*PI)/3.0));
	if(e1 < e2)swap(e1,e2);
	if(e2 < e3)swap(e2,e3);
	if(e1 < e2)swap(e1,e2);
	dia_rg.xx = e1;
	dia_rg.yy = e2;
	dia_rg.zz = e3;
	
	return dia_rg;
}



void Re_time(){
  if(exstep > 0){
    double Re_dx,Re_dy,Re_dz;
    double dx,dy,dz;
    int bead1,bead2;
    for(int j=0;j<box.chains;j++){
      Re_dx=0.0;Re_dy=0.0;Re_dz=0.0;
      for(int i=0;i<sim.N-1;i++)
	{
	  bead1=j*sim.N+i;
	  bead2=bead1+1;
	  dx=atom[bead2].pos.x-atom[bead1].pos.x;
	  dy=atom[bead2].pos.y-atom[bead1].pos.y;
	  dz=atom[bead2].pos.z-atom[bead1].pos.z;
	  distance(dx,dy,dz);
	  Re_dx+=dx;
	  Re_dy+=dy;
	  Re_dz+=dz;
	}
      Re[j]+=sqrt(Re_dx*Re_dx+Re_dy*Re_dy+Re_dz*Re_dz);
    }
  }
  if(exstep>0 && exstep % samp == 0){
    double Re_avg = 0.0;
    for(int i=0;i<box.chains;i++){
      Re_avg+=Re[i]/samp;
      Re[i]=0.0;
    }
    FILE *handle;
    handle=fopen("./Re.out","a");
    fprintf(handle,"\n%7.5lf",Re_avg/box.chains);
    fclose(handle);
  }
}
void nano_traj()
{
  FILE *io;
  io = fopen("nano_trajectory.out","a");
  fprintf(io,"%15.6lf\t%15.6lf\t%15.6lf\t%15.6lf\n",cur_t,nano_v[0].x,nano_v[0].y,nano_v[0].z);
  fclose(io);
}

void poly_traj()
{
  FILE *io;
  io = fopen("polyA_trajectory.out","a");
  fprintf(io,"%15.6lf\t%15.6lf\t%15.6lf\t%15.6lf\n",cur_t,atom_v[0].x,atom_v[0].y,atom_v[0].z);
  fclose(io);
  io = fopen("poly_mid_trajectory.out","a");
  fprintf(io,"%15.6lf\t%15.6lf\t%15.6lf\t%15.6lf\n",cur_t,atom_v[sim.N/2].x,atom_v[sim.N/2].y,atom_v[sim.N/2].z);
  fclose(io);
  if(sim.N1 > 0){
    io = fopen("polyB_trajectory.out","a");
    fprintf(io,"%15.6lf\t%15.6lf\t%15.6lf\t%15.6lf\n",cur_t,atom_v[sim.N-1].x,atom_v[sim.N-1].y,atom_v[sim.N-1].z);
    fclose(io);
  }
}


void ss_dis(int sw){
  if(sw==0){
    delt_ss = 5.0*sim.bb/Nbin;
    for(int i=0;i<Nbin;i++)
      {
	hist_ss[i]=0;
      }
  }else if(sw==1){
    int ibins;
    double dx,dy,dz;
    double dr;
    for(int i=0;i < sim.nss;i++){
      dx = atom[ss[i].end1].pos.x - atom[ss[i].end2].pos.x;
      dy = atom[ss[i].end1].pos.y - atom[ss[i].end2].pos.y;
      dz = atom[ss[i].end1].pos.z - atom[ss[i].end2].pos.z;
      dr = distance(dx,dy,dz);
      dr = sqrt(dr);
      ibins=(int)(dr/delt_ss);
      if(ibins<Nbin)
	{
	  hist_ss[ibins]+=1;
	}
    }
  }else{
    FILE *io;
    double sum = 0.0;


    for(int i=0;i<Nbin;i++)
      {
	sum += hist_ss[i];
      }

    io=fopen("./ss_dist.out","w");
    for(int i=0; i<Nbin; i++)fprintf(io,"%lf\t%lf\n",(i+0.5)*delt_ss,hist_ss[i]/(sum*delt_ss)
				     );
    fclose(io);

  }
}

void slip_spring_check(void){
	//calculate Z distribution
	int ichain,sum_Z = 0,sum_N  = 0,sum_Q  = 0,n_count;
	int *Z_dist;
	Z_dist=(int *)calloc(box.chains,sizeof(int));
	
	for(int i=0;i <sim.nss;i++){
		ichain = ss[i].end1/sim.N;
		Z_dist[ichain]++;
		ichain = ss[i].end2/sim.N;
		Z_dist[ichain]++;
	}
	for(int i=0;i < box.chains;i++){
		Z_dump[Z_dist[i]]++;
	}
	for(int i=0;i < sim.N+1;i++){
		//cout <<Z_dump[i]<<endl;
		sum_Z += Z_dump[i];
	}
	ofstream ofs ("Z_dist.out", ofstream::out);
	for(int i=0;i < sim.N+1;i++){
		ofs <<i<<" "<<(double)Z_dump[i]/sum_Z<<endl;
	}
	ofs.close();

	free(Z_dist);
	vect a_pos;
	vect b_pos;
	double dx,dy,dz,dr;
	int ibin;
	//calculate N distribution
	for(int i=0;i<box.chains;i++){
	n_count = 0;
	int q_count = 0;
	for(int j=0;j<sim.N;j++){
		if(atom[i*sim.N+j].ss_tag == 1){
			if(q_count == 0){
				a_pos = atom[i*sim.N+j].pos;
			}else{
				b_pos = atom[i*sim.N+j].pos;
				dx = b_pos.x - a_pos.x;
				dy = b_pos.y - a_pos.y;
				dz = b_pos.z - a_pos.z;
				dr = distance(dx,dy,dz);
				ibin = (int)(sqrt(dr)/delt_ss);
				if(ibin < Nbin)hist_Q[ibin]++;
				a_pos = b_pos;
			}
			N_dist[n_count]++;
			n_count = 0;
			q_count++;
		}else{
			n_count++;
		}
	}
    }
    for(int i=0;i < sim.N-1;i++){
		//cout <<Z_dump[i]<<endl;
		sum_N += N_dist[i];
	}
    ofs.open("N_dist.out", ofstream::out);
	for(int i=0;i < sim.N-1;i++){
		ofs <<i<<" "<<(double)N_dist[i]/sum_N<<endl;
	}
	ofs.close();
	for(int i=0;i < Nbin;i++){
		//cout <<Z_dump[i]<<endl;
		sum_Q += hist_Q[i];
	}
	
	ofs.open("Q_dist.out", ofstream::out);
	for(int i=0; i<Nbin; i++)ofs <<(i+0.5)*delt_ss<<" "<<hist_Q[i]/(sum_Q*delt_ss)<<endl;
	ofs.close();
	

}
/*void compensate_forces(){
  double compensate_E=0;
  int vdxx = (int)(dx_LE/del_L1);
  double V=box.length1*box.length2*box.length3;
  wxx=0;
  wyy=0;
  wzz=0;
  wyx=0;
  wzy=0;
  wzx=0;
  int link_1,link_2;                    //the link_2 is for the neighboring cell
  int ind,ind_l;
  int m1,n1,l1;
  #pragma omp parallel for default(none) shared(cp_L_1,cp_L_2,cp_L_3,LeesEd,cp_head,cp_lscl,vdxx,force_nano,force_bead,stress_cpxx,stress_xx,stress_cpyy,stress_yy,stress_cpzz,stress_zz,stress_cpyz,stress_yz,stress_cpxy,stress_xy,stress_cpxz,stress_xz) private(m1,n1,l1,link_1,link_2,ss_cp,ind_l,ind) reduction(+:wxx) reduction(+:wyy) reduction(+:wzz)reduction(+:wyx) reduction(+:wzy) reduction(+:wzx)reduction(+:compensate_E)
  for(int i=0;i<cp_L_1;i++)
  for(int j=0;j<cp_L_2;j++)
  for(int k=0;k<cp_L_3;k++)
  {
  if(LeesEd == true){
  if(k==0){
  ind=i*cp_L_2*cp_L_3+j*cp_L_3+k;
  for(int m=i-1;m<=i+1;m++)
  for(int n=j-1;n<=j+1;n++)
  for(int l=k;l<=k+1;l++)
  {
  m1=mod(m,cp_L_1);
  n1=mod(n,cp_L_2);
  l1=l;
  ind_l=m1*cp_L_2*cp_L_3+n1*cp_L_3+l1;
  link_1=head[ind];
  while(link_1!=-1)
  {
  link_2=cp_head[ind_l];
  while(link_2!=-1){
  if(link_1<link_2){
  nonbonded_E += cpforce(link_1,link_2);
  }
  link_2=cp_lscl[link_2];
  }
  link_1=cp_lscl[link_1];
  }			
  }
  for(int m=-1;m<=2;m++)
  for(int n=j-1;n<=j+1;n++)
  {
  m1=mod(i+vdxx+m,cp_L_1);
  n1=mod(n,cp_L_2);
  l1=cp_L_3-1;
  ind_l=m1*cp_L_2*cp_L_3+n1*cp_L_3+l1;
  link_1=cp_head[ind];
  while(link_1!=-1)
  {
  link_2=cp_head[ind_l];
  while(link_2!=-1){
  if(link_1<link_2){
  compensate_E += cpforce(link_1,link_2);
  }
  link_2=cp_lscl[link_2];
  }
  link_1=cp_lscl[link_1];
  }			
  }

  }else if (k == L_3-1){
  ind=i*cp_L_2*cp_L_3+j*cp_L_3+k;
  for(int m=i-1;m<=i+1;m++)
  for(int n=j-1;n<=j+1;n++)
  for(int l=k-1;l<=k;l++)
  {
  m1=mod(m,cp_L_1);
  n1=mod(n,cp_L_2);
  l1=l;
  ind_l=m1*cp_L_2*cp_L_3+n1*cp_L_3+l1;
  link_1=cp_head[ind];
  while(link_1!=-1)
  {
  link_2=cp_head[ind_l];
  while(link_2!=-1){
  if(link_1<link_2){
  compensate_E += cpforce(link_1,link_2);
  }
  link_2=cp_lscl[link_2];
  }
  link_1=cp_lscl[link_1];
  }			
  }
  for(int m=-2;m<=1;m++)
  for(int n=j-1;n<=j+1;n++)
  {
  m1=mod(i-vdxx+m,cp_L_1);
  n1=mod(n,cp_L_2);
  l1=0;
  ind_l=m1*cp_L_2*cp_L_3+n1*cp_L_3+l1;
  link_1=cp_head[ind];
  while(link_1!=-1)
  {
  link_2=cp_head[ind_l];
  while(link_2!=-1){
  if(link_1<link_2){
  nonbonded_E += cpforce(link_1,link_2);
  }
  link_2=cp_lscl[link_2];
  }
  link_1=cp_lscl[link_1];
  }			
  }
  }else{
  ind=i*cp_L_2*cp_L_3+j*cp_L_3+k;
  for(int m=i-1;m<=i+1;m++)
  for(int n=j-1;n<=j+1;n++)
  for(int l=k-1;l<=k+1;l++)
  {
  m1=mod(m,cp_L_1);
  n1=mod(n,cp_L_2);
  l1=mod(l,cp_L_3);
  ind_l=m1*cp_L_2*cp_L_3+n1*cp_L_3+l1;
  link_1=cp_head[ind];
  while(link_1!=-1)
  {
  link_2=cp_head[ind_l];
  while(link_2!=-1){
  if(link_1<link_2){
  compensate_E += cpforce(link_1,link_2);
  }
  link_2=cp_lscl[link_2];
  }
  link_1=cp_lscl[link_1];
  }			
  }
  }
  }else{
  ind=i*cp_L_2*cp_L_3+j*cp_L_3+k;
  for(int m=i-1;m<=i+1;m++)
  for(int n=j-1;n<=j+1;n++)
  for(int l=k-1;l<=k+1;l++)
  {
  m1=mod(m,cp_L_1);
  n1=mod(n,cp_L_2);
  l1=mod(l,cp_L_3);
  ind_l=m1*cp_L_2*cp_L_3+n1*cp_L_3+l1;
  link_1=cp_head[ind];
  while(link_1!=-1)
  {
  link_2=cp_head[ind_l];
  while(link_2!=-1){
  if(link_1<link_2){
  compensate_E += cpforce(link_1,link_2);
  }
  link_2=cp_lscl[link_2];
  }
  link_1=cp_lscl[link_1];
  }			
  }
  }
  }

  en.compensate_E = compensate_E;
  if(STRESS==true){
  wxx /= V;
  wyx /= V;
  wyy /= V;
  wzx /= V;
  wzy /= V;
  wzz /= V;

  SS[0][0]+=wxx;SS[0][1]+=wyx;SS[0][2]+=wzx;
  SS[1][0]+=wyx;SS[1][1]+=wyy;SS[1][2]+=wzy;
  SS[2][0]+=wzx;SS[2][1]+=wzy;SS[2][2]+=wzz;

  SSCP[0][0]+=wxx;SSCP[0][1]+=wyx;SSCP[0][2]+=wzx;
  SSCP[1][0]+=wyx;SSCP[1][1]+=wyy;SSCP[1][2]+=wzy;
  SSCP[2][0]+=wzx;SSCP[2][1]+=wzy;SSCP[2][2]+=wzz;
  }
  }*/
