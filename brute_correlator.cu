#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <omp.h>
#include <vector>
#include <ctime>
#include <iomanip> 
#define THRESHOLD 500000

using namespace std;
float Lx = 2.5,Ly = 2.5, Lz = 2.5;
vector<float> tim;
vector<float>	stress_total,stress_bond,stress_nb,stress_ss,stress_hc;
void correlate(string);
float calc(long int k,vector<float> a);


float	*xy_t,*xy_b,*xy_nb,*xy_ss,*xy_hc;
float	*xz_t,*xz_b,*xz_nb,*xz_ss,*xz_hc;
float	*yz_t,*yz_b,*yz_nb,*yz_ss,*yz_hc;
void import_data(string);
__global__ void Cuda_Correlate(float *,long int ,float **);


void import_data(string filename){
  ifstream ifs(filename.c_str(),std::ifstream::in);
  float data1,data2,data3,data4,data5,data6;
  if (ifs.fail() )exit(1);
  while (!ifs.eof()){
	  ifs>>data1;
	  ifs>>data2;
	  ifs>>data3;
	  ifs>>data4;
	  ifs>>data5;
	  ifs>>data6;
	  tim.push_back(data1);
	  stress_total.push_back(data2);
	  stress_bond.push_back(data3);
	  stress_ss.push_back(data4);
	  stress_nb.push_back(data5);
	  stress_hc.push_back(data6);
  }
  ifs.close();
}


long int size_convert(long int m){
	long int a,b;
	if(m > 10){
		m -= 2;
		a = m/9;
		b = m%9;
		return (b+2)*pow(10,a);
	}else return m;
}

__device__ inline long int cuda_pow(long int a,long int k){
	if(k == 0)return 1;
	for(long int i =0;i<k-1;i++)a*=a;
	return a;
}
__device__ long int Cuda_size_convert(long int m){
	long int a,b;
	if(m > 10){
		m -= 2;
		a = m/9;
		b = m%9;
		return (b+2)*cuda_pow(10,a);
	}else return m;
}

double convert_test(long int ind,vector<float> *aa,long int t_size){
	double value = 0.0;
		for(long int in_y = 0;in_y < t_size-ind;in_y++)
		value += aa->at(in_y) * aa->at(in_y+ind);
		value /= (t_size-ind);
	return value;
}
/*__device__ double Cuda_convert(long int ind,float *aa,long int t_size){
	double value = 0.0;
		for(long int in_y = 0;in_y < t_size-ind;in_y++)
		value += aa[in_y] * aa[in_y+ind];
		value /= (t_size-ind);
	return value;
}

__global__ void Cuda_Correlate(float *c_res,vector<float> *aa,long int act_size,long int t_size){
	long int index = blockIdx.x * blockDim.x +threadIdx.x;
	long int n_range;
	if(index < act_size){
		n_range = Cuda_size_convert(index);
		c_res[index] = Cuda_convert(n_range,aa,t_size);
		for(long int in_y = 0;in_y < t_size-n_range;in_y++)
		c_res[index] += aa[in_y] * aa[in_y+n_range];
		c_res[index] /= (t_size-n_range);
	}
}*/




void correlate(){
  import_data("Gxz_stress.dat");
  long int array_size = tim.size(),size = 0;
  long int k = 1;
  while(size_convert(k-1) < array_size-1)k++;
  size = k-1;
  
  //str = (float *)calloc(array_size,sizeof(float));
  //for(long int i=0;i < array_size;i++)str[i] = stress.at(i);
  
  xz_t =(float *)calloc(size,sizeof(float));
  xz_b =(float *)calloc(size,sizeof(float));
  xz_nb =(float *)calloc(size,sizeof(float));
  xz_ss =(float *)calloc(size,sizeof(float));
  xz_hc =(float *)calloc(size,sizeof(float));

  
  for(long int i = 0;i<size;i++){
	  xz_t[i] = convert_test(size_convert(i),&stress_total,array_size);
	  xz_b[i] = convert_test(size_convert(i),&stress_bond,array_size);
	  xz_nb[i] = convert_test(size_convert(i),&stress_nb,array_size);
	  xz_ss[i] = convert_test(size_convert(i),&stress_ss,array_size);
	  xz_hc[i] = convert_test(size_convert(i),&stress_hc,array_size);
  }
  
  	  tim.clear();
	  stress_total.clear();
	  stress_bond.clear();
	  stress_ss.clear();
	  stress_nb.clear();
	  stress_hc.clear();
  
  import_data("Gxy_stress.dat");
  array_size = tim.size();
  k = 1;
  while(size_convert(k-1) < array_size-1)k++;
  size = k-1;
  
  //str = (float *)calloc(array_size,sizeof(float));
  //for(long int i=0;i < array_size;i++)str[i] = stress.at(i);
  
  xy_t =(float *)calloc(size,sizeof(float));
  xy_b =(float *)calloc(size,sizeof(float));
  xy_nb =(float *)calloc(size,sizeof(float));
  xy_ss =(float *)calloc(size,sizeof(float));
  xy_hc =(float *)calloc(size,sizeof(float));

  
  for(long int i = 0;i<size;i++){
	  xy_t[i] = convert_test(size_convert(i),&stress_total,array_size);
	  xy_b[i] = convert_test(size_convert(i),&stress_bond,array_size);
	  xy_nb[i] = convert_test(size_convert(i),&stress_nb,array_size);
	  xy_ss[i] = convert_test(size_convert(i),&stress_ss,array_size);
	  xy_hc[i] = convert_test(size_convert(i),&stress_hc,array_size);
  }
  
    tim.clear();
	  stress_total.clear();
	  stress_bond.clear();
	  stress_ss.clear();
	  stress_nb.clear();
	  stress_hc.clear();
  
  import_data("Gyz_stress.dat");
  array_size = tim.size();
  k = 1;
  while(size_convert(k-1) < array_size-1)k++;
  size = k-1;
  
  //str = (float *)calloc(array_size,sizeof(float));
  //for(long int i=0;i < array_size;i++)str[i] = stress.at(i);
  
  yz_t =(float *)calloc(size,sizeof(float));
  yz_b =(float *)calloc(size,sizeof(float));
  yz_nb =(float *)calloc(size,sizeof(float));
  yz_ss =(float *)calloc(size,sizeof(float));
  yz_hc =(float *)calloc(size,sizeof(float));

  
  for(long int i = 0;i<size;i++){
	  yz_t[i] = convert_test(size_convert(i),&stress_total,array_size);
	  yz_b[i] = convert_test(size_convert(i),&stress_bond,array_size);
	  yz_nb[i] = convert_test(size_convert(i),&stress_nb,array_size);
	  yz_ss[i] = convert_test(size_convert(i),&stress_ss,array_size);
	  yz_hc[i] = convert_test(size_convert(i),&stress_hc,array_size);
  }
  
  ofstream ofs("Gt_total.out",ofstream::out);
   float vol = Lx * Ly * Lz;
      for(long int i = 0;i < size;i++){
	  ofs<< std::setprecision(10)<<tim.at(size_convert(i))<<" "<<vol*xy_t[i]<<" "<<vol*yz_t[i]<<" "<<vol*xz_t[i]<<" "<<vol*(xy_t[i]+yz_t[i]+xz_t[i])/3.0<<endl;
      }
     free(xy_t);
     free(yz_t);
     free(xz_t);
  ofs.close();
  ofs.open("Gt_bond.out",ofstream::out);
      for(long int i = 0;i < size;i++){
	  ofs<< std::setprecision(10)<<tim.at(size_convert(i))<<" "<<vol*xy_b[i]<<" "<<vol*yz_b[i]<<" "<<vol*xz_b[i]<<" "<<vol*(xy_b[i]+yz_b[i]+xz_b[i])/3.0<<endl;
      }
     free(xy_b);
     free(yz_b);
     free(xz_b);
  ofs.close();
  
  ofs.open("Gt_nonbond.out",ofstream::out);
      for(long int i = 0;i < size;i++){
	  ofs<< std::setprecision(10)<<tim.at(size_convert(i))<<" "<<vol*xy_nb[i]<<" "<<vol*yz_nb[i]<<" "<<vol*xz_nb[i]<<" "<<vol*(xy_nb[i]+yz_nb[i]+xz_nb[i])/3.0<<endl;
      }
     free(xy_nb);
     free(yz_nb);
     free(xz_nb);
  ofs.close();
  
  ofs.open("Gt_ss.out",ofstream::out);
      for(long int i = 0;i < size;i++){
	  ofs<< std::setprecision(10)<<tim.at(size_convert(i))<<" "<<vol*xy_ss[i]<<" "<<vol*yz_ss[i]<<" "<<vol*xz_ss[i]<<" "<<vol*(xy_ss[i]+yz_ss[i]+xz_ss[i])/3.0<<endl;
      }
     free(xy_ss);
     free(yz_ss);
     free(xz_ss);
  ofs.close();
  
  ofs.open("Gt_hc.out",ofstream::out);
      for(long int i = 0;i < size;i++){
	  ofs<< std::setprecision(10)<<tim.at(size_convert(i))<<" "<<vol*xy_hc[i]<<" "<<vol*yz_hc[i]<<" "<<vol*xz_hc[i]<<" "<<vol*(xy_hc[i]+yz_hc[i]+xz_hc[i])/3.0<<endl;
      }
     free(xy_hc);
     free(yz_hc);
     free(xz_hc);
  ofs.close();
}



int main(void){
	float t1 = clock();
	correlate();
	float t2= clock();
	cout<<"time "<<(t2-t1)/(double) CLOCKS_PER_SEC<<"sec"<<endl;
	return 0;

}
