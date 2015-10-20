/**********************************************************/
//              diffusing interface + particles
//        for two-phase flow with insoluble surfactants
//                       YUAN Chengjun
//                  (cjyuan@mail.ustc.edu.cn)
//                      July 19th 2014
/**********************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include "parameter_set.c"
#include "ch_govern.c"
#include "part_surf_govern.c"
#include "ns_govern.c"
/**********Sub-Fuction Definition start**************************/
void flow_initialization(void); void diffuse_initialization(void); void update_c_uv(void);
void save_axissym(void); void save_shape(void); void save_particle(void); void save_file(void);
/**********Sub-Fuction Definition END ***************************/
int main()
{
	parameter_initial();
	diffuse_initialization();
	flow_initialization();
	particle_surfactant_initialization();
	log_initial();
	save_file();
//	exit(0);
	for(iteration=1;iteration<=iterationnumber;iteration++)
	{
		NS();
		CH();
		particle_surfactant_transportation();
		update_c_uv();
		log_res();
		save_file();
//		exit(0);
//		write_data2file("mass_of_surfactant.dat",dt*iteration,sum_surfactant(surfactant_n[0],l+1,m+1,dx));
//		write_data2file("mass_of_surfactant_2.dat",dt*iteration,sum_surfactant_2(surfactant_n[0],l+1,m+1,dx));
//		printf("iteration=%ld \n",iteration);
	}
//	fclose(log_fp);
//	fclose(data_fp);
	exit(0);
}
/***********Sub Function*************************************/
void flow_initialization(void)
{
	int i,j;
	if(coord==1)
	{
		#pragma omp parallel for schedule(static) num_threads(TN)
		for(i=0;i<l;i++)raxis[i]=i*dx;
	}
	else if(coord==0)
	{
		#pragma omp parallel for schedule(static) num_threads(TN)
		for(i=0;i<l;i++)raxis[i]=1.0;
	}
	else {	printf("coord error!"); exit(0); }

	#pragma omp parallel for schedule(static) private(j) num_threads(TN)
    for(i=0;i<l;i++)
	for(j=0;j<m-1;j++)
		{u2[i][j]=(vel_bnd[3]-vel_bnd[1])*j/m+vel_bnd[1]; u3[i][j]=u2[i][j];}

	#pragma omp parallel for schedule(static) private(j) num_threads(TN)
    for(i=0;i<l-1;i++)
	for(j=0;j<m;j++)
		{
			if(fi[i+1][j]+fi[i+1][j+1]>=1.0)v2[i][j]=0.0;
			else v2[i][j]=0;
			v3[i][j]=v2[i][j];
		}

	#pragma omp parallel for schedule(static) private(j) num_threads(TN)
    for(i=0;i<l+1;i++)
	for(j=0;j<m+1;j++)
        p[i][j]=0;

	#pragma omp parallel for schedule(static) private(j) num_threads(TN)
	for(i=0;i<l;i++)
	for(j=0;j<m-1;j++)
		preU[i][j]=0.0;

	#pragma omp parallel for schedule(static) private(j) num_threads(TN)
	for(i=0;i<l-1;i++)
	for(j=0;j<m;j++)
		preV[i][j]=0.0;

}

void diffuse_initialization(void)
{
	int i,j;
	double x,y,temp1,temp2,temp;
	#pragma omp parallel for schedule(static) private(j,x,y,temp,temp1,temp2) num_threads(TN)
	for(i=0;i<l+1;i++)
	for(j=0;j<m+1;j++)
	{
		x=(i-0.5)*dx; y=(j-0.5)*dy;
		temp1=sqrt((x-xc1)*(x-xc1)+(y-yc1)*(y-yc1))-rad1;
		temp2=sqrt((x-xc2)*(x-xc2)+(y-yc2)*(y-yc2))-rad2;
		temp=temp1<temp2?temp1:temp2;
//		temp=-1.0*temp;
//		temp=0.1*cos(2.0*PI*x)+2-y;
//		temp=x-0.1;
		fi[i][j]=0.5*(1.0-tanh(temp/(2.0*1.41421356*epn)));
		fio[i][j]=fi[i][j];
		fin[i][j]=fi[i][j];
	}
}

void update_c_uv(void)
{
	int i,j;
	#pragma omp parallel for schedule(static) private(j) num_threads(TN)
	for(i=0;i<l+1;i++)
	for(j=0;j<m+1;j++)
	{ fio[i][j]=fi[i][j]; fi[i][j]=fin[i][j]; }

	//Advance the time from n level to n+1 level
	#pragma omp parallel for schedule(static) private(j) num_threads(TN)
	for(i=0;i<l;i++)
	for(j=0;j<m-1;j++)
		u2[i][j]=u3[i][j];

	#pragma omp parallel for schedule(static) private(j) num_threads(TN)
	for(i=0;i<l-1;i++)
	for(j=0;j<m;j++)
		v2[i][j]=v3[i][j];
}

void save_axissym(void)
{
	FILE *fp;
	int i,j;
	double x,y,u,v;
	fp=fopen(axis_name,"w");
	fprintf(fp," VARIABLES=\"X \",\"Y  \",\"U \",\"V \",\"P \" \n");
	fprintf(fp," ZONE T=\"Floor\", I=%d  J=%d  F=POINT \n",m-1,l-1);
	for(i=1;i<l;i++)
	{
		x=(i-0.5)*dx;
		for(j=1;j<m;j++)
		{
			y=(j-0.5)*dy;
			u=(u3[i-1][j-1]+u3[i][j-1])/2.0;
			v=(v3[i-1][j-1]+v3[i-1][j])/2.0;
			fprintf(fp,"%6lf %6lf %lf %lf %lf \n",x,y,u,v,p[i][j]);
		}
	}
	fclose(fp);
}

void save_shape(void)
{
	FILE *fp;
	int i,j;
	double x,y;
	fp=fopen(shape_name,"w");
	fprintf(fp," VARIABLES=\"X \",\"Y \",\"fi \",\"P \" \n");
	fprintf(fp," ZONE T=\"Floor\", I=%d  J=%d  F=POINT \n",m-1,l-1);
	for(i=1;i<l;i++)
	{
		x=(i-0.5)*dx;
		for(j=1;j<m;j++)
		{
			y=(j-0.5)*dy;
			fprintf(fp,"%6lf %6lf %5.3lf %5.3lf \n",x,y,fi[i][j],p[i][j]);
		}
	}
	fclose(fp);
}

void save_particle(void)
{
    FILE *fp; int i,next; double temp1,temp2,temp3,surf_sum=0.0;
    fp=fopen(part_name,"w");
	fprintf(fp," VARIABLES=\"xp \",\"yp \",\"f \",\"surf \" \n");
	fprintf(fp," ZONE F=POINT \n");

    cal_length2p();
	for(i=1;i<=np;i++)
    {
        if(i==np)next=1; else next=i+1;
        temp1=0.5*(xp[i]+xp[next]); temp2=0.5*(yp[i]+yp[next]);
        temp3=surf[i]/length2p[i];
        fprintf(fp,"%lf %lf %lf %lf \n",temp1,temp2,temp3,surf[i]);
        surf_sum=surf_sum+surf[i];
    }
    fclose(fp);
    printf("output %s ,surf_sum=%lf \n",part_name,surf_sum);
}

void save_file(void)
{
	int i,j; double a;
	i=iteration%ioutput;
	if(i==0)
	{
		get_save_name();
//		save_axissym();
		save_shape();
		save_particle();
//		printf("np=%d \n",np);
//		write_data2file("Delta_Integration.dat", dt*iteration, IntegrateDeltaFunction());
//		for(j=1;j<l;j++)
//		{
//			a=fi[j][(m-1)/2];
//			write_data2file("Delta_Distribution.dat", (j-0.5)*dx, 3.0*1.41421*a*a*(1.0-a)*(1.0-a)/epn);
//		}
//		print_2D_array("wenox",iteration,"wenox",wenox[0],l,m-1,dx);
//		print_2D_array("wenoy",iteration,"wenoy",wenoy[0],l-1,m,dx);
//		print_2D_array("source",iteration,"source",source[0],l+1,m+1,dx);
//		print_2D_array("Pre_conv_surfactant",iteration,"Pre_conv_surfactant",Pre_conv_surfactant[0],l+1,m+1,dx);
//		print_2D_array("d2surf",iteration,"d2surf",d2surf[0],l+1,m+1,dx);
	}
}


