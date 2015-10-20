
#define l 401
#define m 401
#define TN 20  // thread number
#define coord 0  // 1:axisymmetric coordinate;  0:2D

#define PI 3.14159265358979323846264
double dx,dy,raxis[l];
double dt=1.0E-4;    // the least step time
long ioutput=1000;  //output file every ioutput*dt
long iteration=0,iterationnumber=200000;

long ch_iter,ch_iternum=30;  // for CahnHilliard equation iteration
long surf_iter,surf_iternum=50;
long mo_iter,mo_iternum=20;  // for momentum equation iteration
long div_iter,div_iternum=200; // divergence free iteration

double rho=988.0, miu=2E-3, a=2.5-3.0*0.5, g=9.8; //, sigma=65E-3
double rd, rv;  //ratio of density, ratio of viscosity
double Re,Oh,Ca,epn,Pe,Bo;
//double rad,xc,yc;
double rad,xc,yc;
double adv,rec,tadv,trec;  //Advancing contact angle, Receding contact angle
int bnd[4]={1,1,1,1};
double vel_bnd[4]={0.0,0.0,0.0,0.0};  // velocity boundary;
int gravity=0; double gravity_angle=90.0;  // 0: No gravity, 1: gravity;

double u2[l][m-1],v2[l-1][m],u3[l][m-1],v3[l-1][m],p[l+1][m+1];
double fi[l+1][m+1],fin[l+1][m+1],fio[l+1][m+1],source[l+1][m+1],curv[l+1][m+1];
double den[l+1][m+1],vis[l+1][m+1],wenox[l][m-1],wenoy[l-1][m],surfx[l][m-1]={0.0},surfy[l-1][m]={0.0};;
double surfactant[l+1][m+1],nor_vector_x[l+1][m+1],nor_vector_y[l+1][m+1];/* unit normal vector */
double CH_res,MOM_res,DIV_res,st,delta_mini;

int np; // number of particles
double surf_elas, surf_cove; /* surfactant elaicity, surfactant_coverage */
double Pe_surf; // Pe number for surface surfactant diffusion
double pdr;  //initial distance between two nearby particles
double pdr_min; //the preset minimum distance between two nearby particles
double pdr_max; //the preset maximum distance between two nearby particles

FILE *log_fp,*data_fp;
char shape_name[16]="shape_00000.dat";
char axis_name[18]="axissym_00000.dat";
char part_name[20]="part_00000.dat";

/********************************************************************/

/********************************************************************/
void parameter_initial(void)
{
	dx=4.0/(l-1);dy=dx;
	Oh=0.0053;Re=10;
	rd=1.0; rv=1.0;
	Ca=0.1; Bo=2.0/3.0;
	epn=0.5*dx;Pe=0.4/epn;
	adv=90.0;rec=adv;
	rad=1.0; xc=2.0; yc=2.0;

	Pe_surf=10.0; /* Pe for surfactant */
	surf_elas=0.5; surf_cove=0.3;
	delta_mini=0.01;

	st=-6.0*sqrt(2.0)/epn;
	tadv=tan((90-adv)*PI/180);
	trec=tan((90-rec)*PI/180);

	pdr=0.3*dx; pdr_min=0.5*pdr; pdr_max=1.6*pdr;
}

void log_initial(void)
{
/*	log_fp=fopen("log.dat","w");
	fprintf(log_fp,"Re=%6lf  Oh=%6lf \n dx=%6lf  epn=%6lf*dx  Pe=%6lf/epn \n",Re,Oh,dx,epn/dx,Pe*epn);
	fprintf(log_fp,"gravity=%d  adv=%6lf  rec=%6lf  dt=%8lf  ioutput=%ld \n",gravity,adv,rec,dt,ioutput);
	fprintf(log_fp,"boundary: left=%d  bottom=%d  right=%d  up=%d \n",bnd[0],bnd[1],bnd[2],bnd[3]);
	*/
}

void log_res(void)
{
/*	if((iteration%100)==0)
	{
		fprintf(data_fp,"%lf %lf %lf %lf \n",iteration*dt,shear_force(),interf_area_liquid_gas()/(Re*Re*Oh*Oh),kinetic());
	}
	*/
//	printf("%d \n",iteration);
}

char num_char(int a)
{
	switch(a)
	{
		case 0: return '0'; break;
		case 1: return '1'; break;
		case 2: return '2'; break;
		case 3: return '3'; break;
		case 4: return '4'; break;
		case 5: return '5'; break;
		case 6: return '6'; break;
		case 7: return '7'; break;
		case 8: return '8'; break;
		case 9: return '9'; break;
		default: return '0';break;
	}
}

void get_save_name(void)
{
	int i,j;
	char a[5];
	i=iteration/ioutput;
	a[0]=num_char(i/10000);
	a[1]=num_char((i%10000)/1000);
	a[2]=num_char((i%1000)/100);
	a[3]=num_char((i%100)/10);
	a[4]=num_char(i%10);
	//shape
	for(j=0;j<5;j++)
	{
		shape_name[6+j]=a[j];
		axis_name[8+j]=a[j];
		part_name[5+j]=a[j];
	}
}

//*********************************************************************//
void write_data2file(char *file_name, double x, double y)
{
	FILE *fp;
	fp=fopen(file_name,"a+");
	fprintf(fp,"%9.5lf %9.5lf \n",x,y);
	fclose(fp);
}

