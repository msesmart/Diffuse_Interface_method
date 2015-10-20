
double RHS[l+1][m+1],d2fi[l+1][m+1];
double bulkE[l+1][m+1],ConvPrev[l+1][m+1]={0.0};

void array_bnd(double a[][m+1])
{
	int i,j;

	if(bnd[0]==5&&bnd[2]==5)
	{
		#pragma omp parallel for schedule(static) num_threads(TN)
		for(j=0;j<m+1;j++)
		{ a[0][j]=a[l-1][j]; a[l][j]=a[1][j]; }
	}
	else if((bnd[0]==2||bnd[0]==1)&&(bnd[2]==1||bnd[2]==2))
	{
		#pragma omp parallel for schedule(static) num_threads(TN)
		for(j=0;j<m+1;j++)
		{ a[0][j]=a[1][j]; a[l][j]=a[l-1][j]; }
	}
	else exit(0);

	if(bnd[1]==5&&bnd[3]==5)
	{
		#pragma omp parallel for schedule(static) num_threads(TN)
		for(i=0;i<l+1;i++)
		{ a[i][0]=a[i][m-1]; a[i][m]=a[i][1]; }
	}
	else if(bnd[1]==1&&bnd[3]==1)
	{
		#pragma omp parallel for schedule(static) num_threads(TN)
		for(i=0;i<l+1;i++)
		{ a[i][0]=a[i][1]; a[i][m]=a[i][m-1]; }
	}
	else exit(0);

}

void weno(void)
{
	int i,j;
	double r1,r2,r3,r4,r5,t1,t2,t3,s1,s2,s3,a1,a2,a3,w1,w2,w3,epslon=1.0E-10;

	#pragma omp parallel for schedule(static) private(j,r1,r2,r3,r4,r5,t1,t2,t3,s1,s2,s3,a1,a2,a3,w1,w2,w3) num_threads(TN)
	for(i=0;i<l;i++)
	for(j=0;j<m-1;j++)
	{
		if(i<2&&(bnd[0]==1||bnd[0]==2))
		{
			if(u3[i][j]>0.0)
			{
				if(i==0) wenox[i][j]=fi[i][j+1];
				else wenox[i][j]=(3.0*fi[i+1][j+1]+6.0*fi[i][j+1]-fi[i-1][j+1])/8.0;
			}
			else wenox[i][j]=(3.0*fi[i][j+1]+6.0*fi[i+1][j+1]-fi[i+2][j+1])/8.0;
		}
		else if(i>=l-2&&(bnd[2]==1||bnd[2]==2))
		{
			if(u3[i][j]<0.0)
			{
				if(i==l-1)wenox[i][j]=fi[i][j+1];
				else wenox[i][j]=(3.0*fi[i][j+1]+6.0*fi[i+1][j+1]-fi[i+2][j+1])/8.0;
			}
			else wenox[i][j]=(3.0*fi[i+1][j+1]+6.0*fi[i][j+1]-fi[i-1][j+1])/8.0;
		}
        else 
		{
			if(i<2&&bnd[0]==5)
			{
				if(u3[i][j]>=0.0)
				{
					r1=fi[i+l-1-2][j+1];
            		r2=fi[i+l-1-1][j+1];
            		r3=fi[i][j+1];
            		r4=fi[i+1][j+1];
            		r5=fi[i+2][j+1];
				}
				else
				{
					r1=fi[i+3][j+1];
            		r2=fi[i+2][j+1];
            		r3=fi[i+1][j+1];
            		r4=fi[i][j+1];
            		r5=fi[i+l-1-1][j+1];
				}
			}
			else if(i>=l-2&&bnd[2]==5)
			{
				if(u3[i][j]>=0.0)
				{
					r1=fi[i-2][j+1];
            		r2=fi[i-1][j+1];
            		r3=fi[i][j+1];
            		r4=fi[i+1-l+1][j+1];
            		r5=fi[i+2-l+1][j+1];
				}
				else
				{
					r1=fi[i+3-l+1][j+1];
            		r2=fi[i+2-l+1][j+1];
            		r3=fi[i+1-l+1][j+1];
            		r4=fi[i][j+1];
            		r5=fi[i-1][j+1];
				}
			}
			else
			{
				if(u3[i][j]>=0.0)
				{
					r1=fi[i-2][j+1];
            		r2=fi[i-1][j+1];
            		r3=fi[i][j+1];
            		r4=fi[i+1][j+1];
            		r5=fi[i+2][j+1];
				}
				else
				{
					r1=fi[i+3][j+1];
            		r2=fi[i+2][j+1];
            		r3=fi[i+1][j+1];
            		r4=fi[i][j+1];
            		r5=fi[i-1][j+1];
				}
			}
			t1=r1-2.0*r2+r3;
        	t2=r1-4.0*r2+3.0*r3;
       		s1=13.0/12.0*t1*t1+0.25*t2*t2;
        	t1=r2-2.0*r3+r4;
       		t2=r2-r4;
     		s2=13.0/12.0*t1*t1+0.25*t2*t2;   
        	t1=r3-2.0*r4+r5;
        	t2=3.0*r3-4.0*r4+r5;
        	s3=13.0/12.0*t1*t1+0.25*t2*t2; 

        	a1=0.1/((epslon+s1)*(epslon+s1));
        	a2=0.6/((epslon+s2)*(epslon+s2));
        	a3=0.3/((epslon+s3)*(epslon+s3));
        	w1=a1/(a1+a2+a3);   
        	w2=a2/(a1+a2+a3);
        	w3=a3/(a1+a2+a3);
        	t1=r1/3.0-7.0*r2/6.0+11.0*r3/6.0;
        	t2=-r2/6.0+5.0*r3/6.0+r4/3.0;
        	t3=r3/3.0+5.0*r4/6.0-r5/6.0;
        	wenox[i][j]=w1*t1+w2*t2+w3*t3;
		}
	}
		
	#pragma omp parallel for schedule(static) private(j,r1,r2,r3,r4,r5,t1,t2,t3,s1,s2,s3,a1,a2,a3,w1,w2,w3) num_threads(TN)
	for(i=0;i<l-1;i++)
	for(j=0;j<m;j++)
	{
		if(j<2&&bnd[1]==1)
		{
			if(v3[i][j]>=0.0)
			{
				if(j==0) wenoy[i][j]=fi[i+1][j];
				else wenoy[i][j]=(3.0*fi[i+1][j+1]+6.0*fi[i+1][j]-fi[i+1][j-1])/8.0;
			}
			else wenoy[i][j]=(3.0*fi[i+1][j]+6.0*fi[i+1][j+1]-fi[i+1][j+2])/8.0;
		}
		else if(j>=m-2&&bnd[3]==1)
		{
			if(v3[i][j]<0.0)
			{
				if(j==m-1) wenoy[i][j]=fi[i+1][j];
				else wenoy[i][j]=(3.0*fi[i+1][j]+6.0*fi[i+1][j+1]-fi[i+1][j+2])/8.0;
			}
			else wenoy[i][j]=(3.0*fi[i+1][j+1]+6.0*fi[i+1][j]-fi[i+1][j-1])/8.0;
		}
		else 
		{
			if(j<2&&bnd[1]==5)
			{
				if(v3[i][j]>=0.0)
				{
					r1=fi[i+1][j+m-3];
            		r2=fi[i+1][j+m-2];
            		r3=fi[i+1][j];
            		r4=fi[i+1][j+1];
           			r5=fi[i+1][j+2];
				}
				else
				{
					r1=fi[i+1][j+3];
            		r2=fi[i+1][j+2];
            		r3=fi[i+1][j+1];
            		r4=fi[i+1][j];
            		r5=fi[i+1][j+m-2];
				}
			}
			else if(j>=m-2&&bnd[3]==5)
			{
				if(v3[i][j]>=0.0)
				{
					r1=fi[i+1][j-2];
            		r2=fi[i+1][j-1];
            		r3=fi[i+1][j];
            		r4=fi[i+1][j-m+2];
            		r5=fi[i+1][j-m+3];
				}
				else
				{
					r1=fi[i+1][j-m+4];
            		r2=fi[i+1][j-m+3];
            		r3=fi[i+1][j-m+2];
            		r4=fi[i+1][j];
            		r5=fi[i+1][j-1];
				}
			}
			else
			{
				if(v3[i][j]>=0.0)
				{
					r1=fi[i+1][j-2];
            		r2=fi[i+1][j-1];
            		r3=fi[i+1][j];
            		r4=fi[i+1][j+1];
            		r5=fi[i+1][j+2];
				}
				else
				{
					r1=fi[i+1][j+3];
            		r2=fi[i+1][j+2];
            		r3=fi[i+1][j+1];
            		r4=fi[i+1][j];
            		r5=fi[i+1][j-1];
				}
			}
			
			t1=r1-2.0*r2+r3;
      		t2=r1-4.0*r2+3.0*r3;
			s1=13.0/12.0*t1*t1+0.25*t2*t2;
			t1=r2-2.0*r3+r4;
			t2=r2-r4;
			s2=13.0/12.0*t1*t1+0.25*t2*t2;   
			t1=r3-2.0*r4+r5;
			t2=3.0*r3-4.0*r4+r5;
			s3=13.0/12.0*t1*t1+0.25*t2*t2; 

			a1=0.1/((epslon+s1)*(epslon+s1));
			a2=0.6/((epslon+s2)*(epslon+s2));
       		a3=0.3/((epslon+s3)*(epslon+s3));
       		w1=a1/(a1+a2+a3);
       		w2=a2/(a1+a2+a3);
       		w3=a3/(a1+a2+a3);
       		t1=r1/3.0-7.0*r2/6.0+11.*r3/6.0;
       		t2=-r2/6.0+5.0*r3/6.0+r4/3.0;
       		t3=r3/3.0+5.0*r4/6.0-r5/6.0;
       		wenoy[i][j]=w1*t1+w2*t2+w3*t3;
		}
	}
}

void CH(void)
{
	double nmda=0.25,tao=0.25,alfa=0.6;
	double epn2=epn*epn;
	double dx2=dx*dx,dy2=dy*dy;
	double c1,r1,r2,r3,temp=0.0,temp1,temp2;
	double a2=10.0/dx2/dx2+10.0/dy2/dy2;
	double diagonal=1.0+0.5/Pe*dt*epn2*a2;
	int i,j;

	#pragma omp parallel for schedule(static) private(j) num_threads(TN)
	for(i=0;i<l+1;i++)
	for(j=0;j<m+1;j++)
		{
			RHS[i][j]=0.0;
			source[i][j]=0.0;
			bulkE[i][j]=0.0;
			d2fi[i][j]=0.0;
		}

	weno();
	
	#pragma omp parallel for schedule(static) private(j,r1,r2,r3,c1) num_threads(TN)
	for(i=1;i<l;i++)
	{
		r1=raxis[i-1]; r2=raxis[i]; r3=(r1+r2)/2.0; 
		for(j=1;j<m;j++)
		{
			c1 = fi[i][j];
			bulkE[i][j]=(0.5-1.5*c1+c1*c1)*c1;
			source[i][j]=-0.5*ConvPrev[i][j];

			RHS[i][j]=(r2*u3[i][j-1]*wenox[i][j-1]-r1*u3[i-1][j-1]*wenox[i-1][j-1])/dx/r3+
     			(v3[i-1][j]*wenoy[i-1][j]-v3[i-1][j-1]*wenoy[i-1][j-1])/dy;
		}
	}
/*
	#pragma omp parallel for schedule(static) private(j,c1) num_threads(TN)
	for(i=1;i<l;i++)
	for(j=1;j<m;j++)
		{
			c1 = fi[i][j];
			bulkE[i][j]=(0.5-1.5*c1+c1*c1)*c1;
			source[i][j]=-0.5*ConvPrev[i][j];
		}*/
	array_bnd(bulkE);

	#pragma omp parallel for schedule(static) private(j,r1,r2,r3,temp) num_threads(TN)
    for(i=1;i<l;i++)
	for(j=1;j<m;j++)
		{
			r1=raxis[i-1]; r2=raxis[i]; r3=(r1+r2)/2.0;

            temp=((bulkE[i+1][j]-bulkE[i][j])*r2-(bulkE[i][j]-bulkE[i-1][j])*r1)/dx2/r3
			  +(bulkE[i][j+1]-2.0*bulkE[i][j]+bulkE[i][j-1])/dy2;

			temp = temp/Pe - RHS[i][j];
			ConvPrev[i][j] = temp;
			source[i][j] = source[i][j]+1.5*temp;
		}

	#pragma omp parallel for schedule(static) private(j,r1,r2,r3) num_threads(TN)
	for(i=1;i<l;i++)
	for(j=1;j<m;j++)
		{
			r1=raxis[i-1]; r2=raxis[i]; r3=(r1+r2)/2.0;

			d2fi[i][j]=((fi[i+1][j]-fi[i][j])*r2-(fi[i][j]-fi[i-1][j])*r1)/dx2/r3
              +(fi[i][j+1]-2*fi[i][j]+fi[i][j-1])/dy2;
		 }
	array_bnd(d2fi);

	#pragma omp parallel for schedule(static) private(j,r1,r2,r3,temp) num_threads(TN)
	for(i=1;i<l;i++)
	for(j=1;j<m;j++)
		{
			r1=raxis[i-1]; r2=raxis[i]; r3=(r1+r2)/2.0;

			temp=((d2fi[i+1][j]-d2fi[i][j])*r2-(d2fi[i][j]-d2fi[i-1][j])*r1)/dx2/r3
			  +(d2fi[i][j+1]-2.0*d2fi[i][j]+d2fi[i][j-1])/dy2;
			source[i][j] = source[i][j]-0.5*temp*epn2/Pe;
		}

/***********************************************************/ 
	for(ch_iter=0;ch_iter<ch_iternum;ch_iter++)
	{
		CH_res=0.0;

		#pragma omp parallel for schedule(static) private(j,r1,r2,r3) num_threads(TN)
		for(i=1;i<l;i++)
		for(j=1;j<m;j++)
			{
				r1=raxis[i-1]; r2=raxis[i]; r3=(r1+r2)/2.0;
				d2fi[i][j]=((fin[i+1][j]-fin[i][j])*r2-(fin[i][j]-fin[i-1][j])*r1)/dx2/r3
						+(fin[i][j+1]-2*fin[i][j]+fin[i][j-1])/dy2;
			}
		array_bnd(d2fi);

		#pragma omp parallel for schedule(static) private(j,r1,r2,r3) num_threads(TN)
		for(i=1;i<l;i++)
		for(j=1;j<m;j++)
			{
          		r1=raxis[i-1]; r2=raxis[i]; r3=(r1+r2)/2.0;

         		bulkE[i][j]=((d2fi[i+1][j]-d2fi[i][j])*r2-(d2fi[i][j]-d2fi[i-1][j])*r1)/dx2/r3
			               +(d2fi[i][j+1]-2.0*d2fi[i][j]+d2fi[i][j-1])/dy2;
			}

		#pragma omp parallel for schedule(static) private(j,temp) num_threads(TN)
		for(i=1;i<l;i++)
		for(j=1;j<m;j++)
			{
				temp=dt*(source[i][j]-0.5*epn2*bulkE[i][j]/Pe)+fi[i][j]-fin[i][j];
        		fin[i][j] = fin[i][j]+0.8*temp/diagonal;
        		if(fin[i][j]<0.0)fin[i][j]=0.0;
       			if(fin[i][j]>1.0)fin[i][j]=1.0;
//				CH_res=(CH_res>fabs(temp/dt)?CH_res:fabs(temp/dt));
			}
//		array_bnd(fin);
		if((bnd[0]==1||bnd[0]==2)&&(bnd[2]==1||bnd[2]==2))
		{
			#pragma omp parallel for schedule(static) num_threads(TN)
			for(j=1;j<m;j++)
			{fin[0][j]=fin[1][j];fin[l][j]=fin[l-1][j];}
		}
		if(bnd[0]==5&&bnd[2]==5)
		{
			#pragma omp parallel for schedule(static) num_threads(TN)
			for(j=1;j<m;j++)
			{fin[0][j]=fin[l-1][j];fin[l][j]=fin[1][j];}
		}

		if(bnd[1]==5&&bnd[3]==5)
		{
			#pragma omp parallel for schedule(static) num_threads(TN)
			for(i=1;i<l;i++)
			{fin[i][0]=fin[i][m-1]; fin[i][m]=fin[i][1];}
		}
		if(bnd[1]==1&&bnd[3]==1)
		{
			#pragma omp parallel for schedule(static) private(temp,temp1,temp2) num_threads(TN)
			for(i=1;i<l;i++)
			{
				if(fin[i][1]>0.001&&fin[i][1]<0.999)
				{
					temp=(fi[i+1][1]-fi[i-1][1])/dx;  // ????yuan_2012_11_15****0.5
					temp1=fabs(temp)*tadv*dy;
					temp2=fabs(temp)*trec*dy;
					if(fin[i][0]<fin[i][2]+temp1)
						fin[i][0]=alfa*(fin[i][2]+temp1)+(1.0-alfa)*fin[i][0];
					else if(fin[i][0]>fin[i][2]+temp2)
						fin[i][0]=alfa*(fin[i][2]+temp2)+(1.0-alfa)*fin[i][0];
					else ;
					if(fin[i][0]>1.0)fin[i][0]=1.0;
					if(fin[i][0]<0.0)fin[i][0]=0.0;
				}
				else fin[i][0]=fin[i][1]; 
				//up boundary do not consider wetting condition & contact_angle
				fin[i][m]=fin[i][m-1];  
			}
		}
	}
}


