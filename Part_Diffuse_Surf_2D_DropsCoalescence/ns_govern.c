double sigma[l+1][m+1],st; /* nondimensional surface tension */
double preU[l][m-1],preV[l-1][m]={0},rhsu[l][m-1],rhsv[l-1][m],pot[l+1][m+1];

void surface_tension_geo(void)
{
	int i,j;
	double b1,b2,b3,b4,n_x,n_y;

	#pragma omp parallel for schedule(static) private(j) num_threads(TN)
	for(i=0;i<l+1;i++)
	for(j=0;j<m+1;j++)
		fi[i][j]=0.5*(fio[i][j]+fin[i][j]);

	#pragma omp parallel for schedule(static) private(j) num_threads(TN)
	for(i=0;i<l+1;i++)
	for(j=0;j<m+1;j++)
		sigma[i][j]=1.0+surf_elas*log(1.0-surf_cove*surfactant[i][j]);

	#pragma omp parallel for schedule(static) private(j,b1,b2,b3) num_threads(TN)
	for(i=1;i<l;i++)
	for(j=1;j<m;j++)
	{
		b1=(fi[i+1][j]-fi[i-1][j])*0.5/dx;
		b2=(fi[i][j+1]-fi[i][j-1])*0.5/dy;
		b3=sqrt(b1*b1+b2*b2)+1.0E-10;
		nor_vector_x[i][j]=b1/b3; nor_vector_y[i][j]=b2/b3;
	}

	array_bnd(nor_vector_x); array_bnd(nor_vector_y);
	#pragma omp parallel for schedule(static) private(j,b1,b2,b3,b4,n_x,n_y) num_threads(TN)
	for(i=0;i<l-1;i++)
	for(j=0;j<m-1;j++)
	{
		b1=(fi[i+1][j+1]-fi[i][j+1])/dx;
		b2=(0.5*(fi[i][j+2]+fi[i+1][j+2])-0.5*(fi[i][j]+fi[i+1][j]))*0.5/dy;
		b3=sqrt(b1*b1+b2*b2)+1.0E-10; n_x=b1/b3; n_y=b2/b3;
		// divergence of normal vector : curvature
		b4=(nor_vector_x[i+1][j+1]-nor_vector_x[i][j+1])/dx
			+(0.5*(nor_vector_y[i][j+2]+nor_vector_y[i+1][j+2])-0.5*(nor_vector_y[i][j]+nor_vector_y[i+1][j]))*0.5/dy;

		surfx[i][j]=(1.0-n_x*n_x)*(sigma[i+1][j+1]-sigma[i][j+1])/dx
			-n_x*n_y*(0.5*(sigma[i][j+2]+sigma[i+1][j+2])-0.5*(sigma[i][j]+sigma[i+1][j]))*0.5/dy;
		surfx[i][j]=surfx[i][j]*b3;
		surfx[i][j]=surfx[i][j]-0.5*(sigma[i][j+1]+sigma[i+1][j+1])*b4*b1;
	}

	#pragma omp parallel for schedule(static) private(j,b1,b2,b3,b4,n_x,n_y) num_threads(TN)
	for(i=0;i<l-1;i++)
	for(j=0;j<m-1;j++)
	{
		b1=(0.5*(fi[i+2][j+1]+fi[i+2][j])-0.5*(fi[i][j+1]+fi[i][j]))*0.5/dx;
		b2=(fi[i+1][j+1]-fi[i+1][j])/dy;
		b3=sqrt(b1*b1+b2*b2)+1.0E-10;
		n_x=b1/b3; n_y=b2/b3;

		b4=(nor_vector_y[i+1][j+1]-nor_vector_y[i+1][j])/dy
			+(0.5*(nor_vector_x[i+2][j]+nor_vector_x[i+2][j+1])-0.5*(nor_vector_x[i][j]+nor_vector_x[i][j+1]))*0.5/dx;

		surfy[i][j]=(1.0-n_y*n_y)*(sigma[i+1][j+1]-sigma[i+1][j])/dy
			-n_x*n_y*(0.5*(sigma[i+2][j]+sigma[i+2][j+1])-0.5*(sigma[i][j]+sigma[i][j+1]))*0.5/dx;
		surfy[i][j]=surfy[i][j]*b3;
		surfy[i][j]=surfy[i][j]-0.5*(sigma[i+1][j]+sigma[i+1][j+1])*b4*b2;
	}
}

void surface_tension(void)
{
	int i,j;
	double r1,r2,r3,a1,a2,c1;

	#pragma omp parallel for schedule(static) private(j) num_threads(TN)
	for(i=0;i<l+1;i++)
	for(j=0;j<m+1;j++)
		fi[i][j]=0.5*(fio[i][j]+fin[i][j]);

	#pragma omp parallel for schedule(static) private(j) num_threads(TN)
	for(i=0;i<l+1;i++)
	for(j=0;j<m+1;j++)
		pot[i][j]=0.0;

	#pragma omp parallel for schedule(static) private(j,r1,r2,r3,c1,a1,a2) num_threads(TN)
	for(i=1;i<l;i++)
	for(j=1;j<m;j++)
	{
		r1=raxis[i-1]; r2=raxis[i]; r3=(r1+r2)/2.0;
		c1=fi[i][j];
		a1=(0.5-1.5*c1+c1*c1)*c1;
	    a2=((fi[i+1][j]-fi[i][j])*r2-(fi[i][j]-fi[i-1][j])*r1)/dx/dx/r3+
			(fi[i][j+1]-2.0*fi[i][j]+fi[i][j-1])/dy/dy;
	    pot[i][j]=a1-epn*epn*a2;
	}
	array_bnd(pot);

	#pragma omp parallel for schedule(static) private(j) num_threads(TN)
	for(i=0;i<l-1;i++)
	for(j=0;j<m-1;j++)
		surfx[i][j]=st*(pot[i+1][j+1]-pot[i][j+1])*(fi[i+1][j+1]+fi[i][j+1])/2.0/dx;

	#pragma omp parallel for schedule(static) private(j) num_threads(TN)
	for(i=0;i<l-1;i++)
	for(j=0;j<m-1;j++)
		surfy[i][j]=st*(pot[i+1][j+1]-pot[i+1][j])*(fi[i+1][j+1]+fi[i+1][j])/2.0/dy;
}


void get_den_vis(void) // get density & viscosity (den[l+1][m+1],vis[l+1][m+1])
{
	int i,j; double c1;

	#pragma omp parallel for schedule(static) private(j,c1) num_threads(TN)
	for(i=0;i<l+1;i++)
	for(j=0;j<m+1;j++)
	{
		c1=fi[i][j];
		if(c1<0.0)c1=0.0;
		if(c1>1.0)c1=1.0;
		den[i][j]=c1+rd*(1.0-c1);
		vis[i][j]=c1+rv*(1.0-c1);
	}
}

void x_conv_diff(void) // get X convective & diffusion term
{
	int i,j,is,im1;
	double r1,r2,r3;
	double Xconv,Yconv,Xdiff,Ydiff,tempv,temp,crou,cmiu;
	double ub,um,up,c1,c2,a1,a2,Diffusion,ConvNow;

	if(bnd[0]==5&&bnd[2]==5) is=0;  else is=1;

	#pragma omp parallel for schedule(static) private(j,r1,r2,r3,im1,Xconv,Yconv,Xdiff,Ydiff,tempv,temp,crou,cmiu,ub,um,up,c1,c2,a1,a2,Diffusion,ConvNow) num_threads(TN)
	for(i=is;i<l-1;i++)
	for(j=0;j<m-1;j++)
	{
		if(i==is&&bnd[0]==5)
		{
			if(coord==1) { printf("Periodic contradict axisymmetric coord==1 error \n"); exit(0); }
			r1=1.0;r2=1.0;r3=1.0;
			im1=l-2;
		}
		else
		{
			r1=(raxis[i]+raxis[i-1])/2.0; r2=(raxis[i]+raxis[i+1])/2.0; r3=raxis[i];
			im1 = i-1;
		}

		Xconv=u2[i][j]*0.5*(u2[i+1][j]-u2[i-1][j])/dx;
		if(j==0)
		{
			up=0.5*(u2[i][j+1]+u2[i][j]);
			if(bnd[1]==1){ub=vel_bnd[1];um=ub;}
			if(bnd[1]==5)um=0.5*(u2[i][m-2]+u2[i][0]);
		}
		else if(j==m-2)
		{
			um=0.5*(u2[i][j-1]+u2[i][j]);
			if(bnd[3]==1){ub=vel_bnd[3];up=ub;}
			if(bnd[3]==5)up=0.5*(u2[i][0]+u2[i][j]);
		}
		else { up=0.5*(u2[i][j+1]+u2[i][j]); um=0.5*(u2[i][j-1]+u2[i][j]);}
		tempv=(v2[i-1][j]+v2[i][j]+v2[i-1][j+1]+v2[i][j+1])/4.0;
        Yconv=tempv*(up-um)/dy;

        c2 =(vis[i+1][j+2]+vis[i][j+2]+vis[i+1][j+1]+vis[i][j+1])/4.0;
        c1 =(vis[i+1][j]+vis[i][j]+vis[i+1][j+1]+vis[i][j+1])/4.0;
		if(j==0)
		{
			if(bnd[1]==1){ub=vel_bnd[1];um=2.0*ub-u2[i][j];}
			if(bnd[1]==5)um=u2[i][m-2];
			up=u2[i][j+1];
		}
		else if(j==m-2)
		{
			if(bnd[3]==1){ub=vel_bnd[3];up=2.0*ub-u2[i][j];}
			if(bnd[3]==5)up=u2[i][0];
			um=u2[i][j-1];
		}
		else {up = u2[i][j+1]; um = u2[i][j-1]; }

		a1=((up-u2[i][j])*c2-(u2[i][j]-um)*c1)/dy/dy;
		a2 =((v2[i][j+1]-v2[i-1][j+1])*c2-(v2[i][j]-v2[i-1][j])*c1)/dx/dy;
        Ydiff =a1+a2;
		Xdiff=2.0*((u2[i+1][j]-u2[i][j])*vis[i+1][j+1]*r2-(u2[i][j]-u2[i-1][j])*vis[i][j+1]*r1)/dx/dx/r3;

        crou = (den[i][j+1]+den[i+1][j+1])/2.0;
        cmiu = (vis[i][j+1]+vis[i+1][j+1])/2.0;

        Diffusion=(Xdiff+Ydiff)/Re/crou;
        ConvNow=Xconv+Yconv;
        temp=surfx[i][j]/Ca/Re/crou;
//		temp=0.0;
		if(coord==1)ConvNow=ConvNow+2.0*cmiu*u2[i][j]/(raxis[i]*raxis[i])/Re/crou;
		rhsu[i][j]=0.5*preU[i][j]-1.5*ConvNow+0.5*Diffusion+u2[i][j]/dt+temp;
		preU[i][j]=ConvNow;
	}
}

void y_conv_diff(void) // get Y convective & diffusion term
{
	int i,j,jm1,js;
	double tempu,temp,Xconv,Yconv,Xdiff,Ydiff;
	double um,up,c1,c2,r1,r2,r3,a1,a2,crou,Diffusion,ConvNow;

	if(bnd[1]==5)js=0;else js=1;

	#pragma omp parallel for schedule(static) private(j,r1,r2,r3,jm1,Xconv,Yconv,Xdiff,Ydiff,tempu,temp,crou,um,up,c1,c2,a1,a2,Diffusion,ConvNow) num_threads(TN)
	for(i=0;i<l-1;i++)
	for(j=js;j<m-1;j++)
	{
		if(j==0&&bnd[1]==5) jm1=m-2; else jm1=j-1;

		temp=0.5*(v2[i][j+1]-v2[i][jm1])/dy;
        Yconv=v2[i][j]*temp;
		if(i==0)
		{
			up=0.5*(v2[i+1][j]+v2[i][j]);
			if(bnd[0]==1)um=vel_bnd[0];
			else if(bnd[0]==2)um=v2[i][j];
			else if(bnd[0]==5)um=0.5*(v2[l-2][j]+v2[i][j]);
			else { printf("y_conv_diff i==0 error \n"); exit(0);}
		}
		else if(i==l-2)
		{
			um=0.5*(v2[i-1][j]+v2[i][j]);
			if(bnd[2]==1)up=vel_bnd[2];
			else if(bnd[2]==2)up=v2[i][j];
			else if(bnd[2]==5) up=0.5*(v2[0][j]+v2[i][j]);
			else { printf("y_conv_diff i==l-2 error \n"); exit(0);}
		}
		else { up=0.5*(v2[i+1][j]+v2[i][j]); um=0.5*(v2[i-1][j]+v2[i][j]);}
		tempu=(u2[i][jm1]+u2[i][j]+u2[i+1][jm1]+u2[i+1][j])/4.0;
		Xconv=tempu*(up-um)/dx;
		ConvNow=Xconv+Yconv;

		Ydiff=2.0*((v2[i][j+1]-v2[i][j])*vis[i+1][j+1]-(v2[i][j]-v2[i][jm1])*vis[i+1][j])/dy/dy;
		c2 = (vis[i+2][j+1]+vis[i+2][j]+vis[i+1][j+1]+vis[i+1][j])/4.0;
        c1 = (vis[i][j+1]+vis[i][j]+vis[i+1][j+1]+vis[i+1][j])/4.0;

        r1=raxis[i];
        r2=raxis[i+1];
        r3=(raxis[i]+raxis[i+1])/2.0;
		a1=((u2[i+1][j]-u2[i+1][jm1])*c2*r2-(u2[i][j]-u2[i][jm1])*c1*r1)/dx/dy;

		if(i==0)
		{
			up=v2[i+1][j];
			if(bnd[0]==1) um=2.0*vel_bnd[0]-v2[i][j];
			else if(bnd[0]==2) um=v2[i][j];
			else if(bnd[0]==5) um=v2[l-2][j];
			else { printf("y_conv_diff i==0 error \n"); exit(0);}
		}
		else if(i==l-2)
		{
			um=v2[i-1][j];
			if(bnd[2]==1) up=2.0*vel_bnd[2]-v2[i][j];
			else if(bnd[2]==2) up=v2[i][j];
			else if(bnd[2]==5) up=v2[0][j];
			else { printf("y_conv_diff i==l-2 error \n"); exit(0);}
		}
		else {up=v2[i+1][j]; um=v2[i-1][j];}

		a2=((up-v2[i][j])*c2*r2-(v2[i][j]-um)*c1*r1)/dx/dx;
        Xdiff =(a1+a2)/r3;
		crou = (den[i+1][j]+den[i+1][j+1])/2.0;
        Diffusion=(Xdiff+Ydiff)/crou/Re;
		temp=surfy[i][j]/Ca/Re/crou;
//		temp=0.0;
        rhsv[i][j]=0.5*preV[i][j]-1.5*ConvNow+0.5*Diffusion+v2[i][j]/dt+temp;
		if(gravity==1&&crou>0.5) rhsv[i][j]=rhsv[i][j]-sin(gravity_angle*PI/180.0)*Bo;
		preV[i][j]=ConvNow;
	}
}

void x_diff(void) // get X diffusion term
{
	int i,j,im1,is;
	double r1,r2,r3;
	double Xdiff,Ydiff,crou,cmiu;
	double um,up,c1,c2,a1,a2,Diffusion,residual;
	double diagonal=(1.0/dx/dx+1.0/dy/dy)/Re;

	if(bnd[0]==5) is=0; else is=1;

	#pragma omp parallel for schedule(static) private(j,r1,r2,r3,im1,Xdiff,Ydiff,crou,miu,um,up,c1,c2,a1,a2,Diffusion,residual) num_threads(TN)
	for(i=1;i<l-1;i++)
	for(j=0;j<m-1;j++)
	{
		if(i==is&&bnd[0]==5)
		{
			if(coord==1) { printf("Periodic contradict axisymmetric coord==1 error \n"); exit(0); }
			r1=1.0;r2=1.0;r3=1.0;
			im1=l-2;
		}
		else
		{
			r1=(raxis[i]+raxis[i-1])/2.0; r2=(raxis[i]+raxis[i+1])/2.0; r3=raxis[i];
			im1 = i-1;
		}

		Xdiff=2.0*((u3[i+1][j]-u3[i][j])*vis[i+1][j+1]*r2-(u3[i][j]-u3[i-1][j])*vis[i][j+1]*r1)/dx/dx/r3;
        c2 =(vis[i+1][j+2]+vis[i][j+2]+vis[i+1][j+1]+vis[i][j+1])/4.0;
        c1 =(vis[i+1][j]+vis[i][j]+vis[i+1][j+1]+vis[i][j+1])/4.0;

		if(j==0)
		{
			if(bnd[1]==1)um=2.0*vel_bnd[1]-u3[i][j];
			if(bnd[1]==5)um=u3[i][m-2];
			up=u3[i][j+1];
		}
		else if(j==m-2)
		{
			if(bnd[3]==1)up=2.0*vel_bnd[3]-u3[i][j];
			if(bnd[3]==5)up=u3[i][0];
			um=u3[i][j-1];
		}
		else {up = u3[i][j+1]; um = u3[i][j-1]; }
		a1=((up-u3[i][j])*c2-(u3[i][j]-um)*c1)/dy/dy;
		a2 =((v3[i][j+1]-v3[i-1][j+1])*c2-(v3[i][j]-v3[i-1][j])*c1)/dx/dy;
        Ydiff =a1+a2;

        crou = (den[i][j+1]+den[i+1][j+1])/2.0;
        cmiu = (vis[i][j+1]+vis[i+1][j+1])/2.0;

        Diffusion=(Xdiff+Ydiff)/Re/crou;
        residual = rhsu[i][j]+Diffusion*0.5-u3[i][j]/dt;
//		MOM_res = MOM_res>fabs(residual)?MOM_res:fabs(residual);
		if(fabs(residual)>1.0E10)
		{
			fprintf(log_fp,"i=%d  j=%d  NS_res=%lf \n",i,j,MOM_res);
			fprintf(log_fp,"u3=%10lf  um=%10lf  up=%10lf \n",u3[i][j],um,up);
			fprintf(log_fp,"Xdiff=%10lf  a1=%10lf  a2==%10lf  Ydiff=%10lf \n",Xdiff,a1,a2,Ydiff);
			fprintf(log_fp,"v3(i-1,j+1)=%10lf v3(i-1,j)=%10lf v3(i,j)=%10lf v3(i,j+1)=%10lf \n"
				            ,v3[im1][j+1],v3[im1][j],v3[i][j],v3[i][j+1]);
			fprintf(log_fp,"u3(i-1,j)=%10lf u3(i+1,j)=%10lf u3(i,j-1)=%10lf u3(i,j+1)=%10lf \n"
				            ,u3[im1][j],u3[i+1][j],u3[i][j-1],u3[i][j+1]);
			exit(0);
		}
		u3[i][j]=u3[i][j]+0.8*residual/(cmiu*diagonal/crou+1.0/dt);
	}
}

void y_diff(void) // get Y diffusion term
{
	int i,j,jm1,js;
	double Xdiff,Ydiff,residual;
	double um,up,c1,c2,r1,r2,r3,a1,a2,crou,cmiu,Diffusion;
	double diagonal=(1.0/dx/dx+1.0/dy/dy)/Re;

	if(bnd[1]==5)js=0;else js=1;

	#pragma omp parallel for schedule(static) private(j,r1,r2,r3,jm1,Xdiff,Ydiff,crou,miu,um,up,c1,c2,a1,a2,Diffusion,residual) num_threads(TN)
	for(i=0;i<l-1;i++)
	for(j=js;j<m-1;j++) // for 1&3 are periodic boundary
	{
		if(j==0&&bnd[1]==5) jm1=m-2;
		else jm1=j-1;

		Ydiff=2.0*((v3[i][j+1]-v3[i][j])*vis[i+1][j+1]-(v3[i][j]-v3[i][jm1])*vis[i+1][j])/dy/dy;
		c2 = (vis[i+2][j+1]+vis[i+2][j]+vis[i+1][j+1]+vis[i+1][j])/4.0;
        c1 = (vis[i][j+1]+vis[i][j]+vis[i+1][j+1]+vis[i+1][j])/4.0;

		r1=raxis[i]; r2=raxis[i+1]; r3=(r1+r2)/2.0;
		if(i==0)
		{
			up=v3[i+1][j];
			if(bnd[0]==1) um=2.0*vel_bnd[0]-v3[i][j];
			else if(bnd[0]==2) um=v3[i][j];
			else if(bnd[0]==5) um=v3[l-2][j];
			else { printf("y_diff i==0 error \n"); exit(0);}
		}
		else if(i==l-2)
		{
			um=v3[i-1][j];
			if(bnd[2]==1)up=2.0*vel_bnd[2]-v3[i][j];
			else if(bnd[2]==2)up=v3[i][j];
			else if(bnd[2]==5) up=v3[0][j];
			else { printf("y_diff i==l-2 error \n"); exit(0);}
		}
		else {up=v3[i+1][j]; um=v3[i-1][j];}

		a1=((u3[i+1][j]-u3[i+1][jm1])*c2*r2-(u3[i][j]-u3[i][jm1])*c1*r1)/dx/dy;
		a2=((up-v3[i][j])*c2*r2-(v3[i][j]-um)*c1*r1)/dx/dx;
        Xdiff =(a1+a2)/r3;
		cmiu = (vis[i+1][j]+vis[i+1][j+1])/2.0;
		crou = (den[i+1][j]+den[i+1][j+1])/2.0;
        Diffusion=(Xdiff+Ydiff)/crou/Re;
        residual = rhsv[i][j]+Diffusion*0.5-v3[i][j]/dt;
//		MOM_res = MOM_res>fabs(residual)?MOM_res:fabs(residual);
		v3[i][j]=v3[i][j]+0.8*residual/(cmiu*diagonal/crou+1.0/dt);
	}
}

void uv_boundary(void)
{
	int i,j;

	if((bnd[0]==1||bnd[0]==2)&&(bnd[2]==1||bnd[2]==2))
	{
		#pragma omp parallel for schedule(static) num_threads(TN)
		for(j=0;j<m-1;j++) { u3[0][j]=0.0; u3[l-1][j]=0.0;}
	}
	else if(bnd[0]==5&&bnd[2]==5)
	{
		#pragma omp parallel for schedule(static) num_threads(TN)
		for(j=0;j<m-1;j++) u3[l-1][j]=u3[0][j];
	}
	else { printf("uv_boundary bnd[0]&&bnd[2] error \n"); exit(0);}


	if(bnd[1]==1&&bnd[3]==1)
	{
		#pragma omp parallel for schedule(static) num_threads(TN)
		for(i=0;i<l-1;i++) { v3[i][0]=0.0; v3[i][m-1]=0.0;}
	}
	else if(bnd[1]==5&&bnd[3]==5)
	{
		#pragma omp parallel for schedule(static) num_threads(TN)
		for(i=0;i<l-1;i++) v3[i][m-1]=v3[i][0];
	}
	else { printf("uv_boundary bnd[1]&&bnd[3] error \n"); exit(0);}
}

void momentum(void)
{
	get_den_vis();
	x_conv_diff();
	y_conv_diff();
	for(mo_iter=0;mo_iter<mo_iternum;mo_iter++)
	{
		MOM_res=0.0;
		x_diff();
		y_diff();
		uv_boundary();
	}
}

void divergence(void)
{
	int i,j,is,js;
	double r1,r2,r3,temp,px1,px2,py1,py2,residual;
	double res[l-1][m-1];
	double diagonal=-2.0*(1.0/dx/dx+1.0/dy/dy);

	#pragma omp parallel for schedule(static) private(j,r1,r2,r3) num_threads(TN)
	for(i=1;i<l;i++)
	for(j=1;j<m;j++)
	{
		r1=raxis[i-1]; r2=raxis[i]; r3=(r1+r2)/2.0;
		res[i-1][j-1]=((r2*u3[i][j-1]-r1*u3[i-1][j-1])/dx/r3+(v3[i-1][j]-v3[i-1][j-1])/dy)/dt;
	}

	for(div_iter=0;div_iter<div_iternum;div_iter++)
	{
		DIV_res=0.0;

		#pragma omp parallel for schedule(static) private(j,r1,r2,r3,temp,px1,px2,py1,py2,residual) num_threads(TN)
		for(i=1;i<l;i++)
		for(j=1;j<m;j++)
		{
			r1=raxis[i-1]; r2=raxis[i]; r3=(r1+r2)/2.0;
			temp=(den[i][j]+den[i-1][j])/2.0;
            px1=(p[i][j]-p[i-1][j])/temp/dx;

            temp=(den[i][j]+den[i+1][j])/2.0;
            px2=(p[i+1][j]-p[i][j])/temp/dx;

            temp=(den[i][j]+den[i][j-1])/2.0;
            py1=(p[i][j]-p[i][j-1])/temp/dy;

            temp=(den[i][j]+den[i][j+1])/2.0;
            py2=(p[i][j+1]-p[i][j])/temp/dy;

            residual=res[i-1][j-1]-((r2*px2-r1*px1)/dx/r3+(py2-py1)/dy);
 //           DIV_res =DIV_res>fabs(residual)?DIV_res:fabs(residual);
            p[i][j]=p[i][j]+1.2*residual/(diagonal/den[i][j]);
		}
		array_bnd(p);
	}

	if(bnd[0]==5)is=0; else is=1;
	if(bnd[1]==5)js=0; else js=1;

	#pragma omp parallel for schedule(static) private(j,temp) num_threads(TN)
	for(i=is;i<l-1;i++)
	for(j=1;j<m;j++)
	{
		temp=(den[i+1][j]+den[i][j])/2.0;
        u3[i][j-1]=u3[i][j-1]-dt*(p[i+1][j]-p[i][j])/dx/temp;
	}

	#pragma omp parallel for schedule(static) private(j,temp) num_threads(TN)
	for(i=1;i<l;i++)
	for(j=js;j<m-1;j++)
	{
		temp=(den[i][j+1]+den[i][j])/2.0;
        v3[i-1][j]=v3[i-1][j]-dt*(p[i][j+1]-p[i][j])/dy/temp;
	}

	if(bnd[2]==5){ for(j=0;j<m-1;j++) u3[l-1][j]=u3[0][j];}
	if(bnd[3]==5){ for(i=0;i<l-1;i++) v3[i][m-1]=v3[i][0];}

	//Pressure reference at right upper corner
	temp=p[l-1][m-1];

	#pragma omp parallel for schedule(static) private(j) num_threads(TN)
	for(i=1;i<l;i++)
	for(j=1;j<m;j++)
		p[i][j]=p[i][j]-temp;

	array_bnd(p);
}

void NS(void)
{
	//surface_tension_geo();
	//surface_tension();
//	if(iteration%ioutput==0)
//	{
//		print_2D_array("GeoSurfx.dat",iteration,"GeoSurfx",surfx[0],l-1,m-1,dx);
//		print_2D_array("GeoSurfy.dat",iteration,"GeoSurfy",surfy[0],l-1,m-1,dx);
//	}
/*	surface_tension();
	if(iteration%ioutput==0)
	{
		print_2D_array("FreeSurfx.dat",iteration,"FreeSurfx",surfx[0],l-1,m-1,dx);
		print_2D_array("FreeSurfy.dat",iteration,"FreeSurfy",surfy[0],l-1,m-1,dx);
	}
	*/
	momentum();
	divergence();
//	print_2D_array("surfx",iteration,"surfx",surfx[0],l-1,m-1,dx);
//	print_2D_array("surfy",iteration,"surfy",surfy[0],l-1,m-1,dx);
//	print_2D_array("sigma",iteration,"sigma",sigma[0],l+1,m+1,dx);
}

