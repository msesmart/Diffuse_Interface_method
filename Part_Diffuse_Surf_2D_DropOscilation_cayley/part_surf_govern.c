
#define npar 4500
double xp[npar],yp[npar],up[npar],vp[npar],surf[npar],length2p[npar],surf_new[npar],d_surf[npar],NSTp[npar][2],MSTp[npar][2];//SurfaceTension

void cal_length2p(void)
{
    double drx,dry,sum=0.0;int i,next;
    #pragma omp parallel for schedule(static) private(next,drx,dry) num_threads(TN)
    for(i=1;i<=np;i++)
    {
        if(i==np)next=1; else next=i+1;
        drx=xp[next]-xp[i];
        dry=yp[next]-yp[i];
        length2p[i]=sqrt(drx*drx+dry*dry);
  //      sum=sum+length2p[i];
    }
 //   printf("sum of length = %lf \n",sum);
}

void output_particle(double *a, char *filename, int file_num)
{
    char name[30],num[6]; FILE *fp; int i,before,next; double temp1,temp2,temp3;
    strcpy(name,filename); //itoa(file_num,num,10); strcat(name,num); strcat(name,".dat");
    fp=fopen(name,"w");
	fprintf(fp," VARIABLES=\"xp \",\"yp \",\"f \",\"surf \" \n");
	fprintf(fp," ZONE F=POINT \n");
    cal_length2p();
	for(i=1;i<=np;i++)
    {
        if(i==1)before=np; else before=i-1;
        if(i==np)next=1; else next=i+1;
        temp1=0.5*(xp[i]+xp[next]); temp2=0.5*(yp[i]+yp[next]);
        temp3=a[i]/length2p[i];
        //fprintf(fp,"%lf %lf %lf %lf \n",temp1,temp2,temp3,surf[i]);
        fprintf(fp,"%lf %lf %lf %lf \n",xp[i],yp[i],0.5*(a[i]/length2p[i]+a[before]/length2p[before]),a[i]);
        //fprintf(fp,"%lf %lf %lf %lf \n",xp[i],yp[i],1.0,1.0);
        //surf_sum=surf_sum+surf[i];
    }
    fclose(fp);
}

void particle_surfactant_initialization(void)
{
    double temp1,temp2,angle,deltaAngle,drx,dry,dr,R;
    np=0; R=rad; deltaAngle=asin(0.8*pdr/R); angle=0.0;
    while(angle<2.0*PI)
    {
        np++;
        if(np>npar-1){printf("ERROR: np=%d exceed the npar",np);exit(0);}
        xp[np]=cos(angle)*(R+R*0.4*cos(angle)*cos(angle))+xc;
        yp[np]=sin(angle)*(R+R*0.4*cos(angle)*cos(angle))+yc;
        if(np>1)
        {
            //temp1=0.5*(1.0-cos(angle)); temp2=0.5*(1.0-cos(angle-deltaAngle));
            temp1=1.0; temp2=temp1;
            drx=xp[np]-xp[np-1]; dry=yp[np]-yp[np-1]; dr=sqrt(drx*drx+dry*dry);
            surf[np-1]=0.5*(temp1+temp2)*dr;
        }
        if(2.0*PI-angle<=deltaAngle)
        {
            drx=xp[np]-xp[1]; dry=yp[np]-yp[1]; dr=sqrt(drx*drx+dry*dry);
            surf[np]=0.5*(temp1+temp2)*dr; break;
        }
        else angle=angle+deltaAngle;
    }
    cal_length2p();
    printf("particle & surfactant initialization is finished, np=%d \n",np);
}

void update_surf(void) // surf_new->surf
{
    int i; double sum_surf=0.0;
    #pragma omp parallel for schedule(static) num_threads(TN)
    for(i=1;i<=np;i++){ surf[i]=surf_new[i];sum_surf=sum_surf+surf[i];}
//    printf("sum_surf=%lf \n",sum_surf);
}

void surf_diffuse(double mdt)
{
    int i,next,n=80; double temp1,temp2,ratio=0.4,f_nplus1,temp;
    cal_length2p();
    #pragma omp parallel for schedule(static) private(next,temp1,temp2) num_threads(TN)
    for(i=1;i<=np;i++)
    {
        if(i==np)next=1; else next=i+1;
        temp1=surf[i]/length2p[i];
        temp2=surf[next]/length2p[next];
        d_surf[next]=2.0*(temp2-temp1)/(length2p[i]+length2p[next])*2.0;//very interesting !!!
    }
    #pragma omp parallel for schedule(static) private(next) num_threads(TN)
    for(i=1;i<=np;i++)
    {
        if(i==np)next=1; else next=i+1;
        surf_new[i]=surf[i]/length2p[i]+ratio*mdt*(d_surf[next]-d_surf[i])/length2p[i]/Pe_surf;
        surf_new[i]=surf_new[i]*length2p[i];
    }
//    output_particle(surf_new,"part_diffuse_new",0);
    while(n>0)
    {
        #pragma omp parallel for schedule(static) private(next,temp1,temp2) num_threads(TN)
        for(i=1;i<=np;i++)
        {
            if(i==np)next=1; else next=i+1;
            temp1=surf_new[i]/length2p[i];
            temp2=surf_new[next]/length2p[next];
            d_surf[next]=2.0*(temp2-temp1)/(length2p[i]+length2p[next])*2.0;
        }
        #pragma omp parallel for schedule(static) private(next,temp,f_nplus1) num_threads(TN)
        for(i=1;i<=np;i++)
        {
            f_nplus1=surf_new[i]/length2p[i];
            if(i==np)next=1; else next=i+1;
            temp=f_nplus1-surf[i]/length2p[i]-(1-ratio)*mdt*(d_surf[next]-d_surf[i])/length2p[i]/Pe_surf;
            f_nplus1=f_nplus1-0.6*temp; surf_new[i]=f_nplus1*length2p[i];
            //surf_new[i]=surf[i]/length2p[i]+(1-ratio)*mdt*(d_surf[next]-d_surf[i])/length2p[i]/Pe_surf;
            //surf_new[i]=surf_new[i]*length2p[i];
        }
        n--;
    }
//    output_particle(surf_new,"part_diffuse_new",1);
    update_surf();
}

void get_part_vel(void)
{
    int i,iu,ju,iv,jv; double x1,y1,x2,y2,x3,y3,x4,y4,temp1,temp2,temp3,temp4;
    #pragma omp parallel for schedule(static) private(iu,ju,iv,jv,x1,y1,x2,y2,x3,y3,x4,y4,temp1,temp2,temp3,temp4) num_threads(TN)
    for(i=1;i<=np;i++)
    {   // up
        iu=(int)(xp[i]/dx); ju=(int)((yp[i]-0.5*dy)/dy);
        x1=iu*dx; y1=ju*dy+0.5*dy; x2=x1+dx; y2=y1; x3=x2; y3=y2+dy; x4=x1; y4=y3;
        temp1=fabs((x3-xp[i])*(y3-yp[i])); temp2=fabs((x4-xp[i])*(y4-yp[i]));
        temp3=fabs((x1-xp[i])*(y1-yp[i])); temp4=fabs((x2-xp[i])*(y2-yp[i]));
        up[i]=u3[iu][ju]*temp1+u3[iu+1][ju]*temp2+u3[iu+1][ju+1]*temp3+u3[iu][ju+1]*temp4;
        up[i]=up[i]/dx/dy;
        //vp
        iv=(int)((xp[i]-0.5*dx)/dx); jv=(int)(yp[i]/dy);
        x1=iv*dx+0.5*dx; y1=jv*dy; x2=x1+dx; y2=y1; x3=x2; y3=y2+dy; x4=x1; y4=y3;
        temp1=fabs((x3-xp[i])*(y3-yp[i])); temp2=fabs((x4-xp[i])*(y4-yp[i]));
        temp3=fabs((x1-xp[i])*(y1-yp[i])); temp4=fabs((x2-xp[i])*(y2-yp[i]));
        vp[i]=v3[iv][jv]*temp1+v3[iv+1][jv]*temp2+v3[iv+1][jv+1]*temp3+v3[iv][jv+1]*temp4;
        vp[i]=vp[i]/dx/dy;
    }
}

void get_normal_vectors(void)
{
    int i,j; double b1,b2,b3;
    #pragma omp parallel for schedule(static) private(j,b1,b2,b3) num_threads(TN)
    for(i=1;i<l;i++)
	for(j=1;j<m;j++)
	{
		b1=(fin[i+1][j]-fin[i-1][j])*0.5/dx;
		b2=(fin[i][j+1]-fin[i][j-1])*0.5/dy;
		b3=sqrt(b1*b1+b2*b2)+1.0E-10;
		nor_vector_x[i][j]=b1/b3; nor_vector_y[i][j]=b2/b3;
	}
	//array_bnd(nor_vector_x); array_bnd(nor_vector_y);
}

void return_surface_all(void)
{
    int i,inor,jnor; double x1,y1,x2,y2,x3,y3,x4,y4,temp1,temp2,temp3,temp4,norx,nory,cp,distance;
    #pragma omp parallel for schedule(static) private(inor,jnor,x1,y1,x2,y2,x3,y3,x4,y4,temp1,temp2,temp3,temp4,norx,nory,cp,distance) num_threads(TN)
	for(i=1;i<=np;i++)
    {
        inor=(int)((xp[i]+0.5*dx)/dx); jnor=(int)((yp[i]+0.5*dy)/dy);
        x1=(inor-0.5)*dx; y1=(jnor-0.5)*dy; x2=x1+dx; y2=y1; x3=x2; y3=y2+dy; x4=x1; y4=y3;
        temp1=fabs((x3-xp[i])*(y3-yp[i])); temp2=fabs((x4-xp[i])*(y4-yp[i]));
        temp3=fabs((x1-xp[i])*(y1-yp[i])); temp4=fabs((x2-xp[i])*(y2-yp[i]));
        norx=nor_vector_x[inor][jnor]*temp1+nor_vector_x[inor+1][jnor]*temp2
            +nor_vector_x[inor+1][jnor+1]*temp3+nor_vector_x[inor][jnor+1]*temp4;
        norx=norx/dx/dy;
        nory=nor_vector_y[inor][jnor]*temp1+nor_vector_y[inor+1][jnor]*temp2
            +nor_vector_y[inor+1][jnor+1]*temp3+nor_vector_y[inor][jnor+1]*temp4;
        nory=nory/dx/dy;
        cp=fin[inor][jnor]*temp1+fin[inor+1][jnor]*temp2+fin[inor+1][jnor+1]*temp3+fin[inor][jnor+1]*temp4;
        cp=cp/dx/dy;
        if(cp>0.99||cp<0.01)
        {printf("ERROR: particle leave too far from interface, xp=%lf yp=%lf cp=%lf \n",xp[i],yp[i],cp);exit(0);}
        distance=2.0*1.41421356237*epn*log(sqrt(cp-cp*cp)/(1.0-cp));
        xp[i]=xp[i]-distance*norx; yp[i]=yp[i]-distance*nory;
    }
}

void return_surface_judge(void)
{
    int i,inor,jnor; double x1,y1,x2,y2,x3,y3,x4,y4,temp1,temp2,temp3,temp4,norx,nory,cp,distance;
    #pragma omp parallel for schedule(static) private(inor,jnor,x1,y1,x2,y2,x3,y3,x4,y4,temp1,temp2,temp3,temp4,norx,nory,cp,distance) num_threads(TN)
	for(i=1;i<=np;i++)
    {
        inor=(int)((xp[i]+0.5*dx)/dx); jnor=(int)((yp[i]+0.5*dy)/dy);
        x1=(inor-0.5)*dx; y1=(jnor-0.5)*dy; x2=x1+dx; y2=y1; x3=x2; y3=y2+dy; x4=x1; y4=y3;
        temp1=fabs((x3-xp[i])*(y3-yp[i])); temp2=fabs((x4-xp[i])*(y4-yp[i]));
        temp3=fabs((x1-xp[i])*(y1-yp[i])); temp4=fabs((x2-xp[i])*(y2-yp[i]));
        norx=nor_vector_x[inor][jnor]*temp1+nor_vector_x[inor+1][jnor]*temp2
            +nor_vector_x[inor+1][jnor+1]*temp3+nor_vector_x[inor][jnor+1]*temp4;
        norx=norx/dx/dy;
        nory=nor_vector_y[inor][jnor]*temp1+nor_vector_y[inor+1][jnor]*temp2
            +nor_vector_y[inor+1][jnor+1]*temp3+nor_vector_y[inor][jnor+1]*temp4;
        nory=nory/dx/dy;
        cp=fin[inor][jnor]*temp1+fin[inor+1][jnor]*temp2+fin[inor+1][jnor+1]*temp3+fin[inor][jnor+1]*temp4;
        cp=cp/dx/dy;
        if((cp>0.5&&up[i]*norx+vp[i]*nory>0.0)||(cp<0.5&&up[i]*norx+vp[i]*nory<0.0))
        {
            if(cp>0.995)
            { distance=3.5*dx; xp[i]=xp[i]-distance*norx; yp[i]=yp[i]-distance*nory; }
            else if(cp<0.005)
            { distance=3.5*dx; xp[i]=xp[i]+distance*norx; yp[i]=yp[i]+distance*nory; }
            else
            {
                distance=2.0*1.41421356237*epn*log(sqrt(cp-cp*cp)/(1.0-cp));
                xp[i]=xp[i]-distance*norx; yp[i]=yp[i]-distance*nory;
            }
        }
    }
}

void combine_part(int i)  //combine particle i and particle i+1
{
    int i_before,i_next,j; double surf_before,surf_i,surf_next,slope,ratio,left_new,right_new,xp_new,yp_new;
    if(i==1)i_before=np; else i_before=i-1;
    if(i==np)i_next=1; else i_next=i+1;
    surf_before=surf[i_before]/length2p[i_before]; surf_i=surf[i]/length2p[i]; surf_next=surf[i_next]/length2p[i_next];
    slope=(surf_next-surf_before)/(0.5*length2p[i_before]+0.5*length2p[i_next]+length2p[i]);
    ratio=(2.0*surf_i-0.5*length2p[i]*slope)/(2.0*surf_i+0.5*length2p[i]*slope);
    left_new=surf[i_before]+ratio/(ratio+1.0)*surf[i]; right_new=surf[i_next]+1.0/(ratio+1.0)*surf[i];
    xp_new=0.5*(xp[i]+xp[i_next]); yp_new=0.5*(yp[i]+yp[i_next]);
    if(i==np)
    {   xp[i_next]=xp_new; yp[i_next]=yp_new; surf[i_next]=right_new; length2p[i_next]=length2p[i_next]+0.5*length2p[i]; }
    else
    {
        xp[i]=xp_new; yp[i]=yp_new; surf[i]=right_new; length2p[i]=0.5*length2p[i]+length2p[i_next];
        for(j=i_next;j<np;j++)
        { xp[j]=xp[j+1]; yp[j]=yp[j+1]; surf[j]=surf[j+1]; length2p[j]=length2p[j+1]; }
    }
    surf[i_before]=left_new; length2p[i_before]=0.5*length2p[i]+length2p[i_before];
    np--;
}

void combine_part_deform(int i)  //combine particle i and particle i+1 because of deformation of particles-line
{
    int i_next,i_nextTwo,j; double surf_before,surf_next,surf_new,xp_new,yp_new,pdx,pdy,l_new;
    if(i==np)i_next=1; else i_next=i+1;
    if(i_next==np)i_nextTwo=1; else i_nextTwo=i_next+1;
    surf_new=surf[i_next]+surf[i]; xp_new=xp[i]; yp_new=yp[i];
    pdx=xp[i_nextTwo]-xp[i]; pdy=yp[i_nextTwo]-yp[i]; l_new=sqrt(pdx*pdx+pdy*pdy);
    if(i==np)
    {   xp[i_next]=xp_new; yp[i_next]=yp_new; surf[i_next]=surf_new; length2p[i_next]=l_new; }
    else
    {
        xp[i]=xp_new; yp[i]=yp_new; surf[i]=surf_new; length2p[i]=l_new;
        for(j=i_next;j<np;j++)
        { xp[j]=xp[j+1]; yp[j]=yp[j+1]; surf[j]=surf[j+1]; length2p[j]=length2p[j+1]; }
    }
    np--;
}

void add_part(int i) // add a new particle between particle i and particle i+1;
{
    int i_before,i_next,j; double surf_before,surf_i,surf_next,slope,ratio,left_new,right_new,xp_new,yp_new;
    if(i==1)i_before=np;else i_before=i-1;
    if(i==np)i_next=1; else i_next=i+1;
    surf_before=surf[i_before]/length2p[i_before]; surf_i=surf[i]/length2p[i]; surf_next=surf[i_next]/length2p[i_next];
    slope=(surf_next-surf_before)/(0.5*length2p[i_before]+0.5*length2p[i_next]+length2p[i]);
    ratio=(2.0*surf_i-0.5*length2p[i]*slope)/(2.0*surf_i+0.5*length2p[i]*slope);
    left_new=ratio/(ratio+1.0)*surf[i]; right_new=1.0/(ratio+1.0)*surf[i];
    xp_new=0.5*(xp[i]+xp[i_next]); yp_new=0.5*(yp[i]+yp[i_next]);
    if(i==np)
    {   xp[np+1]=xp_new; yp[np+1]=yp_new; surf[np+1]=right_new; length2p[np+1]=0.5*length2p[i]; }
    else
    {
        for(j=np+1;j>i_next;j--)
        { xp[j]=xp[j-1]; yp[j]=yp[j-1]; surf[j]=surf[j-1]; length2p[j]=length2p[j-1]; }
        xp[i_next]=xp_new; yp[i_next]=yp_new; surf[i_next]=right_new; length2p[i_next]=0.5*length2p[i];
    }
    surf[i]=left_new; length2p[i]=0.5*length2p[i];
    np++;
    if(np>=npar){printf("ERROR: Add particles np=%d exceed npar=%d \n",np,npar); exit(0);}
}

void check_part(void)
{
    int i,mark,next,nextTwo; double temp1,temp2;
    cal_length2p();
combine_aga:i=1;
    while(i<=np)
    {
        if(length2p[i]<=pdr_min)combine_part(i);
        i++;
    }
    if(mark==1){ mark=0; goto combine_aga;}
add_aga:i=1;
    cal_length2p();
    while(i<=np){ if(length2p[i]>=pdr_max){add_part(i); mark=1; /*printf("%d ",np);*/} i++; }
    if(mark==1){ mark=0; goto add_aga;}
}

void surf_advect(double mdt)  //mdt=dt/n;
{
    int i;
    get_part_vel();
    #pragma omp parallel for schedule(static) num_threads(TN)
    for(i=1;i<=np;i++){ xp[i]=xp[i]+up[i]*mdt; yp[i]=yp[i]+vp[i]*mdt; }
//    save_particle();
    return_surface_judge(); //must judge whether return_surface is needed !!!!!!!!!!!
    check_part();
//    check_part_2();
}

void get_curvature(void)
{
    int i,j;
    #pragma omp parallel for schedule(static) private(j) num_threads(TN)
	for(i=1;i<l;i++)
	for(j=1;j<m;j++)
	{ curv[i][j]=(nor_vector_x[i+1][j]-nor_vector_x[i-1][j])/dx/2.0+(nor_vector_y[i][j+1]-nor_vector_y[i][j-1])/dy/2.0; }
}

void get_surfaceTension_Lagrange(void) // get NSTp[npar][2] & MSTp[npar][2];=Normal & Marangoni SurfaceTension
{
    /*   P(k-1)                   P(k)                   P(k+1)      */
    /*   MSTp(k-1)   NSTp(k-1)    MSTp(k)     NSTp(k)    MSTp(k+1)   */
    int i_before,i,i_next,ip,jp;
    double temp1,temp2,temp3,temp4,x1,y1,x2,y2,x3,y3,x4,y4,norx,nory,curv_p,sigma_p,sigma_p_before,d_sigma;
    #pragma omp parallel for schedule(static) private(i_before,i_next,ip,jp,temp1,temp2,temp3,temp4,x1,\
    y1,x2,y2,x3,y3,x4,y4,norx,nory,curv_p,sigma_p,sigma_p_before,d_sigma) num_threads(TN)
    for(i=1;i<=np;i++)
    {
        ip=(int)((xp[i]+0.5*dx)/dx); jp=(int)((yp[i]+0.5*dy)/dy);
        x1=(ip-0.5)*dx; y1=(jp-0.5)*dy; x2=x1+dx; y2=y1; x3=x2; y3=y2+dy; x4=x1; y4=y3;
        temp1=fabs((x3-xp[i])*(y3-yp[i])); temp2=fabs((x4-xp[i])*(y4-yp[i]));
        temp3=fabs((x1-xp[i])*(y1-yp[i])); temp4=fabs((x2-xp[i])*(y2-yp[i]));
        norx=nor_vector_x[ip][jp]*temp1+nor_vector_x[ip+1][jp]*temp2+nor_vector_x[ip+1][jp+1]*temp3+nor_vector_x[ip][jp+1]*temp4;
        norx=norx/dx/dy;
        nory=nor_vector_y[ip][jp]*temp1+nor_vector_y[ip+1][jp]*temp2+nor_vector_y[ip+1][jp+1]*temp3+nor_vector_y[ip][jp+1]*temp4;
        nory=nory/dx/dy;
        curv_p=curv[ip][jp]*temp1+curv[ip+1][jp]*temp2+curv[ip+1][jp+1]*temp3+curv[ip][jp+1]*temp4;
        curv_p=curv_p/dx/dy;
        sigma_p=1.0+surf_elas*log(1.0-surf_cove*surf[i]/length2p[i]);
        NSTp[i][0]=-1.0*curv_p*sigma_p*norx; NSTp[i][1]=-1.0*curv_p*sigma_p*nory;

        if(i==1)i_before=np; else i_before=i-1; if(i==np)i_next=1; else i_next=i+1;
        sigma_p_before=1.0+surf_elas*log(1.0-surf_cove*surf[i_before]/length2p[i_before]);
        d_sigma=(sigma_p-sigma_p_before)/0.5/(length2p[i]+length2p[i_before]);
        temp1=xp[i_next]-xp[i_before]; temp2=yp[i_next]-yp[i_before];
        temp3=sqrt(temp1*temp1+temp2*temp2);
        MSTp[i][0]=d_sigma*temp1/temp3; MSTp[i][1]=d_sigma*temp2/temp3;
    }
}

void distribute_surfaceTension_direct(void) //distribute surf_p[npar][2] to surfx & surfy directly
{
    int i_before,i,j,i_next,i_nextTwo,ieuler,jeuler,ieuler_next,jeuler_next,i_start,i_end,j_start,j_end,ii,jj;
    double xeuler,yeuler,temp1,temp2,temp3,temp4,x_pro,y_pro,deltaF,temp,norx,nory;
    #pragma omp parallel for schedule(static) private(j) num_threads(TN)
    for(i=0;i<l-1;i++)
    for(j=0;j<m-1;j++)
    { surfx[i][j]=0.0; surfy[i][j]=0.0; }

    get_curvature();
    get_surfaceTension_Lagrange();
    /*   P(k-1)                   P(k)                   P(k+1)      */
    /*   MSTp(k-1)   NSTp(k-1)    MSTp(k)     NSTp(k)    MSTp(k+1)   */
    #pragma omp parallel for schedule(static) private(i_before,i_next,i_nextTwo,ieuler,jeuler,ieuler_next,temp,jeuler_next,\
    i_start,i_end,j_start,j_end,ii,jj,xeuler,yeuler,temp1,temp2,temp3,temp4,x_pro,y_pro,deltaF,norx,nory) num_threads(TN)
    for(i=1;i<=np;i++)
    {
        if(i==np)i_next=1; else i_next=i+1; if(i==1)i_before=np; else i_before=i-1;
        if(i_next==np)i_nextTwo=1; else i_nextTwo=i_next+1;
        ieuler=(int)((xp[i]+0.5*dx)/dx); jeuler=(int)((yp[i]+0.5*dy)/dy);
        ieuler_next=(int)((xp[i_next]+0.5*dx)/dx); jeuler_next=(int)((yp[i_next]+0.5*dy)/dy);
        if(ieuler>=ieuler_next){ i_start=ieuler_next; i_end=ieuler; }
        else { i_start=ieuler; i_end=ieuler_next; }
        if(jeuler>=jeuler_next){ j_start=jeuler_next; j_end=jeuler; }
        else { j_start=jeuler; j_end=jeuler_next;}
        i_start=i_start-5; i_end=i_end+5; j_start=j_start-5; j_end=j_end+5;

        //for surfx
        for(ii=i_start-1;ii<=i_end;ii++)
        for(jj=j_start-1;jj<=j_end-1;jj++)
        {
            temp=0.5*(fin[ii][jj+1]+fin[ii+1][jj+1]);
            if(temp>=0.001&&temp<=0.999)
            {
                temp1=xp[i_next]-xp[i]; temp2=yp[i_next]-yp[i];
                deltaF=3.0*1.414213562373095/epn*temp*temp*(1.0-temp)*(1.0-temp);
                norx=0.5*(nor_vector_x[ii][jj+1]+nor_vector_x[ii+1][jj+1]);
                nory=0.5*(nor_vector_y[ii][jj+1]+nor_vector_y[ii+1][jj+1]);
                if(fabs(temp1)<=fabs(temp2))  //use y_pro
                {
                    temp3=(ii*dx*nory-(jj+0.5)*dy*norx)*temp2-(xp[i]*temp2-yp[i]*temp1)*nory;
                    temp4=nory*temp1-norx*temp2;
                    if(fabs(temp4)>=0.2*length2p[i])
                    {
                        y_pro=temp3/temp4;
                        temp4=0.5*(yp[i]+yp[i_next]);
                        //if(y_pro>=yp[i]&&y_pro<y)
                        if(y_pro-yp[i]==0.0||(y_pro-yp[i])*(y_pro-temp4)<0.0)
                        {
                            temp1=0.5*(yp[i_before]+yp[i]); temp2=0.5*(yp[i_next]+yp[i]);
                            surfx[ii][jj]=surfx[ii][jj]+deltaF*((y_pro-temp1)*NSTp[i][0]-(y_pro-temp2)*NSTp[i_before][0])/(temp2-temp1);
                            surfx[ii][jj]=surfx[ii][jj]+deltaF*((y_pro-yp[i])*MSTp[i_next][0]-(y_pro-yp[i_next])*MSTp[i][0])/(yp[i_next]-yp[i]);
                        }
                        else if(y_pro-temp4==0.0||(y_pro-temp4)*(y_pro-yp[i_next])<0.0)
                        {
                            temp1=0.5*(yp[i_next]+yp[i_nextTwo]); temp2=0.5*(yp[i_next]+yp[i]);
                            surfx[ii][jj]=surfx[ii][jj]+deltaF*((y_pro-temp1)*NSTp[i][0]-(y_pro-temp2)*NSTp[i_next][0])/(temp2-temp1);
                            surfx[ii][jj]=surfx[ii][jj]+deltaF*((y_pro-yp[i])*MSTp[i_next][0]-(y_pro-yp[i_next])*MSTp[i][0])/(yp[i_next]-yp[i]);
                        }
                        else ;
                    }
                }
                else  //use x_pro
                {
                    temp3=(ii*dx*nory-(jj+0.5)*dy*norx)*temp1-(xp[i]*temp2-yp[i]*temp1)*norx;
                    temp4=nory*temp1-norx*temp2;
                    if(fabs(temp4)>=0.2*length2p[i])
                    {
                        x_pro=temp3/temp4;
                        temp4=0.5*(xp[i]+xp[i_next]);
                        if(x_pro-xp[i]==0.0||(x_pro-xp[i])*(x_pro-temp4)<0.0)
                        {
                            temp1=0.5*(xp[i_before]+xp[i]); temp2=0.5*(xp[i_next]+xp[i]);
                            surfx[ii][jj]=surfx[ii][jj]+deltaF*((x_pro-temp1)*NSTp[i][0]-(x_pro-temp2)*NSTp[i_before][0])/(temp2-temp1);
                            surfx[ii][jj]=surfx[ii][jj]+deltaF*((x_pro-xp[i])*MSTp[i_next][0]-(x_pro-xp[i_next])*MSTp[i][0])/(xp[i_next]-xp[i]);
                        }
                        else if(x_pro-temp4==0.0||(x_pro-temp4)*(x_pro-xp[i_next])<0.0)
                        {
                            temp1=0.5*(xp[i_next]+xp[i_nextTwo]); temp2=0.5*(xp[i_next]+xp[i]);
                            surfx[ii][jj]=surfx[ii][jj]+deltaF*((x_pro-temp1)*NSTp[i][0]-(x_pro-temp2)*NSTp[i_next][0])/(temp2-temp1);
                            surfx[ii][jj]=surfx[ii][jj]+deltaF*((x_pro-xp[i])*MSTp[i_next][0]-(x_pro-xp[i_next])*MSTp[i][0])/(xp[i_next]-xp[i]);
                        }
                        else ;
                    }
                }
            }
        }
        // for surfy
        for(ii=i_start-1;ii<=i_end-1;ii++)
        for(jj=j_start-1;jj<=j_end;jj++)
        {
            temp=0.5*(fin[ii+1][jj]+fin[ii+1][jj+1]);
            if(temp>=0.001&&temp<=0.999)
            {
                temp1=xp[i_next]-xp[i]; temp2=yp[i_next]-yp[i];
                deltaF=3.0*1.414213562373095/epn*temp*temp*(1.0-temp)*(1.0-temp);
                norx=0.5*(nor_vector_x[ii+1][jj]+nor_vector_x[ii+1][jj+1]);
                nory=0.5*(nor_vector_y[ii+1][jj]+nor_vector_y[ii+1][jj+1]);
                if(fabs(temp1)<=fabs(temp2))  //use y_pro
                {
                    temp3=((ii+0.5)*dx*nory-jj*dy*norx)*temp2-(xp[i]*temp2-yp[i]*temp1)*nory;
                    temp4=nory*temp1-norx*temp2;
                    if(fabs(temp4)>=0.2*length2p[i])
                    {
                        y_pro=temp3/temp4;
                        temp4=0.5*(yp[i]+yp[i_next]);
                        //if(y_pro>=yp[i]&&y_pro<y)
                        if(y_pro-yp[i]==0.0||(y_pro-yp[i])*(y_pro-temp4)<0.0)
                        {
                            temp1=0.5*(yp[i_before]+yp[i]); temp2=0.5*(yp[i_next]+yp[i]);
                            surfy[ii][jj]=surfy[ii][jj]+deltaF*((y_pro-temp1)*NSTp[i][1]-(y_pro-temp2)*NSTp[i_before][1])/(temp2-temp1);
                            surfy[ii][jj]=surfy[ii][jj]+deltaF*((y_pro-yp[i])*MSTp[i_next][1]-(y_pro-yp[i_next])*MSTp[i][1])/(yp[i_next]-yp[i]);
                        }
                        else if(y_pro-temp4==0.0||(y_pro-temp4)*(y_pro-yp[i_next])<0.0)
                        {
                            temp1=0.5*(yp[i_next]+yp[i_nextTwo]); temp2=0.5*(yp[i_next]+yp[i]);
                            surfy[ii][jj]=surfy[ii][jj]+deltaF*((y_pro-temp1)*NSTp[i][1]-(y_pro-temp2)*NSTp[i_next][1])/(temp2-temp1);
                            surfy[ii][jj]=surfy[ii][jj]+deltaF*((y_pro-yp[i])*MSTp[i_next][1]-(y_pro-yp[i_next])*MSTp[i][1])/(yp[i_next]-yp[i]);
                        }
                        else ;
                    }
                }
                else  //use x_pro
                {
                    temp3=((ii+0.5)*dx*nory-jj*dy*norx)*temp1-(xp[i]*temp2-yp[i]*temp1)*norx;
                    temp4=nory*temp1-norx*temp2;
                    if(fabs(temp4)>=0.2*length2p[i])
                    {
                        x_pro=temp3/temp4;
                        temp4=0.5*(xp[i]+xp[i_next]);
                        if(x_pro-xp[i]==0.0||(x_pro-xp[i])*(x_pro-temp4)<0.0)
                        {
                            temp1=0.5*(xp[i_before]+xp[i]); temp2=0.5*(xp[i_next]+xp[i]);
                            surfy[ii][jj]=surfy[ii][jj]+deltaF*((x_pro-temp1)*NSTp[i][1]-(x_pro-temp2)*NSTp[i_before][1])/(temp2-temp1);
                            surfy[ii][jj]=surfy[ii][jj]+deltaF*((x_pro-xp[i])*MSTp[i_next][1]-(x_pro-xp[i_next])*MSTp[i][1])/(xp[i_next]-xp[i]);
                        }
                        else if(x_pro-temp4==0.0||(x_pro-temp4)*(x_pro-xp[i_next])<0.0)
                        {
                            temp1=0.5*(xp[i_next]+xp[i_nextTwo]); temp2=0.5*(xp[i_next]+xp[i]);
                            surfy[ii][jj]=surfy[ii][jj]+deltaF*((x_pro-temp1)*NSTp[i][1]-(x_pro-temp2)*NSTp[i_next][1])/(temp2-temp1);
                            surfy[ii][jj]=surfy[ii][jj]+deltaF*((x_pro-xp[i])*MSTp[i_next][1]-(x_pro-xp[i_next])*MSTp[i][1])/(xp[i_next]-xp[i]);
                        }
                        else ;
                    }
                }
            }
        }
    }
}

void distribute_surfactant(void)
{
    int i_before,i,i_next,i_nextTwo,j,ieuler,jeuler,ieuler_next,jeuler_next,i_start,i_end,j_start,j_end,ii,jj;
    double xeuler,yeuler,temp1,temp2,temp3,temp4,x_pro,y_pro;
    #pragma omp parallel for schedule(static) private(j) num_threads(TN)
    for(i=1;i<l;i++)
    for(j=1;j<m;j++)
        surfactant[i][j]=0.0;

    #pragma omp parallel for schedule(static) private(i_before,i_next,i_nextTwo,ieuler,jeuler,ieuler_next,\
    jeuler_next,i_start,i_end,j_start,j_end,ii,jj,xeuler,yeuler,temp1,temp2,temp3,temp4,x_pro,y_pro) num_threads(TN)
    for(i=1;i<=np;i++)
    {
        if(i==np)i_next=1; else i_next=i+1; if(i==1)i_before=np; else i_before=i-1;
        if(i_next==np)i_nextTwo=1; else i_nextTwo=i_next+1;
        ieuler=(int)((xp[i]+0.5*dx)/dx); jeuler=(int)((yp[i]+0.5*dy)/dy);
        ieuler_next=(int)((xp[i_next]+0.5*dx)/dx); jeuler_next=(int)((yp[i_next]+0.5*dy)/dy);
        if(ieuler>=ieuler_next){ i_start=ieuler_next; i_end=ieuler; }
        else { i_start=ieuler; i_end=ieuler_next; }
        if(jeuler>=jeuler_next){ j_start=jeuler_next; j_end=jeuler; }
        else { j_start=jeuler; j_end=jeuler_next;}
        i_start=i_start-4; i_end=i_end+4; j_start=j_start-4; j_end=j_end+4;

        for(ii=i_start;ii<=i_end;ii++)
        for(jj=j_start;jj<=j_end;jj++)
        {
            if(fin[ii][jj]>=0.005&&fin[ii][jj]<=0.995)
            {
                temp1=xp[i_next]-xp[i]; // delta x
                temp2=yp[i_next]-yp[i]; // delta y
                if(fabs(temp1)<=fabs(temp2))  //use y_pro
                {
                    temp3=((ii-0.5)*dx*nor_vector_y[ii][jj]-(jj-0.5)*dy*nor_vector_x[ii][jj])*temp2
                        -(xp[i]*temp2-yp[i]*temp1)*nor_vector_y[ii][jj];
                    temp4=nor_vector_y[ii][jj]*temp1-nor_vector_x[ii][jj]*temp2;
                    if(fabs(temp4)>=0.2*length2p[i])
                    {
                        y_pro=temp3/temp4;
                        temp4=0.5*(yp[i]+yp[i_next]);
                        if((y_pro-yp[i])*(y_pro-temp4)<=0.0)
                        {
                            temp1=0.5*(yp[i_before]+yp[i]); temp2=0.5*(yp[i_next]+yp[i]);
                            surfactant[ii][jj]=((y_pro-temp1)*surf[i]/length2p[i]-
                                                (y_pro-temp2)*surf[i_before]/length2p[i_before])/(temp2-temp1);
                        }
                        else if((y_pro-temp4)*(y_pro-yp[i_next])<=0.0)
                        {
                            temp1=0.5*(yp[i_next]+yp[i_nextTwo]); temp2=0.5*(yp[i_next]+yp[i]);
                            surfactant[ii][jj]=((y_pro-temp1)*surf[i]/length2p[i]-
                                                (y_pro-temp2)*surf[i_next]/length2p[i_next])/(temp2-temp1);
                        }
                        else ;
                    }
                }
                else  //use x_pro
                {
                    temp3=((ii-0.5)*dx*nor_vector_y[ii][jj]-(jj-0.5)*dy*nor_vector_x[ii][jj])*temp1
                        -(xp[i]*temp2-yp[i]*temp1)*nor_vector_x[ii][jj];
                    temp4=nor_vector_y[ii][jj]*temp1-nor_vector_x[ii][jj]*temp2;
                    if(fabs(temp4)>=0.2*length2p[i])
                    {
                        x_pro=temp3/temp4;
                        temp4=0.5*(xp[i]+xp[i_next]);
                        if((x_pro-xp[i])*(x_pro-temp4)<=0.0)
                        {
                            temp1=0.5*(xp[i_before]+xp[i]); temp2=0.5*(xp[i_next]+xp[i]);
                            surfactant[ii][jj]=((x_pro-temp1)*surf[i_before]/length2p[i_before]
                                                -(x_pro-temp2)*surf[i]/length2p[i])/(temp2-temp1);
                        }
                        else if((x_pro-temp4)*(x_pro-xp[i_next])<=0.0)
                        {
                            temp1=0.5*(xp[i_next]+xp[i_nextTwo]); temp2=0.5*(xp[i_next]+xp[i]);
                            surfactant[ii][jj]=((x_pro-temp1)*surf[i_next]/length2p[i_next]
                                                -(x_pro-temp2)*surf[i]/length2p[i])/(temp2-temp1);
                        }
                        else ;
                    }
                }
            }
        }
    }
}

void particle_surfactant_transportation(void)
{
    int n=8; double mdt=dt/n;
    get_normal_vectors();
    while(n>0)
    {
        surf_advect(mdt);
        surf_diffuse(mdt);
//        output_particle(surf,"part_diffuse",n);
        n--;
    }
    return_surface_all();
    distribute_surfaceTension_direct();
    //distribute_surfaceTension();
    //distribute_surfactant();
}

