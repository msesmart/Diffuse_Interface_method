
#define npar 16000
double xp[npar],yp[npar],up[npar],vp[npar],surf[npar],length2p[npar],surf_new[npar],d_surf[npar],surf_p[npar][2];


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

void particle_surfactant_initialization(void)
{
    double temp1,temp2,angle,deltaAngle,drx,dry,dr,R;
    np=0; R=rad; deltaAngle=asin(0.8*pdr/R); angle=0.0;
    while(angle<2.0*PI)
    {
        np++;
        if(np>npar-1){printf("ERROR: np=%d exceed the npar",np);exit(0);}
        xp[np]=cos(angle)*R+xc;
        yp[np]=sin(angle)*R+yc;
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

    while(n>0)
    {
        #pragma omp parallel for schedule(static) private(next,temp1,temp2) num_threads(TN)
        for(i=1;i<=np;i++)
        {
            if(i==np)next=1; else next=i+1;
            temp1=surf_new[i]/length2p[i];
            temp2=surf_new[next]/length2p[next];
            d_surf[next]=2.0*(temp2-temp1)/(length2p[i]+length2p[next]);
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
    int i=1,mark;
    cal_length2p();
    while(i<=np){ if(length2p[i]<=pdr_min)combine_part(i); i++; }
aga:i=1;
    cal_length2p();
    while(i<=np){ if(length2p[i]>=pdr_max){add_part(i); mark=1; /*printf("%d ",np);*/} i++; }
    if(mark==1){ mark=0; goto aga;}
}

void check_part_2(void)
{
    int i=1;
    cal_length2p();
    for(i=1;i<=np;i++)
    {
        if(length2p[i]<=pdr_min||length2p[i]>=pdr_max){printf("ERROR: length2p %d %lf \n",i,length2p[i]); exit(0);}
    }
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

void particle_surfactant_transportation(void)
{
    int n=8; double mdt=dt/n;
    get_normal_vectors();
    while(n>0)
    {
        surf_advect(mdt);
//        surf_diffuse(mdt);
        n--;
    }
    return_surface_all();
//    distribute_surfactant();
    //distribute_surfaceTension();
}

