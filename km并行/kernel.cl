#pragma OPENCL EXTENSION cl_amd_printf : enable
__kernel void arrangement(int nx,int ny,__global int *link,__global double *lx,__global double *ly,__global double*cost)
{
int i=get_global_id(0);
if(i<ny)
{
	ly[i]=0;
}
barrier(CLK_GLOBAL_MEM_FENCE);
if(i<nx)
{link[i]=-1;
for (int j = 0; j < ny; j++)
{	if(j==0) lx[i]=-DBL_MAX;
	if (cost[i*ny+j] > lx[i])
	{lx[i] = cost[i*ny+j];
	 link[i]=j;
	}
}
}
}

