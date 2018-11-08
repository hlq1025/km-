#include "km.h"

using namespace std;

AssignmentProblemSolver::AssignmentProblemSolver(int nx,int ny,double **w)
{
	this->nx = nx;
	
	this->ny = ny;
	
	visx = new int [nx];

	visy=new int [ny];

	link=new int [nx];

	lx=new double[nx];

	ly=new double[ny];

	slack=new double[ny];
	cost = new double[nx*ny];


	for (int i = 0; i < nx;i++)
	for (int j = 0; j < ny; j++)
		cost[i*ny+j] = w[i][j];
	GpuInitial();

}

AssignmentProblemSolver::~AssignmentProblemSolver()
{
}


double AssignmentProblemSolver::solve()
{
	double start = static_cast<double>(cvGetTickCount());
	errNum = clEnqueueWriteBuffer(commandQueue, memObjects[0], CL_TRUE,

		0, nx *sizeof(int), link,

		0, NULL, NULL);

	errNum |= clEnqueueWriteBuffer(commandQueue, memObjects[1], CL_TRUE,



		0, nx * sizeof(double), lx,



		0, NULL, NULL);

	errNum |= clEnqueueWriteBuffer(commandQueue, memObjects[2], CL_TRUE,



		0, ny* sizeof(double), ly,



		0, NULL, NULL);

	errNum |= clEnqueueWriteBuffer(commandQueue, memObjects[3], CL_TRUE,



		0, nx*ny* sizeof(double), cost,



		0, NULL, NULL);

	errNum |= clEnqueueNDRangeKernel(commandQueue, kernel1, 1, NULL,



		globalWorkSize, NULL,



		0, NULL, NULL);

	errNum |= clEnqueueReadBuffer(commandQueue, memObjects[1], CL_TRUE,



		0, nx * sizeof(double), lx,



		0, NULL, NULL);

	errNum |= clEnqueueReadBuffer(commandQueue, memObjects[0], CL_TRUE,



		0, nx * sizeof(int), link,



		0, NULL, NULL);
	errNum |= clEnqueueReadBuffer(commandQueue, memObjects[2], CL_TRUE,



		0, ny * sizeof(double), ly,



		0, NULL, NULL);

	/*for (int i = 0; i < nx; i++)
	{
		cout << link[i] << endl;
	}*/
	
	bool flag = true;
	while (flag)
	{
		flag = false;
		for (int i = 0; i < nx; i++)
		{
			for (int j = i + 1; j < nx; j++)
			{
				if (link[i] == link[j])
				{
					
					
					double h = DBL_MAX;
					int substitution = -1;
					for (int p = 0; p < ny; p++)
					{
						if (p == link[i]) continue;
						double slack1 = lx[i] + ly[p] - cost[i*ny + p];
						double slack2 = lx[j] + ly[p] - cost[j*ny + p];
						if (slack1 > 0 && slack1<h)
						{
							h = slack1;
							substitution = p;
						}//ע��˴�Ҫ��Ϊ����0
						if (slack2 > 0 && slack2 < h)
						{
							h = slack2;
							substitution = p;
						}
					}
					lx[i] -= h;
					lx[j] -= h;
					ly[link[i]] += h;
					if (lx[i] + ly[substitution] == cost[i*ny + substitution])
						link[i] = substitution;
					if (lx[j] + ly[substitution] == cost[j*ny + substitution])
						link[j] = substitution;
					flag = true;
					break;
			}
		}
			if (flag)  break;
	}
	}

	double res = 0;
	for (int i = 0; i < nx; i++)
	{
		//cout << lx[i] << endl;
		res = res + cost[i*ny + link[i]];
	}
	/*for (int i = 0; i < nx; i++)
	{
		cout << link[i] << endl;
	}*/
	double time = ((double)cvGetTickCount() - start) / cvGetTickFrequency();
	cout << "������ʱ��Ϊ:" << time << "us" << endl;
	return res;
}
cl_int AssignmentProblemSolver::ConvertToString(const char *pFileName, std::string &Str)

{

	size_t		uiSize = 0;

	size_t		uiFileSize = 0;

	char		*pStr = NULL;

	std::fstream fFile(pFileName, (std::fstream::in | std::fstream::binary));

	if (fFile.is_open())

	{

		fFile.seekg(0, std::fstream::end);

		uiSize = uiFileSize = (size_t)fFile.tellg();  // ����ļ���С

		fFile.seekg(0, std::fstream::beg);

		pStr = new char[uiSize + 1];

		if (NULL == pStr)

		{

			fFile.close();

			return 0;

		}

		fFile.read(pStr, uiFileSize);				// ��ȡuiFileSize�ֽ�

		fFile.close();

		pStr[uiSize] = '\0';

		Str = pStr;

		delete[] pStr;

		return 0;

}

//cout << "Error: Failed to open cl file\n:" << pFileName << endl;

return -1;

}
cl_context AssignmentProblemSolver::CreateContext()
{
	cl_int errNum;

	cl_uint numPlatforms;

	cl_platform_id firstPlatformId;

	cl_context context = NULL;



	//ѡ����õ�ƽ̨�еĵ�һ��

	errNum = clGetPlatformIDs(1, &firstPlatformId, &numPlatforms);

	if (errNum != CL_SUCCESS || numPlatforms <= 0)

	{

		std::cerr << "Failed to find any OpenCL platforms." << std::endl;

		return NULL;

	}

	//����һ��OpenCL�����Ļ���

	cl_context_properties contextProperties[] =

	{

		CL_CONTEXT_PLATFORM,

		(cl_context_properties)firstPlatformId,

		0

	};

	context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_GPU,

		NULL, NULL, &errNum);



	return context;

}
cl_command_queue AssignmentProblemSolver::CreateCommandQueue(cl_context context, cl_device_id *device)

{

	cl_int errNum;

	cl_device_id *devices;

	cl_command_queue commandQueue = NULL;

	size_t deviceBufferSize = -1;

	// ��ȡ�豸��������С

	errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &deviceBufferSize);

	if (deviceBufferSize <= 0)

	{

		std::cerr << "No devices available.";

		return NULL;

	}


	// Ϊ�豸���仺��ռ�

	devices = new cl_device_id[deviceBufferSize / sizeof(cl_device_id)];

	errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, deviceBufferSize, devices, NULL);

	//ѡȡ�����豸�еĵ�һ��

	commandQueue = clCreateCommandQueue(context, devices[0], 0, NULL);

	*device = devices[0];

	delete[] devices;

	return commandQueue;

}
cl_program AssignmentProblemSolver::CreateProgram(cl_context context, cl_device_id device, const char* fileName)

{

	cl_int errNum;

	cl_program program;

	std::ifstream kernelFile(fileName, std::ios::in);

	if (!kernelFile.is_open())

	{

		std::cerr << "Failed to open file for reading: " << fileName << std::endl;

		return NULL;

	}

	std::ostringstream oss;

	oss << kernelFile.rdbuf();

	std::string srcStdStr = oss.str();

	const char *srcStr = srcStdStr.c_str();

	program = clCreateProgramWithSource(context, 1,

		(const char**)&srcStr,

		NULL, NULL);

	errNum = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);

	return program;

}
bool AssignmentProblemSolver::CreateMemObjects(cl_context context, cl_mem memObjects[4])

{
	// ���������ڴ����
	memObjects[0] = clCreateBuffer(context,

		CL_MEM_READ_WRITE,  // �����ڴ�Ϊֻ���������Դ��������ڴ渴�Ƶ��豸�ڴ�

		ny* sizeof(int),		  // �����ڴ�ռ��С

		NULL,

		NULL);

	memObjects[1] = clCreateBuffer(context,

		CL_MEM_READ_WRITE,  // �����ڴ�Ϊֻ���������Դ��������ڴ渴�Ƶ��豸�ڴ�

		nx * sizeof(double),		  // �����ڴ�ռ��С

		NULL,

		NULL);

	memObjects[2] = clCreateBuffer(context,

		CL_MEM_READ_WRITE,  // �����ڴ�Ϊֻ���������Դ��������ڴ渴�Ƶ��豸�ڴ�

		ny * sizeof(double),		  // �����ڴ�ռ��С

		NULL,

		NULL);

	memObjects[3] = clCreateBuffer(context,

		CL_MEM_READ_WRITE,  // �����ڴ�Ϊֻ���������Դ��������ڴ渴�Ƶ��豸�ڴ�

		nx*ny * sizeof(double),		  // �����ڴ�ռ��С

		NULL,

		NULL);

	
		

	if ((NULL == memObjects[0]) || (NULL == memObjects[1]) || (NULL == memObjects[2]) || NULL == memObjects[3] )
	{
		cout << "Error creating memory objects" << endl;

		return false;
	}
	return true;
	
}
void Cleanup(cl_context context, cl_command_queue commandQueue,

	cl_program program, cl_kernel kernel, cl_mem memObjects[4])

{

	for (int i = 0; i < 4; i++)

	{

		if (memObjects[i] != 0)

			clReleaseMemObject(memObjects[i]);

	}

	if (commandQueue != 0)

		clReleaseCommandQueue(commandQueue);



	if (kernel != 0)

		clReleaseKernel(kernel);



	if (program != 0)

		clReleaseProgram(program);



	if (context != 0)

		clReleaseContext(context);
	return;
}
void AssignmentProblemSolver::GpuInitial()
{
	// һ��ѡ��OpenCLƽ̨������һ��������

	context = CreateContext();

	// ���� �����豸�������������

	commandQueue = CreateCommandQueue(context, &device);

	CreateMemObjects(context, memObjects);

	//���������͹����������

	program = CreateProgram(context, device, "kernel.cl");


	kernel1 = clCreateKernel(program, "arrangement", NULL);


	// �ġ� ����OpenCL�ں˲������ڴ�ռ�
	

	errNum = clSetKernelArg(kernel1, 0, sizeof(cl_mem), (void *)(&nx));

	errNum |= clSetKernelArg(kernel1, 1, sizeof(cl_mem), (void *)(&ny));

	errNum |= clSetKernelArg(kernel1, 2, sizeof(cl_mem), (void *)&memObjects[0]);

	errNum |= clSetKernelArg(kernel1, 3, sizeof(cl_mem), (void *)&memObjects[1]);

	errNum |= clSetKernelArg(kernel1, 4, sizeof(cl_mem), (void *)&memObjects[2]);

	errNum |= clSetKernelArg(kernel1, 5, sizeof(cl_mem), (void *)&memObjects[3]);

	if (CL_SUCCESS != errNum)
	{
		cout << "Error setting kernel arguments" << endl;
	}
	// --------------------------10.�����ں�---------------------------------
	globalWorkSize[0] = max(nx,ny);
}
/*
// --------------------------------------------------------------------------
// Usage example
// --------------------------------------------------------------------------
void main(void)
{
// Matrix size
int N=8; // tracks
int M=9; // detects
// Random numbers generator initialization
srand (time(NULL));
// Distance matrix N-th track to M-th detect.
vector< vector<double> > Cost(N,vector<double>(M));
// Fill matrix with random values
for(int i=0; i<N; i++)
{
for(int j=0; j<M; j++)
{
Cost[i][j] = (double)(rand()%1000)/1000.0;
std::cout << Cost[i][j] << "\t";
}
std::cout << std::endl;
}

AssignmentProblemSolver APS;

vector<int> Assignment;

cout << APS.Solve(Cost,Assignment) << endl;

// Output the result
for(int x=0; x<N; x++)
{
std::cout << x << ":" << Assignment[x] << "\t";
}

getchar();
}
*/
// --------------------------------------------------------------------------