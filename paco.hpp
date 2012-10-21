// TODO: symmterize rhs in order to get machine precision energy conservation
// verify basis functions - why do asymmetries appear?

#include <iostream>
#include <vector>
#include <list>
#include <math.h>
#include <assert.h>
#include <stdio.h>

using namespace std;

#include "quad.hpp"

#include "_hypre_utilities.h"
#include "HYPRE.h"
#include "HYPRE_IJ_mv.h"
#include "HYPRE_seq_mv.h" 
#include "HYPRE_IJ_mv.h"
#include "HYPRE_krylov.h"
#include "HYPRE_parcsr_ls.h"

#define XMIN -5.0
#define XMAX  5.0
#define YMIN -5.0
#define YMAX  5.0

#define XM_BIT 1
#define XP_BIT 2
#define YM_BIT 4
#define YP_BIT 8

#define FOR_IP for (int ip = 0; ip < np; ++ip)
#define FOR_IT for (int it = 0; it < triVec.size(); ++it)

#define FOR_I2 for (int i = 0; i < 2; ++i)
#define FOR_I3 for (int i = 0; i < 3; ++i)
#define FOR_J2 for (int j = 0; j < 2; ++j)
#define FOR_J3 for (int j = 0; j < 3; ++j)
#define FOR_IVE for (int ive = 0; ive < 3; ++ive) 
#define FOR_IQ for (int iq = 0; iq < NQUAD; ++iq) 

extern "C" {
  void dgesv_(int* n, int* nrhs, double* A, int* lda, int* ipiv,
              double* b, int* ldb, int* info );
}

struct Basis {
  int index;
  int ip;
  double phi;
  double grad_phi[2];
};

template <int NQUAD >
class Paco { 
public:  
  class Tri {
  public: 
    int ve_ip[3];
    int ve_bit[3];
    int ve_itnbr[3];
    int ve_ivenbr[3];
    double ve_x[3][2];
    
    double area;
    
    list<Basis> basisList[NQUAD];
    
    Tri() {}
    
    void calcArea() {
      area = 0.5*((ve_x[1][0]-ve_x[0][0])*(ve_x[2][1]-ve_x[0][1])-(ve_x[1][1]-ve_x[0][1])*(ve_x[2][0]-ve_x[0][0]));
      assert(area > 0.0);
    }
    
    void updateVeX(double (*xp)[2]) {
      FOR_IVE {
	if (ve_bit[ive] & XM_BIT) ve_x[ive][0] = xp[ve_ip[ive]][0] - (XMAX - XMIN);
	else if (ve_bit[ive] & XP_BIT) ve_x[ive][0] = xp[ve_ip[ive]][0] + (XMAX - XMIN);
	else ve_x[ive][0] = xp[ve_ip[ive]][0];
	
	if (ve_bit[ive] & YM_BIT) ve_x[ive][1] = xp[ve_ip[ive]][1] - (YMAX - YMIN);
	else if (ve_bit[ive] & YP_BIT) ve_x[ive][1] = xp[ve_ip[ive]][1] + (YMAX - YMIN);
	else ve_x[ive][1] = xp[ve_ip[ive]][1];
      }
    }
    
    void toggleXMBit(const int ive) {
      if (ve_bit[ive] & XP_BIT) 
	ve_bit[ive] &= ~XP_BIT;
      else {
	assert( ~(ve_bit[ive] & XM_BIT) );
	ve_bit[ive] |= XM_BIT;
      }
    }
    void toggleXPBit(const int ive) {
      if (ve_bit[ive] & XM_BIT) 
	ve_bit[ive] &= ~XM_BIT;
      else {
	assert( ~(ve_bit[ive] & XP_BIT) );
	ve_bit[ive] |= XP_BIT;
      }
    }
    void toggleYMBit(const int ive) {
      if (ve_bit[ive] & YP_BIT) 
	ve_bit[ive] &= ~YP_BIT;
      else {
	assert( ~(ve_bit[ive] & YM_BIT) );
	ve_bit[ive] |= YM_BIT;
      }
    }
    void toggleYPBit(const int ive) {
      if (ve_bit[ive] & YM_BIT) 
	ve_bit[ive] &= ~YM_BIT;
      else {
	assert( ~(ve_bit[ive] & YP_BIT) );
	ve_bit[ive] |= YP_BIT;
      }
    }
    
    void resetBits() {      
      // if all bits are set, then triangle has completely migrated across periodic boundary...
      if ((ve_bit[0] & XM_BIT) && (ve_bit[1] & XM_BIT) && (ve_bit[2] & XM_BIT)) 
	FOR_IVE ve_bit[ive] &= ~XM_BIT;
      else if ((ve_bit[0] & XP_BIT) && (ve_bit[1] & XP_BIT) && (ve_bit[2] & XP_BIT)) 
	FOR_IVE ve_bit[ive] &= ~XP_BIT;
      if ((ve_bit[0] & YM_BIT) && (ve_bit[1] & YM_BIT) && (ve_bit[2] & YM_BIT)) 
	FOR_IVE ve_bit[ive] &= ~YM_BIT;
      else if ((ve_bit[0] & YP_BIT) && (ve_bit[1] & YP_BIT) && (ve_bit[2] & YP_BIT)) 
	FOR_IVE ve_bit[ive] &= ~YP_BIT;
    }
    
  };
  
  Quad<NQUAD> quad;
  vector<Tri> triVec;
  
  struct Point {
    double x[2];
    int ip;
    double tmp;
    double tmp2[2];
    double tmptmp;
  };

  vector<list <Point> > cells;  
  int ncells;

  int np;
  double (*xp)[2];
  double *Mp;
  double *rhop;
  double *pp;
  double (*Gp)[2];
  double *Ep;
  double (*up)[2];
  double *volp;
  double (*lambdap)[2];
  double *denomp;
  int *ip_flag;

  double Et_init;
 
  double *rho_error;
  double (*u_error)[2];
  double *p_error;

  double gam;
  double mu;

  int step;
  int nsteps;
  double time;
  double dt;
  int check_interval;
  int write_interval;

  int * nbopa_i;
  int * nbopa_v;
  int nbopa_s;

  double * M;
      
  MPI_Comm mpi_comm;
  HYPRE_IJMatrix Aij;
  HYPRE_IJVector bij;
  HYPRE_IJVector xij;
    
  HYPRE_CSRMatrix A;
  HYPRE_Vector x;
  HYPRE_Vector b;

  HYPRE_Solver hypre_solver;

  int nrows;
  int *ncols; 
                 
  int *rows;
  int *cols;
  
  Paco() {
    cout << "Paco()" << endl;
    
    Mp        = NULL;
    rhop      = NULL;
    Gp        = NULL;
    Ep        = NULL;
    up        = NULL;
    volp      = NULL;
    ip_flag   = NULL;
    
    nbopa_i = NULL;
    nbopa_v = NULL;
    M = NULL;
   
    ncols = NULL;
    rows = NULL;
    cols = NULL;
  }

  virtual ~Paco() {
    
    if (Mp != NULL)      delete[] Mp;
    if (rhop != NULL)    delete[] rhop;
    if (Gp != NULL)      delete[] Gp;
    if (Ep != NULL)      delete[] Ep;
    if (up != NULL)      delete[] up;
    if (volp != NULL)    delete[] volp;
    if (ip_flag != NULL) delete[] ip_flag;
   
    if (nbopa_i != NULL) delete[] nbopa_i;
    if (nbopa_v != NULL) delete[] nbopa_v;
    if (M != NULL) delete[] M;
       
  }

  virtual void initialHook() {
    cout << "initialHook()" << endl;
  
  }
  
  virtual void temporalHook() {
    cout << "temporalHook()" << endl;
  }

  virtual void finalHook() {
    cout << "finalHook()" << endl;
  }

  inline double pEOS(double _rho) {
    return(pow(_rho,gam));
  }

  void setDt() {
    dt = 0.1;
  }

  void initLinearSolver() {
    
    nrows = np;
    assert(ncols == NULL); ncols = new int[np];
    assert(rows == NULL); rows = new int[np];
    FOR_IP rows[ip] = ip;

    HYPRE_IJMatrixCreate(mpi_comm, 0, np, 0, np, &Aij);
    HYPRE_IJMatrixSetObjectType(Aij, HYPRE_PARCSR);
   
    HYPRE_IJVectorCreate(mpi_comm,0,np,&bij);
    HYPRE_IJVectorSetObjectType(bij,HYPRE_PARCSR);
   
    HYPRE_IJVectorCreate(mpi_comm,0,np,&xij);
    HYPRE_IJVectorSetObjectType(xij,HYPRE_PARCSR);
     
    /*
    HYPRE_ParCSRBiCGSTABCreate(mpi_comm,&hypre_solver);
    HYPRE_BiCGSTABSetMaxIter(hypre_solver,1000);
    HYPRE_BiCGSTABSetTol(hypre_solver, 1e-8);
    HYPRE_BiCGSTABSetPrintLevel(hypre_solver,3);
    */
    
    HYPRE_ParCSRPCGCreate(mpi_comm,&hypre_solver);
    HYPRE_PCGSetMaxIter(hypre_solver,1000);
    HYPRE_PCGSetTol(hypre_solver, 1e-9);
    HYPRE_PCGSetPrintLevel(hypre_solver,0);
    
  }
  
  void init() {
    cout << "init()" << endl;
    
    nsteps = 800;
    check_interval = 1;
    write_interval = 1;
            
    gam = 1.4;
    mu = 0.0; //1.0;

    ncells = 6;
    list<Point> cell_list;
    for (int i = 0; i < ncells; ++i) 
      for (int j = 0; j < ncells; ++j) 
	cells.push_back(cell_list);
    
    buildCartMesh(31, 32);   
    //buildCartMesh(31, 32);   

    //assert(Mp == NULL);        
    Mp = new double[np];
    //assert(rhop == NULL);      
    rhop = new double[np];
    //assert(Gp == NULL);        
    Gp = new double[np][2]; 
    //assert(Ep == NULL);        
    Ep = new double[np]; 
    //assert(up == NULL);        
    up = new double[np][2];
    //assert(pp == NULL);        
    pp = new double[np];
    //assert(volp == NULL);      
    volp = new double[np];
    //assert(ip_flag == NULL);   
    ip_flag = new int[np];
    //assert(lambdap == NULL);   
    lambdap = new double[np][2];
    //assert(denomp == NULL);    
    denomp = new double[np];
    //assert(rho_error == NULL); 
    rho_error = new double[np];
    //assert(u_error == NULL);   
    u_error = new double[np][2];
    //assert(p_error == NULL);   
    p_error = new double[np];
    
    FOR_IP FOR_I2 lambdap[ip][i] = 0.0;
    nbopa_i = new int[np + 1];
    initLinearSolver();

    repairTris(); 
    updateCells();
    
    cout << "np: " << np << " ntris: " << triVec.size() << " nquad: " << NQUAD << endl;
    initPhisFromDelaunay();
    updatePhisOMP();  
    updateVol();
   
    
    initialHook();
    
    //char filename[32];
    //sprintf(filename,"lp.%08d.dat",step);
    //writeLp(filename);
    
    buildMassMatrix();   
    solveForRho();    
    solveForP();
    
    buildMomentumMatrix();   
    solveForU();

    step = 0;
    time = 0.0;

    //sprintf(filename,"tri.%08d.dat",step);
    //writeTri(filename);
  }
  
  void checkCount() {
    cout << "checkCount()" << endl;
    int * count = new int[np];
    FOR_IP count[ip] = 0;
    
    FOR_IT {
      FOR_IVE {
	++count[triVec[it].ve_ip[ive]];
      }
    }
    
    FOR_IP {
      if (count[ip] != 6) {
	cout << ip << " | " << count[ip] << endl;
	getchar();
      }
    }
    delete[] count;
  }
  
  void updateAfterAdvect() {
    repairTris();
    initPhisFromDelaunay();
    updatePhisOMP();
    
    updateVol();

    buildMassMatrix();   
    solveForRho();
    solveForP();
    
    buildMomentumMatrix();   
    solveForU();
  }
  
  void writeLp(char * filename) {
    
    cout << "writing " << filename << endl;
    
    FILE * fp = fopen(filename,"w");
    
    /*
    int count = 0;
    for (int i = 0; i < ncells; ++i) {
      for (int j = 0; j < ncells; ++j) {
	for (typename list<Point>::iterator il = cells[i + ncells*j].begin(); il != cells[i + ncells*j].end(); ++il) {
	  ++count;
	}
      }
    }
    */    
    
    fprintf(fp,"TITLE = \"%s\"\n",filename);
    fprintf(fp,"VARIABLES = \"X\"\n");
    fprintf(fp,"\"Y\"\n");
    fprintf(fp,"\"PHI\"\n");
    fprintf(fp,"\"GRAD_PHI-X\"\n");
    fprintf(fp,"\"GRAD_PHI-Y\"\n");
    
    //fprintf(fp,"ZONE T=\"%s, np=%d\" I=%d, J=1, K=1, ZONETYPE=Ordered, DATAPACKING=POINT\n",filename,nquadp,nquadp);
    
    /*
    for (int i = 0; i < ncells; ++i) {
      for (int j = 0; j < ncells; ++j) {
	for (typename list<Point>::iterator il = cells[i + ncells*j].begin(); il != cells[i + ncells*j].end(); ++il) {
	  fprintf(fp,"%18.15le %18.15le %d %d\n", il->x[0], il->x[1], i, j);
	}
      }
    } 
    */
    /*
    FOR_IP if (fabs(xp[ip][0] - 0.6666) < 0.5 && fabs(xp[ip][1] - 0.6666) < 0.5) {
      cout << ip << endl;
      getchar();
    }
    */
    const int index = 136;
    
    FOR_IT {    
      if ((triVec[it].ve_bit[0] == 0) && (triVec[it].ve_bit[1] == 0) && (triVec[it].ve_bit[2] == 0)) {
	FOR_IQ {
	  double xquad[2] = {0.0, 0.0};
	  FOR_I2 FOR_IVE xquad[i] += triVec[it].ve_x[ive][i]*quad.ve_wgt[iq][ive];
	  
	  Basis * basis_index = NULL;
	  for (typename list<Basis>::iterator ib = triVec[it].basisList[iq].begin(); 
	       ib != triVec[it].basisList[iq].end(); ++ib) {
	    if (ib->ip == index) {
	      basis_index = &(*ib);
	      break;
	    }
	  }
	  
	  if (basis_index == NULL) {
	    fprintf(fp,"%18.15le %18.15le %18.15le %18.15le %18.15le \n", xquad[0], xquad[1], 0.0,
		    0.0, 0.0);
	  }
	  else {
	    fprintf(fp,"%18.15le %18.15le %18.15le %18.15le %18.15le \n", xquad[0], xquad[1], basis_index->phi,
		    basis_index->grad_phi[0], basis_index->grad_phi[1]);
	  }
	}
      }
    }
    
    fclose(fp);
    
  }

 
  void buildMassMatrix() {
    cout << "buildMassMatrix()" << endl;
    
    // first build full mass matrix, then threshold and build csr..
    double * Mfull = new double[np*np];
    for (int ii = 0; ii < np*np; ++ii) Mfull[ii] = 0.0;

    FOR_IT {    
      FOR_IQ {
	const double weight = quad.p_wgt[iq]*triVec[it].area;
	for (typename list<Basis>::iterator ib = triVec[it].basisList[iq].begin(); 
	       ib != triVec[it].basisList[iq].end(); ++ib) {
	  const int ip = ib->ip;
	  for (typename list<Basis>::iterator jb = triVec[it].basisList[iq].begin(); 
	       jb != triVec[it].basisList[iq].end(); ++jb) {
	    const int jp = jb->ip;
	    Mfull[ip + jp*np] += weight*(ib->phi)*(jb->phi);
	  }
	}
      }
    }

      
    // build csr...
    assert(nbopa_i != NULL);
    // first count...
    FOR_IP nbopa_i[ip+1] = 0; 
    for (int iter = 0; iter < 2; iter++) {
      FOR_IP {
	for (int ip_nbr = 0; ip_nbr < np; ++ip_nbr) {
	  if (Mfull[ip*np + ip_nbr] > 1e-8) {
	    if (iter == 0) 
	      ++nbopa_i[ip+1];
	    else {
	      nbopa_v[nbopa_i[ip]] = ip_nbr;
	      M[nbopa_i[ip]] = Mfull[ip*np + ip_nbr];
	      ++nbopa_i[ip];
	    }
	  }
	}
      }
      if (iter == 0) {
	nbopa_i[0] = 0;
	FOR_IP nbopa_i[ip+1] += nbopa_i[ip]; 
	nbopa_s = nbopa_i[np];
	if (nbopa_v != NULL) delete[] nbopa_v;
	nbopa_v = new int[nbopa_s];
	if (M != NULL) delete[] M;
	M = new double[nbopa_s];
      }
      else {
	for (int ip = np; ip > 0; --ip) 
	  nbopa_i[ip] = nbopa_i[ip-1];
	nbopa_i[0] = 0;	
      }
    }
        
    delete[] Mfull;
    /*
    cout << triVec[0].area << endl;
    FOR_IP {
      int nbopa_f = nbopa_i[ip];
      int nbopa_l = nbopa_i[ip+1];
      cout << ip << " | ";
      for (int inbopa = nbopa_f; inbopa < nbopa_l; ++inbopa) {
	const int ip_nbr = nbopa_v[inbopa];
	cout << ip_nbr << " > " << M[inbopa] << "   ";
      }
      cout << endl;
      getchar();
    }
    */  

  }

  
  void buildMomentumMatrix() {
    cout << "buildMomentumMatrix()" << endl;
    double * Mfull = new double[np*np];
    for (int ii = 0; ii < np*np; ++ii) Mfull[ii] = 0.0;

    FOR_IT {    
      FOR_IQ {
	double this_rho = 0.0;
	for (typename list<Basis>::iterator ib = triVec[it].basisList[iq].begin(); 
	       ib != triVec[it].basisList[iq].end(); ++ib) {
	  const int ip = ib->ip;
	  this_rho += rhop[ip]*ib->phi;
	}

	const double weight = quad.p_wgt[iq]*triVec[it].area;
	for (typename list<Basis>::iterator ib = triVec[it].basisList[iq].begin(); 
	       ib != triVec[it].basisList[iq].end(); ++ib) {
	  const int ip = ib->ip;
	  for (typename list<Basis>::iterator jb = triVec[it].basisList[iq].begin(); 
	       jb != triVec[it].basisList[iq].end(); ++jb) {
	    const int jp = jb->ip;
	    Mfull[ip + jp*np] += weight*this_rho*ib->phi*jb->phi;
	  }
	}
      }
    }
    
    // same sparsity as M...
    if (M != NULL) delete[] M;
    M = new double[nbopa_s];
    FOR_IP {
      int nbopa_f = nbopa_i[ip];
      int nbopa_l = nbopa_i[ip+1];
      for (int inbopa = nbopa_f; inbopa < nbopa_l; ++inbopa) {
	const int ip_nbr = nbopa_v[inbopa];
	assert(Mfull[ip*np + ip_nbr] > 0.0);
	M[inbopa] = Mfull[ip*np + ip_nbr];
      }
    }
    delete[] Mfull;
  }

  void updateVol() {
    
   FOR_IP volp[ip] = 0.0;

    FOR_IT {    
      FOR_IQ {
	const double weight = quad.p_wgt[iq]*triVec[it].area;
	for (typename list<Basis>::iterator ib = triVec[it].basisList[iq].begin(); 
	       ib != triVec[it].basisList[iq].end(); ++ib) {
	  const int ip = ib->ip;
	  assert(ip < np);
	  //assert(fabs(ib->phi - 1.0/3.0) < 1e-7);
	  volp[ip] += weight*ib->phi;
	}
      }
    }
        
    double vol_sum = 0.0;
    FOR_IP vol_sum += volp[ip];
    assert( fabs(vol_sum - (XMAX-XMIN)*(YMAX-YMIN)) < 1e-9); 
    //cout << "vol_sum: " << vol_sum << endl;
  }

  
  void solveForRho() {
    
    assert(M != NULL);
    FOR_IP {
      ncols[ip] = nbopa_i[ip+1] - nbopa_i[ip];
      assert(ncols[ip] < np);
    }       
    if (cols != NULL) delete[] cols;
    cols = new int[nbopa_s];
    
    int count = 0;
    FOR_IP {
      int nbopa_f = nbopa_i[ip];
      int nbopa_l = nbopa_i[ip+1];
      for (int inbopa = nbopa_f; inbopa < nbopa_l; ++inbopa) {
	const int ip_nbr = nbopa_v[inbopa];
	cols[count] = ip_nbr;
	++count;
      }
    }
    assert(count == nbopa_s);

    HYPRE_IJMatrixInitialize(Aij);
    HYPRE_IJMatrixSetValues(Aij,nrows,ncols,rows,cols,M);    
    HYPRE_IJMatrixAssemble(Aij);
    HYPRE_IJMatrixGetObject(Aij, (void **) &A);
    
    HYPRE_IJVectorInitialize(bij);
    HYPRE_IJVectorSetValues(bij,np,rows,Mp);
    HYPRE_IJVectorAssemble(bij);
    HYPRE_IJVectorGetObject(bij,(void **) &b);
    
    HYPRE_IJVectorInitialize(xij);
    HYPRE_IJVectorSetValues(xij,np,rows,rhop);
    HYPRE_IJVectorAssemble(xij);
    HYPRE_IJVectorGetObject(xij,(void **) &x);
        
    /*
    HYPRE_BiCGSTABSetup(hypre_solver, (HYPRE_Matrix)A,
		      (HYPRE_Vector)b, (HYPRE_Vector)x);

    HYPRE_BiCGSTABSolve(hypre_solver, (HYPRE_Matrix)A,
		      (HYPRE_Vector)b, (HYPRE_Vector)x);
    */
    
    HYPRE_PCGSetup(hypre_solver, (HYPRE_Matrix)A,
		      (HYPRE_Vector)b, (HYPRE_Vector)x);

    HYPRE_PCGSolve(hypre_solver, (HYPRE_Matrix)A,
		      (HYPRE_Vector)b, (HYPRE_Vector)x);
    
    HYPRE_IJVectorGetValues(xij,np,rows,rhop);
    
    double error = 0.0;
    FOR_IP {
      double this_mp = 0.0;
      int nbopa_f = nbopa_i[ip];
      int nbopa_l = nbopa_i[ip+1];
      for (int inbopa = nbopa_f; inbopa < nbopa_l; ++inbopa) {
	const int ip_nbr = nbopa_v[inbopa];
	this_mp += M[inbopa]*rhop[ip_nbr];
      }
      error += (this_mp - Mp[ip])*(this_mp - Mp[ip]);
    }
    assert(error < 1e-7);    
        
       
  }

  
  void solveForP() {
    double * rhs = new double[np];
    FOR_IP rhs[ip] = 0.0;
    
    FOR_IT {
      FOR_IQ {
	double this_rho = 0.0;
	for (typename list<Basis>::iterator ib = triVec[it].basisList[iq].begin(); 
	       ib != triVec[it].basisList[iq].end(); ++ib) {
	  const int ip = ib->ip;
	  this_rho += rhop[ip]*(ib->phi);
	}
	
	//const double this_p = pEOS(this_rho);
	const double this_p = pow(this_rho, gam);
	const double weight = quad.p_wgt[iq]*triVec[it].area;
	for (typename list<Basis>::iterator ib = triVec[it].basisList[iq].begin(); 
	       ib != triVec[it].basisList[iq].end(); ++ib) {
	  const int ip = ib->ip;
	  rhs[ip] += weight*(ib->phi)*this_p;
	}
      }
    }
    
    HYPRE_IJVectorInitialize(bij);
    HYPRE_IJVectorSetValues(bij,np,rows,rhs);
    HYPRE_IJVectorAssemble(bij);
    HYPRE_IJVectorGetObject(bij,(void **) &b);
    
    HYPRE_IJVectorInitialize(xij);
    HYPRE_IJVectorSetValues(xij,np,rows,pp);
    HYPRE_IJVectorAssemble(xij);
    HYPRE_IJVectorGetObject(xij,(void **) &x);
       
    HYPRE_PCGSetup(hypre_solver, (HYPRE_Matrix)A,
		      (HYPRE_Vector)b, (HYPRE_Vector)x);

    HYPRE_PCGSolve(hypre_solver, (HYPRE_Matrix)A,
		      (HYPRE_Vector)b, (HYPRE_Vector)x);
    
    HYPRE_IJVectorGetValues(xij,np,rows,pp);

    HYPRE_IJMatrixDestroy(Aij);
    HYPRE_IJMatrixCreate(mpi_comm, 0, np, 0, np, &Aij);
    HYPRE_IJMatrixSetObjectType(Aij, HYPRE_PARCSR);

    double error = 0.0;
    FOR_IP {
      double this_rhs = 0.0;
      int nbopa_f = nbopa_i[ip];
      int nbopa_l = nbopa_i[ip+1];
      for (int inbopa = nbopa_f; inbopa < nbopa_l; ++inbopa) {
	const int ip_nbr = nbopa_v[inbopa];
	this_rhs += M[inbopa]*pp[ip_nbr];
      }
      error += (this_rhs - rhs[ip])*(this_rhs - rhs[ip]);
    }
    assert(error < 1e-7);    
    delete[] rhs;
  }

  void solveForU() {
    
    assert(M != NULL);
            
    FOR_IP ncols[ip] = nbopa_i[ip+1] - nbopa_i[ip];
             
    if (cols != NULL) delete[] cols;
    cols = new int[nbopa_s];
    
    int count = 0;
    FOR_IP {
      int nbopa_f = nbopa_i[ip];
      int nbopa_l = nbopa_i[ip+1];
      for (int inbopa = nbopa_f; inbopa < nbopa_l; ++inbopa) {
	const int ip_nbr = nbopa_v[inbopa];
	cols[count] = ip_nbr;
	++count;
      }
    }
    assert(count == nbopa_s);
    
    HYPRE_IJMatrixInitialize(Aij);
    HYPRE_IJMatrixSetValues(Aij,nrows,ncols,rows,cols,M);
    HYPRE_IJMatrixAssemble(Aij);
    HYPRE_IJMatrixGetObject(Aij, (void **) &A);
       
    double * gp_tmp = new double[np];
    double * up_tmp = new double[np];
    FOR_I2 {
      FOR_IP gp_tmp[ip] = Gp[ip][i];
      HYPRE_IJVectorInitialize(bij);
      HYPRE_IJVectorSetValues(bij,np,rows,gp_tmp);
      HYPRE_IJVectorAssemble(bij);
      HYPRE_IJVectorGetObject(bij,(void **) &b);
      
      FOR_IP up_tmp[ip] = up[ip][i]; 
      HYPRE_IJVectorInitialize(xij);
      HYPRE_IJVectorSetValues(xij,np,rows,up_tmp);
      HYPRE_IJVectorAssemble(xij);
      HYPRE_IJVectorGetObject(xij,(void **) &x);
            
      HYPRE_PCGSetup(hypre_solver, (HYPRE_Matrix)A,
		     (HYPRE_Vector)b, (HYPRE_Vector)x);
      
      HYPRE_PCGSolve(hypre_solver, (HYPRE_Matrix)A,
		     (HYPRE_Vector)b, (HYPRE_Vector)x);
      
      HYPRE_IJVectorGetValues(xij,np,rows,up_tmp);
      FOR_IP up[ip][i] = up_tmp[ip];
    }
    delete[] gp_tmp;
    delete[] up_tmp;
    

    HYPRE_IJMatrixDestroy(Aij);
    HYPRE_IJMatrixCreate(mpi_comm, 0, np, 0, np, &Aij);
    HYPRE_IJMatrixSetObjectType(Aij, HYPRE_PARCSR);
  }
  
  void findCell(int &icell, int &jcell, const double x[2]) {
    
    double tmp = XMIN;
    icell = 0;
    while(x[0] > tmp) {
      tmp += (XMAX-XMIN)/((double) ncells-2);
      ++icell;
    }

    tmp = YMIN;
    jcell = 0;
    while(x[1] > tmp) {
      tmp += (YMAX-YMIN)/((double) ncells-2);
      ++jcell;
    }

    icell = max(1,min(icell,ncells-2));
    jcell = max(1,min(jcell,ncells-2));
    
  }
  
  void updateCells() {
    
    for (int i = 0; i < ncells; ++i) 
      for (int j = 0; j < ncells; ++j) 
	cells[i + ncells*j].clear();

    FOR_IP {    
      assert((xp[ip][0] >= XMIN) && (xp[ip][0] <= XMAX) && (xp[ip][1] >= YMIN) && (xp[ip][1] <= YMAX));
      
      int icell, jcell;
      findCell(icell, jcell, xp[ip]);
      Point this_point;
      this_point.ip = ip;      
      FOR_I2 this_point.x[i] = xp[ip][i];
      cells[icell + ncells*jcell].push_back(this_point);
      
      // bottom row
      if (icell == 1) {
	for (int j = 1; j < ncells-1; ++j) {
	  if (jcell == j) {
	    this_point.x[0] = xp[ip][0] + (XMAX - XMIN);
	    this_point.x[1] = xp[ip][1];
	    cells[(ncells-1) + ncells*j].push_back(this_point);
	  }
	}
      }

      // top row
      if (icell == (ncells-2)) {
	for (int j = 1; j < ncells-1; ++j) {
	  if (jcell == j) {
	    this_point.x[0] = xp[ip][0] - (XMAX - XMIN);
	    this_point.x[1] = xp[ip][1];
	    cells[0 + ncells*j].push_back(this_point);
	  }
	}
      }

      // left column
      if (jcell == 1) {
	for (int i = 1; i < ncells-1; ++i) {
	  if (icell == i) {
	    this_point.x[0] = xp[ip][0];
	    this_point.x[1] = xp[ip][1]  + (YMAX - YMIN);
	    cells[i + ncells*(ncells-1)].push_back(this_point);
	  }
	}
      }

      // left column
      if (jcell == (ncells-2)) {
	for (int i = 1; i < ncells-1; ++i) {
	  if (icell == i) {
	    this_point.x[0] = xp[ip][0];
	    this_point.x[1] = xp[ip][1]  - (YMAX - YMIN);
	    cells[i + ncells*0].push_back(this_point);
	  }
	}
      }
      
      // cornors
      if (icell == 1 && jcell == 1) {
	this_point.x[0] = xp[ip][0]  + (XMAX - XMIN);
	this_point.x[1] = xp[ip][1]  + (YMAX - YMIN);
	cells[(ncells-1) + ncells*(ncells-1)].push_back(this_point);
      }
      else if (icell == (ncells-2) && jcell == 1) {
	this_point.x[0] = xp[ip][0]  - (XMAX - XMIN);
	this_point.x[1] = xp[ip][1]  + (YMAX - YMIN);
	cells[0 + ncells*(ncells-1)].push_back(this_point);
      }
      else if (icell == 1 && jcell == (ncells-2)) {
	this_point.x[0] = xp[ip][0]  + (XMAX - XMIN);
	this_point.x[1] = xp[ip][1]  - (YMAX - YMIN);
	cells[(ncells-1) + ncells*0].push_back(this_point);
      }
      else if (icell == (ncells-2) && jcell == (ncells-2)) {
	this_point.x[0] = xp[ip][0]  - (XMAX - XMIN);
	this_point.x[1] = xp[ip][1]  - (YMAX - YMIN);
	cells[0 + ncells*0].push_back(this_point);
      }

    }
        
  }
  
  void buildCartMesh(const int nx, const int ny) {
    
    const double dx = (XMAX - XMIN)/(double) nx;
    const double dy = (YMAX - YMIN)/(double) ny;
    
    // way to set to const?
    np = nx*ny;

    // allocate and populate coordinates... 
    xp = new double[np][2];
    for (int i = 0; i < nx; ++i) {
      for (int j = 0; j < ny; ++j) {
	xp[i*ny + j][0] = XMIN + i*dx + 0.5*dx; // - 0.5*(j%2)*dx;
	xp[i*ny + j][1] = YMIN + j*dy + 0.5*dy; // //- 0.5*(i%2)*dy;
      }
    }
    
    FOR_IP {
      //xp[ip][0] += 0.25*dx*rand()/RAND_MAX; 
      //xp[ip][1] += 0.25*dy*rand()/RAND_MAX; 
    }
    
    // build tri's manually...
    // first interior...
    for (int i = 0; i < (nx-1); ++i) {
      for (int j = 0; j < (ny-1); ++j) {
	// "upward" pointing
	Tri tri1;
	// right handed tet..
	tri1.ve_ip[0] = i*ny + j;
	tri1.ve_ip[1] = (i+1)*ny + j;
	tri1.ve_ip[2] = (i+1)*ny + (j+1);
	
	// no periodicity...
	tri1.ve_bit[0] = 0;
	tri1.ve_bit[1] = 0;
	tri1.ve_bit[2] = 0;
	
	triVec.push_back(tri1);
	
	// "downward" pointing
	Tri tri2;
	// right handed tet..
	tri2.ve_ip[0] = i*ny + j;
	tri2.ve_ip[1] = (i+1)*ny + (j+1);
	tri2.ve_ip[2] = i*ny + (j+1);
	
	// no periodicity...
	tri2.ve_bit[0] = 0;
	tri2.ve_bit[1] = 0;
	tri2.ve_bit[2] = 0;
	
	triVec.push_back(tri2);     
	
      }
    }
    
    // now build periodic boundaries...
    // x boundaries..
    for (int j = 0; j < (ny-1); ++j) {
      // "upward" pointing
      Tri tri1;
      // right handed tet..
      tri1.ve_ip[0] = j;
      tri1.ve_ip[1] = j+1;
      tri1.ve_ip[2] = (nx-1)*ny + j;
      
      tri1.ve_bit[0] = 0;
      tri1.ve_bit[1] = 0;
      tri1.ve_bit[2] = XM_BIT;
    
      triVec.push_back(tri1);
      
      // "downward" pointing
      Tri tri2;
      // right handed tet..
      tri2.ve_ip[0] = (nx-1)*ny + j;
      tri2.ve_ip[1] = j+1;
      tri2.ve_ip[2] = (nx-1)*ny + (j+1);
      
      tri2.ve_bit[0] = 0;
      tri2.ve_bit[1] = XP_BIT;
      tri2.ve_bit[2] = 0;
      
      triVec.push_back(tri2);
    }
    
    // y boundaries..
    for (int i = 0; i < (nx-1); ++i) {
      // "upward" pointing
      Tri tri1;
      // right handed tet..
      tri1.ve_ip[0] = i*ny + (ny-1);
      tri1.ve_ip[1] = (i+1)*ny + (ny-1);
      tri1.ve_ip[2] = (i+1)*ny;
    
      tri1.ve_bit[0] = 0;
      tri1.ve_bit[1] = 0;
      tri1.ve_bit[2] = YP_BIT;
      
      triVec.push_back(tri1);
      
      // "downward" pointing
      Tri tri2;
      // right handed tet..
      tri2.ve_ip[0] = i*ny;
      tri2.ve_ip[1] = i*ny + (ny-1);
      tri2.ve_ip[2] = (i+1)*ny;;
      
      tri2.ve_bit[0] = 0;
      tri2.ve_bit[1] = YM_BIT;
      tri2.ve_bit[2] = 0;
      
      triVec.push_back(tri2);
    }
  
    // and finally the cornors...
    {
      // xm, ym cornor
      Tri tri1;
      tri1.ve_ip[0] = 0;
      tri1.ve_ip[1] = ny*(nx-1) + (ny-1);
      tri1.ve_ip[2] = (ny-1);
      
      tri1.ve_bit[0] = 0;
      tri1.ve_bit[1] = XM_BIT|YM_BIT;
      tri1.ve_bit[2] = YM_BIT;
      
      triVec.push_back(tri1);
      
      Tri tri2;
      tri2.ve_ip[0] = 0;
      tri2.ve_ip[1] = ny*(nx-1);
      tri2.ve_ip[2] = ny*(nx-1) + (ny-1);
      
      tri2.ve_bit[0] = 0;
      tri2.ve_bit[1] = XM_BIT;
      tri2.ve_bit[2] = XM_BIT|YM_BIT;
      
      triVec.push_back(tri2);    
    }
    
    
    // build neighbors.. (we only do this once, so efficiency doesn't matter)
    FOR_IT {
      triVec[it].updateVeX(xp);  // make sure all new tris have vertex coords
      triVec[it].calcArea();
      FOR_IVE {
	bool found = false;
	for (int it_nbr = 0; it_nbr < triVec.size(); ++it_nbr) {
	  if (it_nbr != it) {
	    for (int ive_nbr = 0; ive_nbr < 3; ++ive_nbr) {
	      if (triVec[it].ve_ip[(ive+1)%3] == triVec[it_nbr].ve_ip[ive_nbr]) {
		if (triVec[it].ve_ip[(ive+2)%3] == triVec[it_nbr].ve_ip[(ive_nbr+1)%3]) {
		  found = true;
		  triVec[it].ve_itnbr[ive] = it_nbr;
		  triVec[it].ve_ivenbr[ive] = (ive_nbr+2)%3;	      
		  break;
		}
		else if (triVec[it].ve_ip[(ive+2)%3] == triVec[it_nbr].ve_ip[(ive_nbr+2)%3]) {
		  found = true;
		  triVec[it].ve_itnbr[ive] = it_nbr;
		  triVec[it].ve_ivenbr[ive] = (ive_nbr+1)%3;	      
		  break;
		} 
	      }
	    }
	    if (found == true) break; 
	  }
	}						
	assert(found);
      }
    }
    
  }
  
  void repairTris() {
    
    // flip points that have crossed periodic boundaries..
    FOR_IP {
      // store flipping info in ip_flag...
      ip_flag[ip] = 0;
      if (xp[ip][0] > XMAX) {
	xp[ip][0] -= (XMAX - XMIN);
	ip_flag[ip] |= XP_BIT;
      }
      else if (xp[ip][0] < XMIN) {
	xp[ip][0] += (XMAX - XMIN);
	ip_flag[ip] |= XM_BIT;
      }
      
      if (xp[ip][1] > YMAX) {
	xp[ip][1] -= (YMAX - YMIN);
	ip_flag[ip] |= YP_BIT;
      }
      else if (xp[ip][1] < YMIN) {
	xp[ip][1] += (YMAX - YMIN);
	ip_flag[ip] |= YM_BIT;
      }
    }
   
    FOR_IT {
      FOR_IVE {
	if      (ip_flag[triVec[it].ve_ip[ive]] & XM_BIT) triVec[it].toggleXMBit(ive);
	else if (ip_flag[triVec[it].ve_ip[ive]] & XP_BIT) triVec[it].toggleXPBit(ive);
	if      (ip_flag[triVec[it].ve_ip[ive]] & YM_BIT) triVec[it].toggleYMBit(ive);
	else if (ip_flag[triVec[it].ve_ip[ive]] & YP_BIT) triVec[it].toggleYPBit(ive);
      }
    }
    
    
    FOR_IT triVec[it].resetBits();
  
    FOR_IT {
      triVec[it].updateVeX(xp);
      triVec[it].calcArea();
    }
    //if (step == 0)
    flipEdges();
    
  }

  void flipEdges() {
    
    bool done = false;
    int iter = 0;
    int nflip = 0;
    while (!done) {
      done = true;
      assert(iter < 100);
      FOR_IT {
	FOR_IVE {
	  
	  int it_nbr  = triVec[it].ve_itnbr[ive];
	  int ive_nbr = triVec[it].ve_ivenbr[ive];
	  
	  assert(it != it_nbr);
	  	  	  
	  int offset;
	  // transform bits so they are equal on verticies they share
	  if ( triVec[it].ve_ip[(ive+1)%3] == triVec[it_nbr].ve_ip[(ive_nbr+1)%3]) {
	    offset = 0;
	  }
	  else {
	    assert(triVec[it].ve_ip[(ive+1)%3] == triVec[it_nbr].ve_ip[(ive_nbr+2)%3]);
	    offset = 1;
	  }
	  
	  assert(triVec[it].ve_ip[(ive+1)%3] == triVec[it_nbr].ve_ip[(ive_nbr+offset+1)%3]);
	  assert(triVec[it].ve_ip[(ive+2)%3] == triVec[it_nbr].ve_ip[(ive_nbr-offset+2)%3]);
	
	  if ( (triVec[it].ve_bit[(ive+1)%3] & XM_BIT) != (triVec[it_nbr].ve_bit[(ive_nbr+offset+1)%3] & XM_BIT)) {
	    if (triVec[it].ve_bit[(ive+1)%3] & XM_BIT) {
	      FOR_I3 triVec[it].toggleXPBit(i);
	    }
	    else {
	      FOR_I3 triVec[it_nbr].toggleXPBit(i);
	    }
	  }
	  else if ( (triVec[it].ve_bit[(ive+1)%3] & XP_BIT) != (triVec[it_nbr].ve_bit[(ive_nbr+offset+1)%3] & XP_BIT)) {
	    if (triVec[it].ve_bit[(ive+1)%3] & XP_BIT) {
	      FOR_I3 triVec[it].toggleXMBit(i);
	    }
	    else {
	      FOR_I3 triVec[it_nbr].toggleXMBit(i);
	    }
	  }
	  if ( (triVec[it].ve_bit[(ive+1)%3] & YM_BIT) != (triVec[it_nbr].ve_bit[(ive_nbr+offset+1)%3] & YM_BIT)) {
	    if (triVec[it].ve_bit[(ive+1)%3] & YM_BIT) {
	      FOR_I3 triVec[it].toggleYPBit(i);
	    }
	    else {
	      FOR_I3 triVec[it_nbr].toggleYPBit(i);
	    }
	  }
	  else if ( (triVec[it].ve_bit[(ive+1)%3] & YP_BIT) != (triVec[it_nbr].ve_bit[(ive_nbr+offset+1)%3] & YP_BIT)) {
	    if (triVec[it].ve_bit[(ive+1)%3] & YP_BIT) {
	    FOR_I3 triVec[it].toggleYMBit(i);
	    }
	    else {
	      FOR_I3 triVec[it_nbr].toggleYMBit(i);
	    }
	  }
	  
	  if ( (triVec[it].ve_bit[(ive+2)%3] & XM_BIT) != (triVec[it_nbr].ve_bit[(ive_nbr-offset+2)%3] & XM_BIT)) {
	    if (triVec[it].ve_bit[(ive+2)%3] & XM_BIT) {
	      FOR_I3 triVec[it].toggleXPBit(i);
	    }
	    else {
	      FOR_I3 triVec[it_nbr].toggleXPBit(i);
	    }
	  }
	  else if ( (triVec[it].ve_bit[(ive+2)%3] & XP_BIT) != (triVec[it_nbr].ve_bit[(ive_nbr-offset+2)%3] & XP_BIT)) {
	    if (triVec[it].ve_bit[(ive+2)%3] & XP_BIT) {
	    FOR_I3 triVec[it].toggleXMBit(i);
	    }
	    else {
	      FOR_I3 triVec[it_nbr].toggleXMBit(i);
	    }
	  }
	  if ( (triVec[it].ve_bit[(ive+2)%3] & YM_BIT) != (triVec[it_nbr].ve_bit[(ive_nbr-offset+2)%3] & YM_BIT)) {
	    if (triVec[it].ve_bit[(ive+2)%3] & YM_BIT) {
	      FOR_I3 triVec[it].toggleYPBit(i);
	    }
	    else {
	      FOR_I3 triVec[it_nbr].toggleYPBit(i);
	    }
	  }
	  else if ( (triVec[it].ve_bit[(ive+2)%3] & YP_BIT) != (triVec[it_nbr].ve_bit[(ive_nbr-offset+2)%3] & YP_BIT)) {
	    if (triVec[it].ve_bit[(ive+2)%3] & YP_BIT) {
	      FOR_I3 triVec[it].toggleYMBit(i);
	    }
	    else {
	      FOR_I3 triVec[it_nbr].toggleYMBit(i);
	    }
	  }
	  
	  triVec[it].updateVeX(xp);
	  triVec[it_nbr].updateVeX(xp);
	  
	  double xc[2];
	  {
	    double bx = triVec[it].ve_x[1][0] - triVec[it].ve_x[0][0];
	    double by = triVec[it].ve_x[1][1] - triVec[it].ve_x[0][1];
	    double cx = triVec[it].ve_x[2][0] - triVec[it].ve_x[0][0];
	    double cy = triVec[it].ve_x[2][1] - triVec[it].ve_x[0][1];
	    double d = 2.0*(bx*cy - by*cx);
	    xc[0] = (cy*(bx*bx+by*by) - by*(cx*cx+cy*cy))/d + triVec[it].ve_x[0][0];
	    xc[1] = (bx*(cx*cx+cy*cy) - cx*(bx*bx+by*by))/d + triVec[it].ve_x[0][1];
	  }
	  
	  
	  double r_it = sqrt((triVec[it].ve_x[ive][0] - xc[0])*(triVec[it].ve_x[ive][0] - xc[0]) + 
			     (triVec[it].ve_x[ive][1] - xc[1])*(triVec[it].ve_x[ive][1] - xc[1])); 
	  double r_it_nbr = sqrt((triVec[it_nbr].ve_x[ive_nbr][0] - xc[0])*(triVec[it_nbr].ve_x[ive_nbr][0] - xc[0]) + 
				 (triVec[it_nbr].ve_x[ive_nbr][1] - xc[1])*(triVec[it_nbr].ve_x[ive_nbr][1] - xc[1])); 
	  
	  if ( (r_it - r_it_nbr) > 1e-9*r_it ) {
	    
	    // need temporary tri to store old info...
	    Tri tmp = triVec[it];
	    Tri tmp_nbr = triVec[it_nbr];
	    
	    triVec[it].ve_ip[(ive+1)%3] = tmp_nbr.ve_ip[ive_nbr];
	    triVec[it].ve_bit[(ive+1)%3] = tmp_nbr.ve_bit[ive_nbr];
	    
	    triVec[it].ve_itnbr[ive] =  tmp_nbr.ve_itnbr[(ive_nbr+offset+1)%3];
	    triVec[it].ve_ivenbr[ive] =  tmp_nbr.ve_ivenbr[(ive_nbr+offset+1)%3];
	    triVec[triVec[it].ve_itnbr[ive]].ve_itnbr[triVec[it].ve_ivenbr[ive]] = it;
	    triVec[triVec[it].ve_itnbr[ive]].ve_ivenbr[triVec[it].ve_ivenbr[ive]] = ive;
	    
	    triVec[it].ve_itnbr[(ive+2)%3] =  it_nbr;
	    triVec[it].ve_ivenbr[(ive+2)%3] =  (ive_nbr+offset+1)%3;  
	    
	    triVec[it_nbr].ve_ip[(ive_nbr-offset+2)%3] = tmp.ve_ip[ive];
	    triVec[it_nbr].ve_bit[(ive_nbr-offset+2)%3] = tmp.ve_bit[ive];
	    
	    triVec[it_nbr].ve_itnbr[ive_nbr] =  tmp.ve_itnbr[(ive+2)%3];
	    triVec[it_nbr].ve_ivenbr[ive_nbr] =  tmp.ve_ivenbr[(ive+2)%3];
	    triVec[triVec[it_nbr].ve_itnbr[ive_nbr]].ve_itnbr[triVec[it_nbr].ve_ivenbr[ive_nbr]] = it_nbr;
	    triVec[triVec[it_nbr].ve_itnbr[ive_nbr]].ve_ivenbr[triVec[it_nbr].ve_ivenbr[ive_nbr]] = ive_nbr;
	    
	    triVec[it_nbr].ve_itnbr[(ive_nbr+offset+1)%3] =  it;
	    triVec[it_nbr].ve_ivenbr[(ive_nbr+offset+1)%3] =  (ive+2)%3;
	    
	    done = false;
	    ++nflip;
	    
	  }
	  
	  // make sure vertex coords and areas are current...
	  triVec[it].updateVeX(xp); 
	  triVec[it_nbr].updateVeX(xp);
	  triVec[it].calcArea();
	  triVec[it_nbr].calcArea();
	  	  
	  triVec[it].resetBits();
	  triVec[it_nbr].resetBits();
	  
	  
	}
      }
      ++iter;
    }
        
  }
  
  void initPhisFromDelaunay() {
    
    FOR_IT {
      const double denom = ( (triVec[it].ve_x[1][1] - triVec[it].ve_x[2][1])*(triVec[it].ve_x[0][0] - triVec[it].ve_x[2][0])
			     + (triVec[it].ve_x[2][0] - triVec[it].ve_x[1][0])*(triVec[it].ve_x[0][1] - triVec[it].ve_x[2][1]) );
      FOR_IQ {
	triVec[it].basisList[iq].clear();
	double xquad[2] = {0.0, 0.0};
	FOR_I2 FOR_IVE xquad[i] += triVec[it].ve_x[ive][i]*quad.ve_wgt[iq][ive];

	const double phi0 = ( (triVec[it].ve_x[1][1] - triVec[it].ve_x[2][1])*(xquad[0] - triVec[it].ve_x[2][0])
			      + (triVec[it].ve_x[2][0] - triVec[it].ve_x[1][0])*(xquad[1] - triVec[it].ve_x[2][1]) )/denom; 
	{
	  Basis this_basis;
	  this_basis.index = 0;
	  this_basis.ip = triVec[it].ve_ip[0];
	  this_basis.phi = phi0;
	  this_basis.grad_phi[0] = (triVec[it].ve_x[1][1] - triVec[it].ve_x[2][1])/denom;
	  this_basis.grad_phi[1] = (triVec[it].ve_x[2][0] - triVec[it].ve_x[1][0])/denom;
	  triVec[it].basisList[iq].push_back(this_basis);
	}
	const double phi1 = ( (triVec[it].ve_x[2][1] - triVec[it].ve_x[0][1])*(xquad[0] - triVec[it].ve_x[2][0])
			      + (triVec[it].ve_x[0][0] - triVec[it].ve_x[2][0])*(xquad[1] - triVec[it].ve_x[2][1]) )/denom;
	{
	  Basis this_basis;
	  this_basis.index = 1;
	  this_basis.ip = triVec[it].ve_ip[1];
	  this_basis.phi = phi1;
	  this_basis.grad_phi[0] = (triVec[it].ve_x[2][1] - triVec[it].ve_x[0][1])/denom;
	  this_basis.grad_phi[1] = (triVec[it].ve_x[0][0] - triVec[it].ve_x[2][0])/denom;
	  triVec[it].basisList[iq].push_back(this_basis);
	}
	
	{
	  Basis this_basis;
	  this_basis.index = 2;
	  this_basis.ip = triVec[it].ve_ip[2];
	  this_basis.phi = 1.0 - phi0 - phi1;
	  this_basis.grad_phi[0] = (triVec[it].ve_x[0][1] - triVec[it].ve_x[1][1])/denom;
	  this_basis.grad_phi[1] = (triVec[it].ve_x[1][0] - triVec[it].ve_x[0][0])/denom;
	  triVec[it].basisList[iq].push_back(this_basis);
	}
	
	//cout << phi0 << " " << phi1 << endl;
		
	/*
	cout << xquad[0] << " " << xquad[1] << endl;
	cout << triVec[it].ve_x[0][0] << " " << triVec[it].ve_x[0][1] << endl;
	cout << triVec[it].ve_x[1][0] << " " << triVec[it].ve_x[1][1] << endl;
	cout << triVec[it].ve_x[2][0] << " " << triVec[it].ve_x[2][1] << endl;
	cout << "phi: " << endl;
	FOR_IP cout << triVec[it].phi[iq][ip] << " ";
	cout << endl;
	getchar();
	*/
      }
    }
  }
    
  void updatePhis() {
    cout << "updatePhis() " << endl;
    double (*xp_trans)[2] = new double[np][2];

    int max_iter = 0;
    FOR_IT {
      FOR_IQ {
	//cout << it << " " << iq << endl;
	double xquad[2] = {0.0, 0.0};
	FOR_I2 FOR_IVE xquad[i] += triVec[it].ve_x[ive][i]*quad.ve_wgt[iq][ive];
	//cout << "xquad: " << xquad[0] << " " << xquad[1] << endl;
	
	FOR_IP {
	  FOR_I2 xp_trans[ip][i] = xp[ip][i];
	  if ((xquad[0] - xp_trans[ip][0]) > 0.5*(XMAX-XMIN)) {
	    xp_trans[ip][0] += (XMAX-XMIN);
	  }
	  else if ((xp_trans[ip][0] - xquad[0]) > 0.5*(XMAX-XMIN)) {
	    xp_trans[ip][0] -= (XMAX-XMIN);
	  }
	  if ((xquad[1] - xp_trans[ip][1]) > 0.5*(YMAX-YMIN)) {
	    xp_trans[ip][1] += (YMAX-YMIN);
	  }
	  else if ((xp_trans[ip][1] - xquad[1]) > 0.5*(YMAX-YMIN)) {
	    xp_trans[ip][1] -= (YMAX-YMIN);
	  }	  
	}

	bool done = false;
	int iter = 0;
	while (!done) {
	  	    
	  int nactive = triVec[it].basisList[iq].size();
	  assert(nactive > 1);
	  
	  double * A = new double[(nactive+1)*(nactive+1)]; // collumn major!!! (though symmteric, so doesn't matter for now...)
	  double * A2 = new double[(nactive+1)*(nactive+1)];
	  double * b = new double[nactive+1];
	  for (int ii = 0; ii < nactive; ++ii) b[ii] = 0.0;
	  
	  for (typename list<Basis>::iterator ib = triVec[it].basisList[iq].begin(); 
	       ib != triVec[it].basisList[iq].end(); ++ib) {
	    const int ip = ib->ip;
	    const int ii = ib->index;
	    for (typename list<Basis>::iterator jb = triVec[it].basisList[iq].begin(); 
		 jb != triVec[it].basisList[iq].end(); ++jb) {
	      const int jp = jb->ip;
	      const int jj = jb->index;
	      const double dist = sqrt((xp_trans[ip][0] - xp_trans[jp][0])*(xp_trans[ip][0] - xp_trans[jp][0]) +
				       (xp_trans[ip][1] - xp_trans[jp][1])*(xp_trans[ip][1] - xp_trans[jp][1]));
	      A[ii*(nactive+1) + jj] = -dist;
	      b[ii] += dist*(jb->phi);
	    }
	    A[ii*(nactive+1) + ii] = 0.0; // diagonal
	    // lagrange multipliers..
	    A[ii*(nactive+1) + nactive] = 1.0;
	    A[nactive*(nactive+1) + ii] = 1.0;
	    
	    // distance to active set...
	    b[ii] -= sqrt((xp_trans[ip][0] - xquad[0])*(xp_trans[ip][0] - xquad[0]) +
			  (xp_trans[ip][1] - xquad[1])*(xp_trans[ip][1] - xquad[1]));
	  }
	
	  A[(nactive+1)*(nactive+1) - 1] = 0.0;
	  b[nactive] = 0.0;
	    
	  for (int ii = 0; ii < (nactive+1)*(nactive+1); ++ii) A2[ii] = A[ii];
	  double * b2 = new double[nactive+1];
	  for (int ii = 0; ii < (nactive+1); ++ii) b2[ii] = b[ii];
	  
	  /*
	  if (it == 150 && iq == 3) {
	    cout << "A: " << endl;
	    for (int ii = 0; ii < nactive+1; ++ii) {
	      cout << ii << " | ";
	      for (int jj = 0; jj < nactive+1; ++jj) {
		cout << A[ii*(nactive+1) + jj] << " ";
	      }
	      cout << endl;
	    }
	    
	    cout << "phi: " << endl;
	    for (typename list<Basis>::iterator ib = triVec[it].basisList[iq].begin(); 
		 ib != triVec[it].basisList[iq].end(); ++ib)  
	      cout << ib->phi << " ";
	    cout << endl;
	    
	    cout << "b: " << endl;
	    for (int ii = 0; ii < nactive+1; ++ii) cout << b[ii] << " ";
	    cout << endl;
	  m}
	  */
	  {
	    int nrhs = 1;
	    int N = nactive+1;
	    int * ipiv = new int[N];
	    int info;	    
	    dgesv_(&N, &nrhs, A, &N, ipiv,
		   b, &N, &info);
	    if (info != 0) {
	      cout << "info: " << info << endl;
	      cout << "A2: " << endl;
	      for (int ii = 0; ii < nactive+1; ++ii) {
		cout << ii << " | ";
		for (int jj = 0; jj < nactive+1; ++jj) {
		  cout << A2[ii*(nactive+1) + jj] << " ";
		}
		cout << endl;
	      }
	      
	      cout << "phi: " << endl;
	      for (typename list<Basis>::iterator ib = triVec[it].basisList[iq].begin(); 
		   ib != triVec[it].basisList[iq].end(); ++ib)  
		cout << ib->phi << " ";
	      cout << endl;
	      
	      cout << "b2: " << endl;
	      for (int ii = 0; ii < nactive+1; ++ii) cout << b2[ii] << " ";
	      cout << endl;
	      getchar();
	    }
	    //assert(info == 0);
	  
	    delete[] ipiv;
	  }
	  delete[] b2;
	  /*
	  cout << "dphi: " << endl;
	  for (int ii = 0; ii < nactive+1; ++ii) cout << b[ii] << " ";
	  cout << endl;
	  */
	  const double nu = b[nactive];
	  
	  // compute step size...
	  double step = 1.0;	  
	  double normsq = 0.0;
	  {
	    for (typename list<Basis>::iterator ib = triVec[it].basisList[iq].begin(); 
		 ib != triVec[it].basisList[iq].end(); ++ib) {
	      const int ii = ib->index;
	      assert(ii < nactive);
	      normsq += b[ii]*b[ii];
	      if (b[ii] < 0.0) {
		step = min(step, -(ib->phi)/b[ii]);
	      }
	    }
	    if (normsq < 1e-8) {
	      step = 0.0;
	    }
	  }
	  
	  //cout << "step: " << step << endl;
	  
	  if (step > 0.0) {
	    list<Basis>::iterator ib = triVec[it].basisList[iq].begin();
	    while (ib != triVec[it].basisList[iq].end()) {
	      const int ii = ib->index;
	      ib->phi += step*b[ii];
	      if (ib->phi < 1e-10) {
		ib = triVec[it].basisList[iq].erase(ib);
	      }
	      else {
		++ib;
	      }
	    }
	    int index = 0;
	    for (typename list<Basis>::iterator ib = triVec[it].basisList[iq].begin(); 
		 ib != triVec[it].basisList[iq].end(); ++ib) {
	      ib->index = index;
	      ++index;
	    }
	  }
	  else {
	    double min_lambda = 0.0;
	    int min_ip = -1;
	    for (int ip = 0; ip < np; ++ip) {
	      double lambda = nu + sqrt((xp_trans[ip][0] - xquad[0])*(xp_trans[ip][0] - xquad[0]) +
					(xp_trans[ip][1] - xquad[1])*(xp_trans[ip][1] - xquad[1]));
	      bool active = false;
	      for (typename list<Basis>::iterator ib = triVec[it].basisList[iq].begin(); 
		   ib != triVec[it].basisList[iq].end(); ++ib) {
		lambda -= ib->phi*sqrt((xp_trans[ib->ip][0] - xp_trans[ip][0])*(xp_trans[ib->ip][0] - xp_trans[ip][0]) +
				       (xp_trans[ib->ip][1] - xp_trans[ip][1])*(xp_trans[ib->ip][1] - xp_trans[ip][1]));
		if (ip == ib->ip) {
		  active = true;
		  break;
		}
	      }
	      if (lambda < min_lambda && !active) {
		min_lambda = lambda;
		min_ip = ip;
	      }
	    }
	    
	    //cout << "lambda: " << min_lambda << endl;
	    if (min_lambda > -1e-8) {
	      done = true;
	      if (normsq > 1e-8) {
		cout << normsq << " " << step << endl;
		//getchar();
	      }
	      //cout << "done!" << endl;
	    }
	    else {
	      Basis this_basis;
	      this_basis.index = nactive;
	      assert(min_ip >= 0);
	      this_basis.ip = min_ip;
	      this_basis.phi = 0.0;
	      triVec[it].basisList[iq].push_back(this_basis);
	    }
	  }
	  
	  /*
	  cout << "phi: " << endl;
	  for (int ii = 0; ii < nactive; ++ii) cout << triVec[it].phi[iq][ind[ii]] << " ";
	  cout << endl;
	  //if (it == 48 && iq == 10) 
	  //getchar();
	  */
	  
	  ++iter;
	  if (iter > 300) {
	    cout << " iter > max iter!!!! : " << it << " " << iq << endl;
	    getchar();
	    done = true;
	  }
	  
	  // compute gradients...
	  if (done) {
	    for (typename list<Basis>::iterator ib = triVec[it].basisList[iq].begin(); 
		   ib != triVec[it].basisList[iq].end(); ++ib) {
	      const int ii = ib->index;
	      b[ii] = (xp_trans[ib->ip][0] - xquad[0])/sqrt((xp_trans[ib->ip][0] - xquad[0])*(xp_trans[ib->ip][0] - xquad[0]) +
							     (xp_trans[ib->ip][1] - xquad[1])*(xp_trans[ib->ip][1] - xquad[1]));
	    }
	    b[nactive] = 0.0;
	    
	    for (int ii = 0; ii < (nactive+1)*(nactive+1); ++ii) A[ii] = A2[ii];
	    {
	      int nrhs = 1;
	      int N = nactive+1;
	      int * ipiv = new int[N];
	      int info;	    
	      dgesv_(&N, &nrhs, A, &N, ipiv,
		     b, &N, &info);
	      assert(info == 0);
	      delete[] ipiv;
	    }
	    
	    for (typename list<Basis>::iterator ib = triVec[it].basisList[iq].begin(); 
		   ib != triVec[it].basisList[iq].end(); ++ib) {
	      const int ii = ib->index;
	      ib->grad_phi[0] = b[ii]; 
	    }
	    
	    for (typename list<Basis>::iterator ib = triVec[it].basisList[iq].begin(); 
		   ib != triVec[it].basisList[iq].end(); ++ib) {
	      const int ii = ib->index;
	      b[ii] = (xp_trans[ib->ip][1] - xquad[1])/sqrt((xp_trans[ib->ip][0] - xquad[0])*(xp_trans[ib->ip][0] - xquad[0]) +
							     (xp_trans[ib->ip][1] - xquad[1])*(xp_trans[ib->ip][1] - xquad[1]));
	    }
	    b[nactive] = 0.0;
	    	    
	    for (int ii = 0; ii < (nactive+1)*(nactive+1); ++ii) A[ii] = A2[ii];
	    {
	      int nrhs = 1;
	      int N = nactive+1;
	      int * ipiv = new int[N];
	      int info;	    
	      dgesv_(&N, &nrhs, A, &N, ipiv,
		     b, &N, &info);
	      assert(info == 0);
	      delete[] ipiv;
	    }
	    
	    for (typename list<Basis>::iterator ib = triVec[it].basisList[iq].begin(); 
		   ib != triVec[it].basisList[iq].end(); ++ib) {
	      const int ii = ib->index;
	      ib->grad_phi[1] = b[ii]; 
	    }
	  }
	  
	  delete[] A;
	  delete[] A2;
	  delete[] b;
	  
	  max_iter = max(max_iter, iter);
	}
	
	// compute gradients down here eventually, only call solver once...
	
      }
    }
    
    cout << "max_iter: " << max_iter << endl;
    delete[] xp_trans;
    
  }
      
  void updatePhisOMP() {
    cout << "updatePhisOMP() " << endl;

    #pragma omp parallel for
    FOR_IT {
      FOR_IQ {
	//cout << it << " " << iq << endl;
	double xquad[2] = {0.0, 0.0};
	FOR_I2 FOR_IVE xquad[i] += triVec[it].ve_x[ive][i]*quad.ve_wgt[iq][ive];
	//cout << "xquad: " << xquad[0] << " " << xquad[1] << endl;
	
	double (*xp_trans)[2] = new double[np][2];
	FOR_IP {
	  FOR_I2 xp_trans[ip][i] = xp[ip][i];
	  if ((xquad[0] - xp_trans[ip][0]) > 0.5*(XMAX-XMIN)) {
	    xp_trans[ip][0] += (XMAX-XMIN);
	  }
	  else if ((xp_trans[ip][0] - xquad[0]) > 0.5*(XMAX-XMIN)) {
	    xp_trans[ip][0] -= (XMAX-XMIN);
	  }
	  if ((xquad[1] - xp_trans[ip][1]) > 0.5*(YMAX-YMIN)) {
	    xp_trans[ip][1] += (YMAX-YMIN);
	  }
	  else if ((xp_trans[ip][1] - xquad[1]) > 0.5*(YMAX-YMIN)) {
	    xp_trans[ip][1] -= (YMAX-YMIN);
	  }	  
	}
	
	bool done = false;
	int iter = 0;
	while (!done) {
	  	    
	  int nactive = triVec[it].basisList[iq].size();
	  assert(nactive > 1);
	  
	  double * A = new double[(nactive+1)*(nactive+1)]; // collumn major!!! (though symmteric, so doesn't matter for now...)
	  double * A2 = new double[(nactive+1)*(nactive+1)];
	  double * b = new double[nactive+1];
	  for (int ii = 0; ii < nactive; ++ii) b[ii] = 0.0;
	  
	  for (typename list<Basis>::iterator ib = triVec[it].basisList[iq].begin(); 
	       ib != triVec[it].basisList[iq].end(); ++ib) {
	    const int ip = ib->ip;
	    const int ii = ib->index;
	    for (typename list<Basis>::iterator jb = triVec[it].basisList[iq].begin(); 
		 jb != triVec[it].basisList[iq].end(); ++jb) {
	      const int jp = jb->ip;
	      const int jj = jb->index;
	      const double dist = sqrt((xp_trans[ip][0] - xp_trans[jp][0])*(xp_trans[ip][0] - xp_trans[jp][0]) +
				       (xp_trans[ip][1] - xp_trans[jp][1])*(xp_trans[ip][1] - xp_trans[jp][1]));
	      A[ii*(nactive+1) + jj] = -dist;
	      b[ii] += dist*(jb->phi);
	    }
	    A[ii*(nactive+1) + ii] = 0.0; // diagonal
	    // lagrange multipliers..
	    A[ii*(nactive+1) + nactive] = 1.0;
	    A[nactive*(nactive+1) + ii] = 1.0;
	    
	    // distance to active set...
	    b[ii] -= sqrt((xp_trans[ip][0] - xquad[0])*(xp_trans[ip][0] - xquad[0]) +
			  (xp_trans[ip][1] - xquad[1])*(xp_trans[ip][1] - xquad[1]));
	  }
	
	  A[(nactive+1)*(nactive+1) - 1] = 0.0;
	  b[nactive] = 0.0;
	    
	  for (int ii = 0; ii < (nactive+1)*(nactive+1); ++ii) A2[ii] = A[ii];
	  double * b2 = new double[nactive+1];
	  for (int ii = 0; ii < (nactive+1); ++ii) b2[ii] = b[ii];
	  
	  {
	    int nrhs = 1;
	    int N = nactive+1;
	    int * ipiv = new int[N];
	    int info;	    
	    dgesv_(&N, &nrhs, A, &N, ipiv,
		   b, &N, &info);
	    if (info != 0) {
	      cout << "info: " << info << endl;
	      cout << "A2: " << endl;
	      for (int ii = 0; ii < nactive+1; ++ii) {
		cout << ii << " | ";
		for (int jj = 0; jj < nactive+1; ++jj) {
		  cout << A2[ii*(nactive+1) + jj] << " ";
		}
		cout << endl;
	      }
	      
	      cout << "phi: " << endl;
	      for (typename list<Basis>::iterator ib = triVec[it].basisList[iq].begin(); 
		   ib != triVec[it].basisList[iq].end(); ++ib)  
		cout << ib->phi << " ";
	      cout << endl;
	      
	      cout << "b2: " << endl;
	      for (int ii = 0; ii < nactive+1; ++ii) cout << b2[ii] << " ";
	      cout << endl;
	      getchar();
	    }
	    //assert(info == 0);
	  
	    delete[] ipiv;
	  }
	  delete[] b2;
	  
	  const double nu = b[nactive];
	  
	  // compute step size...
	  double step = 1.0;	  
	  double normsq = 0.0;
	  {
	    for (typename list<Basis>::iterator ib = triVec[it].basisList[iq].begin(); 
		 ib != triVec[it].basisList[iq].end(); ++ib) {
	      const int ii = ib->index;
	      assert(ii < nactive);
	      normsq += b[ii]*b[ii];
	      if (b[ii] < 0.0) {
		step = min(step, -(ib->phi)/b[ii]);
	      }
	    }
	    if (normsq < 1e-8) {
	      step = 0.0;
	    }
	  }
	  
	  //cout << "step: " << step << endl;
	  
	  if (step > 0.0) {
	    list<Basis>::iterator ib = triVec[it].basisList[iq].begin();
	    while (ib != triVec[it].basisList[iq].end()) {
	      const int ii = ib->index;
	      ib->phi += step*b[ii];
	      if (ib->phi < 1e-10) {
		ib = triVec[it].basisList[iq].erase(ib);
	      }
	      else {
		++ib;
	      }
	    }
	    int index = 0;
	    for (typename list<Basis>::iterator ib = triVec[it].basisList[iq].begin(); 
		 ib != triVec[it].basisList[iq].end(); ++ib) {
	      ib->index = index;
	      ++index;
	    }
	  }
	  else {
	    double min_lambda = 0.0;
	    int min_ip = -1;
	    for (int ip = 0; ip < np; ++ip) {
	      double lambda = nu + sqrt((xp_trans[ip][0] - xquad[0])*(xp_trans[ip][0] - xquad[0]) +
					(xp_trans[ip][1] - xquad[1])*(xp_trans[ip][1] - xquad[1]));
	      bool active = false;
	      for (typename list<Basis>::iterator ib = triVec[it].basisList[iq].begin(); 
		   ib != triVec[it].basisList[iq].end(); ++ib) {
		lambda -= ib->phi*sqrt((xp_trans[ib->ip][0] - xp_trans[ip][0])*(xp_trans[ib->ip][0] - xp_trans[ip][0]) +
				       (xp_trans[ib->ip][1] - xp_trans[ip][1])*(xp_trans[ib->ip][1] - xp_trans[ip][1]));
		if (ip == ib->ip) {
		  active = true;
		  break;
		}
	      }
	      if (lambda < min_lambda && !active) {
		min_lambda = lambda;
		min_ip = ip;
	      }
	    }
	    
	    //cout << "lambda: " << min_lambda << endl;
	    if (min_lambda > -1e-8) {
	      done = true;
	      if (normsq > 1e-8) {
		cout << normsq << " " << step << endl;
		//getchar();
	      }
	      //cout << "done!" << endl;
	    }
	    else {
	      Basis this_basis;
	      this_basis.index = nactive;
	      assert(min_ip >= 0);
	      this_basis.ip = min_ip;
	      this_basis.phi = 0.0;
	      triVec[it].basisList[iq].push_back(this_basis);
	    }
	  }
	  
	  ++iter;
	  if (iter > 300) {
	    cout << " iter > max iter!!!! : " << it << " " << iq << endl;
	    getchar();
	    done = true;
	  }
	  
	  // compute gradients...
	  if (done) {
	    for (typename list<Basis>::iterator ib = triVec[it].basisList[iq].begin(); 
		   ib != triVec[it].basisList[iq].end(); ++ib) {
	      const int ii = ib->index;
	      b[ii] = (xp_trans[ib->ip][0] - xquad[0])/sqrt((xp_trans[ib->ip][0] - xquad[0])*(xp_trans[ib->ip][0] - xquad[0]) +
							     (xp_trans[ib->ip][1] - xquad[1])*(xp_trans[ib->ip][1] - xquad[1]));
	    }
	    b[nactive] = 0.0;
	    
	    for (int ii = 0; ii < (nactive+1)*(nactive+1); ++ii) A[ii] = A2[ii];
	    {
	      int nrhs = 1;
	      int N = nactive+1;
	      int * ipiv = new int[N];
	      int info;	    
	      dgesv_(&N, &nrhs, A, &N, ipiv,
		     b, &N, &info);
	      assert(info == 0);
	      delete[] ipiv;
	    }
	    
	    for (typename list<Basis>::iterator ib = triVec[it].basisList[iq].begin(); 
		   ib != triVec[it].basisList[iq].end(); ++ib) {
	      const int ii = ib->index;
	      ib->grad_phi[0] = b[ii]; 
	    }
	    
	    for (typename list<Basis>::iterator ib = triVec[it].basisList[iq].begin(); 
		   ib != triVec[it].basisList[iq].end(); ++ib) {
	      const int ii = ib->index;
	      b[ii] = (xp_trans[ib->ip][1] - xquad[1])/sqrt((xp_trans[ib->ip][0] - xquad[0])*(xp_trans[ib->ip][0] - xquad[0]) +
							     (xp_trans[ib->ip][1] - xquad[1])*(xp_trans[ib->ip][1] - xquad[1]));
	    }
	    b[nactive] = 0.0;
	    	    
	    for (int ii = 0; ii < (nactive+1)*(nactive+1); ++ii) A[ii] = A2[ii];
	    {
	      int nrhs = 1;
	      int N = nactive+1;
	      int * ipiv = new int[N];
	      int info;	    
	      dgesv_(&N, &nrhs, A, &N, ipiv,
		     b, &N, &info);
	      assert(info == 0);
	      delete[] ipiv;
	    }
	    
	    for (typename list<Basis>::iterator ib = triVec[it].basisList[iq].begin(); 
		   ib != triVec[it].basisList[iq].end(); ++ib) {
	      const int ii = ib->index;
	      ib->grad_phi[1] = b[ii]; 
	    }
	  }
	  
	  delete[] A;
	  delete[] A2;
	  delete[] b;
	  
	}
	
	// compute gradients down here eventually, only call solver once...
	delete[] xp_trans;
	
      }
    }
    
        
  }
  
  void calcRhs(double (*rhs_x)[2], double (*rhs_gp)[2], double * rhs_ep) {
    cout << "calcRhs()" << endl;
    
    // compute filtered advection velocity...
    double (*u_advect)[2] = new double[np][2];
    double * vol = new double[np];
    FOR_IP FOR_I2 u_advect[ip][i] = 0.0;
    FOR_IP vol[ip] = 0.0;
    FOR_IT {
      FOR_IQ {
	double this_u[2] = {0.0, 0.0};
	for (typename list<Basis>::iterator ib = triVec[it].basisList[iq].begin(); 
	     ib != triVec[it].basisList[iq].end(); ++ib) {
	  const int ip = ib->ip;
	  FOR_I2 this_u[i] += up[ip][i]*(ib->phi);
	}
	
	const double weight = quad.p_wgt[iq]*triVec[it].area;
	for (typename list<Basis>::iterator ib = triVec[it].basisList[iq].begin(); 
	       ib != triVec[it].basisList[iq].end(); ++ib) {
	  const int ip = ib->ip;
	  FOR_I2 u_advect[ip][i] += weight*(ib->phi)*this_u[i];
	  vol[ip] += weight*(ib->phi);
	}
      }
    }
    FOR_IP FOR_I2 u_advect[ip][i] /= vol[ip];

    // since basis interpolatory, rhs is just nodal values...
    FOR_IP {
      //FOR_I2 rhs_x[ip][i] = up[ip][i];
      FOR_I2 rhs_x[ip][i] = u_advect[ip][i];
    }
    delete[] u_advect;
    delete[] vol;

    FOR_IP FOR_I2 rhs_gp[ip][i] = 0.0;
    FOR_IP        rhs_ep[ip]    = 0.0;
    
    // calc momentum rhs...
    FOR_IT {
      FOR_IQ {
	/*
	double xquad[2] = {0.0, 0.0};
	FOR_I2 FOR_IVE xquad[i] += triVec[it].ve_x[ive][i]*quad.ve_wgt[iq][ive];
	
	double exact_rho, exact_p;
	double exact_u[2];
	double exact_dp[2];
	calcEulerVortex(exact_rho, exact_u, exact_p, exact_dp, xquad);
	*/

	const double weight = quad.p_wgt[iq]*triVec[it].area;
	for (typename list<Basis>::iterator ib = triVec[it].basisList[iq].begin(); 
	       ib != triVec[it].basisList[iq].end(); ++ib) {
	  const int ip = ib->ip;
	  //FOR_I2 rhs_gp[ip][i] += weight*ib->phi*exact_dp[i];
	  for (typename list<Basis>::iterator jb = triVec[it].basisList[iq].begin(); 
	       jb != triVec[it].basisList[iq].end(); ++jb) {
	    const int jp = jb->ip;
	    FOR_I2 rhs_gp[ip][i] -= weight*(ib->phi)*pp[jp]*(jb->grad_phi[i]);
	    //FOR_I2 rhs_gp[ip][i] -= weight*(jb->phi)*pp[jp]*(ib->grad_phi[i]);
	    //cout << (jb->phi)*(ib->grad_phi[0]) << " " << (ib->phi)*(jb->grad_phi[0]) << " " << (jb->phi)*(ib->grad_phi[1]) << " " << (ib->phi)*(jb->grad_phi[1]) << endl; 
	    //getchar();
	    //FOR_I2 rhs_gp[ip][i] += 0.5*(pp[ip] + pp[jp])*weight*(-(jb->phi)*(ib->grad_phi[i]) - (ib->phi)*(jb->grad_phi[i]));
	  }
	}
      }
    }
    
   }

  void run() {
    cout << "run()" << endl;
    
    double (*dg0)[2] = new double[np][2];
    double (*dg1)[2] = new double[np][2];
    double (*dg2)[2] = new double[np][2];
    
    double (*dx0)[2] = new double[np][2];
    double (*dx1)[2] = new double[np][2];
    double (*dx2)[2] = new double[np][2];

    double (*de0)    = new double[np];
    double (*de1)    = new double[np];
    double (*de2)    = new double[np];
    
    temporalHook();
        
    char filename[32];
    sprintf(filename,"tri.%08d.dat",step);
    writeTri(filename);
    //sprintf(filename,"lp.%08d.dat",step);
    //writeLp(filename);

    bool done = false;
    while(!done) {
      
      ++step;
      setDt();
      time += dt;

      if (step%check_interval == 0) {
	cout <<
	  "\n----------------------------------------------------------\n" <<
	  " starting step: " << step << " time: " << time << " dt: " << dt <<
	  "\n----------------------------------------------------------" << endl;
      }
         
      calcRhs(dx0,dg0,de0);
      FOR_IP FOR_I2 Gp[ip][i] += dt*dg0[ip][i];
      FOR_IP FOR_I2 xp[ip][i] += dt*dx0[ip][i];
      FOR_IP        Ep[ip]    += dt*de0[ip];
      updateAfterAdvect();
      
      calcRhs(dx1,dg1,de1);
      FOR_IP FOR_I2 Gp[ip][i] += dt*(-3.0*dg0[ip][i] + dg1[ip][i])/4.0;
      FOR_IP FOR_I2 xp[ip][i] += dt*(-3.0*dx0[ip][i] + dx1[ip][i])/4.0;
      FOR_IP        Ep[ip]    += dt*(-3.0*de0[ip]    + de1[ip]   )/4.0;
      updateAfterAdvect();
      
      calcRhs(dx2,dg2,de2);
      FOR_IP FOR_I2 Gp[ip][i] += dt*(-dg0[ip][i] - dg1[ip][i] + 8.0*dg2[ip][i])/12.0;
      FOR_IP FOR_I2 xp[ip][i] += dt*(-dx0[ip][i] - dx1[ip][i] + 8.0*dx2[ip][i])/12.0;
      FOR_IP        Ep[ip]    += dt*(-de0[ip]    - de1[ip]    + 8.0*de2[ip]   )/12.0;
      updateAfterAdvect();
      
      if (step%check_interval == 0) {
      	temporalHook();
      }
      
      if (step%write_interval == 0) {
	char filename[32];
	sprintf(filename,"tri.%08d.dat",step);
	writeTri(filename);
	//sprintf(filename,"lp.%08d.dat",step);
	//writeLp(filename);
      }
      
      if (step >= nsteps) done = true;
      
    }
    
    delete[] dg0;
    delete[] dg1;
    delete[] dg2;

    delete[] dx0;
    delete[] dx1;
    delete[] dx2;
      
    delete[] de0;
    delete[] de1;
    delete[] de2;
    
  }

  void writeTecplot(char * filename) {
    
    cout << "writing " << filename << endl;
    
    FILE * fp = fopen(filename,"w");
    
    int tri_count = 0;
    FOR_IT {
      if ((triVec[it].ve_bit[0] == 0) && (triVec[it].ve_bit[1] == 0) && (triVec[it].ve_bit[2] == 0))
      ++tri_count;
    }
    
    fprintf(fp,"TITLE = \"%s\"\n",filename);
    fprintf(fp,"VARIABLES = \"X\"\n");
    fprintf(fp,"\"Y\"\n");
    fprintf(fp,"\"VOL\"\n");
    fprintf(fp,"\"M\"\n");
    fprintf(fp,"\"RHO\"\n");
    fprintf(fp,"\"RHO_ERROR\"\n");
    fprintf(fp,"\"U_ERROR-X\"\n");
    fprintf(fp,"\"U_ERROR-Y\"\n");
    fprintf(fp,"\"U-X\"\n");
    fprintf(fp,"\"U-Y\"\n");
    
    fprintf(fp,"ZONE T=\"%s\"\n",filename);
    fprintf(fp,"N=%d, E=%d, F=FEPOINT, ET=TRIANGLE\n", np, tri_count);
    
    // back out pointwise quantity 
    double * my_rho = new double[np]; 
    double (*my_u)[2] = new double[np][2];

    #pragma omp parallel for 
    FOR_IP {
      my_rho[ip] = 0.0;
      FOR_I2 my_u[ip][i] = 0.0;

      int icell, jcell;
      findCell(icell, jcell, xp[ip]);
    
      const double coeff = 1.0/denomp[ip];
            	
      for (int ivar = -1; ivar <= 1; ++ivar) {
	for (int jvar = -1; jvar <= 1; ++jvar) {
	  const int this_icell = icell + ivar;
	  const int this_jcell = jcell + jvar;
	  for (typename list<Point>::iterator il = cells[this_icell + ncells*this_jcell].begin(); il != cells[this_icell + ncells*this_jcell].end(); ++il) {
	    my_rho[ip] += coeff*rhop[il->ip]*evalExp(xp[ip], il->x, lambdap[ip]);
	    FOR_I2 my_u[ip][i] += coeff*up[il->ip][i]*evalExp(xp[ip], il->x, lambdap[ip]);
	  }
	}
      }
    }
    

    FOR_IP  {      
      fprintf(fp,"%18.15le %18.15le %18.15le %18.15le %18.15le %18.15le %18.15le %18.15le %18.15le %18.15le\n", xp[ip][0], xp[ip][1], volp[ip], Mp[ip], my_rho[ip], rho_error[ip], u_error[ip][0], u_error[ip][1], my_u[ip][0], my_u[ip][1]);
    }

    delete[] my_rho;
    delete[] my_u;

    FOR_IT {
    if ((triVec[it].ve_bit[0] == 0) && (triVec[it].ve_bit[1] == 0) && (triVec[it].ve_bit[2] == 0))
      fprintf(fp,"%d %d %d\n",triVec[it].ve_ip[0]+1,triVec[it].ve_ip[1]+1,triVec[it].ve_ip[2]+1);
    }
    
    fclose(fp);
    
  }

  void writeTri(char * filename) {
    
    cout << "writing " << filename << endl;
    
    FILE * fp = fopen(filename,"w");
    
    int tri_count = 0;
    FOR_IT {
      if ((triVec[it].ve_bit[0] == 0) && (triVec[it].ve_bit[1] == 0) && (triVec[it].ve_bit[2] == 0))
      ++tri_count;
    }
    
    fprintf(fp,"TITLE = \"%s\"\n",filename);
    fprintf(fp,"VARIABLES = \"X\"\n");
    fprintf(fp,"\"Y\"\n");
    fprintf(fp,"\"M\"\n");
    fprintf(fp,"\"G-X\"\n");
    fprintf(fp,"\"G-Y\"\n");
    fprintf(fp,"\"RHO\"\n");
    fprintf(fp,"\"VOL\"\n");
    fprintf(fp,"\"P\"\n");
    fprintf(fp,"\"U-X\"\n");
    fprintf(fp,"\"U-Y\"\n");
    fprintf(fp,"\"U_EXACT-X\"\n");
    fprintf(fp,"\"U_EXACT-Y\"\n");
    
    fprintf(fp,"ZONE T=\"%s\"\n",filename);
    fprintf(fp,"N=%d, E=%d, F=FEPOINT, ET=TRIANGLE\n", np, tri_count);
    
    FOR_IP  {      
      double exact_rho, exact_p;
      double exact_u[2];
      double exact_dp[2];
      calcEulerVortex(exact_rho, exact_u, exact_p, exact_dp, xp[ip]);
      
      fprintf(fp,"%18.15le %18.15le %18.15le %18.15le %18.15le %18.15le %18.15le %18.15le %18.15le %18.15le %18.15le %18.15le\n", 
                  xp[ip][0], xp[ip][1], Mp[ip],
	      Gp[ip][0], Gp[ip][1], rhop[ip], volp[ip],
                  pp[ip], up[ip][0], up[ip][1], 
                  exact_u[0], exact_u[1]);
    }
    
    FOR_IT {
      if ((triVec[it].ve_bit[0] == 0) && (triVec[it].ve_bit[1] == 0) && (triVec[it].ve_bit[2] == 0))
	fprintf(fp,"%d %d %d\n",triVec[it].ve_ip[0]+1,triVec[it].ve_ip[1]+1,triVec[it].ve_ip[2]+1);
    }
    
    fclose(fp);
    
  }

  inline void calcEulerVortex(double &rho, double u[2], double &p, double dp[2], const double x[2]) {
    
    const double u_inf = 0.5;
    
    // x,y position of the vortex
    const double x0 = 0.0;
    const double y0 = 0.0;
  
    const double Ma_inf = u_inf/sqrt(gam);
    const double rc = 1.0;
  
    // circulation parameter...
    const double e_twopi = 0.08; // normal
    //const double e_twopi = 0.5;
    
    // setup...
    const double coeff = 0.5*e_twopi*e_twopi*(gam-1.0)*Ma_inf*Ma_inf;
    
    double dx = x[0] - x0;
    double dy = x[1] - y0;
    
    const double f0 = 1.0 - (dx*dx  + dy*dy)/(rc*rc);
    rho = pow(1.0 - coeff*exp(f0), 1.0/(gam-1.0));
      
    u[0] = -u_inf*(e_twopi*dy/rc*exp(0.5*f0));
    u[1] = u_inf*(e_twopi*dx/rc*exp(0.5*f0));
    
    p = pow(1.0 - coeff*exp( f0 ), gam/(gam-1.0) );
       
    dp[0] = 2.0*coeff*gam*dx*exp(f0)*rho/((gam-1.0)*rc*rc);
    dp[1] = 2.0*coeff*gam*dy*exp(f0)*rho/((gam-1.0)*rc*rc);
              
  }

  /*
  void writeTecplot2(char * filename) {
    
    cout << "writing " << filename << endl;
    
    FILE * fp = fopen(filename,"w");
    
    int tri_count = 0;
    FOR_IT {
      if ((triVec[it].ve_bit[0] == 0) && (triVec[it].ve_bit[1] == 0) && (triVec[it].ve_bit[2] == 0))
      ++tri_count;
    }
    
    fprintf(fp,"TITLE = \"%s\"\n",filename);
    fprintf(fp,"VARIABLES = \"X\"\n");
    fprintf(fp,"\"Y\"\n");
    fprintf(fp,"\"RHO\"\n");
    fprintf(fp,"\"U-X\"\n");
    fprintf(fp,"\"U-Y\"\n");
    fprintf(fp,"\"P\"\n");
    fprintf(fp,"\"DP-X\"\n");
    fprintf(fp,"\"DP-Y\"\n");
        
    fprintf(fp,"ZONE T=\"%s\"\n",filename);
    fprintf(fp,"N=%d, E=%d, F=FEPOINT, ET=TRIANGLE\n", np, tri_count);
    
    FOR_IP  {  
      double this_rho;
      double this_u[2];
      double this_p;
      double this_dp[2];
      calcEulerVortex2(this_rho,this_u,this_p,this_dp,xp[ip]);
      fprintf(fp,"%18.15le %18.15le %18.15le %18.15le %18.15le %18.15le %18.15le %18.15le\n", xp[ip][0], xp[ip][1], this_rho, this_u[0], this_u[1], this_p, this_dp[0], this_dp[1]);
    }

    FOR_IT {
    if ((triVec[it].ve_bit[0] == 0) && (triVec[it].ve_bit[1] == 0) && (triVec[it].ve_bit[2] == 0))
      fprintf(fp,"%d %d %d\n",triVec[it].ve_ip[0]+1,triVec[it].ve_ip[1]+1,triVec[it].ve_ip[2]+1);
    }
    
    fclose(fp);
    
  }
  inline void calcEulerVortex2(double &rho, double u[2], double &p, double dp[2], const double x[2]) {
    
    const double u_inf = 0.5;
    
    // x,y position of the vortex
    const double x0 = 0.0;
    const double y0 = 0.0;
  
    const double Ma_inf = u_inf/sqrt(gam);
    const double rc = 1.0;
  
    // circulation parameter...
    const double e_twopi = 0.08; // normal
    //const double e_twopi = 0.16;
    
    // setup...
    const double coeff = 0.5*e_twopi*e_twopi*(gam-1.0)*Ma_inf*Ma_inf;
    
    double dx = x[0] - x0;
    double dy = x[1] - y0;
    const double f0 = 1.0 - (dx*dx  + dy*dy)/(rc*rc);
    rho = pow(1.0 - coeff*exp(f0), 1.0/(gam-1.0));
      
    u[0] = -u_inf*(e_twopi*dy/rc*exp(0.5*f0));
    u[1] = u_inf*(e_twopi*dx/rc*exp(0.5*f0));
    
    p = pow(1.0 - coeff*exp( f0 ), gam/(gam-1.0) );
       
    dp[0] = 2.0*coeff*gam*dx*exp(f0)*rho/((gam-1.0)*rc*rc);
    dp[1] = 2.0*coeff*gam*dy*exp(f0)*rho/((gam-1.0)*rc*rc);
    
    double v;
    double r = sqrt(dx*dx + dy*dy);
    v = sqrt(sqrt(dp[0]*dp[0] + dp[1]*dp[1])/rho*r);
    
    cout << v << " " << sqrt(u[0]*u[0] + u[1]*u[1]) << " " << v - sqrt(u[0]*u[0] + u[1]*u[1])<< endl;
    getchar();
          
  }
  */
};
