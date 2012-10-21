#include "paco.hpp"

template <int NQUAD>
class MyPaco : public Paco<175> {
public:

  double M_init;
  double G_init[2];
  double E_init;
  double E_init2;

  MyPaco() {
    cout << "MyPaco()" << endl;
    assert(NQUAD == 175);
  }
  
  void initialHook() {
    cout << "MyPaco::initialHook()" << endl;

    FOR_IP {
      Mp[ip] = 0.0;
      FOR_I2 Gp[ip][i] = 0.0;   
      Ep[ip] = 0.0;
    }

    FOR_IT {
      FOR_IQ {
	double xquad[2] = {0.0, 0.0};
	FOR_I2 FOR_IVE xquad[i] += triVec[it].ve_x[ive][i]*quad.ve_wgt[iq][ive];
		
	double this_rho, this_p;
	double this_u[2];
	double this_dp[2];
	calcEulerVortex(this_rho, this_u, this_p, this_dp, xquad);

	const double weight = quad.p_wgt[iq]*triVec[it].area;

	for (typename list<Basis>::iterator ib = triVec[it].basisList[iq].begin(); 
	       ib != triVec[it].basisList[iq].end(); ++ib) {
	  const int ip = ib->ip;
	  const double this_phi = ib->phi;
	  Mp[ip] += weight*this_phi*this_rho;
	  FOR_I2 Gp[ip][i] += weight*this_phi*this_rho*this_u[i];
	  Ep[ip] += weight*this_phi*(0.5*this_rho*(this_u[0]*this_u[0] + this_u[1]*this_u[1]) + this_p/(gam-1.0));
	}
      }
    }
    
  }

  void temporalHook() {
    cout << "MyPaco::temporalHook()" << endl;
    
    double rhol2_error = 0.0;
    double ul2_error[2] = {0.0, 0.0};
    double pl2_error = 0.0;
    
    double rhol2_denom = 0.0;
    double ul2_denom[2] = {0.0, 0.0};
    double pl2_denom = 0.0;

    double this_E = 0.0;
    double this_M = 0.0;
    double this_G[2] = {0.0, 0.0};

    FOR_IT {
      FOR_IQ {
	double this_rho = 0.0;
	double this_p = 0.0;
	double this_u[2] = {0.0, 0.0};

	for (typename list<Basis>::iterator ib = triVec[it].basisList[iq].begin(); 
	       ib != triVec[it].basisList[iq].end(); ++ib) {
	  const int ip = ib->ip;
	  this_rho += rhop[ip]*ib->phi;
	  this_p   += pp[ip]*ib->phi;
	  FOR_I2 this_u[i] += up[ip][i]*ib->phi;
	}

	double xquad[2] = {0.0, 0.0};
	FOR_I2 FOR_IVE xquad[i] += triVec[it].ve_x[ive][i]*quad.ve_wgt[iq][ive];
	
	double exact_rho, exact_p;
	double exact_u[2];
	double exact_dp[2];
	calcEulerVortex(exact_rho, exact_u, exact_p, exact_dp, xquad);
	
	const double weight = quad.p_wgt[iq]*triVec[it].area;
	
	rhol2_error += weight*(exact_rho - this_rho)*(exact_rho - this_rho);
	rhol2_denom += weight*exact_rho*exact_rho;
	pl2_error += weight*(exact_p - this_p)*(exact_p - this_p);
	pl2_denom += weight*exact_p*exact_p;
	FOR_I2 ul2_error[i] += weight*(exact_u[i] - this_u[i])*(exact_u[i] - this_u[i]);
	FOR_I2 ul2_denom[i] += weight*exact_u[i]*exact_u[i];
	
	this_M += weight*this_rho;
	FOR_I2 this_G[i] += weight*this_rho*this_u[i];
	this_E += weight*(0.5*this_rho*(this_u[0]*this_u[0] + this_u[1]*this_u[1]) + this_p/(gam-1.0));
      }
    }
    
    cout << " > time, rhoL2, uxL2, uyL2, pl2: " << time << " " << sqrt(rhol2_error/rhol2_denom) 
	 << " " << sqrt(ul2_error[0]/ul2_denom[0]) 
	 << " " << sqrt(ul2_error[1]/ul2_denom[1]) 
	 << " " << sqrt(pl2_error/pl2_denom)  << endl;
        
    if (step == 0) {
      M_init = this_M;
      FOR_I2 G_init[i] = this_G[i];
      E_init = this_E;
    }
    
    cout << " > time, mass, momentum, energy: " << time << " " << this_M - M_init << " " << this_G[0] - G_init[0] << " " << this_G[1] - G_init[1] << " " << this_E - E_init  << endl;
    
  }  

};

int main(int argc,char * argv[]) {
  
  MyPaco<175> * solver = new MyPaco<175>;
  solver->init();
  solver->run();
  
  return(0);
}
