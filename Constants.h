#ifndef __CONSTANTS_H_INCLUDED__   // if Constants.h hasn't been included yet...
#define __CONSTANTS_H_INCLUDED__

#ifdef CALCULATE_MOM
#define DECLARE_LMSUM()	threevector LinearMomentumSum(0.0,0.0,0.0);
#define DECLARE_AMSUM()	threevector AngularMomentumSum(0.0,0.0,0.0);
#define SAVE_MOM(x)	RunDataFile << x
#else
#define DECLARE_LMSUM()
#define DECLARE_AMSUM()
#define SAVE_MOM(x)
#endif

#ifdef SELF_CONS_CHARGE
#define ADD_CHARGE()	Charge += SPEC_CHARGE;
#define SAVE_CHARGE(x)	RunDataFile << x;
#else
#define ADD_CHARGE()
#define SAVE_CHARGE(x)
#endif

#ifdef SAVE_TRACKS 
#define DECLARE_TRACK()	std::ofstream TrackDataFiles	
#define PRIVATE_FILES()	private(TrackDataFiles)
#define OPEN_TRACK(x)	TrackDataFiles.open(x)
#define RECORD_TRACK(x)	TrackDataFiles << x
#define CLOSE_TRACK()	TrackDataFiles.close()
#else
#define DECLARE_TRACK()
#define PRIVATE_FILES()
#define OPEN_TRACK(x)
#define RECORD_TRACK(x)
#define CLOSE_TRACK()
#endif 

#ifdef SAVE_ANGULAR_VEL
#define DECLARE_AVEL()	std::ofstream AngularDataFile;
#define OPEN_AVEL()	AngularDataFile.open("Data/DMP_AngVel.txt");
#define SAVE_AVEL()	AngularDataFile << "\n" << j << "\t" << TotalAngularVel;
#define CLOSE_AVEL()	AngularDataFile.close();
#else
#define DECLARE_AVEL()
#define OPEN_AVEL()
#define SAVE_AVEL()
#define CLOSE_AVEL()
#endif 

#ifdef SAVE_LINEAR_MOM
#define DECLARE_LMOM()	std::ofstream LinearDataFile;
#define OPEN_LMOM()	LinearDataFile.open("Data/DMP_LinMom.txt");
#define SAVE_LMOM()	LinearDataFile << "\n" << j << "\t" << SpeciesMass*FinalVelocity;
#define CLOSE_LMOM()	LinearDataFile.close();
#else
#define DECLARE_LMOM()
#define OPEN_LMOM()
#define SAVE_LMOM()
#define CLOSE_LMOM()
#endif 

#ifdef TEST_VELPOSDIST
#define PRINT_VPD(x)	std::cout << x;
#else
#define PRINT_VPD(x)
#endif

#ifdef TEST_FINALPOS
#define PRINT_FP(x)	std::cout << x;
#else
#define PRINT_FP(x)
#endif

#ifdef TEST_CHARGING
#define PRINT_CHARGE(x)	std::cout << x;
#else
#define PRINT_CHARGE(x)
#endif

#ifdef TEST_ANGVEL
#define PRINT_AVEL(x)	std::cout << x;
#else
#define PRINT_AVEL(x)
#endif

#if defined TEST_ENERGY 
#define INITIAL_VEL()	threevector InitialVel = Velocity;
#define INITIAL_POT()	double InitialPot = Charge/(4.0*PI*e0norm*Position.mag3());
#define RESET_VEL()	InitialVel = Velocity;
#define RESET_POT()	InitialPot = Charge/(4.0*PI*e0norm*Position.mag3());
#define FINAL_POT()	double FinalPot = Charge/(4.0*PI*e0norm*FinalPosition.mag3());
#define PRINT_ENERGY(x)	std::cout << x
#else
#define INITIAL_VEL()
#define INITIAL_POT()
#define RESET_VEL()
#define RESET_POT()
#define FINAL_POT()
#define PRINT_ENERGY(x)
#endif

#ifdef PAUSE
#define Pause(); std::cin.get();
#else
#define Pause();
#endif

extern double Kb;  		// (kg m^2 s^-2 K^1) || (J K^-1)
extern double R;		// https://en.wikipedia.org/wiki/Gas_constant 
extern double echarge; 		// C 
extern double Me;		// kg, mass of electron
extern double Mp;		// kg, mass of ion (And Neutrons)
extern double AvNo; 		// mol^-1
extern double PI;
extern double AMU;	 	// kg, Atomic Mass unit
extern double Richardson;	// A/(metres K^2), Richardson constant
extern double c; 		// m/s, Speed of light
extern double h; 		// m^2 kg / s, Planck's Constant
extern double epsilon0; 	// F/m, vacuum permittivity
extern double Sigma;		// Boltsmann constant: W/(m^2 K^4) 
extern double naturale;		// Mathematical constant

#endif
