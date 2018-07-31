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
#define ADD_CHARGE()	PotentialNorm += Particles[i].Species_Charge;
#define SAVE_CHARGE(x)	RunDataFile << x;
#else
#define ADD_CHARGE()
#define SAVE_CHARGE(x)
#endif

#ifdef SAVE_TRACKS 
#define DECLARE_TRACK()	std::ofstream TrackDataFiles	
#define PRIVATE_FILES()	private(TrackDataFiles)
#define OPEN_TRACK(x)	TrackDataFiles.open(x,std::ofstream::app)
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
#define OPEN_AVEL()	AngularDataFile.open("Data/DiMPl_AngVel.txt");
#define HEAD_AVEL()	AngularDataFile << "#Collect num\tSimulated num\tLx\tLy\tLz";
#define SAVE_AVEL()	AngularDataFile << "\n" << j << "\t" << i << "\t" << TotalAngularVel;
#define CLOSE_AVEL()	AngularDataFile.close();
#else
#define DECLARE_AVEL()
#define HEAD_AVEL()
#define OPEN_AVEL()
#define SAVE_AVEL()
#define CLOSE_AVEL()
#endif 

#ifdef SAVE_LINEAR_MOM
#define DECLARE_LMOM()	std::ofstream LinearDataFile;
#define OPEN_LMOM()	LinearDataFile.open("Data/DiMPl_LinMom.txt");
#define HEAD_LMOM()	LinearDataFile << "#Collect num\tSimulated num\tPx\tPy\tPz";
#define SAVE_LMOM()	LinearDataFile << "\n" << j << "\t" << i << "\t" << Particles[i].Species_Mass*Particles[i].Velocity;
#define CLOSE_LMOM()	LinearDataFile.close();
#else
#define DECLARE_LMOM()
#define OPEN_LMOM()
#define HEAD_LMOM()
#define SAVE_LMOM()
#define CLOSE_LMOM()
#endif 

#ifdef SAVE_CHARGING
#define DECLARE_CHA()	std::ofstream ChargeDataFile;
#define OPEN_CHA()	ChargeDataFile.open("Data/DiMPl_Charge.txt");
#define HEAD_CHA()	ChargeDataFile << "#Collect num\tSimulated num\tPotential (1/echarge)";
#define SAVE_CHA()	ChargeDataFile << "\n" << j << "\t" << i << "\t" << PotentialNorm;
#define CLOSE_CHA()	ChargeDataFile.close();
#else
#define DECLARE_CHA()
#define OPEN_CHA()
#define HEAD_CHA()
#define SAVE_CHA()
#define CLOSE_CHA()
#endif 

#ifdef SAVE_ENDPOS
#define DECLARE_EPOS()	std::ofstream EndPosDataFile;
#define OPEN_EPOS()	EndPosDataFile.open("Data/DiMPl_EndPos.txt");
#define HEAD_EPOS()	EndPosDataFile << "#Collect num\tSimulated num\tx\ty\tz\tvx\tvy\tvz\tv·r";
#define SAVE_EPOS()	EndPosDataFile << "\n" << j << "\t" << i << "\t" << Particles[i].Position << "\t" << Particles[i].Velocity << "\t" << Particles[i].Velocity*(Particles[i].Position.getunit());
#define CLOSE_EPOS()	EndPosDataFile.close();
#else
#define DECLARE_EPOS()
#define OPEN_EPOS()
#define HEAD_EPOS()
#define SAVE_EPOS()
#define CLOSE_EPOS()
#endif 

#ifdef SAVE_SPECIES
#define DECLARE_SPEC()	std::ofstream SpeciesDataFile;
#define OPEN_SPEC()	SpeciesDataFile.open("Data/DiMPl_Species.txt");
#define HEAD_SPEC()	SpeciesDataFile << "#Collect num\tSimulated num\tSpecies Charge(echarge)";
#define SAVE_SPEC()	SpeciesDataFile << "\n" << j << "\t" << i << "\t" << Particles[i].Species_Charge;
#define CLOSE_SPEC()	SpeciesDataFile.close();
#else
#define DECLARE_SPEC()
#define OPEN_SPEC()
#define HEAD_SPEC()
#define SAVE_SPEC()
#define CLOSE_SPEC()
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

#ifdef TEST_ANGMOM
#define DECLARE_AMOM()	threevector INITIAL_AMOM(0.0,0.0,0.0),FINAL_AMOM(0.0,0.0,0.0);
#define SAVE_I_AMOM(x)	INITIAL_AMOM = x;
#define SAVE_F_AMOM(x)	FINAL_AMOM = x;
#define PRINT_AMOM(x)	std::cout << x;
#else
#define DECLARE_AMOM()
#define SAVE_I_AMOM(x)
#define SAVE_F_AMOM(x)
#define PRINT_AMOM(x)
#endif

#if defined TEST_ENERGY 
#define INITIAL_VEL()	threevector InitialVel = Particles[i].Velocity;
#define INITIAL_POT()	double InitialPot = echarge*echarge*PotentialNorm/(4.0*PI*epsilon0*Particles[i].Position.mag3()*Radius);
#define RESET_VEL()	InitialVel = Particles[i].Velocity;
#define RESET_POT()	InitialPot = echarge*echarge*PotentialNorm/(4.0*PI*epsilon0*Particles[i].Position.mag3()*Radius);
#define FINAL_POT()	double FinalPot = echarge*echarge*PotentialNorm/(4.0*PI*epsilon0*FinalParticles[i].Position.mag3()*Radius);
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
