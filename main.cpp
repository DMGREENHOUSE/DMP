#define CALCULATE_MOM
#define SELF_CONS_CHARGE

//#define SAVE_TRACKS
#define SAVE_ANGULAR_VEL
#define SAVE_CHARGING
#define SAVE_LINEAR_MOM
#define SAVE_ENDPOS
#define SAVE_SPECIES

#define SPHERICAL_INJECTION
//#define POINT_INJECTION
//#define NO_SPHERE

//#define TEST_VELPOSDIST
//#define TEST_FINALPOS
//#define TEST_CLOSEST_APPROACH
//#define TEST_CHARGING
//#define TEST_ANGMOM
//#define TEST_ENERGY

/* If `SCEPTIC_REINJECTION` is defined, and the simulation domain is
   spherical (i.e. if `SPHERICAL` is also defined), try to mimic how Ian
   Hutchinson's SCEPTIC reinjects particles.
   If running a Rutherford scattering test, this should not be defined. */
//#define SCEPTIC_REINJECTION

#include <omp.h>	// For parallelisation

#include <chrono>	// for chrono::high_resolution_clock::now().time_since_epoch().count();
#include <iostream>	// for std::cout
#include <array>	
#include <random>	// for std::normal_distribution<> etc.
#include <fstream>	// for std::ofstream
#include <ctime>	// for clock()
#include <math.h> 	// for fabs()
#include <sstream>	// for std::stringstream
#include <assert.h>	// for assert()
#include <algorithm>	// for std::min()
#include <vector>	// for std::vector

//#include "Function.h"	// for LambertW() Function to calculate Debye-Huckel Screening Length
#include "Constants.h"	// Define Pre-processor Directives and constants
#include "threevector.h"// for threevector class
#include "rand_mwts.h"	// Functions to generate correct 1-way maxwellian flux


struct ParticleData{
	threevector 	Position;
	threevector 	Velocity;
	int 		Species_Charge;
	double		Species_Mass;
	unsigned int	Reflections;
};

#ifdef SCEPTIC_REINJECTION


/* Convert a velocity in spherical coordinates to a velocity in Cartesian
   coordinates, at a position given in spherical coordinates. */
void velocity_in_cartesian_coords(double* v, double phi, double theta, double* vx, double* vy, double* vz)
{
	*vx = (v[0] * cos(phi) * sin(theta))
	      - (v[1] * sin(phi)) + (v[2] * cos(phi) * cos(theta));
	*vy = (v[0] * sin(phi) * sin(theta))
	      + (v[1] * cos(phi)) + (v[2] * sin(phi) * cos(theta));
	*vz = (v[0] * cos(theta)) - (v[2] * sin(theta));
}

/* Compute the `q`th quantile of the inverse cumulative distribution
   function of a particle's speed at infinity, according to Hutchinson
   (2003)'s eqs. 18 & 19 in the special case of no flow.

   19/05/2014: I've compared this function's output against SCEPTIC's
   `finvtfunc` and Mathematica's for `chi` = 0; this function matches
   Mathematica's results except at extreme `q` values (where the inverse
   CDF is nearly flat) and differs in the 2nd/3rd decimal place from
   SCEPTIC's results. Since the Mathematica results are derived semi-
   analytically and this function pretty much matches those, the
   SCEPTIC function appears to be the one that's (slightly) wrong here! */
double u_no_flow_icdf(double chi, double q)
{
	double omc = 1.0 - chi;
	double p;
	double u = 1.29951;  /* use (`chi` = 0) median as starting guess */
	/* FIXME: if/when I start using nonzero `chi` for this function,
	   use an approximate expression for the median as a func. of `chi`
	   to initialize `u`. */
	double usq;

	/* The Newton-Raphson method does the inversion; it needs 4.75
	   iterations on average when `chi` is zero. */
	do {
		usq = u * u;
		p = 1.0 - ((usq + omc) * exp(-usq) / omc) - q;
		u -= p / (2.0 * u * exp(-usq) * (chi - usq) / -omc);
	} while ((p * p) > 1e-16);  /* squaring because p may be negative */

	return u;
}

/* Rotate the position vector (`x`, `y`, `z`) about the direction vector `d`
   (with no z-component!) through an angle with sine `s` and cosine `c`.
   Cf. <http://inside.mines.edu/~gmurray/ArbitraryAxisRotation/>,
   end of section 5.2. */
void rotate_vec_about_xy_dir(const double* d, double s, double c, double *x, double* y, double* z, double offset)
{
	const double omc = 1.0 - c;
	double temp[3];

	temp[0] = d[0] * ((d[0] * *x) + (d[1] * *y)) * omc;
	temp[0] += (*x * c) + (d[1] * *z * s);
	temp[1] = d[1] * ((d[0] * *x) + (d[1] * *y)) * omc;
	temp[1] += (*y * c) - (d[0] * *z * s);
	temp[2] = *z * c;
	temp[2] += ((d[0] * *y) - (d[1] * *x)) * s;
	*x = offset + temp[0];
	*y = offset + temp[1];
	*z = offset + temp[2];
}

/* Solve the orbit integral for the angle `alpha` between an initially
   distant particle's initial velocity and the position where it would enter
   the simulation domain, as Ian Hutchinson's SCEPTIC does. The arguments
   `s`, `chi_b` & `u2` have the same meaning as in Hutchinson's work (the
   ratio of the impact parameter `b` to the simulation domain radius, the
   normalized edge electric potential, and the square of the normalized
   velocity at infinity respectively); `Rol` denotes the ratio of the domain
   radius to the relevant (Debye) shielding length. The integration
   algorithm is a crudely adaptive Simpson's rule; usually this works well
   because the orbit integrand is usually a smooth, well-defined curve.

   But not always! Certain parameter values represent particles which
   would encounter effective potential barriers on their way to the
   simulation domain. For such particles the orbit integrand is imaginary
   at some radii and `alpha` is undefined and physically meaningless.
   In such situations this function is liable to return NaN; it is the
   caller's responsibility to detect that (or to avoid the entire situation
   by never calling this function with parameter values that contradict its
   implicit OML model).

   (This function integrates a simpler, less exact integrand than SCEPTIC.
    When computing the integral, SCEPTIC incorporates a more complicated
    version of the Debye-Hueckel potential profile which accounts for ion
    depletion, whereas this function uses the ordinary D-H profile,
    neglecting ion depletion. This makes little difference for small
    grains because for small grains the profiles differ only very slightly
    in absolute terms.) */
double integrate_for_alpha(double s, double chi_b, double u2, double Rol)
{
	const double chu2 = chi_b / u2;
	const double max_lin_dev = 0.01;
	const double s2 = s * s;
	const double y_one = s / sqrt(1.0 - chu2 - s2);  /* `xi` = 1 integ'nd */

	double alpha = 0.0;
	double xi[3] = { 0.0, 0.5, 1.0 };  /* current `xi` interval */
	double y[3] = { s, -9.9, y_one };  /* integrand values at `xi` */

	/* Handle the special case of head-on motion towards the
	   simulation domain's centre (i.e. zero impact parameter). */
	if (s == 0.0) {
		return 0.0;
	}

	do {
		/* Try defining the current integration interval as starting
		   at `xi[0]` and stretching all the way to 1. */
		xi[2] = 1.0;
		y[2] = y_one;
		xi[1] = (xi[0] + xi[2]) / 2.0;
		y[1] = 1.0 - (chu2 * xi[1] * exp(Rol * (1.0 - (1.0 / xi[1]))));
		y[1] = 1.0 / sqrt((y[1] / s2) - (xi[1] * xi[1]));

		/* If the integrand (`y`) appears to be linear over this interval,
		   excellent -- don't bother entering the loop. Otherwise, enter
		   the loop and iteratively shrink the interval until it's small
		   enough that `y` is close to linear throughout. */
		while (fabs(y[2] - y[1] - (y[1] - y[0])) > max_lin_dev) {
			/* `y` deviates substantially from linearity over the
			   current `xi` interval, so shrink the interval to half
			   its size and recompute `y` at the new mid-point and end
		       point, before checking again for near-linearity. */
			xi[2] = xi[1];
			y[2] = y[1];
			xi[1] = (xi[0] + xi[2]) / 2.0;
			y[1] = 1.0 - (chu2 * xi[1] * exp(Rol * (1.0 - (1.0 / xi[1]))));
			y[1] = 1.0 / sqrt((y[1] / s2) - (xi[1] * xi[1]));
		}

		/* `y` is approximately linear over the interval `xi`,
		   so apply Simpsons's rule over `xi`, incrementing `alpha`.
		   Then shift the interval's start to where the interval's end
		   currently is. */
		alpha += (xi[2] - xi[0]) * (y[0] + (4.0 * y[1]) + y[2]) / 6.0;
		xi[0] = xi[2];
		y[0] = y[2];
	} while (xi[2] < 1.0);

	/* Display a warning about this integral if it evaluated to NaN
	   (implying bad parameters for this function). */
	if (std::isnan(alpha)) {
		fprintf(stderr, "integrate_for_alpha got NaN: %.8g %.9g %g %g\n",
		        s, u2, chi_b, Rol);
	}

	return alpha;
}

/* Compute the `q`th quantile of the inverse cumulative distribution
   function of a particle's speed at infinity according to Hutchinson
   (2003)'s eq. 19 given nonzero flow (i.e. the normalized velocity `U` > 0).

   Warning: this function's caller must ensure `U` is greater than zero.

   19/05/2014: I've tested this function too against Mathematica and
   SCEPTIC for `chi` = 0 and it appears to work (so long as one bears in
   mind that SCEPTIC's equivalent of `U` is `Uc`, not `vd`). */
/* FIXME: when `chi` = 0, this function works when `U` is less than 26.48.
   I should either make it more robust for larger `U`, or make it explicitly
   detect when `U` is >= 26.48 and fail with a warning. */
#ifndef SQRT_PI
#define SQRT_PI 1.77245385090552
#endif

double u_icdf(double U, double chi, double q)
{
	const double Usq = U * U;
	const double psi = SQRT_PI * (1.0 + (2.0 * (Usq - chi))) / 2.0;
	const double p_den = 2.0 * ((U * exp(-Usq)) + (psi * erf(U)));

	double dp_den;
	double dp_num;
	double p;
	double p_num;
	double temp;
	double u;
	double usq;

	/* Try to find a good starting estimate for `u` for the Newton-Raphson
	   iterations to come. */
	if (U > 2.1) {
		/* Use the median in the large `U` limit (when `chi` = 0). */
		u = U + (1.0 / U);
	} else {
		/* Use a quadratic (in `U`) approximation for the median `u`
		   (assuming `chi` = 0). */
		u = 1.2586147 + (0.221 * U) + (0.1903312 * U * U);
	}

	/* Precompute the u-independent denominator of the unwieldy fraction
	   in the definition of the cumulative dist. func.'s derivative. */
	dp_den = 2.0 * SQRT_PI * (U + (psi * exp(Usq) * erf(U)));

	/* Do the inversion with the Newton-Raphson method. */
	do {
		usq = u * u;
		temp = - (U + u) * (U + u);
		p_num = (U - u) * exp(temp);
		p_num += (U + u) * exp((4.0 * U * u) + temp);
		p_num += psi * (erf(U - u) + erf(U + u));
		p = 1.0 - (p_num / p_den) - q;
		dp_num = (SQRT_PI * (1.0 + (2.0 * (Usq - usq)))) - (2.0 * psi);
		dp_num *= exp(-u * ((2.0 * U) + u)) - exp((2.0 * U * u) - usq);
		u -= p / (dp_num / dp_den);
	} while ((p * p) > 1e-16);  /* squaring because p may be negative */

	return u;
}

/* Reinject a particle at the simulation domain edge using a reinjection
   algorithm like that in Ian Hutchinson's SCEPTIC. */
void sceptically_reinject(ParticleData& part, int imax, double Ti, double Te, double ImpactParameter, double v_drift, double chi_b, std::mt19937 &mt)
{
	double a[2];   /* general-purpose rotation axis in the x-y plane */
	double alpha;  /* Hutchinson's alpha angle */
	double b;      /* normalized impact parameter */
	double c;      /* cosine of Hutchinson's theta angle */
	double co_z;   /* cosine of angle zeta, used below */
	double phi;    /* azimuthal angle in spherical coordinate system */
	double psi;    /* auxiliary rotation angle */
	double q;      /* u.r.v. quantile for sampling from CDFs */
	double Rol;    /* ratio of simul'n domain radius to shielding length */
	double U;      /* normalized drift velocity */
	double u_mag;  /* initial speed at infinity */
	double u2;     /* square of initial speed at infinity */
	double u[3];   /* particle's velocity at infinity (Cart. coords.) */
	double v[3];   /* reinj. particle's veloc. at boundary (sph. coords.) */
	double v_n;    /* characteristic normalization velocity */

	/* Compute the normalization velocity (square root of 2 k_B T / m).
	   Also, if `part` represents an electron, renormalize `chi_b` and
	   flip its sign so that it makes sense for an electron. */
	v_n = 2.0 * Kb / part.Species_Mass;
	if ( part.Species_Charge == -1 ) {
		v_n *= Te;
		chi_b *= -Ti / Te;
	} else {
		v_n *= Ti;
	}
	v_n = sqrt(v_n);

	/* Choose a random initial speed at infinity `u_mag`, ensuring its
	   square is no less than `chi_b` (otherwise the implied impact
	   parameter `b`, to be sampled below, would be imaginary --
	   corresponding to a particle which lacks the energy to reach the
	   simulation domain under the OML assumption). Then choose a random
	   cosine `c` of theta, theta being the angle between the drift
	   velocity and the initial velocity at infinity. */
	std::uniform_real_distribution<double> rad(0.0, 1.0); // IONS
	if (v_drift) {
		U = v_drift / v_n;  /* constant, so compute before the loop */
		do {
			u_mag = u_icdf(U, chi_b, rad(mt));
			u2 = u_mag * u_mag;
		} while (u2 < chi_b);
		q = rad(mt);
		c = log(q + (1.0 - q) * exp(-4.0 * u_mag * U)) / (2.0 * u_mag * U);
		c += 1.0;
	} else {
		do {
			u_mag = u_no_flow_icdf(chi_b, rad(mt));
			u2 = u_mag * u_mag;
		} while (u2 < chi_b);
		c = rad(mt) - 1.0;
	}

	/* With `u_mag` & `c` now in hand, it's time to randomly decide
	   the three components of the initial velocity `u`.
	   First, initialize `u` to be entirely in the z-direction by setting
	   the z-component to `u_mag`. Then rotate `u` about the y-axis
	   through an angle of pi/2 minus arccos `c` radians, to bring it
	   onto the cone of possible velocities defined as having angle theta
	   about the x-axis (the drift direction). Finally, to randomize `u`'s
	   precise direction, rotate it again, this time about the x-axis,
	   through a uniformly distributed random angle (call it `psi`). */
	u[0] = 0.0;
	u[1] = 0.0;
	u[2] = u_mag;
	a[0] = 0.0;
	a[1] = 1.0;
	rotate_vec_about_xy_dir(a, c, sqrt(1.0 - (c * c)),
	                        &(u[0]), &(u[1]), &(u[2]), 0.0);
	a[0] = 1.0;
	a[1] = 0.0;
	psi = 2.0*PI*rad(mt);
	rotate_vec_about_xy_dir(a, sin(psi), cos(psi),
	                        &(u[0]), &(u[1]), &(u[2]), 0.0);

	/* Choose a random impact parameter `b` at infinity.
	   Bad things would happen here were `chi_b` > `u2`, but the code
	   sampling `u_mag` above should ensure `chi_b` <= `u2`. */
	std::uniform_real_distribution<double> rad2(0.0, 1.0-(chi_b/u2)); // IONS
	b = sqrt(rad2(mt));

	/* Compute Hutchinson's alpha angle from `b`, `chi_b` and `u2`. */
	Rol = echarge * echarge * (imax - 1) / 2.0;
	Rol /= (4.0 * PI / 3.0) * ImpactParameter * epsilon0 * Kb * Te;
	Rol = sqrt(Rol);
	alpha = integrate_for_alpha(b, chi_b, u2, Rol);
	if (std::isnan(alpha)) {
		/* `alpha` isn't well defined; given the sampled `b`, `chi_b`,
		   `u2` and potential profile used by `integrate_for_alpha`,
		   this particle would apparently have encountered an effective
		   potential barrier on its way to the simulation domain.
		   So try to reinject it again. Since this problem is unlikely
		   to occur many times in a row, this function reattempts
		   reinjection by calling itself; since reinjection attempts
		   usually succeed, this should never overflow the stack. */
		if (part.Species_Charge == -1) {
			/* Undo the electron-specific renormalization of `chi_b` done
			   at the start of this function before passing it to this
			   function again. */
			chi_b /= -Ti / Te;
		}
		sceptically_reinject(part, imax, Ti, Te, ImpactParameter, v_drift, chi_b, mt);
		return;
	}

	/* Time to generate the particle's position at the simulation edge
	   based on its `u` & `b`. Begin by putting the particle at the edge at
	   an angle `alpha` from the z-axis, since that's easy. */
	phi = 2.0*PI*rad(mt);
	part.Position.setx(ImpactParameter * sin(alpha) * cos(phi));
	part.Position.sety(ImpactParameter * sin(alpha) * sin(phi));
	part.Position.setz(ImpactParameter * cos(alpha));

	/* But the particle still needs to be moved so that it's at an
	   angle `alpha` from the vector `u`, not from the z-axis! So
	   derive a rotation axis `a` by taking the cross product of
	   `u` and a unit z-axis vector. Then normalize `a`. */
	a[0] = u[1];
	a[1] = -u[0];
	a[0] /= sqrt((u[0] * u[0]) + (u[1] * u[1]));
	a[1] /= sqrt((u[0] * u[0]) + (u[1] * u[1]));

	/* Deduce (the cosine of) the angle zeta through which to rotate the
	   particle about `a`, then rotate the particle through zeta. Also,
	   increment each of the particle's coordinates by the simulation
	   radius to bring the particle inside the actual simulation domain
	   (which has an origin different to the ideal mathematical origin of
	   (0, 0, 0)). */
	co_z = u[2] / u_mag;
	double x_temp = part.Position.getx();
	double y_temp = part.Position.gety();
	double z_temp = part.Position.getz();
	rotate_vec_about_xy_dir(a, sin(acos(co_z)), co_z, &x_temp,
	                        &y_temp, &z_temp, ImpactParameter);

	/* Now the particle's velocity must be set. As before, set the velocity
	   based on the pretence that `u` is along the z-axis. In this case the
	   reinjection velocity has no azimuthal component (by c.o.a.m.), which
	   beautifully simplifies the velocity calculation; the zenith comp't
	   follows directly from angular momentum conservation and the radial
	   comp't follows from Pythagoras' theorem and energy conservation. */
	v[1] = 0.0;
	v[2] = u_mag * b;
	v[0] = -sqrt(u2 - chi_b - (v[2] * v[2]));
	velocity_in_cartesian_coords(v, phi, alpha,
		                         &x_temp, &y_temp, &z_temp);

	/* Like the position, the velocity vector must also be rotated
	   through zeta about `a`, since `u` is (generally) not actually
	   parallel to the z-axis. */
	double vx_temp = part.Velocity.getx();
        double vy_temp = part.Velocity.gety();
        double vz_temp = part.Velocity.getz();
	rotate_vec_about_xy_dir(a, sin(acos(co_z)), co_z, &vx_temp,
	                        &vy_temp, &vz_temp, 0.0);

	part.Position.setx(x_temp);
	part.Position.sety(y_temp);
	part.Position.setz(z_temp);

	part.Velocity.setx(vx_temp);
	part.Velocity.sety(vy_temp);
	part.Velocity.setz(vz_temp);
}

#endif /* #ifdef SCEPTIC_REINJECTION */



static void show_usage(std::string name){
	std::cerr << "Usage: int main(int argc, char* argv[]) <option(s)> SOURCES"
	<< "\n\nOptions:\n"
	<< "\t-h,--help\t\t\tShow this help message\n\n"
	<< "\t-r,--radius RADIUS\t\t(m), Specify radius of Dust grain\n"
	<< "\t\tRadius(=1e-6m) DEFAULT,\t\tBy Default, simulate sphere of size 1um in radius\n\n"
	<< "\t-a1,--semix SEMIX\t\t(arb), Specify the semi-axis for x in dust radii\n"
	<< "\t\ta1(=1) DEFAULT,\t\t\tBy Default, simulate perfect sphere\n\n"
	<< "\t-a2,--semiy SEMIZ\t\t(arb), Specify the semi-axis for y in dust radii\n"
	<< "\t\ta2(=1) DEFAULT,\t\t\tBy Default, simulate perfect sphere\n\n"
	<< "\t-a3,--semiz SEMIY\t\t(arb), Specify the semi-axis for z in dust radii\n"
	<< "\t\ta3(=1) DEFAULT,\t\t\tBy Default, simulate perfect sphere\n\n"
	<< "\t-d,--density DENSITY\t\t(kgm^-^3), Specify density of Dust grain\n"
	<< "\t\tDensity(=19600kgm^-^3) DEFAULT,\tBy Default, Tungsten Density\n\n"
	<< "\t-c,--ichance ICHANCE\t\t\t(eV), Specify the probability of generating an Ion\n"
	<< "\t\tiChance(=-0.5) DEFAULT,\tFicticious ion generation probability: i.e Self-consistently generate ions & electrons\n\n"
	<< "\t-p,--potential POTENTIAL\t(double), Specify the potential of Dust grain normalised to electron temperature\n"
	<< "\t\tPotential(=-2.5eV) DEFAULT,\tBy Default, OML Potential in Ti=Te Hydrogen plasma\n\n"
	<< "\t-m,--magfield MAGFIELD\t\t(T), Specify the magnetic field (z direction)\n"
	<< "\t\tBMag(=1.0T) DEFAULT,\t\tBy Default, magnetic field is 1.0T upwards in vertical z\n\n"
	<< "\t-n,--normalised NORMALISED\t(bool), whether normalisation (following Sonmor & Laframboise) is on or off\n"
	<< "\t\tNormalisedVars(=0) DEFAULT,\tFalse, By Default use Tesla and electron volts\n\n"
	<< "\t-te,--etemp ETEMP\t\t(eV), Specify the temperature of plasma electrons\n"
	<< "\t\teTemp(=1eV) DEFAULT,\n\n"
	<< "\t-ne,--edensity EDENSITY\t\t(m^-^3), Specify the plasma electron density\n"
	<< "\t\teDensity(=1e18m^-^3) DEFAULT,\tTokamak density\n\n"
	<< "\t-ti,--itemp ITEMP\t\t(eV), Specify the temperature of Ions\n"
	<< "\t\tiTemp(=1eV) DEFAULT,\n\n"
	<< "\t-ni,--idensity IDENSITY\t\t(m^-^3), Specify the plasma ion density\n"
	<< "\t\tiDensity(=1e18m^-^3) DEFAULT,\tTokamak density\n\n"
	<< "\t-z,--zboundforce ZBOUNDFORCE\t(double), Force the absolute value of simulation domain upper and lower boundaries\n"
	<< "\t\tZBoundForce(=1.5) DEFAULT,\tBy Default,\n\n"
	<< "\t-b,--impactpar IMPACTPAR\t(double), Specify the radial limit of simulation domain as number of distances\n"
	<< "\t\tImpactPar(=2.0) DEFAULT,\tBy Default, Radial extent of injection is three gyro-radii from centre\n\n"
	<< "\t-j,--jmax JMAX\t\t\t(int), Specify the number of particles to be collected\n"
	<< "\t\tjmax(=5000) DEFAULT,\t\tBy Default, stop simulation if 5,000 particles are collected\n\n"
	<< "\t-t,--timestep TIMESTEP\t\t(double), Specify the multiplicative factor for the length of a time step\n"
	<< "\t\tTimeStepFactor(=0.0005) DEFAULT,\n\n"
	<< "\t-t,--maxtime MAXTIME\t\t(int), Specify the multiplicative factor for the number of timesteps to be taken\n"
	<< "\t\tMaxTimeFactor(=50) DEFAULT,\n\n"
	<< "\t-no,--number NUMBER\t\t\t(int), Specify the number of particles to be captured before saving\n"
	<< "\t\tNumber(=1000) DEFAULT,\t\tBy Default, store average values after collecting 1000 particles\n\n"
	<< "\t-v,--driftvel DRIFTVEL\t\t(m s^-^1), Specify the drift velocity of the plasma\n"
	<< "\t\tDriftVel(=0.0) DEFAULT,\t\tBy Default, No drift velocity\n\n"
	<< "\t-se,--seed SEED\t\t\t(double), Specify the seed for the random number generator\n"
	<< "\t\tSeed(=1.0) DEFAULT,\t\tBy Default, Seed is 1.0\n\n"
	<< "\t-sa,--Saves SAVES\t\t\t(int), Specify the number of saves in a run\n"
	<< "\t\tSaves(=1.0) DEFAULT,\tBy Default, Save data to Meta datafile once per run\n\n"
	<< "\t-o,--output OUTPUT\t\t(string), Specify the suffix of the output file\n"
	<< "\t\tsuffix(='.txt') DEFAULT,\tBy Default, Save data to Data/DiMPl.txt\n\n"
	<< std::endl;
}

template<typename T> int InputFunction(int &argc, char* argv[], int &i, std::stringstream &ss0, T &Temp){
	if (i + 1 < argc) { // Make sure we aren't at the end of argv!
		i+=1;
		ss0 << argv[i]; // Increment 'i' so we don't get the argument as the next argv[i].
		ss0 >> Temp;
		ss0.clear(); ss0.str("");
		return 0;
	}else{ // Uh-oh, there was no argument to the destination option.
		std::cerr << "\noption requires argument." << std::endl;
		return 1;
	}

}

void RegenerateOrbit(ParticleData& PD, const double &ImpactParameter, const double &zBound, const double DriftNorm, 
			const double ThermalVel, std::mt19937 &mt){

	// ***** DEFINE RANDOM NUMBER GENERATOR ***** //
	std::normal_distribution<double> Gaussdist(0.0,ThermalVel);
	std::normal_distribution<double> GaussDriftdist(DriftNorm,ThermalVel);
	std::uniform_real_distribution<double> rad(0.0, 1.0); // IONS

	// ***** RANDOMISE POSITION CYLINDRICALLY ***** //
	do{
		double radial_pos=ImpactParameter*sqrt(rad(mt));
		double theta_pos =2*PI*rad(mt);
		PD.Position.setx(radial_pos*cos(theta_pos));
		PD.Position.sety(radial_pos*sin(theta_pos));
		PD.Position.setz(-zBound);
	}while( PD.Position.mag3() <= 1.0 );

	// ***** RANDOMISE VELOCITY CYLINDRICALLY ***** //
	PD.Velocity.setx(Gaussdist(mt));
	PD.Velocity.sety(Gaussdist(mt));
	double invel = rand_mwts(DriftNorm,ThermalVel,mt);
	if( rad(mt) > 0.5 ){
		PD.Position.setz(zBound);
		invel=-invel+2.0*DriftNorm;
	}
	PD.Velocity.setz(invel);

	#ifdef SPHERICAL_INJECTION
		// ***** RANDOMISE POSITION SPHERICALLY ***** //
		do{

			double radial_pos = ImpactParameter;
			double theta_pos  = 2.0*PI*rad(mt);
			double phi_pos    = asin(2.0*rad(mt)-1.0);
			PD.Position.setx(radial_pos*cos(phi_pos)*cos(theta_pos));
			PD.Position.sety(radial_pos*cos(phi_pos)*sin(theta_pos));
			PD.Position.setz(radial_pos*sin(phi_pos));
		}while( PD.Position.mag3() <= 1.0 );
	
		// ***** RANDOMISE VELOCITY SPHERICALLY ***** //
		PD.Velocity.setx(Gaussdist(mt));
		PD.Velocity.sety(Gaussdist(mt));
		PD.Velocity.setz(GaussDriftdist(mt));
		if ( PD.Velocity*PD.Position >= 0.0 ){
			PD.Position = -1.0*PD.Position;
		}
	#endif
	#ifdef POINT_INJECTION
		// ***** INJECT AT SINGLE POINT WITH DRIFT VELOCITY ***** //
		PD.Position.setx(0.0);
		PD.Position.sety(0.0);
		PD.Position.setz(zBound);
		
		PD.Velocity.setx(0.0);
		PD.Velocity.sety(0.0);
		PD.Velocity.setz(-DriftNorm);
	#endif
	PD.Reflections=0;
}

ParticleData GenerateOrbit(const double &ImpactParameter, const double &zBound, const double DriftNorm, const double ThermalVel,
			std::mt19937 &mt){
	ParticleData PD;

	// ***** DEFINE RANDOM NUMBER GENERATOR ***** //
	std::normal_distribution<double> Gaussdist(0.0,ThermalVel);
	std::normal_distribution<double> GaussDriftdist(DriftNorm,ThermalVel);
	std::uniform_real_distribution<double> rad(0.0, 1.0); // IONS

	// ***** RANDOMISE POSITION CYLINDRICALLY ***** //
	do{
		double radial_pos=ImpactParameter*sqrt(rad(mt));
		double theta_pos =2*PI*rad(mt);
		double vertical_pos = zBound*(2.0*rad(mt)-1.0);
		PD.Position.setx(radial_pos*cos(theta_pos));
		PD.Position.sety(radial_pos*sin(theta_pos));
		PD.Position.setz(vertical_pos);
	}while( PD.Position.mag3() <= 1.0 );

	// ***** RANDOMISE VELOCITY CYLINDRICALLY ***** //
	PD.Velocity.setx(Gaussdist(mt));
	PD.Velocity.sety(Gaussdist(mt));
	PD.Velocity.setz(GaussDriftdist(mt)); // This is currently the incorrect distribution!

	#ifdef SPHERICAL_INJECTION
		// ***** RANDOMISE POSITION SPHERICALLY ***** //
		do{
			double radial_pos = ImpactParameter*sqrt(rad(mt));
			double theta_pos  = 2.0*PI*rad(mt);
			double phi_pos    = asin(2.0*rad(mt)-1.0);
			PD.Position.setx(radial_pos*cos(phi_pos)*cos(theta_pos));
			PD.Position.sety(radial_pos*cos(phi_pos)*sin(theta_pos));
			PD.Position.setz(radial_pos*sin(phi_pos));
		}while( PD.Position.mag3() <= 1.0 );
	
		// ***** RANDOMISE VELOCITY SPHERICALLY ***** //
		PD.Velocity.setx(Gaussdist(mt));
		PD.Velocity.sety(Gaussdist(mt));
		PD.Velocity.setz(GaussDriftdist(mt));
	#endif
	#ifdef POINT_INJECTION
		// ***** INJECT AT SINGLE POINT WITH DRIFT VELOCITY ***** //
		PD.Position.setx(0.0);
		PD.Position.sety(0.0);
		PD.Position.setz(zBound);
		
		PD.Velocity.setx(0.0);
		PD.Velocity.sety(0.0);
		PD.Velocity.setz(-DriftNorm);
	#endif
	return PD;
}

/*updates velocity using the Boris method, Birdsall, Plasma Physics via Computer Simulation, p.62*/
static void UpdateVelocityBoris(threevector Efield, threevector BField, double dt, ParticleData& PD){

	/*t vector*/
	threevector t = BField*0.5*PD.Species_Charge*dt*PD.Species_Mass;

	/*magnitude of t, squared*/
//	double t_mag2 = t.square();

	/*s vector*/
	threevector s = 2.0*t*(1.0/(1.0+t.square()));
	
	/*v minus*/
	threevector v_minus = PD.Velocity + Efield*0.5*(PD.Species_Charge*PD.Species_Mass)*dt; 
	
	/*v prime*/
	threevector v_prime = v_minus + (v_minus^t);
	
	/*v prime*/
	threevector v_plus = v_minus + (v_prime^s);
	
	/*v n+1/2*/
	PD.Velocity = v_plus + Efield*0.5*(PD.Species_Charge*PD.Species_Mass)*dt;
}

threevector CoulombField(threevector Position, double Charge, double COULOMB_NORM){
	threevector Efield = (COULOMB_NORM*Charge/Position.square())*Position.getunit(); // 0.07, 0.1475
	return Efield;
}

threevector DebyeHuckelField(threevector Position, double Charge, double DebyeLength, double COULOMB_NORM){
	if(Charge==0.0) return threevector(0.0,0.0,0.0);
//	threevector Efield = COULOMB_NORM*Charge*(1.0/(Position.mag3()))*exp(-Position.mag3()/DebyeLength)
//				*(1.0/Position.mag3()+1.0/DebyeLength)*Position.getunit();

	threevector Efield = COULOMB_NORM*Charge*(1.0/(Position.square()))*exp(-Position.mag3()/DebyeLength)
				*Position.getunit();
	return Efield;
}

int main(int argc, char* argv[]){
	
	// ***** TIMER AND FILE DECLERATIONS 		***** //
	clock_t begin = clock();
	std::string filename = "Data/DiMPl";
	std::string suffix	= ".txt";
	DECLARE_TRACK();
	DECLARE_AVEL();			// Data file for angular momentum information
	DECLARE_LMOM();			// Data file for linear momentum information
	DECLARE_CHA();			// Data file for charge
	DECLARE_EPOS();			// Data file for end positions
	DECLARE_SPEC();			// Data file for the species
	std::ofstream RunDataFile;	// Data file for containing the run information

	// ************************************************** //


	// ***** DEFINE DUST PARAMETERS 		***** //
	double Radius 		= 1e-6;		// m, Radius of dust
	double Density 		= 19600;	// kg m^-^3, Tungsten
	double Potential	= -2.5;		// Normalised Potential, 
	double BMag 		= 1.0; 		// Tesla, Magnitude of magnetic field
	double BMagIn		= BMag;		// (arb), input magnetic field,in normalised units or Tesla
	bool   NormalisedVars	= false;	// Is magnetic field Normalised according to Sonmor & Laframboise?
	double a1		= 1.0;		// Semi-axis for x in dust-grain radii
	double a2		= 1.0;		// Semi-axis for y in dust-grain radii 
	double a3		= 1.0;		// Semi-axis for z in dust-grain radii

	// ************************************************** //


	// ***** DEFINE PLASMA PARAMETERS 		***** //
	double iTemp 		= 1.0;	// Ion Temperature, eV
	double eTemp 		= 1.0;	// Electron Temperature, eV
	double eDensity		= 1e18;	//1e14;	// m^(-3), Electron density
	double iDensity		= 1e18;	//1e14;	// m^(-3), Ion density
	double DriftVel 	= 0.0;	// m s^-1, This is the Temperature of the species being considered
	double ZBoundForce	= 1.5;	// Arb, Number of dust grain radii to vertical edge of simulation domain
	double ImpactPar	= 2.0;	// Species gyro-radii, Multiplicative factor for the Impact Parameter
	double iChance		= 0.5;	// Manually set probability of Generating an ion, negative will cause self-consistent
	unsigned long long jmax	= 5000; // Arb, Number of particles to be collected
	unsigned long long num	= 1000; // Arb, Number of particles to be collected before saving
	double TimeStepFactor	= 0.0005;// Arb, Multiplicative factor used to determine size of the timestep
	double MaxTimeFactor	= 50;	// Arb, Multiplicative factor used to determine number of timesteps to take
	unsigned int Saves(1);		// Arb, Number of Saves to be performed in a run
	unsigned int reflectionsmax(15);// Arb, Number of reflections before rejecting particles


	// ************************************************** //


	// ***** RANDOM NUMBER GENERATOR 		***** //
	// Arb, Seed for the random number generator
	double seed		= 2.0; //std::chrono::high_resolution_clock::now().time_since_epoch().count();
	
	// ************************************************** //


	// ***** DETERMINE USER INPUT ***** //
	std::vector <std::string> sources;
	std::stringstream ss0;
	for (int i = 1; i < argc; ++i){ // Read command line input
		std::string arg = argv[i];
		if     ( arg == "--help" 	|| arg == "-h" ){	show_usage( argv[0]); return 0; 		}
		else if( arg == "--radius" 	|| arg == "-r" ) 	InputFunction(argc,argv,i,ss0,Radius);
		else if( arg == "--semix" 	|| arg == "-a1") 	InputFunction(argc,argv,i,ss0,a1);
		else if( arg == "--semiy" 	|| arg == "-a2") 	InputFunction(argc,argv,i,ss0,a2);
		else if( arg == "--semiz" 	|| arg == "-a3") 	InputFunction(argc,argv,i,ss0,a3);
		else if( arg == "--density" 	|| arg == "-d" )	InputFunction(argc,argv,i,ss0,Density);
		else if( arg == "--potential" 	|| arg == "-p" )	InputFunction(argc,argv,i,ss0,Potential);
		else if( arg == "--magfield" 	|| arg == "-m" )	InputFunction(argc,argv,i,ss0,BMagIn);
		else if( arg == "--normalised" 	|| arg == "-n" )	InputFunction(argc,argv,i,ss0,NormalisedVars);
		else if( arg == "--etemp" 	|| arg == "-te")	InputFunction(argc,argv,i,ss0,eTemp);
		else if( arg == "--edensity" 	|| arg == "-ne")	InputFunction(argc,argv,i,ss0,eDensity);
		else if( arg == "--itemp" 	|| arg == "-ti")	InputFunction(argc,argv,i,ss0,iTemp);
		else if( arg == "--idensity" 	|| arg == "-ni")	InputFunction(argc,argv,i,ss0,iDensity);
		else if( arg == "--ichance" 	|| arg == "-c" )	InputFunction(argc,argv,i,ss0,iChance);
		else if( arg == "--zboundforce"	|| arg == "-z" )	InputFunction(argc,argv,i,ss0,ZBoundForce);
		else if( arg == "--impactpar"	|| arg == "-b" )	InputFunction(argc,argv,i,ss0,ImpactPar);
		else if( arg == "--jmax"	|| arg == "-j" )	InputFunction(argc,argv,i,ss0,jmax);
		else if( arg == "--time"	|| arg == "-t" )	InputFunction(argc,argv,i,ss0,TimeStepFactor);
		else if( arg == "--maxtime"	|| arg == "-l" )	InputFunction(argc,argv,i,ss0,MaxTimeFactor);
		else if( arg == "--number"	|| arg == "-no")	InputFunction(argc,argv,i,ss0,num);
		else if( arg == "--driftvel"	|| arg == "-v" )	InputFunction(argc,argv,i,ss0,DriftVel);
		else if( arg == "--seed"	|| arg == "-se")	InputFunction(argc,argv,i,ss0,seed);
		else if( arg == "--Saves"	|| arg == "-sa")	InputFunction(argc,argv,i,ss0,Saves);
		else if( arg == "--output"	|| arg == "-o" )	InputFunction(argc,argv,i,ss0,suffix);
                else{
			sources.push_back(argv[i]);
		}
	}
	if( Radius < 0.0 )
		std::cerr << "\nError! Probe Radius is negative\nRadius : " << Radius;
	if( Density < 0.0 )
		std::cerr << "\nError! Probe Density is negative\nDensity : " << Density;
	if( eTemp < 0.0 )
		std::cerr << "\nError! Electron Temperature is negative\neTemp : " << eTemp;
	if( eDensity < 0.0 )
		std::cerr << "\nError! Electron Density is negative\neDensity : " << eDensity;
	if( iTemp < 0.0 )
		std::cerr << "\nError! Ion Temperature is negative\niTemp : " << iTemp;
	if( iDensity < 0.0 )
		std::cerr << "\nError! Ion Density is negative\niDensity : " << iDensity;
	if( iChance < 0.0 )
		std::cout << "\nWarning! Chance of generating Ion is negative, self-consistent flux assumed\niChance : " << iChance;
	if( ZBoundForce <= 0.0 )
		std::cerr << "\nError! Force Vertical Boundaries Parameter is negative\nZBoundForce : " << ZBoundForce;
	if( ImpactPar < 0.0 )
		std::cerr << "\nError! Impact Parameter is negative\nImpactPar : " << ImpactPar;
	if( iDensity != eDensity )
		std::cerr << "\nError! Quasineutrality is not observed\niDensity != eDensity : " 
			<< iDensity << " != " << eDensity;
	if( jmax < num )
		std::cout << "\nWarning! Save interval less than captured particle goal. No Angular data recorded\nnum < jmax : " 
			<< num << " < " << jmax;

	// If species is positively charged, we assume it's a singly charged ion. Otherwise, singly charged electron
	double MASS 		= Mp;		// kg, This is the Mass to which quantities are normalised 
	a1 = 1.0/(a1*a1);
	a2 = 1.0/(a2*a2);
	a3 = 1.0/(a3*a3);
	double MassRatio 	= sqrt(Mp/Me);
	double DustMass 	= (4.0/3.0)*PI*pow(Radius,3)*Density;
	if( NormalisedVars ){
		BMag = pow(2.0/PI,2.0)*BMagIn*sqrt(PI*Mp*iTemp/(2*echarge))/Radius;	// BMag normalised to Ions
		Potential = Potential;
		if( iChance == 0.0 ){ // If we are simulating only Electrons
        	        BMag = pow(2.0/PI,2.0)*BMagIn*sqrt(PI*Me*eTemp/(2*echarge))/Radius;	// BMag normalised to Electrons
		}else{	// If we are simulating only Ions or otherwise Ions and electrons.
			BMag = pow(2.0/PI,2.0)*BMagIn*sqrt(PI*Mp*iTemp/(2*echarge))/Radius;	// BMag normalised to Ions
		}

	}else{
		BMag = BMagIn;
                Potential = Potential*echarge/(echarge*eTemp);	// Convert from SI Potential to normalised potential
	}

	// ************************************************** //


	// ***** NORMALISATION 				***** //
	// Normalise TIME to the Gyro-Radius of an Ion at B=100T
	// Normalise MASS to Ion Mass
	// Normalise DISTANCE to Dust Radius
	// Normalise CHARGE to fundamental charge
	double MAGNETIC(100);
        double ELECTRIC = Radius*MAGNETIC*MAGNETIC*echarge/Mp;  // Electric Field Normalisation
        double POTENTIAL = echarge/(4.0*PI*epsilon0*Radius);    // Potential Normalisation
        double Tau = MASS/(echarge*MAGNETIC);

 	double eDensityNorm	= eDensity*pow(Radius,3.0);	//1e14;	// m^(-3), Electron density
	double iDensityNorm	= iDensity*pow(Radius,3.0);	//1e14;	// m^(-3), Ion density
        double PotentialNorm    = Potential*(eTemp*echarge)/(echarge*POTENTIAL);        // Normalised Charge,
        double DriftNorm        = DriftVel*Tau/(Radius);
        double DebyeLength      = sqrt((epsilon0*echarge*eTemp)/(eDensity*pow(echarge,2.0)))/Radius;
        double A_Coulomb        = POTENTIAL/(Radius*ELECTRIC);

	// ************************************************** //


	// ***** DEFINE SIMULATION SPACE 		***** //
	threevector Bhat(0.0,0.0,1.0);	// Direction of magnetic field, z dir.
	threevector BField 	= BMag*Bhat*(1.0/MAGNETIC);
	double iThermalVel	= sqrt(echarge*iTemp/(Mp*2.0*PI))*(Tau/Radius);		// Normalised Ion Thermal velocity
	double eThermalVel	= sqrt(echarge*eTemp/(Me*2.0*PI))*(Tau/Radius);	// Normalised Electron Thermal velocity

	double iRhoTherm = 0.0;					// Ion gyro-radii are zero by Default
	double eRhoTherm = 0.0;					// Electron gyro-radii are zero by Default
	double TimeStep = TimeStepFactor;
	double MaxTime = MaxTimeFactor*TimeStep;
	double ImpactParameter = ImpactPar;
	if( BMag != 0.0 ){						// If Magnetic field is non-zero
		// Calculate thermal GyroRadius for ions and electrons normalised to dust grain radii
		iRhoTherm	= iThermalVel/(BMag/MAGNETIC); 
		eRhoTherm	= eThermalVel/(pow(MassRatio,2)*BMag/MAGNETIC); 
		
		if( NormalisedVars ){
			iRhoTherm	= 1.0/BMagIn;
			eRhoTherm       = 1.0/(BMagIn*MassRatio);
		}
	}
	double SimulationVolume = PI*pow(ImpactParameter,2.0)*2.0*ZBoundForce;
	#ifdef SPHERICAL_INJECTION
		SimulationVolume = (4.0/3.0)*PI*pow(ImpactParameter,3.0);
	#endif
	double imax = SimulationVolume*iDensityNorm;

	double ProbabilityOfIon = 0.5;
	if( iChance >= 0.0 && iChance <= 1.0)
		ProbabilityOfIon = iChance;


	// ************************************************** //


	// ***** SEED RANDOM NUMBER GENERATOR IN THREADS***** //
	std::vector<std::mt19937> randnumbers;
	for(int p = 0; p < omp_get_max_threads(); p ++){
		randnumbers.push_back(std::mt19937(seed+p));
	}

	// ************************************************** //


	// ***** OPEN DATA FILE WITH HEADER 		***** //
	time_t now = time(0);		// Get the time of simulation
	char * dt = ctime(&now);
	OPEN_AVEL();	HEAD_AVEL();
	OPEN_LMOM();	HEAD_LMOM();
	OPEN_CHA();	HEAD_CHA();
	OPEN_EPOS();	HEAD_EPOS();
	OPEN_SPEC();	HEAD_SPEC();

	RunDataFile.open(filename + suffix);
	RunDataFile << "## Run Data File ##\n";
	RunDataFile << "#Date: " << dt;
	RunDataFile << "#Input:\t\tValue\n\njmax (arb #):\t\t"<<jmax<<"\nProbOfIon (arb):\t"<<ProbabilityOfIon<<"\n\nElectron Gyro (1/Radius):\t"<<eRhoTherm<<"\nElectron Temp (eV):\t\t"<<eTemp<<"\nElectron Density (m^-^3):\t"<<eDensity<< "\n\nIon Gyro (1/Radius):\t"<<iRhoTherm<<"\nIon Temp (eV):\t\t"<<iTemp<<"\nIon Density (m^-^3):\t"<<iDensity<<"\n\nRadius (m):\t\t"<<Radius<<"\na1 (1/Radius):\t\t"<<(1.0/sqrt(a1))<<"\na2 (1/Radius):\t\t"<<(1.0/sqrt(a2))<<"\na3 (1/Radius):\t\t"<<(1.0/sqrt(a3))<<"\nDensity (kg m^-^3):\t"<<Density<<"\nCharge (1/echarge):\t\t"<<PotentialNorm<<"\nB Field (T or Radius/GyroRad):\t"<<BMag<<"\nDebyeLength (1/Radius):\t\t"<<DebyeLength <<"\nDrift Norm (Radius/Tau):\t"<<DriftNorm<<"\nTime Step (Tau):\t\t"<<TimeStep<<"\nMax Time (Tau):\t\t"<<MaxTime<<"\nZBound (1/Radius):\t"<<ZBoundForce<<"\nImpact Par (1/Radius):\t"<<ImpactParameter<<"\nSimulationVolume (1/Radius^3):\t"<<SimulationVolume<<"\nParticle Number (arb):\t"<<imax<<"\nTime Norm [Tau] (s):\t\t"<<Tau<<"\n\n"<<"RNG Seed (arb):\t\t"<<seed<<"\nOMP_THREADS (arb):\t"<<omp_get_max_threads()<<"\n\n";

	// ************************************************** //


	// ***** BEGIN LOOP OVER PARTICLE ORBITS 	***** //
	threevector TotalAngularVel(0.0,0.0,0.0);
	threevector TotalAngularMom(0.0,0.0,0.0);
	DECLARE_LMSUM();
	DECLARE_AMSUM();
	DECLARE_AMOM();

	unsigned long long j(0), i(0), RegeneratedParticles(0), TrappedParticles(0), MissedParticles(0), TotalNum(imax);
	long long CapturedCharge(0), RegeneratedCharge(0), TrappedCharge(0), MissedCharge(0), TotalCharge(0);
	double TotalTime(0.0);

	std::vector<ParticleData> Particles;
	std::random_device rd;		// Create Random Device
	std::mt19937 randnumber = std::mt19937(rd());
	std::uniform_real_distribution<double> rad(0.0, 1.0); // IONS
	

	// Initialise particles in master thread
	for( int n=0; n < imax; n ++ ){
		double IonOrElectron = rad(randnumber);
		if( IonOrElectron < ProbabilityOfIon ){
			Particles.push_back(GenerateOrbit(ImpactParameter,ZBoundForce,DriftNorm,iThermalVel,randnumber));
			Particles[n].Species_Charge	=1;			// Charge Of Ion
			Particles[n].Species_Mass	=1.0;			// Normalised Mass Of Ion
			TotalCharge += 1;
		}else{
			Particles.push_back(GenerateOrbit(ImpactParameter,ZBoundForce,DriftNorm,eThermalVel,randnumber));
			Particles[n].Species_Charge	=-1;			// Charge Of Electron
			Particles[n].Species_Mass	=pow(MassRatio,2.0);	// Normalise Mass Of Electron
			TotalCharge += -1;
		}
		Particles[n].Reflections = 0;

		// ***** TESTING AREA  				***** //
		// ***** ANGULAR-MOMENTUM TEST 			***** //
		#ifdef TEST_ANGMOM
			SAVE_I_AMOM(Particles[n].Species_Mass*(Particles[n].Position^Particles[n].Velocity));		// For Angular Momentum Calculations
			PRINT_AMOM("\n"); PRINT_AMOM(i); PRINT_AMOM("\t"); PRINT_AMOM(INITIAL_AMOM); PRINT_AMOM("");
		#endif
		// ***** VELOCITY-POSITION DISTRIBUTION TEST 	***** //

		#ifdef TEST_VELPOSDIST
			PRINT_VPD(Particles[n].Position); PRINT_VPD("\t");	// For debugging look at initial positions
			PRINT_VPD(Particles[n].Velocity); PRINT_VPD("\t");	// For debugging look at initial velocities
			PRINT_VPD(Particles[n].Velocity*(Particles[n].Position.getunit())); PRINT_VPD("\t");	
			PRINT_VPD( sqrt(pow(Particles[n].Velocity.getx(),2)+pow(Particles[n].Velocity.gety(),2))*Particles[n].Species_Mass); 
			PRINT_VPD("\n");	// For debugging look at gyro-radii
		#endif
		// ***** ENERGY TEST: MEASURE INITIAL ENERGY	***** //
		INITIAL_VEL();						// For energy calculations
		INITIAL_POT();						// For energy calculations

		#ifdef TEST_CLOSEST_APPROACH
		double C1 = fabs(2.0*echarge*echarge*PotentialNorm/(Mp*4.0*PI*epsilon0));
		double ri = Particles[n].Position.mag3()*Radius;
		double vmag = Particles[n].Velocity.mag3()*Radius/Tau;
		double vperp = (Particles[n].Position.getunit()^Particles[n].Velocity).mag3()*Radius/Tau;
		double Min_r1 = (C1+sqrt(C1*C1+4.0*(vmag*vmag+C1/ri)*ri*ri*vperp*vperp))/(2.0*(vmag*vmag+C1/ri));
		double Min_r2 = (C1-sqrt(C1*C1+4.0*(vmag*vmag+C1/ri)*ri*ri*vperp*vperp))/(2.0*(vmag*vmag+C1/ri));
		double MinPos = sqrt(zmax*zmax+ImpactParameter*ImpactParameter);
		#endif

	}

	// loop over saves
	for( unsigned int s=1; s <= Saves; s ++){
		unsigned long long IMAX = (s*imax)/Saves;
		RunDataFile.close();
		RunDataFile.clear();
		RunDataFile.open(filename+suffix, std::fstream::app);

		// loop over the total time to run the simulation
		for( TotalTime = 0.0; TotalTime < MaxTime; TotalTime += TimeStep ){
			// Loop until we reach a certain number of particles jmax
			if( j <= jmax ){	

			//  Parallelised Loop over maximum number of particles in system
			#pragma omp parallel for shared(TotalAngularVel,TotalAngularMom,j) PRIVATE_FILES()
			for( i=(IMAX-imax/Saves); i < IMAX; i ++){ 	
	
//				std::cout << "\n" << omp_get_thread_num() << "/" << omp_get_num_threads();

				// ***** RECORD TRACK DATA, DEBUG AND TEST	***** //
				OPEN_TRACK(filename + "_Track_" + std::to_string(i) + ".txt");
				RECORD_TRACK("\n"); RECORD_TRACK(TotalTime); 
				RECORD_TRACK("\t"); RECORD_TRACK(Particles[i].Position);
				RECORD_TRACK("\t"); RECORD_TRACK(Particles[i].Velocity); RECORD_TRACK("\t");
				RECORD_TRACK(sqrt(Particles[i].Position.getx()*Particles[i].Position.getx()
                                                        +Particles[i].Position.gety()*Particles[i].Position.gety()
							+Particles[i].Position.getz()*Particles[i].Position.getz()));
	
				// ************************************************** //
	
	
				// ***** DO PARTICLE PATH INTEGRATION 		***** //

				// Condition for exiting edge of cylindrical simulation domain
				bool EdgeCondition = (Particles[i].Position.getz() < -ZBoundForce 
							|| Particles[i].Position.getz() > ZBoundForce);

				bool CylinderCondition = (sqrt(Particles[i].Position.getx()*Particles[i].Position.getx()+
								Particles[i].Position.gety()*Particles[i].Position.gety()) 
								> ImpactParameter);
				#ifdef SPHERICAL_INJECTION
                                // Condition for exiting edge of spherical simulation domain
				EdgeCondition = ((Particles[i].Position.getx()*Particles[i].Position.getx()
							+Particles[i].Position.gety()*Particles[i].Position.gety()
							+Particles[i].Position.getz()*Particles[i].Position.getz()) >= ImpactParameter*ImpactParameter*1.01);
				#endif
                                // Condition for particle crossing into the sphere
				bool SphereCondition = (sqrt(Particles[i].Position.getx()*Particles[i].Position.getx()*a1
								+Particles[i].Position.gety()*Particles[i].Position.gety()*a2
								+Particles[i].Position.getz()*Particles[i].Position.getz()*a3) < 1.0);
				#ifdef NO_SPHERE
					SphereCondition = false;
				#endif

				#pragma omp critical
				{
				threevector OldPos = Particles[i].Position; // For Angular Momentum Calculations
				threevector FinalPos = 0.5*(OldPos+Particles[i].Position);
				threevector AngularMom = Particles[i].Species_Mass*(FinalPos^Particles[i].Velocity);
				if( SphereCondition || Particles[i].Reflections >= reflectionsmax ){ // Particle captured
 		
					double AngVelNorm = 5.0*Particles[i].Species_Mass*MASS/(2.0*DustMass);
					threevector AngularVel = (AngVelNorm)*
					((FinalPos^Particles[i].Velocity)-(FinalPos^(TotalAngularVel^FinalPos)));
//					ADD_F_AMOM(AngularVel*(2.0*DustMass/5.0));


					PRINT_FP(fabs(FinalPos.mag3()-1)); PRINT_FP("\n");
					TotalAngularVel += AngularVel;
					TotalAngularMom += AngularMom;

					j ++;
					CapturedCharge += Particles[i].Species_Charge;
					PRINT_CHARGE(j)			PRINT_CHARGE("\t")
					PRINT_CHARGE(PotentialNorm) 	PRINT_CHARGE("\t")
					PRINT_CHARGE(Particles[i].Species_Charge);	PRINT_CHARGE("\n")
//					PRINT_AMOM((AngVelNorm)*(FinalPos^Particles[i].Velocity)); PRINT_AMOM("\t");
//					PRINT_AMOM((AngVelNorm)*(FinalPos^Particles[i].Velocity)*(1.0/Tau)); PRINT_AMOM("\n");
					ADD_CHARGE()
					if(j % num == 0){
						SAVE_AVEL()
						SAVE_LMOM()
						SAVE_CHA()
						SAVE_EPOS()
						SAVE_SPEC()
					}
					if( Particles[i].Reflections >= reflectionsmax ){        // In this case it was trapped!
        	                                TrappedParticles ++;
	                                        TrappedCharge += Particles[i].Species_Charge;
                                	}

					if( Particles[i].Species_Charge == 1 ){
						#ifdef SCEPTIC_REINJECTION
							double chi_b = (echarge*PotentialNorm/ImpactParameter)
								*exp(-ImpactParameter/DebyeLength);
							sceptically_reinject(Particles[i], imax, iTemp, eTemp, ImpactParameter, 
									DriftNorm, chi_b, randnumbers[omp_get_thread_num()]);
						#else
							RegenerateOrbit(Particles[i],ImpactParameter,ZBoundForce,DriftNorm,
        								iThermalVel,randnumbers[omp_get_thread_num()]);
						#endif
					}else if(Particles[i].Species_Charge == -1){
						#ifdef SCEPTIC_REINJECTION
							double chi_b = (echarge*PotentialNorm/ImpactParameter)
								*exp(-ImpactParameter/DebyeLength);
							sceptically_reinject(Particles[i], imax, iTemp, eTemp, ImpactParameter, 
									DriftNorm, chi_b, randnumbers[omp_get_thread_num()]);
						#else
							RegenerateOrbit(Particles[i],ImpactParameter,ZBoundForce,DriftNorm,
        								eThermalVel,randnumbers[omp_get_thread_num()]);
						#endif
					}else{
						std::cout << "ERROR!"; exit(1);
					}
					// ***** VELOCITY-POSITION DISTRIBUTION TEST 	***** //
					#ifdef TEST_VELPOSDIST
					PRINT_VPD(Particles[i].Position); PRINT_VPD("\t");	// For debugging look at initial positions
					PRINT_VPD(Particles[i].Velocity); PRINT_VPD("\t");	// For debugging look at initial velocities
					PRINT_VPD(Particles[i].Velocity*(Particles[i].Position.getunit())); PRINT_VPD("\t");	
					PRINT_VPD( sqrt(pow(Particles[i].Velocity.getx(),2)+pow(Particles[i].Velocity.gety(),2))*Particles[i].Species_Mass); 
					PRINT_VPD("\n");	// For debugging look at gyro-radii
				
					#endif


					RegeneratedParticles ++;
                                        RegeneratedCharge += Particles[i].Species_Charge;
					TotalNum ++;
                                        TotalCharge += Particles[i].Species_Charge;
				}else if( EdgeCondition || CylinderCondition ){ // Particle left simulation domain
					LinearMomentumSum += Particles[i].Species_Mass*Particles[i].Velocity;	
					AngularMomentumSum += AngularMom;
					MissedParticles ++;
					MissedCharge += Particles[i].Species_Charge;
					if( Particles[i].Species_Charge == 1 ){
						#ifdef SCEPTIC_REINJECTION
							double chi_b = (echarge*PotentialNorm/ImpactParameter)
								*exp(-ImpactParameter/DebyeLength);
                                                        sceptically_reinject(Particles[i], imax, iTemp, eTemp, ImpactParameter, 
                                                                        DriftNorm, chi_b, randnumbers[omp_get_thread_num()]);
                                                #else
							RegenerateOrbit(Particles[i],ImpactParameter,ZBoundForce,DriftNorm,
        								iThermalVel,randnumbers[omp_get_thread_num()]);
						#endif
					}else if(Particles[i].Species_Charge == -1){
						#ifdef SCEPTIC_REINJECTION
							double chi_b = (echarge*PotentialNorm/ImpactParameter)
								*exp(-ImpactParameter/DebyeLength);
                                                        sceptically_reinject(Particles[i], imax, iTemp, eTemp, ImpactParameter, 
                                                                        DriftNorm, chi_b, randnumbers[omp_get_thread_num()]);
                                                #else
							RegenerateOrbit(Particles[i],ImpactParameter,ZBoundForce,DriftNorm,
        								eThermalVel,randnumbers[omp_get_thread_num()]);
						#endif
					}else{
						std::cout << "ERROR!"; exit(1);
					}
					// ***** VELOCITY-POSITION DISTRIBUTION TEST 	***** //
					#ifdef TEST_VELPOSDIST
					PRINT_VPD(Particles[i].Position); PRINT_VPD("\t");	// For debugging look at initial positions
					PRINT_VPD(Particles[i].Velocity); PRINT_VPD("\t");	// For debugging look at initial velocities
					PRINT_VPD(Particles[i].Velocity*(Particles[i].Position.getunit())); PRINT_VPD("\t");	
					PRINT_VPD( sqrt(pow(Particles[i].Velocity.getx(),2)+pow(Particles[i].Velocity.gety(),2))*Particles[i].Species_Mass); 
					PRINT_VPD("\n");	// For debugging look at gyro-radii
					#endif

					#ifdef TEST_CLOSEST_APPROACH	
//					if( Min_r1 <= Radius ){//|| fabs(Min_r2) <= Radius){	
						std::cout << "\n" << i << "\t" << j << "\t" << MinPos*Radius
						<< "\t" << Min_r1 << "\t" << Min_r2 << "\t" << std::min(Min_r1,fabs(Min_r2));
//					}
					#endif				
					SAVE_F_AMOM(Particles[i].Species_Mass*(Particles[i].Position^Particles[i].Velocity));
					PRINT_AMOM("\n"); PRINT_AMOM(i); PRINT_AMOM("\t"); PRINT_AMOM(FINAL_AMOM); PRINT_AMOM("");
					FINAL_POT(); 
					PRINT_ENERGY(i); PRINT_ENERGY("\t"); 
					PRINT_ENERGY(100*(Particles[i].Velocity.square()/InitialVel.square()-1.0));  PRINT_ENERGY("\t");
					PRINT_ENERGY(0.5*Mp*Particles[i].Species_Mass*Particles[i].Velocity.square()+Particles[i].Species_Charge*FinalPot-
							(0.5*Mp*Particles[i].Species_Mass*InitialVel.square()+Particles[i].Species_Charge*InitialPot));  
					PRINT_ENERGY("\n");
					RegeneratedParticles ++;
					RegeneratedCharge += Particles[i].Species_Charge;
					TotalNum ++;
					TotalCharge += Particles[i].Species_Charge;
				}else{	// Particle hasn't satisfied Sphere or Edge condition yet, so we step it
					threevector EField = DebyeHuckelField(Particles[i].Position,
										PotentialNorm,DebyeLength,A_Coulomb);
//					EField = CoulombField(Particles[i].Position,PotentialNorm,A_Coulomb);
					UpdateVelocityBoris(EField,BField,TimeStep,Particles[i]);
					
					double PreviousVel = Particles[i].Velocity.getz();
					if( (Particles[i].Velocity.getz() > 0.0 && PreviousVel <= 0.0) || 
						(Particles[i].Velocity.getz() < 0.0 && PreviousVel >= 0.0) ){
						Particles[i].Reflections ++;
					}

					Particles[i].Position+=TimeStep*Particles[i].Velocity;
					#ifdef TEST_CLOSEST_APPROACH
					if( OldPos.mag3() > Particles[i].Position.mag3() || Particles[i].Position.mag3() < MinPos ){
						MinPos = Particles[i].Position.mag3();
					}
					#endif

					RECORD_TRACK("\n"); RECORD_TRACK(TotalTime); 
					RECORD_TRACK("\t");RECORD_TRACK(Particles[i].Position);
					RECORD_TRACK("\t");RECORD_TRACK(Particles[i].Velocity);
					RECORD_TRACK("\t");
					RECORD_TRACK(sqrt(Particles[i].Position.getx()*Particles[i].Position.getx()
							+Particles[i].Position.gety()*Particles[i].Position.gety()+Particles[i].Position.getz()*Particles[i].Position.getz()));


				} // end of determining particle fate
				CLOSE_TRACK();
				} // end of #pragma omp critical
			} // END OF PARALLELISED FOR LOOP
			} // end of if( j < jmax ) 
		} // end of for( TotalTime < MaxTime )

		// ***** PRINT ANGULAR MOMENTUM AND CHARGE DATA	***** //
		RunDataFile << "*****\n\n\nSave : " << s << "\t\tTotalTime:\t" << TotalTime << "\n\n";
	
		SAVE_MOM("LinMom\t\t\t\tAngMom\n");
		SAVE_MOM(LinearMomentumSum); SAVE_MOM("\t"); SAVE_MOM(AngularMomentumSum); SAVE_MOM("\n\n");
		// Save the Charge in units of electron charges and normalised potential.
		SAVE_CHARGE("Charge\tPotential\n")
		SAVE_CHARGE(PotentialNorm) SAVE_CHARGE("\t") SAVE_CHARGE(PotentialNorm*POTENTIAL/eTemp) 
		SAVE_CHARGE("\n\n")
	
		RunDataFile << "Normalised Ion Current:\tNormalised Electron current\n";
//		Calculate currents for cylindrical geometry
		double ThermaliCurrent = 4.0*PI*pow(Radius,2.0)*iDensity*iThermalVel*echarge*Radius/Tau;
		double ThermaleCurrent = 4.0*PI*pow(Radius,2.0)*eDensity*eThermalVel*echarge*Radius/Tau;
		RunDataFile << 0.5*echarge*(j+CapturedCharge)/(TotalTime*Tau*ThermaliCurrent);
		RunDataFile << "\t\t" << 0.5*echarge*(j-CapturedCharge)/(TotalTime*Tau*ThermaleCurrent) << "\n\n";

		// ************************************************** //
	
	
		// ***** PRINT CHARGE AND PATH COUNTERS 	***** //
		RunDataFile << "j\tjCharge\tMissed\tMCharge\tRegen\tRCharge\tTrapped\tTCharge\tGross\tGCharge\n"; 
		RunDataFile << j << "\t" << CapturedCharge << "\t" << MissedParticles << "\t" << MissedCharge << "\t" << RegeneratedParticles << "\t" << RegeneratedCharge << "\t" << TrappedParticles << "\t" << TrappedCharge << "\t" << TotalNum << "\t" << TotalCharge << "\n";
	
		clock_t end = clock();
		double elapsd_secs = double(end-begin)/CLOCKS_PER_SEC;
		RunDataFile << "\n\n*****\n\nCompleted in " << elapsd_secs << "s\n\n";
	
		// ************************************************** //
	
	
		// ***** CLOSE DATA FILES 			***** //
		RunDataFile.close();
	} // end of for( saves )
	CLOSE_AVEL();
	CLOSE_LMOM();
	CLOSE_CHA();
	CLOSE_EPOS();
	CLOSE_SPEC();

	// ************************************************** //


	return 0;
}
