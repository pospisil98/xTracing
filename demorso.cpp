#define _USE_MATH_DEFINES
#include <cassert>
#include <math.h>
#include <cstdlib>
#include <cstdio>
#include <vector>

#include <string>
#include <iostream>

#if defined(__APPLE__)
#include <OpenGL/gl.h>
#include <OpenGL/glu.h>
#include <GLUT/glut.h>
#else
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__)
#include <windows.h>
#endif
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glut.h>
#endif

//#include <vld.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"


#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

struct vec3;

// Pseudocolor mapping of a scalar value
// http://www.kennethmoreland.com/color-maps/, CoolWarmFloat33.csv 
float pscols[4 * 33] = { // 33 colors RGB
  0,0.2298057,0.298717966,0.753683153, 0.03125,0.26623388,0.353094838,0.801466763,
  0.0625,0.30386891,0.406535296,0.84495867, 0.09375,0.342804478,0.458757618,0.883725899,
  0.125,0.38301334,0.50941904,0.917387822, 0.15625,0.424369608,0.558148092,0.945619588,
  0.1875,0.46666708,0.604562568,0.968154911, 0.21875,0.509635204,0.648280772,0.98478814,
  0.25,0.552953156,0.688929332,0.995375608, 0.28125,0.596262162,0.726149107,0.999836203,
  0.3125,0.639176211,0.759599947,0.998151185, 0.34375,0.681291281,0.788964712,0.990363227,
  0.375,0.722193294,0.813952739,0.976574709, 0.40625,0.761464949,0.834302879,0.956945269,
  0.4375,0.798691636,0.849786142,0.931688648, 0.46875,0.833466556,0.860207984,0.901068838,
  0.5,0.865395197,0.86541021,0.865395561, 0.53125,0.897787179,0.848937047,0.820880546,
  0.5625,0.924127593,0.827384882,0.774508472, 0.59375,0.944468518,0.800927443,0.726736146,
  0.625,0.958852946,0.769767752,0.678007945, 0.65625,0.96732803,0.734132809,0.628751763,
  0.6875,0.969954137,0.694266682,0.579375448, 0.71875,0.966811177,0.650421156,0.530263762,
  0.75,0.958003065,0.602842431,0.481775914, 0.78125,0.943660866,0.551750968,0.434243684,
  0.8125,0.923944917,0.49730856,0.387970225, 0.84375,0.89904617,0.439559467,0.343229596,
  0.875,0.869186849,0.378313092,0.300267182, 0.90625,0.834620542,0.312874446,0.259301199,
  0.9375,0.795631745,0.24128379,0.220525627, 0.96875,0.752534934,0.157246067,0.184115123,
  1.0,0.705673158,0.01555616,0.150232812
};

// Structure to compute variance
struct SVAR
{
	unsigned int cnt; // the number of samples taken to compute statistics
	double mean; // mean value1
	double M2; // sum for variance1
	void Reset() { cnt = 0; mean = M2 = 0; }
	// Statistical support
	SVAR() { Reset(); }
	// add a single sample
	void Update(const double newSampleValue)
	{
		cnt++;
		double delta = newSampleValue - mean;
		mean += delta / (double)cnt;
		M2 += delta * (newSampleValue - mean);
	}
	// It returns unbiased sample variance (so not for finite population)
	double Evaluate() { return (double)M2 / ((double)cnt - 1); }
};

const double epsilon = 1e-9; // Small value
const int rainbowPSC = 0; // 0 .. use CoolWarm mapping, 1 .. use rainbow color mapping
const int showBargraph = 1; // 0/1 .. dont use/use bargraph on the right for color mapping

enum Show { DIFF, WEIGHT, WEIGHT_PSEUDOCOLOR };
const Show showFlag = WEIGHT_PSEUDOCOLOR;

// The cost of sampling - should be measured and set
double costBRDF = 1.0, costLight = 1.0, referenceEfficiency = 1.0;

int nIterations = 1; // how many iterations to render
int nTotalSamples = 128; // samples in one render iteration - should be even number

// Compute random number in range [0,1], uniform distribution
double drandom() { return (double)rand() / RAND_MAX; }

template<typename T>
T Clamp(T value, T lowerBound, T upperBound)
{
	if (value < lowerBound) return lowerBound;
	else if (value > upperBound) return upperBound;
	else return value;
}

// -------------------- VECTORS
// Vector 3D
struct vec3
{
	double x, y, z;

	vec3() { x = y = z = 0; }
	vec3(double x0, double y0, double z0 = 0) { x = x0; y = y0; z = z0; }
	vec3 operator*(double a)  const { return vec3(x * a, y * a, z * a); }
	vec3 operator*(const vec3 r)  const { return vec3(x * r.x, y * r.y, z * r.z); }
	vec3 operator/(const double r)  const
	{
		if (fabs(r) > epsilon)
			return vec3(x / r, y / r, z / r);
		else
			return vec3(0, 0, 0);
	}
	vec3 operator+(const vec3& v)  const { return vec3(x + v.x, y + v.y, z + v.z); }
	vec3 operator-(const vec3& v)  const { return vec3(x - v.x, y - v.y, z - v.z); }
	void operator+=(const vec3& v) { x += v.x, y += v.y, z += v.z; }
	void operator*=(double a) { x *= a, y *= a, z *= a; }
	double length()  const { return sqrt(x * x + y * y + z * z); }
	vec3 normalize() const
	{
		double l = length();
		if (l > epsilon)
			return (*this) / l;
		else
			return vec3(0, 0, 0);
	}
	double average() { return (x + y + z) / 3; }
};
// dot product of two vectors
double dot(const vec3& v1, const vec3& v2)
{
	return (v1.x * v2.x + v1.y * v2.y + v1.z * v2.z);
}
// cross product of two vectors
vec3 cross(const vec3& v1, const vec3& v2)
{
	return vec3(v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z, v1.x * v2.y - v1.y * v2.x);
}
// reflection of vector around normal
vec3 reflect(const vec3& N, const vec3& L)
{
	return  (N * (2.0 * dot(N, L))) - L;
}
// computes lumimance from RGB vector
float lumimance(const float r, const float g, const float b)
{
	return 0.2126f * r + 0.7152f * g + 0.0722f * b;
}

// return theta (spherical coords) from some vector 
inline float SphericalTheta(const vec3& v) {
	return acos(Clamp(v.y, -1.0, 1.0));
}
// return phi (spherical coords) from some vector 
inline float SphericalPhi(const vec3& v) {
	float p = atan2(v.z, v.x) - M_PI/2.0f;
	return (p < 0) ? (p + 2.0f * M_PI) : p;
}

struct Distribution1D
{
	float* func;
	float* cdf;

	float funcInt, invFuncInt, invCount;

	int count;

	Distribution1D(float* f, int n)
	{
		func = new float[n];
		cdf = new float[n + 1];
		count = n;
		memcpy(func, f, n * sizeof(float));

		ComputeStep1dCDF(func, n, &funcInt, cdf);
		
		invFuncInt = 1.0f / funcInt;
		invCount = 1.0f / count;
	}

	~Distribution1D()
	{
		delete[] func;
		delete[] cdf;
	}

	void ComputeStep1dCDF(float* f, int nSteps, float* c, float* cdf)
	{
		// Init first to 0 to be able to compute from it
		cdf[0] = 0.0;

		for (int i = 1; i < nSteps + 1; i++) {
			cdf[i] = cdf[i - 1] + f[i - 1] / nSteps;
		}

		// save sum
		*c = cdf[nSteps];

		// normalize
		for (int i = 0; i < nSteps + 1; i++) {
			cdf[i] = cdf[i] / *c;
		}
	}

	float Sample(float u, float* pdf)
	{
		// std version of binary search of first element that is not less than value
		float* ptr = std::lower_bound(cdf, cdf + count + 1, u);

		// index of that pointer in array
		int offset = (int)(ptr - cdf - 1);

		u = (u - cdf[offset]) / (cdf[offset + 1] - cdf[offset]);
		*pdf = func[offset] * invFuncInt;

		return offset + u;
	}
};


// -------------------- MATERIALS
// The definition of material surface (BRDF + emission)
struct Material
{
	vec3 Le; // the emmited power
	vec3 diffuseAlbedo; // albedo for diffuse component
	vec3 specularAlbedo; // albedo for specular component
	double  shininess;

	Material() {
		shininess = 0;
	}

	// Evaluate the BRDF given normal, view direction (outgoing) and light direction (incoming)
	vec3 BRDF(const vec3& N, const vec3& V, const vec3& L)
	{
		vec3 brdf(0, 0, 0);

		double cosThetaL = dot(N, L);
		double cosThetaV = dot(N, V);

		if (cosThetaL <= epsilon || cosThetaV <= epsilon)
			return brdf;

		brdf = diffuseAlbedo / M_PI; // diffuse part

		vec3 R = N * (dot(N, L) * 2.0) - L;
		double cosPhi = dot(V, R);

		if (cosPhi <= 0)
			return brdf; // farther by PI/2 from reflected direction

		// max-Phong specular BRDF: symmetric and energy conserving 
		return brdf + specularAlbedo * ((shininess + 1.0) / 2.0 / M_PI * pow(cosPhi, shininess) / fmax(cosThetaL, cosThetaV));
	}

	// BRDF.cos(theta) importance sampling for input normal, outgoing direction
	bool sampleDirection(const vec3& N, const vec3& V, vec3& L)
	{ // output - the incoming light direction
		L = vec3(0, 0, 0);

		const double randomType = drandom();
		const double um = drandom();
		const double vm = drandom();

		// only two randoms
		// prvni random musime podelit tim rozsahem abchom zas edostali 0-1
		// popripade podelit tim mezi koncem diffuse a koncem specular

		vec3 koeffs;
		if (randomType < diffuseAlbedo.average()) {
			const double alpham = acos(sqrt(um));
			const double thetam = 2.0 * M_PI * vm;

			koeffs.x = sin(alpham) * cos(thetam);
			koeffs.y = sin(alpham) * sin(thetam);
			koeffs.z = cos(alpham);

			const vec3 w = vec3(2.0 * drandom() - 1.0, 2.0 * drandom() - 1.0, 2.0 * drandom() - 1.0).normalize();
			vec3 k = N.normalize();
			// TODO: opatrne na ten vector (1, 0, 0) muze byt stejny jako N a muze to bejt spatny
			vec3 i = cross(N, w).normalize();
			vec3 j = cross(i, k).normalize();

			const vec3 Lm = i * koeffs.x + j * koeffs.y + k * koeffs.z;

			if (dot(N, Lm) < 0)
				return false;

			L = Lm;
		} else if (randomType < (diffuseAlbedo.average() + specularAlbedo.average())) {
			const double alpham = acos(pow(um, 1 / (shininess + 1)));
			const double thetam = 2.0 * M_PI * vm;

			koeffs.x = sin(alpham) * cos(thetam);
			koeffs.y = sin(alpham) * sin(thetam);
			koeffs.z = cos(alpham);

			const vec3 k = V.normalize();
			//const vec3 k = reflect(N, V).normalize();
			const vec3 i = cross(V, N).normalize();
			const vec3 j = cross(i, k).normalize();

			const vec3 Rm = i * koeffs.x + j * koeffs.y + k * koeffs.z;
			const vec3 Lm = (N * dot(N, Rm) * 2.0 - Rm).normalize();

			if (dot(N, Lm) < 0)
				return false;

			L = Lm;
		} else {
			return false;
		}

		return true; // error - no value
	}

	// Evaluate the probability given input normal, view (outgoing) direction and incoming light direction
	double sampleProb(const vec3& N, const vec3& V, const vec3& L)
	{
		double p = 0.0;

		p += diffuseAlbedo.average() * max(0.0, dot(L, N)) / M_PI;
		p += specularAlbedo.average() * ((shininess + 1.0) / (2.0 * M_PI)) * pow(max(0.0, dot(V, reflect(N, L))), shininess);

		return p;
	}
};

// Material used for light source
struct LightMaterial : Material
{
	LightMaterial(vec3 _Le) { Le = _Le; }
};

// Material used for objects, given how much is reflective/shiny
struct TableMaterial : Material
{
	TableMaterial(double shine)
	{
		shininess = shine;
		//diffuseAlbedo = vec3(0.8, 0.8, 0.8);
		diffuseAlbedo = vec3(0.5, 0.5, 0.5);
		specularAlbedo = vec3(0.5, 0.5, 0.5);
		//specularAlbedo = vec3(0.05, 0.05, 0.05);
	}
};


// Structure for a ray
struct Ray
{
	vec3 start, dir;
	Ray(const vec3& _start, const vec3& _dir) { start = _start; dir = _dir.normalize(); }
};

class Intersectable;

// Structure to store the result of ray tracing
struct Hit
{
	double t;
	vec3 position;
	vec3 normal;
	Material* material;
	Intersectable* object;
	Hit() { t = -1; }
};

// Abstract 3D object 
struct Intersectable
{
	Material* material;
	double power;
	virtual Hit intersect(const Ray& ray) = 0;
	virtual double pointSampleProb(double totalPower)
	{
		printf("Point sample on table\n");
		return 0;
	}
};

// Rectangle 2D in 3D space
class Rect :
	public Intersectable
{
	// anchor point, normal, 
	vec3 r0, normal, right, forward;
	double width, height; // size
public:
	Rect(vec3 _r0, vec3 _r1, vec3 _r2,
		double _width, double _height, Material* mat)
	{
		r0 = _r0;
		vec3 L = _r1 - r0;
		vec3 V = _r2 - r0;
		// compute normal
		normal = (L.normalize() + V.normalize()).normalize();
		material = mat;
		power = 0; // default - does not emit light
		width = _width; height = _height;
		// recompute directions to get rectangle
		right = cross(vec3(0, 0, 1), normal).normalize();
		forward = cross(normal, right).normalize();
	}

	// Compute intersection between a ray and the rectangle
	Hit intersect(const Ray& ray)
	{
		Hit hit;
		double denom = dot(normal, ray.dir);
		if (fabs(denom) > epsilon) {
			hit.t = dot(normal, r0 - ray.start) / denom;
			if (hit.t < 0) return hit;
			hit.position = ray.start + ray.dir * hit.t;
			double x = dot(hit.position - r0, right);
			double y = dot(hit.position - r0, forward);
			if (fabs(x) > width || fabs(y) > height) {
				hit.t = -1;
				return hit;
			}
			hit.normal = normal;
			hit.position = ray.start + ray.dir * hit.t;
			hit.material = material;
			hit.object = this;
		}
		return hit;
	}
};

// Sphere used as light source
struct Sphere :
	public Intersectable
{
	vec3 center;
	double  radius;

	Sphere(const vec3& cent, double rad, Material* mat)
	{
		const double targetPower = 60;
		center = cent; radius = rad;
		material = mat;
		power = material->Le.average() * (4 * radius * radius * M_PI) * M_PI;
		material->Le = material->Le * (targetPower / power);
		power = targetPower;
	}

	Hit intersect(const Ray& r)
	{
		Hit hit;
		vec3 dist = r.start - center;
		double b = dot(dist, r.dir) * 2.0;
		double a = dot(r.dir, r.dir);
		double c = dot(dist, dist) - radius * radius;
		double discr = b * b - 4.0 * a * c;
		if (discr < 0) return hit;
		double sqrt_discr = sqrt(discr);
		double t1 = (-b + sqrt_discr) / 2.0 / a;
		double t2 = (-b - sqrt_discr) / 2.0 / a;
		if (t1 <= 0 && t2 <= 0) return hit;
		if (t1 <= 0 && t2 > 0)
			hit.t = t2;
		else
			if (t2 <= 0 && t1 > 0)
				hit.t = t1;
			else
				if (t1 < t2)
					hit.t = t1;
				else
					hit.t = t2;
		hit.position = r.start + r.dir * hit.t;
		hit.normal = (hit.position - center) / radius;
		hit.material = material;
		hit.object = this;
		return hit;
	}

	// find a random point with uniform distribution on that half sphere, which can be visible
	void sampleUniformPoint(const vec3& illuminatedPoint, vec3& point, vec3& normal)
	{
		do {
			// uniform in a cube of edge size 2
			normal = vec3(drandom() * 2 - 1, drandom() * 2 - 1, drandom() * 2 - 1);
			if (dot(illuminatedPoint - center, normal) < 0) continue;	// ignore surely non visible points
		} while (dot(normal, normal) > 1);	// finish if the point is in the unit sphere
		normal = normal.normalize();	// project points onto the surface of the unit sphere
		point = center + normal * radius;	// project onto the real sphere
	}

	double pointSampleProb(double totalPower)
	{
		return power / totalPower / (4 * radius * radius * M_PI);
	}
};

// -------------------- LIGHTS
struct Light
{
	vec3 normal;

	// get sample 
	virtual vec3 sampleLight(const vec3& point, vec3* wi, float* pdf) = 0;
	 
	// probability of sample
	virtual float sampleProbability(const vec3& point, const vec3& wi, float totalPower) = 0;
	
	// get ilumination at dir
	virtual vec3 getIlumination(vec3 dir) = 0;
};

// The light source represented by a sphere
struct SphereLight : Light
{
	Sphere* sphere;
	vec3 point;

	SphereLight(Sphere* _sphere, vec3 _point, vec3 _normal) {
		sphere = _sphere;
		point = _point;
		normal = _normal;
	}

	vec3 sampleLight(const vec3& point, vec3* wi, float* pdf) {
		return sphere->material->Le;
	}

	float sampleProbability(const vec3& point, const vec3& wi, float totalPower) {
		return sphere->pointSampleProb(totalPower);
	}

	vec3 getIlumination(vec3 dir) {
		return sphere->material->Le;
	}
};

struct InfiniteAreaLight : Light
{
	float* img;
	float* luminanceImg;
	int nu;
	int nv;
	int n;

	vec3 point;

	bool debug = false;

	Distribution1D* uDistrib;
	Distribution1D** vDistribs;

	InfiniteAreaLight(const char* path)
	{
		img = stbi_loadf(path, &nu, &nv, &n, 0);

		luminanceImg = new float[nu * nv];
		RGBToLuminanceImage(img, nu, nv, luminanceImg);

		// calculate sin value for every row of image for later weighting
		float* sinVals = new float[nv * sizeof(float)];
		for (int i = 0; i < nv; i++) {
			sinVals[i] = sinf(M_PI * float(i + 0.5f) / float(nv));
		}

		// Buffer for storing sin weighted luminance values
		float* func = new float[max(nu, nv)];

		vDistribs = new Distribution1D * [nu];

		for (int u = 0; u < nu; ++u) {
			for (int v = 0; v < nv; ++v) {		
				func[v] = luminanceImg[v * nu + u] * sinVals[v];
			}
			vDistribs[u] = new Distribution1D(func, nv);
		}

		for (int u = 0; u < nu; ++u) {
			func[u] = vDistribs[u]->funcInt;
		}

		uDistrib = new Distribution1D(func, nu);
	}

	~InfiniteAreaLight()
	{
		stbi_image_free(img);
		delete[] luminanceImg;
	}

	vec3 sampleLight(const vec3& point, vec3* wi, float* pdf) {
		float u;
		float v;
		return Sample(wi, pdf, &u, &v);
	}

	float sampleProbability(const vec3& point, const vec3& wi, float totalPower) {
		return pdf(point, wi);
	}

	vec3 getIlumination(vec3 dir) {
		return sampleMapFromDirection(dir, vec3(1, 0, 0));
		//return vec3(0.0, 0.0, 0.0);
	}


	vec3 Sample(vec3* wi, float* pdf, float* uOpt, float* vOpt)
	{
		const float u1 = drandom();
		const float u2 = drandom();

		float pdfs[2];

		float fu = uDistrib->Sample(u1, &pdfs[0]);
		int u = Clamp((int)fu, 0, uDistrib->count - 1);
		float fv = vDistribs[u]->Sample(u2, &pdfs[1]);

	
		//*uOpt = fu / uDistrib->count;
		//*vOpt = fv / vDistribs[u]->count;

		float theta = fv * vDistribs[u]->invCount * M_PI;
		float phi = fu * uDistrib->invCount * 2.0f * M_PI + M_PI / 2.0f;

		*wi = vec3(sin(theta) * cos(phi), cos(theta), sin(theta) * sin(phi));
		// TABULE
		//*wi = vec3(sin(theta) * sin(phi), sin(theta) * cos(phi), cos(theta));

		normal = ( * wi * (-1.0f)).normalize();

		*pdf = (pdfs[0] * pdfs[1]) / (2.0f * M_PI * M_PI * sinf(theta));
		if (sin(theta) == 0.0f) *pdf = 0.0f;
		
		float c1, c2, c3;
		// spatne ale nepouziva se to
		c1 = img[(int)(fu * uDistrib->invCount) + (int)(fv * vDistribs[u]->invCount * nv)];
		c2 = img[(int)(fu * uDistrib->invCount) + (int)(fv * vDistribs[u]->invCount * nv) + 1];
		c3 = img[(int)(fu * uDistrib->invCount) + (int)(fv * vDistribs[u]->invCount * nv) + 2];
		
		if (debug) {
			int v = Clamp((int)fv, 0, nv - 1);

			int index = v * nu + u;
			img[3 * index] = 1.0f;
			img[3 * index + 1] = 0.0f;
			img[3 * index + 2] = 0.0f;
		}
		
		return vec3(c1, c2, c3);
	}

	float pdf(const vec3& point, const vec3& wi) {
		const float theta = SphericalTheta(wi);
		const float phi = SphericalPhi(wi);

		float sintheta = sinf(theta);
		if (sintheta == 0.0f) return 0.f;

		const int u = Clamp((int)(phi * (1 / (2 * M_PI) * uDistrib->count)), 0, uDistrib->count - 1);
		const int v = Clamp((int)(theta * (1 / M_PI) * vDistribs[u]->count), 0, vDistribs[u]->count - 1);

		if (vDistribs[v]->funcInt * uDistrib->funcInt == 0.0f) return 0.0f;

		return ((uDistrib->func[u] * vDistribs[u]->func[v]) /
			(uDistrib->funcInt * vDistribs[u]->funcInt)) *
			1.0f / (2.0f * M_PI * M_PI * sintheta);
	}

	void RGBToLuminanceImage(float* image, int nu, int nv, float* output)
	{
		for (int u = 0; u < nu; u++) {
			for (int v = 0; v < nv; v++) {
				int index = v * nu + u;

				float r = img[index * 3] * 255;
				float g = img[index * 3 + 1] * 255;
				float b = img[index * 3 + 2] * 255;

				float l = lumimance(r, g, b) / 256;

				output[index] = l;
			}
		}
	}

	vec3 sampleMapFromDirection(const vec3& dir, const vec3& color) {
		const float phi = SphericalPhi(dir.normalize());
		const float theta = SphericalTheta(dir.normalize());

		const int u = Clamp((int)(phi * (1 / (2 * M_PI) * uDistrib->count)), 0, uDistrib->count - 1);
		const int v = Clamp((int)(theta * (1 / M_PI) * vDistribs[u]->count), 0, vDistribs[u]->count - 1);

		const int index = v * nu + u;

		const float r = img[index * 3];
		const float g = img[index * 3 + 1];
		const float b = img[index * 3 + 2];

		if (debug) {
			img[index * 3] = color.x;
			img[index * 3 + 1] = color.y;
			img[index * 3 + 2] = color.z;
		}

		return vec3(r, g, b);
	}

	void sampleRandomUniformDirection(vec3* wi, float* pdf) {
		const double u1 = drandom();
		const double u2 = drandom();

		const float phi = 2.0f * M_PI * u1;
		const float theta = acos(1.0f - 2.0f * u2);


		 *wi = vec3(sin(theta) * cos(phi), cos(theta), sin(theta) * sin(phi));
		
		 normal = *wi * (-1.0f);

		 *pdf = 1 / (4 * M_PI);
	}
};

const int screenWidth = 600;
const int screenHeight = 600;
vec3 image[screenWidth * screenHeight]; // computed image
vec3 reference[screenWidth * screenHeight]; // reference image
// alpha of light source sampling, 0 .. BRDF only, 1.0 .. light only
double weight[screenWidth * screenHeight] = { 0 };

// Definition of the camera
class Camera
{
	// center of projection and orthogonal basis of the camera
	vec3 eye, lookat, right, up;
public:
	void set(const vec3& _eye, const vec3& _lookat, const vec3& _vup, double fov)
	{
		eye = _eye;
		lookat = _lookat;
		vec3 w = eye - lookat;
		double f = w.length();
		right = cross(_vup, w).normalize() * f * tan(fov / 2);
		up = cross(w, right).normalize() * f * tan(fov / 2);
	}
	Ray getRay(int X, int Y)
	{ // X,Y - pixel coordinates, compute a primary ray
		vec3 dir = lookat +
			right * (2.0 * (X + 0.5) / screenWidth - 1) +
			up * (2.0 * (Y + 0.5) / screenHeight - 1) - eye;
		return Ray(eye, dir.normalize());
	}
};

// Which sampling method should be used
enum Method { BRDF, LIGHT_SOURCE, LIGHT_SOURCE_ENV, NAIVE } method;

// The scene definition with main rendering method
class Scene
{
	std::vector<Intersectable*> objects;
	double totalPower;
	int nLightSamples, nBRDFSamples;
public:
	Camera camera;
	//InfiniteAreaLight envMap = InfiniteAreaLight("./EM/raw023.hdr");;
	//InfiniteAreaLight envMap = InfiniteAreaLight("./EM/rotated/raw004.hdr");;
	InfiniteAreaLight envMap = InfiniteAreaLight("./EM/rotated/raw013.hdr");;
	//InfiniteAreaLight envMap = InfiniteAreaLight("./debugEnv.hdr");;

	void build()
	{
		// Create a simple scene
		vec3 eyePos(0, 6, 18); // camera center
		vec3 lightCenterPos(0, 4, -6); // first light source

		// Create geometry - 4 rectangles
		bool sameShininies = false;
		if (sameShininies == true) {
			objects.push_back(new Rect(vec3(0, -4, +2), eyePos, lightCenterPos, 4, 1, new TableMaterial(800)));
			objects.push_back(new Rect(vec3(0, -3.5, -2), eyePos, lightCenterPos, 4, 1, new TableMaterial(800)));
			objects.push_back(new Rect(vec3(0, -2.5, -6), eyePos, lightCenterPos, 4, 1, new TableMaterial(800)));
			objects.push_back(new Rect(vec3(0, -1, -10), eyePos, lightCenterPos, 4, 1, new TableMaterial(800)));

		} else {
			objects.push_back(new Rect(vec3(0, -4, +2), eyePos, lightCenterPos, 4, 1, new TableMaterial(500)));
			objects.push_back(new Rect(vec3(0, -3.5, -2), eyePos, lightCenterPos, 4, 1, new TableMaterial(1000)));
			objects.push_back(new Rect(vec3(0, -2.5, -6), eyePos, lightCenterPos, 4, 1, new TableMaterial(5000)));
			objects.push_back(new Rect(vec3(0, -1, -10), eyePos, lightCenterPos, 4, 1, new TableMaterial(10000)));
		}
		
		// Create 4 light sources
		objects.push_back(new Sphere(lightCenterPos + vec3(-4.5, 0, 0), 0.07, new LightMaterial(vec3(4, 2, 1))));
		objects.push_back(new Sphere(lightCenterPos + vec3(-1.5, 0, 0), 0.16, new LightMaterial(vec3(2, 4, 1))));
		objects.push_back(new Sphere(lightCenterPos + vec3(1.5, 0, 0), 0.4, new LightMaterial(vec3(2, 1, 4))));
		objects.push_back(new Sphere(lightCenterPos + vec3(4.5, 0, 0), 1, new LightMaterial(vec3(4, 1, 2))));

		// Set the camera
		camera.set(eyePos, vec3(0, 0, 0), vec3(0, 1, 0), 35.0 * M_PI / 180.0);

		totalPower = 0;
		for (int i = 0; i < objects.size(); i++) {
			totalPower += objects[i]->power; //  hit.t < 0 if no intersection
		}
	}

	// Set the weight for the sampling method
	void setWeight(double wval)
	{
		for (int Y = 0; Y < screenHeight; Y++)
			for (int X = 0; X < screenWidth; X++)
				weight[Y * screenWidth + X] = wval;
	}

	// Render the scene
	void render()
	{
		// Total number of samples per pixel is: nIterators*nTotalSamples
		srand(1);
		char buffer[100];
		FILE* errorFile = 0;

		switch (method) {
		case BRDF:
			nBRDFSamples = nTotalSamples;
			nLightSamples = 0;
			errorFile = fopen("BRDF.txt", "w");
			setWeight(0.0);
			break;
		case NAIVE:
		case LIGHT_SOURCE_ENV:
		case LIGHT_SOURCE:
			nBRDFSamples = 0;
			nLightSamples = nTotalSamples;
			errorFile = fopen("light.txt", "w");
			setWeight(1.0);
			break;
		} // switch

		double cost = 0;
		bool debug = true;
		// How many iterations
		for (int iIter = 1; iIter <= nIterations; iIter++) {
			double error = 0;
			for (int Y = 0; Y < screenHeight; Y++) { // for all rows

				#pragma omp parallel for
				for (int X = 0; X < screenWidth; X++) { // for all pixels in a row
					//if (debug) { // debug particular pixel x,y, coordinates from pfsv (pfstools)
					//	X = 287;
					//	Y = 270;
					//}

					nLightSamples = (int)(weight[Y * screenWidth + X] * nTotalSamples + 0.5);
					nBRDFSamples = nTotalSamples - nLightSamples;
					cost += nBRDFSamples * costBRDF + nLightSamples * costLight;

					// For a primary ray at pixel (X,Y) compute the color
					vec3 color;
					if (method == NAIVE) {
						color = traceNaive(camera.getRay(X, Y));
					} else {
						color = trace2(camera.getRay(X, Y));
					}
					
					double w = 1.0 / iIter; // the same weight for all samples for computing mean incrementally
					image[Y * screenWidth + X] = color * w + image[Y * screenWidth + X] * (1.0 - w);

					w = 1.0 / sqrt(iIter); // emphasize later samples
					vec3 diff = reference[Y * screenWidth + X] - image[Y * screenWidth + X];
					error += dot(diff, diff);
				} // for X
			} // for Y
			double eff = 100000.0 * nIterations * nTotalSamples * screenWidth * screenHeight / error / cost;
			printf("Iter: %d, Error: %4.2f, Efficiency: %f, Relative Efficiency: %f\n", iIter, sqrt(error), eff, eff / referenceEfficiency);
			fprintf(errorFile, "%d, %f\n", iIter * nTotalSamples, sqrt(error));
		} // for iTer
		fclose(errorFile);
	} // render

	// Compute intersection between a ray and primitive
	Hit firstIntersect(const Ray& ray, Intersectable* skip)
	{
		Hit bestHit;
		for (int i = 0; i < objects.size(); i++) {
			if (objects[i] == skip) continue;
			if (method == LIGHT_SOURCE_ENV || method == BRDF || method == NAIVE) {
				if (i >= 4 && i <= 7) {
					continue;
				}
			}
			Hit hit = objects[i]->intersect(ray); //  hit.t < 0 if no intersection
			if (hit.t > epsilon) {
				if (bestHit.t < 0 || hit.t < bestHit.t) bestHit = hit;
			}
		}
		return bestHit;
	}

	// Sample the light source from all the light sources in the scene
	SphereLight* sampleLightSource(const vec3& illuminatedPoint) // the 3D point on an object
	{
		while (true) { // if no light source is selected due to floating point inaccuracies, repeat
			double threshold = totalPower * drandom();
			double running = 0;
			for (int i = 0; i < objects.size(); i++) {
				running += objects[i]->power; // select light source with the probability proportional to its power
				if (running > threshold) {
					Sphere* sphere = (Sphere*)objects[i];
					vec3 point, normal;
					// select a point on the visible half of the light source
					((Sphere*)objects[i])->sampleUniformPoint(illuminatedPoint, point, normal);
					return new SphereLight(sphere, point, normal);
				} // if
			} // for i
		} // for ever
	}

	Light& sampleLight(vec3& point, vec3& direction) {
		if (method == LIGHT_SOURCE) {
			SphereLight* lightSample = sampleLightSource(point); // generate a light sample

			vec3 outDir = lightSample->point - point; // compute direction towards sample

			direction = outDir;

			return *lightSample;
		} else {
			// return env map
			float pdf;
			envMap.sampleLight(point, &direction, &pdf);
			return envMap;
		}
	}

	vec3 trace2(const Ray& r)
	{
		// error measures for the two combined techniques: used for adaptation
		Hit hit = firstIntersect(r, NULL);	// find visible point
		if (hit.t < 0) {
			//if (method == LIGHT_SOURCE_ENV) {
				return envMap.sampleMapFromDirection(r.dir, vec3(0,1,0));
			//} else {
			//	return vec3(0, 0, 0);
			//}
		}

		// The energy emanated from the material
		vec3 radianceEmitted = hit.material->Le;

		if (hit.material->diffuseAlbedo.average() < epsilon && hit.material->specularAlbedo.average() < epsilon) {
			return radianceEmitted; // if albedo is low, no energy can be reefleted
		}

		// Compute the contribution of reflected lgiht
		vec3 radianceBRDFSampling(0, 0, 0);
		vec3 radianceLightSourceSampling(0, 0, 0);
		vec3 inDir = r.dir * (-1);	// incident direction

		int nTotalSamples = (nLightSamples + nBRDFSamples);
		double alpha = (double)nLightSamples / nTotalSamples;

		// The direct illumination for chosen number of samples
		for (int i = 0; i < nLightSamples; i++) {
			vec3 outDir = vec3(0, 0, 0);
			Light& lightSample = sampleLight(hit.position, outDir);

			double distance2 = dot(outDir, outDir);
			if (sqrt(distance2) < epsilon) {
				continue;
			}

			outDir = outDir.normalize();

			//vec3 col = lightSample.getIlumination(outDir);

			double cosThetaLight = dot(lightSample.normal, outDir * (-1));

			if (cosThetaLight > epsilon) {
				// visibility is not needed to handle, all lights are visible
				double pdfLightSourceSampling = lightSample.sampleProbability(hit.position, outDir, totalPower) * distance2 / cosThetaLight;
				double pdfBRDFSampling = hit.material->sampleProb(hit.normal, inDir, outDir);

				// the theta angle on the surface between normal and light direction
				double cosThetaSurface = dot(hit.normal, outDir);
				if (cosThetaSurface > 0) {
					// yes, the light is visible and contributes to the output power
					// The evaluation of rendering equation locally: (light power) * brdf * cos(theta)

					vec3 f = lightSample.getIlumination(outDir) * hit.material->BRDF(hit.normal, inDir, outDir) * cosThetaSurface;

					double p = pdfLightSourceSampling;
					// importance sample = 1/n . \sum (f/prob)
					radianceLightSourceSampling += f / p / nTotalSamples;
				} // if
			}
		} // for all the samples from light

		// The contribution from importance sampling BRDF.cos(theta)
		for (int i = 0; i < nBRDFSamples; i++) {
			// BRDF.cos(theta) sampling should be implemented first!
			vec3 outDir;

			// BRDF sampling with Russian roulette
			if (hit.material->sampleDirection(hit.normal, inDir, outDir)) {
				double pdfBRDFSampling = hit.material->sampleProb(hit.normal, inDir, outDir);
				double cosThetaSurface = dot(hit.normal, outDir);

				if (cosThetaSurface > 0) {
					vec3 brdf = hit.material->BRDF(hit.normal, inDir, outDir);

					// Trace a ray to the scene
					//Hit lightSource = firstIntersect(Ray(hit.position, outDir), hit.object);
					vec3 Le = envMap.getIlumination(outDir);

					// Do we hit a light source
					//if (lightSource.t > 0 && lightSource.material->Le.average() > 0) {
					if (Le.average() > 0) {
						// squared distance between an illuminated point and light source
						//double distance2 = lightSource.t * lightSource.t;
						//double cosThetaLight = dot(lightSource.normal, outDir * (-1));
						double cosThetaLight = dot(outDir * (-1), outDir * (-1));

						if (cosThetaLight > epsilon) {
							//double pdfLightSourceSampling = lightSource.object->pointSampleProb(totalPower) * distance2 / cosThetaLight;
							double pdfLightSourceSampling = envMap.sampleProbability(hit.position, outDir, totalPower) / cosThetaLight;

							// The evaluation of rendering equation locally: (light power) * brdf * cos(theta)
							//vec3 f = lightSource.material->Le * brdf * cosThetaSurface;
							vec3 f = Le * brdf * cosThetaSurface;
							double p = pdfBRDFSampling;
							radianceBRDFSampling += f / p / nTotalSamples;
						}
						else
							printf("ERROR: Sphere hit from back\n");
					}
				}
			}
		} // for i
		return radianceEmitted + radianceLightSourceSampling + radianceBRDFSampling;
	}

	vec3 traceNaive(const Ray& r)
	{
		// error measures for the two combined techniques: used for adaptation
		Hit hit = firstIntersect(r, NULL);	// find visible point
		if (hit.t < 0) {
			return envMap.sampleMapFromDirection(r.dir, vec3(0, 1, 0));
		}

		// The energy emanated from the material
		vec3 radianceEmitted = hit.material->Le;

		if (hit.material->diffuseAlbedo.average() < epsilon && hit.material->specularAlbedo.average() < epsilon) {
			return radianceEmitted; // if albedo is low, no energy can be reefleted
		}

		// Compute the contribution of reflected lgiht
		vec3 radianceBRDFSampling(0, 0, 0);
		vec3 radianceLightSourceSampling(0, 0, 0);
		vec3 inDir = r.dir * (-1);	// incident direction

		int nTotalSamples = (nLightSamples + nBRDFSamples);
		double alpha = (double)nLightSamples / nTotalSamples;

		// The direct illumination for chosen number of samples
		for (int i = 0; i < nLightSamples; i++) {
			vec3 outDir = vec3(0, 0, 0);
			
			float pdf;
			envMap.sampleRandomUniformDirection(&outDir, &pdf);

			double distance2 = dot(outDir, outDir);
			if (sqrt(distance2) < epsilon) {
				continue;
			}

			outDir = outDir.normalize();

			double cosThetaLight = dot(envMap.normal, outDir * (-1));
			if (cosThetaLight > epsilon) {
				// visibility is not needed to handle, all lights are visible
				double pdfLightSourceSampling = pdf;
				double pdfBRDFSampling = hit.material->sampleProb(hit.normal, inDir, outDir);

				// the theta angle on the surface between normal and light direction
				double cosThetaSurface = dot(hit.normal, outDir * (-1.f));
				if (cosThetaSurface > 0) {
					// yes, the light is visible and contributes to the output power
					// The evaluation of rendering equation locally: (light power) * brdf * cos(theta)

					vec3 f = envMap.getIlumination(outDir) * hit.material->BRDF(hit.normal, inDir, outDir) * cosThetaSurface;

					//std::cout << f.x << " " << f.y << " " << f.z << std::endl;

					double p = pdfLightSourceSampling;
					// importance sample = 1/n . \sum (f/prob)
					radianceLightSourceSampling += f / p / nTotalSamples;
				} // if
			}
		} // for all the samples from light

		// The contribution from importance sampling BRDF.cos(theta)
		for (int i = 0; i < nBRDFSamples; i++) {
			// BRDF.cos(theta) sampling should be implemented first!
			vec3 outDir;

			// BRDF sampling with Russian roulette
			if (hit.material->sampleDirection(hit.normal, inDir, outDir)) {
				double pdfBRDFSampling = hit.material->sampleProb(hit.normal, inDir, outDir);
				double cosThetaSurface = dot(hit.normal, outDir);

				if (cosThetaSurface > 0) {
					vec3 brdf = hit.material->BRDF(hit.normal, inDir, outDir);

					// Trace a ray to the scene
					Hit lightSource = firstIntersect(Ray(hit.position, outDir), hit.object);

					/*if (hit.t < 0) {
						radianceBRDFSampling += envMap.getIlumination(outDir) / envMap.pdf(hit.position, outDir) / nTotalSamples;
						continue;
					}*/

					// Do we hit a light source
					if (lightSource.t > 0 && lightSource.material->Le.average() > 0) {
						// squared distance between an illuminated point and light source
						double distance2 = lightSource.t * lightSource.t;
						double cosThetaLight = dot(lightSource.normal, outDir * (-1));

						if (cosThetaLight > epsilon) {
							double pdfLightSourceSampling = lightSource.object->pointSampleProb(totalPower) * distance2 / cosThetaLight;

							// The evaluation of rendering equation locally: (light power) * brdf * cos(theta)
							vec3 f = lightSource.material->Le * brdf * cosThetaSurface;
							double p = pdfBRDFSampling;
							radianceBRDFSampling += f / p / nTotalSamples;
						}
						else
							printf("ERROR: Sphere hit from back\n");
					}
				}
			}
		} // for i
		return radianceEmitted + radianceLightSourceSampling + radianceBRDFSampling;
	}


	// Trace a primary ray towards the scene
	/*
	vec3 trace(const Ray& r)
	{
		// error measures for the two combined techniques: used for adaptation
		Hit hit = firstIntersect(r, NULL);	// find visible point
		if (hit.t < 0) {
			if (method == LIGHT_SOURCE_ENV) {
				return envMap.sampleMapFromDirection(r.dir);
			} else {
				return vec3(0, 0, 0);
			}
		} 
		
		// The energy emanated from the material
		vec3 radianceEmitted = hit.material->Le;
		
		if (hit.material->diffuseAlbedo.average() < epsilon && hit.material->specularAlbedo.average() < epsilon) {
			return radianceEmitted; // if albedo is low, no energy can be reefleted
		}
		
		// Compute the contribution of reflected lgiht
		vec3 radianceBRDFSampling(0, 0, 0);
		vec3 radianceLightSourceSampling(0, 0, 0);
		vec3 inDir = r.dir * (-1);	// incident direction

		int nTotalSamples = (nLightSamples + nBRDFSamples);
		double alpha = (double)nLightSamples / nTotalSamples;

		// The direct illumination for chosen number of samples
		for (int i = 0; i < nLightSamples; i++) {
			SphereLight lightSample = *sampleLightSource(hit.position); // generate a light sample

			vec3 outDir = lightSample.point - hit.position; // compute direction towards sample
			
			double distance2 = dot(outDir, outDir);
			double distance = sqrt(distance2);
			
			if (distance < epsilon) {
				continue;
			}

			outDir = outDir / distance; // normalize the direction
			
			double cosThetaLight = dot(lightSample.normal, outDir * (-1));
			double random = drandom();
			
			if (cosThetaLight > epsilon) {
				// visibility is not needed to handle, all lights are visible
				double pdfLightSourceSampling = lightSample.sphere->pointSampleProb(totalPower) * distance2 / cosThetaLight;
				double pdfBRDFSampling = hit.material->sampleProb(hit.normal, inDir, outDir);
				
				// the theta angle on the surface between normal and light direction
				double cosThetaSurface = dot(hit.normal, outDir);
				if (cosThetaSurface > 0) {
					// yes, the light is visible and contributes to the output power
					// The evaluation of rendering equation locally: (light power) * brdf * cos(theta)
					vec3 f = lightSample.sphere->material->Le * hit.material->BRDF(hit.normal, inDir, outDir) * cosThetaSurface;
					double p = pdfLightSourceSampling;
					// importance sample = 1/n . \sum (f/prob)
					radianceLightSourceSampling += f / p / nTotalSamples;
				} // if
			}
		} // for all the samples from light

		// The contribution from importance sampling BRDF.cos(theta)
		for (int i = 0; i < nBRDFSamples; i++) {
			// BRDF.cos(theta) sampling should be implemented first!
			vec3 outDir;

			// BRDF sampling with Russian roulette
			if (hit.material->sampleDirection(hit.normal, inDir, outDir)) {
				double pdfBRDFSampling = hit.material->sampleProb(hit.normal, inDir, outDir); 
				double cosThetaSurface = dot(hit.normal, outDir);
				
				if (cosThetaSurface > 0) {
					vec3 brdf = hit.material->BRDF(hit.normal, inDir, outDir);
					
					// Trace a ray to the scene
					Hit lightSource = firstIntersect(Ray(hit.position, outDir), hit.object);
					
					// Do we hit a light source
					if (lightSource.t > 0 && lightSource.material->Le.average() > 0) {
						// squared distance between an illuminated point and light source
						double distance2 = lightSource.t * lightSource.t;
						double cosThetaLight = dot(lightSource.normal, outDir * (-1));
						
						if (cosThetaLight > epsilon) {
							double pdfLightSourceSampling = lightSource.object->pointSampleProb(totalPower) * distance2 / cosThetaLight;
							
							// The evaluation of rendering equation locally: (light power) * brdf * cos(theta)
							vec3 f = lightSource.material->Le * brdf * cosThetaSurface;
							double p = pdfBRDFSampling;
							radianceBRDFSampling += f / p / nTotalSamples;
						} else
							printf("ERROR: Sphere hit from back\n");
					}
				}
			}
		} // for i
		return radianceEmitted + radianceLightSourceSampling + radianceBRDFSampling;
	}
	*/

	// Only testing routine for debugging
	void testRay(int X, int Y)
	{
		nBRDFSamples = nLightSamples = 1000;
		vec3 current = trace2(camera.getRay(X, Y));
		printf("Pixel %d, %d Value = %f, %f, %f\n", X, Y, current.x, current.y, current.z);
	}
};

// Global variable
Scene scene;

void onInitialization()
{
	for (int Y = 0; Y < screenHeight; Y++) {
		for (int X = 0; X < screenWidth; X++) {
			reference[Y * screenWidth + X] = image[Y * screenWidth + X] = vec3(0, 0, 0);
		}
	}
	// Read the reference image from binary file
	FILE* refImage = fopen("image.bin", "rb");
	if (!refImage) {
		printf("No reference file\n");
	} else {
		int sz = fread(reference, sizeof(vec3), screenWidth * screenHeight, refImage);
		fclose(refImage);
		for (int Y = 0; Y < screenHeight; Y++) {
			for (int X = 0; X < screenWidth; X++) {
				image[Y * screenWidth + X] = reference[Y * screenWidth + X];
			}
		}
	}
	glViewport(0, 0, screenWidth, screenHeight);
	scene.build(); // create scene objects
	method = LIGHT_SOURCE; // the method to compute an image
}

 void getPseudocolorRainbow(double val, double minVal, double maxVal, double& r, double& g, double& b)
{
	if (isnan(val) || isinf(val)) {
		r = g = b = 0; // black ... exception
		return;
	}
	if (val < minVal) val = minVal;
	if (val > maxVal) val = maxVal;
	double ratio = (val - minVal) / (maxVal - minVal);
	double value = 1.0f - ratio;
	float val4 = value * 4.0f;
	value = val4 - (int)val4;
	switch ((int)(val4)) {
	case 0: r = 1.0; g = value; b = 0.f; break;
	case 1: r = 1.0 - value; g = 1.0; b = 0.f; break;
	case 2: r = 0.f; g = 1.0; b = value; break;
	case 3: r = 0.f; g = 1.0 - value; b = 1.0; break;
	default: r = value * 1.0; g = 0.f; b = 1.0; break;
	}
	return;
}

void getPseudocolorCoolWarm(double val, double minVal, double maxVal, double& r, double& g, double& b)
{
	if (isnan(val) || isinf(val)) {
		r = g = b = 0; // black ... exception
		return;
	}
	if (val < minVal) val = minVal;
	if (val > maxVal) val = maxVal;
	double ratio = (val - minVal) / (maxVal - minVal);
	int i = int(ratio * 31.999);
	assert(i < 33); assert(i >= 0);
	float alpha = i + 1.0 - (ratio * 31.999);
	r = pscols[4 * i + 1] * alpha + pscols[4 * (i + 1) + 1] * (1.0 - alpha);
	g = pscols[4 * i + 2] * alpha + pscols[4 * (i + 1) + 2] * (1.0 - alpha);
	b = pscols[4 * i + 3] * alpha + pscols[4 * (i + 1) + 3] * (1.0 - alpha);
	//printf("rgb=%f %f %f index=%d a=%g\n",r,g,b,i, alpha);
}

void onDisplay()
{
	glClearColor(0.1f, 0.2f, 0.3f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	static float displayImage[screenWidth * screenHeight * 3] = { 0 };
	// Image on the left
	for (int Y = 0; Y < screenHeight; Y++) {
		for (int X = 0; X < screenWidth; X++) {
			displayImage[3 * (Y * screenWidth + X)] = image[Y * screenWidth + X].x;
			displayImage[3 * (Y * screenWidth + X) + 1] = image[Y * screenWidth + X].y;
			displayImage[3 * (Y * screenWidth + X) + 2] = image[Y * screenWidth + X].z;
		}
	}
	glRasterPos2i(-1, -1);
	glDrawPixels(screenWidth, screenHeight, GL_RGB, GL_FLOAT, displayImage);
	// Image on the right
	for (int Y = 0; Y < screenHeight; Y++) {
		for (int X = 0; X < screenWidth; X++) {
			if (showFlag == DIFF) {
				double diff = (image[Y * screenWidth + X] - reference[Y * screenWidth + X]).average();
				displayImage[3 * (Y * screenWidth + X)] = diff > 0 ? diff : 0;
				displayImage[3 * (Y * screenWidth + X) + 1] = diff < 0 ? -diff : 0;
				displayImage[3 * (Y * screenWidth + X) + 2] = 0;
			}
			if (showFlag == WEIGHT) { // black to white
				double w = weight[Y * screenWidth + X];
				displayImage[3 * (Y * screenWidth + X)] = w;
				displayImage[3 * (Y * screenWidth + X) + 1] = w;
				displayImage[3 * (Y * screenWidth + X) + 2] = w;
			}
			if (showFlag == WEIGHT_PSEUDOCOLOR) {
				double w = weight[Y * screenWidth + X];
				if (showBargraph && (X > 0.98 * screenWidth))
					w = (double)Y / screenHeight; // thin bar on the right showing the mapping
				double r, g, b;
				if (rainbowPSC)
					getPseudocolorRainbow(w, 0.0, 1.0, r, g, b); // is more common but wrong perceptually
				else
					getPseudocolorCoolWarm(w, 0.0, 1.0, r, g, b); // is perceptually better
				displayImage[3 * (Y * screenWidth + X)] = r;
				displayImage[3 * (Y * screenWidth + X) + 1] = g;
				displayImage[3 * (Y * screenWidth + X) + 2] = b;
			}
		}
	}
	glRasterPos2i(0, -1);
	glDrawPixels(screenWidth, screenHeight, GL_RGB, GL_FLOAT, displayImage);
	glutSwapBuffers();

	// Save TGA file for the image, simple format
	FILE* ofile = 0;
	switch (method) {
	case BRDF:	ofile = fopen("brdf.tga", "wb"); break;
	case LIGHT_SOURCE:	ofile = fopen("lightsource.tga", "wb"); break;
	}
	if (!ofile) return;

	fputc(0, ofile); fputc(0, ofile); fputc(2, ofile);
	for (int i = 3; i < 12; i++) { fputc(0, ofile); }
	int width = screenWidth * 2, height = screenHeight;
	fputc(width % 256, ofile); fputc(width / 256, ofile);
	fputc(height % 256, ofile); fputc(height / 256, ofile);
	fputc(24, ofile);
	fputc(32, ofile);

	for (int Y = screenHeight - 1; Y >= 0; Y--) {
		for (int X = 0; X < width; X++) {
			double r, g, b;
			if (X < screenWidth) {
				r = image[Y * screenWidth + X].x;
				g = image[Y * screenWidth + X].y;
				b = image[Y * screenWidth + X].z;
			} else {
				int XX = X - screenWidth;
				double w = weight[Y * screenWidth + XX];
				if (showBargraph && (XX > 0.98 * screenWidth))
					w = (double)Y / screenHeight; // thin bar on the right showing the mapping
				if (rainbowPSC)
					getPseudocolorRainbow(w, 0.0, 1.0, r, g, b); // is more common but wrong perceptually
				else
					getPseudocolorCoolWarm(w, 0.0, 1.0, r, g, b); // is perceptually better
			}
			int R = fmax(fmin(r * 255.5, 255), 0);
			int G = fmax(fmin(g * 255.5, 255), 0);
			int B = fmax(fmin(b * 255.5, 255), 0);
			fputc(B, ofile); fputc(G, ofile); fputc(R, ofile);
		}
	}
	fclose(ofile);
}

void Usage()
{
	printf("Usage:\n");
	printf(" 'b': BRDF sampling \n");
	printf(" 'l': light source sampling \n");
	printf(" 'r': Show reference\n");
	printf(" 'w': Print current as a ground truth reference file for the future\n");
	printf(" 'o': Write output HDR file of rendered image\n\n");
	printf(" 'O': Write output HDR file render+pseudocolor visualization\n\n");
}

// Mouse click event
void onMouse(int button, int state, int pX, int pY)
{
	if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {
		// select GLUT_LEFT_BUTTON / GLUT_RIGHT_BUTTON and GLUT_DOWN / GLUT_UP
		scene.testRay(pX, screenHeight - pY);
	}
}

void onKeyboard(unsigned char key, int x, int y)
{
	switch (key) {
		printf("%c", key);
	case 'h':
	{
		printf("Try to load HDR image\n");

	
		InfiniteAreaLight light = InfiniteAreaLight("./EM/raw015.hdr");
		vec3 wi; float pdf;
		float u;
		float v;

		for (int i = 0; i < 50000; i++) {
			// generate point in map according to samplin
			vec3 pos = light.Sample(&wi, &pdf, &u, &v);

			u = Clamp(u, 0.0f, 1.0f);
			v = Clamp(v, 0.0f, 1.0f);
			
			// make pixel there red
			int index = int((light.nv - 1) * v) * light.nu + int((light.nu - 1) * u);
			light.img[index * 3] = 1.0f;
			light.img[index * 3 + 1] = 0.0f;
			light.img[index * 3 + 2] = 0.0f;
		}

		// save HDR map
		stbi_write_hdr("swag.hdr", light.nu, light.nv, 3, light.img);


		stbi_write_hdr("swagLumi.hdr", light.nu, light.nv, 1, light.luminanceImg);


		break;
	}
	case 'd':
	{
		printf("Creating debug env map");

		InfiniteAreaLight light = InfiniteAreaLight("./EM/raw023.hdr");
		vec3 wi; float pdf;
		float u;
		float v;

		for (int u = 0; u < light.nu;u++) {
			for (int v = 0; v < light.nv; v++)
			{
				int index = v * light.nu + u;
				if (u >= 15 && u <= 50 && v >= 680 && v <= 710) {
					light.img[index * 3] = 1.0f;
					light.img[index * 3 + 1] = 1.0f;
					light.img[index * 3 + 2] = 1.0f;
				} else {
					light.img[index * 3] = 0.3f;
					light.img[index * 3 + 1] = 0.3f;
					light.img[index * 3 + 2] = 0.3f;
				}
			}
		}

		// save HDR map
		stbi_write_hdr("debugEnv.hdr", light.nu, light.nv, 3, light.img);

		break;
	}
	case 'l':
		method = LIGHT_SOURCE;
		printf("Light source sampling\n");
		scene.render();
		break;
	case 'e':
		method = LIGHT_SOURCE_ENV;
		printf("ENV map sampling\n");
		scene.render();
		
		//stbi_write_hdr("swag.hdr", scene.envMap.nu, scene.envMap.nv, 3, scene.envMap.img);

		break;
	case 'b':
		method = BRDF;
		printf("BRDF sampling\n");
		scene.render();
		break;
	case 'n':
		method = NAIVE;
		printf("Naive env map render\n");
		scene.render();

		stbi_write_hdr("swag.hdr", scene.envMap.nu, scene.envMap.nv, 3, scene.envMap.img);

		break;
	case 'w':
	{
		printf("Writing reference file\n");
		FILE* refImage = fopen("image.bin", "wb");
		if (refImage) {
			fwrite(image, sizeof(vec3), screenWidth * screenHeight, refImage);
			fclose(refImage);
		}
	}
	case 'O':
	case 'o':
	{
		printf("Writing output HDR file (extension .hdr)\n");
		FILE* fp = nullptr;
		switch (method) {
		case LIGHT_SOURCE: fp = fopen("lightsource.hdr", "wb"); break;
		case BRDF: fp = fopen("brdf.hdr", "wb"); break;
		case LIGHT_SOURCE_ENV: fp = fopen("lightsourceEnv.hdr", "wb"); break;
		case NAIVE: fp = fopen("naive.hdr", "wb"); break;
		}
		int width = screenWidth;
		bool psf = false;
		if (key == 'O') { width *= 2;psf = true; }
		if (fp) {
			size_t nmemb = width * screenHeight;
			typedef unsigned char RGBE[4];
			RGBE* data = new RGBE[nmemb];
			for (int ii = 0; ii < nmemb; ii++) {
				RGBE& rgbe = data[ii];
				int x = (ii % width);
				int y = screenHeight - (ii / width) + 1;
				vec3 vv;
				assert(image);
				vv = image[y * screenWidth + x];
				if (psf) {
					if (x < screenWidth) {
						vv = image[y * screenWidth + x];
					} else {
						x -= screenWidth;
						double w = weight[y * screenWidth + x];
						if (showBargraph && (x > 0.98 * screenWidth))
							w = (double)y / screenHeight; // thin bar on the right showing the mapping
						if (rainbowPSC)
							getPseudocolorRainbow(w, 0.0, 1.0, vv.x, vv.y, vv.z); // is more common but wrong perceptually
						else
							getPseudocolorCoolWarm(w, 0.0, 1.0, vv.x, vv.y, vv.z); // is perceptually better
					}
				}
				float v; int e;
				v = vv.x; if (vv.y > v) v = vv.y; if (vv.z > v) v = vv.z;
				if (v < 1e-32) {
					rgbe[0] = rgbe[1] = rgbe[2] = rgbe[3] = 0x0;
				} else {
					v = (float)(frexp(v, &e) * 256.0 / v);
					rgbe[0] = (unsigned char)(vv.x * v);
					rgbe[1] = (unsigned char)(vv.y * v);
					rgbe[2] = (unsigned char)(vv.z * v);
					rgbe[3] = (unsigned char)(e + 128);
				}
			}
			fflush(stdout);
			const char* programtype = "RADIANCE";
			if (fprintf(fp, "#?%s\n", programtype) < 0) { abort(); }
			float gamma = 2.2; float exposure = 1.0;
			if (fprintf(fp, "GAMMA=%g\n", gamma) < 0) { abort(); }
			if (fprintf(fp, "EXPOSURE=%g\n", exposure) < 0) { abort(); }
			if (fprintf(fp, "FORMAT=32-bit_rle_rgbe\n\n") < 0) { abort(); }
			if (fprintf(fp, "-Y %d +X %d\n", screenHeight, width) < 0) { abort(); }
			// Write data
			size_t kk = fwrite(data, (size_t)4, nmemb, fp);
			fclose(fp);
			if (kk != nmemb) {
				printf("ERROR - was not able to save all kk=%d entries to file, exiting\n",
					(int)nmemb); fflush(stdout);
				abort(); // error
			}
		}
	}
	} // switch (key)
	Usage();
	glutPostRedisplay();
}

int main(int argc, char** argv)
{
	glutInit(&argc, argv);
	glutInitWindowSize(2 * screenWidth, screenHeight);
	glutInitWindowPosition(100, 100);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);

	glutCreateWindow("DEMO RSO 2018");

	Usage();
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();

	onInitialization();

	glutDisplayFunc(onDisplay);
	glutKeyboardFunc(onKeyboard);
	glutMouseFunc(onMouse);
	glutMainLoop();
	return 0;
}
