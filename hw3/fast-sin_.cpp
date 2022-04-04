#include <math.h>
#include <stdio.h>
#include "utils.h"
#include "intrin-wrapper.h"

// Headers for intrinsics
#ifdef __SSE__
#include <xmmintrin.h>
#endif
#ifdef __SSE2__
#include <emmintrin.h>
#endif
#ifdef __AVX__
#include <immintrin.h>
#endif


// coefficients in the Taylor series expansion of sin(x)
static constexpr double c3  = -1/(((double)2)*3);
static constexpr double c5  =  1/(((double)2)*3*4*5);
static constexpr double c7  = -1/(((double)2)*3*4*5*6*7);
static constexpr double c9  =  1/(((double)2)*3*4*5*6*7*8*9);
static constexpr double c11 = -1/(((double)2)*3*4*5*6*7*8*9*10*11);
// sin(x) = x + c3*x^3 + c5*x^5 + c7*x^7 + x9*x^9 + c11*x^11

// coefficients in the Taylor series expansion of cos(x)
static constexpr double c2  = -1/(((double)2));
static constexpr double c4  =  1/(((double)2)*3*4);
static constexpr double c6  = -1/(((double)2)*3*4*5*6);
static constexpr double c8  =  1/(((double)2)*3*4*5*6*7*8);
static constexpr double c10 = -1/(((double)2)*3*4*5*6*7*8*9*10);
static constexpr double c12 =  1/(((double)2)*3*4*5*6*7*8*9*10*11*12);
// cos(x) = 1 + c2*x^2 + c4*x^4 + c6^x^6 + c^8*x^8 + c^10*x^10 + c^12*x^12

void sin4_reference(double* sinx, const double* x) {
  for (long i = 0; i < 4; i++) sinx[i] = sin(x[i]);
}
 
void cos_reference(double* cosx, const double* x){
  for(long i = 0; i < 4; i++) cosx[i] = cos(x[i]);
}
void sin4_taylor(double* sinx, const double* x) {
  for (int i = 0; i < 4; i++) {
    double x1  = x[i];
    double x2  = x1 * x1;
    double x3  = x1 * x2;
    double x5  = x3 * x2;
    double x7  = x5 * x2;
    double x9  = x7 * x2;
    double x11 = x9 * x2;

    double s = x1;
    s += x3  * c3;
    s += x5  * c5;
    s += x7  * c7;
    s += x9  * c9;
    s += x11 * c11;
    sinx[i] = s;
  }
}

void cos4_taylor(double* cosx, const double* x){
    for (int i = 0; i < 4; i++) {
    double x1  = x[i];
    double x2  = x1 * x1;
    double x4  = x2 * x2;
    double x6  = x4 * x2;
    double x8  = x6 * x2;
    double x10  = x8 * x2;
    double x12 = x10 * x2;

    double s = 1;
    s += x2  * c2;
    s += x4  * c4;
    s += x6  * c6;
    s += x8  * c8;
    s += x10 * c10;
    s += x12 * c12;
    cosx[i] = s;
  }
}

void sin4_taylor_full(double* sinx, const double* x)
{
  double sc[4];
  double* x_ = (double*) aligned_malloc(4*sizeof(double));
  for(int i = 0; i < 4; i++)
  {

    double x1 = fmod(x[i], 2 * M_PI); // modulo 2pi gives same value 
    if (x1 < 0)
        x1 += 2 * M_PI;
    if (3.0/2 * M_PI <=  x1 && x1 <= 2.0 * M_PI)
      x1 = x1 - (2.0 * M_PI);
    else if (1.0/2 * M_PI <=  x1 && x1 < 3.0/2 * M_PI)
      x1 = 2 * M_PI/2.0 - x1;

    //Now x1 is in the range [-pi/2,pi/2]
    if (- M_PI / 4.0 <= x1 && x1 <= M_PI / 4.0)
    {
      x_[i] = x1; 
      sc[i] = 0;
    }
    else if( x1 <  - M_PI / 4.0)
    {
      x_[i] = (M_PI) / 2.0 + x1; 
      sc[i] = -1;
    }
    else
    {
      x_[i] = (M_PI) / 2.0 - x1; 
      sc[i] = 1;
    }
  
  }

  double* sinx_ = (double*) aligned_malloc(4*sizeof(double));
  double* cosx_ = (double*) aligned_malloc(4*sizeof(double));
  sin4_taylor(sinx_, x_);
  cos4_taylor(cosx_, x_);

  for(int i = 0; i < 4; i++)
    sinx[i] = (1 - abs(sc[i])) * sinx_[i] + (sc[i]) * cosx_[i];

  aligned_free(x_);
  aligned_free(sinx_);
  aligned_free(cosx_);


}




void sin4_vector(double* sinx, const double* x) {
  // The Vec class is defined in the file intrin-wrapper.h
  typedef Vec<double,4> Vec4;
  Vec4 x1, x2, x3, x5, x7, x9, x11;
   
  x1  = Vec4::LoadAligned(x);
  x2  = x1 * x1;
  x3  = x1 * x2;
  x5  = x3 * x2;
  x7  = x5 * x2;
  x9  = x7 * x2;
  x11 = x9 * x2;
  
  Vec4 s = x1; //Stores the final sinx
  s += x3 * c3;
  s += x5 * c5;
  s += x7 * c7;
  s += x9 * c9;
  s += x11 * c11;
  s.StoreAligned(sinx);

  
}

void cos4_vector(double* cosx, const double* x) {
  // The Vec class is defined in the file intrin-wrapper.h
  typedef Vec<double,4> Vec4;
  Vec4 x1, x2, x4, x6, x8, x10, x12;

  x1  = Vec4::LoadAligned(x);
  x2  = x1 * x1;
  x4  = x2 * x2;
  x6  = x4 * x2;
  x8  = x6 * x2;
  x10 = x8 * x2;
  x12 = x10 * x2;
  
  double p[4] = {1, 1, 1, 1};
  Vec4 s = Vec4::LoadAligned(p); //Stores the final cosx
  s += x2  * c2;
  s += x4  * c4;
  s += x6  * c6;
  s += x8  * c8;
  s += x10 * c10;
  s += x12 * c12;
  s.StoreAligned(cosx);
}


void sin4_vector_full(double* sinx, const double* x)
{
  double sc[4];
  double* x_ = (double*) aligned_malloc(4*sizeof(double));
  for(int i = 0; i < 4; i++)
  {

    double x1 = fmod(x[i], 2 * M_PI); // modulo 2pi gives same value 
    if (x1 < 0)
        x1 += 2 * M_PI;
    if (3.0/2 * M_PI <=  x1 && x1 <= 2.0 * M_PI)
      x1 = x1 - (2.0 * M_PI);
    else if (1.0/2 * M_PI <=  x1 && x1 < 3.0/2 * M_PI)
      x1 = 2 * M_PI/2.0 - x1;

    //Now x1 is in the range [-pi/2,pi/2]
    if (- M_PI / 4.0 <= x1 && x1 <= M_PI / 4.0)
    {
      x_[i] = x1; 
      sc[i] = 0;
    }
    else if( x1 <  - M_PI / 4.0)
    {
      x_[i] = (M_PI) / 2.0 + x1; 
      sc[i] = -1;
    }
    else
    {
      x_[i] = (M_PI) / 2.0 - x1; 
      sc[i] = 1;
    }
  
  }

  double* sinx_ = (double*) aligned_malloc(4*sizeof(double));
  double* cosx_ = (double*) aligned_malloc(4*sizeof(double));
  sin4_vector(sinx_, x_);
  cos4_vector(cosx_, x_);

  for(int i = 0; i < 4; i++)
    sinx[i] = (1 - abs(sc[i])) * sinx_[i] + (sc[i]) * cosx_[i];

  aligned_free(x_);
  aligned_free(sinx_);
  aligned_free(cosx_);
}


void sin4_intrin(double* sinx, const double* x) {
  // The definition of intrinsic functions can be found at:
  // https://software.intel.com/sites/landingpage/IntrinsicsGuide/#
#if defined(__SSE2__)
  constexpr int sse_length = 2;
  for (int i = 0; i < 4; i+=sse_length) {
    __m128d x1, x2, x3, x5, x7, x9, x11;
    x1  = _mm_load_pd(x+i);
    x2  = _mm_mul_pd(x1, x1);
    x3  = _mm_mul_pd(x1, x2);
    x5 =  _mm_mul_pd(x3, x2);
    x7 =  _mm_mul_pd(x5, x2);
    x9 =  _mm_mul_pd(x7, x2);
    x11 =  _mm_mul_pd(x9, x2);

    __m128d s = x1;
    s = _mm_add_pd(s, _mm_mul_pd(x3 , _mm_set1_pd(c3)));
    s = _mm_add_pd(s, _mm_mul_pd(x5 , _mm_set1_pd(c5)));
    s = _mm_add_pd(s, _mm_mul_pd(x7 , _mm_set1_pd(c7)));
    s = _mm_add_pd(s, _mm_mul_pd(x9 , _mm_set1_pd(c9)));
    s = _mm_add_pd(s, _mm_mul_pd(x11 , _mm_set1_pd(c11 )));
    _mm_store_pd(sinx+i, s);
  }
#else
  sin4_reference(sinx, x);
#endif
}

void cos4_intrin(double* cosx, const double* x) {
  // The definition of intrinsic functions can be found at:
  // https://software.intel.com/sites/landingpage/IntrinsicsGuide/#
#if defined(__SSE2__)
  constexpr int sse_length = 2;
  for (int i = 0; i < 4; i+=sse_length) {
    __m128d x1, x2, x4, x6, x8, x10, x12;
    x1  = _mm_load_pd(x+i);
    x2  = _mm_mul_pd(x1, x1);
    x4  = _mm_mul_pd(x2, x2);
    x6 =  _mm_mul_pd(x4, x2);
    x8 =  _mm_mul_pd(x6, x2);
    x10 =  _mm_mul_pd(x8, x2);
    x12 =  _mm_mul_pd(x10, x2);

    double p[2] = {1,1};
    __m128d s = _mm_load_pd(p);
    s = _mm_add_pd(s, _mm_mul_pd(x2 , _mm_set1_pd(c2)));
    s = _mm_add_pd(s, _mm_mul_pd(x4 , _mm_set1_pd(c4)));
    s = _mm_add_pd(s, _mm_mul_pd(x6 , _mm_set1_pd(c6)));
    s = _mm_add_pd(s, _mm_mul_pd(x8 , _mm_set1_pd(c8)));
    s = _mm_add_pd(s, _mm_mul_pd(x10 , _mm_set1_pd(c10)));
    s = _mm_add_pd(s, _mm_mul_pd(x12 , _mm_set1_pd(c12)));
    _mm_store_pd(cosx+i, s);
  }
#else
  sin4_reference(cosx, x);
#endif
}

void sin4_intrin_full(double* sinx, const double* x)
{
    double sc[4];
  double* x_ = (double*) aligned_malloc(4*sizeof(double));
  for(int i = 0; i < 4; i++)
  {
    double x1 = fmod(x[i], 2 * M_PI); // modulo 2pi gives same value 
    if (x1 < 0)
        x1 += 2 * M_PI;
    if (3.0/2 * M_PI <=  x1 && x1 <= 2.0 * M_PI)
      x1 = x1 - (2.0 * M_PI);
    else if (1.0/2 * M_PI <=  x1 && x1 < 3.0/2 * M_PI)
      x1 = 2 * M_PI/2.0 - x1;

    //Now x1 is in the range [-pi/2,pi/2]
    if (- M_PI / 4.0 <= x1 && x1 <= M_PI / 4.0)
    {
      x_[i] = x1; 
      sc[i] = 0;
    }
    else if( x1 <  - M_PI / 4.0)
    {
      x_[i] = (M_PI) / 2.0 + x1; 
      sc[i] = -1;
    }
    else
    {
      x_[i] = (M_PI) / 2.0 - x1; 
      sc[i] = 1;
    }
  
  }

  double* sinx_ = (double*) aligned_malloc(4*sizeof(double));
  double* cosx_ = (double*) aligned_malloc(4*sizeof(double));
  sin4_intrin(sinx_, x_);
  cos4_intrin(cosx_, x_);

  for(int i = 0; i < 4; i++)
    sinx[i] = (1 - abs(sc[i])) * sinx_[i] + (sc[i]) * cosx_[i];

  aligned_free(x_);
  aligned_free(sinx_);
  aligned_free(cosx_);
}



double err(double* x, double* y, long N) {
  double error = 0;
  for (long i = 0; i < N; i++) error = std::max(error, fabs(x[i]-y[i]));
  return error;
}

int main() {
  Timer tt;
  long N = 1000000;
  double* x = (double*) aligned_malloc(N*sizeof(double));
  double* sinx_ref = (double*) aligned_malloc(N*sizeof(double));
  double* sinx_taylor = (double*) aligned_malloc(N*sizeof(double));
  double* sinx_intrin = (double*) aligned_malloc(N*sizeof(double));
  double* sinx_vector = (double*) aligned_malloc(N*sizeof(double));

  double* cosx_ref = (double*) aligned_malloc(N*sizeof(double));
  double* cosx_taylor = (double*) aligned_malloc(N*sizeof(double));

  double* sinx_ref_ = (double*) aligned_malloc(N*sizeof(double));
  double* sinx_taylor_ = (double*) aligned_malloc(N*sizeof(double));
  double* sinx_vector_ = (double*) aligned_malloc(N*sizeof(double));
  double* sinx_intrin_ = (double*) aligned_malloc(N*sizeof(double));

  for (long i = 0; i < N; i++) {
    //x[i] = (drand48()-0.5) * M_PI/2; // [-pi/4,pi/4]
    //x[i] = (drand48()) * M_PI/2; // [0,pi/2]
    //x[i] = (drand48()-0.5) * M_PI; // [-pi/2, -pi/2]
    x[i] = (drand48()-0.5) * 8 * M_PI; // [-4pi, 4pi]
    sinx_ref[i] = 0;
    sinx_taylor[i] = 0;
    sinx_intrin[i] = 0;
    sinx_vector[i] = 0;

    cosx_ref[i] = 0;
    cosx_taylor[i] = 0; 

    sinx_ref_[i] = 0;
    sinx_taylor_[i] = 0;
  }


  printf("Sine\n");

  tt.tic();
  for (long rep = 0; rep < 1000; rep++) {
    for (long i = 0; i < N; i+=4) {
      sin4_reference(sinx_ref_+i, x+i);
    }
  }
  printf("Reference time: %6.4f\n", tt.toc());


  tt.tic();
  for (long rep = 0; rep < 1000; rep++) {
    for (long i = 0; i < N; i+=4) {
      sin4_taylor_full(sinx_taylor_+i, x+i);
    }
  }
  printf("Taylor time:    %6.4f      Error: %e\n", tt.toc(), err(sinx_ref_, sinx_taylor_, N));


  // tt.tic();
  // for (long rep = 0; rep < 1000; rep++) {
  //   for (long i = 0; i < N; i+=4) {
  //     sin4_vector_full(sinx_vector+i, x+i);
  //   }
  // }
  // printf("Vector time:    %6.4f      Error: %e\n", tt.toc(), err(sinx_ref_, sinx_vector_, N));


  tt.tic();
  for (long rep = 0; rep < 1000; rep++) {
    for (long i = 0; i < N; i+=4) {
      sin4_intrin_full(sinx_intrin_+i, x+i);
    }
  }
  printf("Intrin time:    %6.4f      Error: %e\n", tt.toc(), err(sinx_ref_, sinx_intrin_, N));

  aligned_free(x);
  aligned_free(sinx_ref);
  aligned_free(sinx_taylor);
  aligned_free(sinx_intrin);
  aligned_free(sinx_vector);
  aligned_free(cosx_ref);
  aligned_free(cosx_taylor);
  aligned_free(sinx_ref_);
  aligned_free(sinx_taylor_);
  aligned_free(sinx_vector_);
  aligned_free(sinx_intrin_);

}

