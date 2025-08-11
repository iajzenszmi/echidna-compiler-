#include <stdio.h>
#include <math.h>
#include <stdbool.h>
#include <string.h>

static inline void __p_int(long long x){ printf("%lld", x); }
static inline void __p_double(double x){ printf("%.15g", x); }
static inline void __p_bool(bool x){ printf("%s", x ? "true" : "false"); }
static inline void __p_string(const char* s){ printf("%s", s); }

int main(void);

int main(void) {
  __p_string("Hello, world!");
  printf("\n");
  return 0;
}
