/* Fixed time(): both versions of a program then draw the same random inputs. */
#include <time.h>
time_t time(time_t *t)
{
    if (t) *t = 42;
    return 42;
}
