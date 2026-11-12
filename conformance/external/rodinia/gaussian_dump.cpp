/* Driver for Rodinia gaussian. The program prints its solution only with
   verbose output, which no command-line flag turns on, so this file calls
   the program's own solver functions (the same calls its main() makes with
   -f) and prints the solution. Built with -Dmain=ocu_prog_main for the
   program's sources; the #undef below keeps this file's main. */
#undef main
#include <cstdio>
extern int Size;
extern float *finalVec;
void InitProblemOnce(char *filename);
void InitPerRun();
void ForwardSub();
void BackSub();
int main(int argc, char **argv)
{
    if (argc != 2) return 2;
    InitProblemOnce(argv[1]);
    InitPerRun();
    ForwardSub();
    BackSub();
    for (int i = 0; i < Size; ++i) printf("%.9g\n", finalVec[i]);
    return 0;
}
