/* Driver for Rodinia backprop. The program prints nothing about its results
   (only "Training done"), so this file makes the same calls its facetrain
   setup() makes (seed 7, create, load, one training step) and prints the
   units, deltas and hidden-layer weights after the step.
   Built with -Dmain=ocu_prog_main for the program's sources; the #undef keeps
   this file's main. -DOMP_REF calls the OpenMP version's training function.
   backprop.h is not included: its empty prototypes do not compile as C++;
   the BPNN struct below is copied from it (identical in both versions). */
#undef main
#include <cstdio>
#include <cstdlib>
extern "C" {
typedef struct {
  int input_n;
  int hidden_n;
  int output_n;
  float *input_units;
  float *hidden_units;
  float *output_units;
  float *hidden_delta;
  float *output_delta;
  float *target;
  float **input_weights;
  float **hidden_weights;
  float **input_prev_weights;
  float **hidden_prev_weights;
} BPNN;
void bpnn_initialize(int seed);
BPNN *bpnn_create(int n_in, int n_hidden, int n_out);
void load(BPNN *net);
#ifdef OMP_REF
void bpnn_train_kernel(BPNN *net, float *eo, float *eh);
#define TRAIN bpnn_train_kernel
#else
void bpnn_train_cuda(BPNN *net, float *eo, float *eh);
#define TRAIN bpnn_train_cuda
#endif
}
int main(int argc, char **argv)
{
    if (argc != 2) return 2;
    int layer_size = atoi(argv[1]);
    bpnn_initialize(7);
    BPNN *net = bpnn_create(layer_size, 16, 1);
    load(net);
    float out_err = 0, hid_err = 0;
    TRAIN(net, &out_err, &hid_err);
    for (int j = 1; j <= net->hidden_n; ++j) printf("h %.9g\n", net->hidden_units[j]);
    for (int j = 1; j <= net->output_n; ++j) printf("o %.9g\n", net->output_units[j]);
    for (int j = 1; j <= net->output_n; ++j) printf("od %.9g\n", net->output_delta[j]);
    for (int j = 1; j <= net->hidden_n; ++j) printf("hd %.9g\n", net->hidden_delta[j]);
    for (int j = 0; j <= net->hidden_n; ++j)
        for (int k = 0; k <= net->output_n; ++k) printf("hw %.9g\n", net->hidden_weights[j][k]);
    return 0;
}
