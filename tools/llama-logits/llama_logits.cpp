// llama_logits: teacher-forced logit dump for comparing llama.cpp builds.
//
// Feeds one fixed token sequence through a model and writes the raw float32
// logits of every position, so two builds (for example the stock CPU backend
// and the CUDA backend built through OpenCuda) can be compared position by
// position without greedy decoding amplifying the first small difference.
//
// Build: compile this file against a llama.cpp build tree and link it with the
// same libraries as that tree's llama-simple (see build.sh next to this file).
//
// Usage:
//   llama_logits -m model.gguf -o out.bin [-t threads] [-ngl n] [--prefix P]
//                [--dump-pos N --dump-dir DIR] (-p text | --ids-file f)
//
//   --prefix P   decode the first P tokens as one batch, then the remaining
//                tokens one at a time (the shape of prompt + generation).
//                Default: the whole sequence in one batch.
//   --dump-pos N write every float32 graph tensor of the decode call that
//                contains position N to DIR/<name>.bin (one-token decode
//                calls only, i.e. N >= P), plus DIR/index.txt.
//
// Output file: int32 n_tokens, int32 n_vocab, int32 tokens[n_tokens],
//              float32 logits[n_tokens][n_vocab] (logits[i] predicts token i+1).

#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <clocale>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

struct dump_state {
    bool        active = false;
    std::string dir;
    std::map<std::string, int> seen;
    FILE *      index = nullptr;
};

static bool dump_cb(struct ggml_tensor * t, bool ask, void * user) {
    auto * st = (dump_state *) user;
    if (ask) {
        return st->active;
    }
    if (!st->active || t->type != GGML_TYPE_F32) {
        return true;
    }
    std::string name = t->name;
    for (char & c : name) {
        if (c == '/' || c == ' ' || c == '(' || c == ')') c = '_';
    }
    int k = st->seen[name]++;
    if (k) name += "#" + std::to_string(k);
    std::vector<float> buf(ggml_nelements(t));
    if (ggml_is_contiguous(t)) {
        ggml_backend_tensor_get(t, buf.data(), 0, ggml_nbytes(t));
    } else {
        return true; // views: skip, the producing op is dumped separately
    }
    std::string path = st->dir + "/" + name + ".bin";
    FILE * f = fopen(path.c_str(), "wb");
    if (f) {
        fwrite(buf.data(), sizeof(float), buf.size(), f);
        fclose(f);
    }
    fprintf(st->index, "%s %s %lld %lld %lld %lld\n", name.c_str(), ggml_op_desc(t),
            (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3]);
    return true;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    std::string model_path, out_path, text, ids_file, dump_dir;
    int threads = 8, ngl = 99, prefix = -1, dump_pos = -1;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", a.c_str()); exit(1); }
            return argv[++i];
        };
        if (a == "-m") model_path = next();
        else if (a == "-o") out_path = next();
        else if (a == "-t") threads = std::stoi(next());
        else if (a == "-ngl") ngl = std::stoi(next());
        else if (a == "-p") text = next();
        else if (a == "--ids-file") ids_file = next();
        else if (a == "--prefix") prefix = std::stoi(next());
        else if (a == "--dump-pos") dump_pos = std::stoi(next());
        else if (a == "--dump-dir") dump_dir = next();
        else { fprintf(stderr, "unknown argument %s\n", a.c_str()); return 1; }
    }
    if (model_path.empty() || out_path.empty() || (text.empty() && ids_file.empty())) {
        fprintf(stderr, "usage: %s -m model -o out.bin (-p text | --ids-file f) [-t n] [-ngl n] [--prefix P] [--dump-pos N --dump-dir D]\n", argv[0]);
        return 1;
    }

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = ngl;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) { fprintf(stderr, "cannot load model\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    std::vector<llama_token> toks;
    if (!ids_file.empty()) {
        std::ifstream f(ids_file);
        long v;
        while (f >> v) toks.push_back((llama_token) v);
    } else {
        int n = -llama_tokenize(vocab, text.c_str(), text.size(), nullptr, 0, true, true);
        toks.resize(n);
        llama_tokenize(vocab, text.c_str(), text.size(), toks.data(), n, true, true);
    }
    const int N = (int) toks.size();
    if (prefix < 0 || prefix > N) prefix = N;
    if (prefix < 1) prefix = 1;

    dump_state st;
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = N + 8;
    cp.n_batch = prefix;
    cp.n_ubatch = prefix;
    cp.n_threads = threads;
    cp.n_threads_batch = threads;
    if (dump_pos >= 0) {
        st.dir = dump_dir.empty() ? "dump" : dump_dir;
        st.index = fopen((st.dir + "/index.txt").c_str(), "w");
        if (!st.index) { fprintf(stderr, "cannot write %s/index.txt\n", st.dir.c_str()); return 1; }
        cp.cb_eval = dump_cb;
        cp.cb_eval_user_data = &st;
    }
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "cannot create context\n"); return 1; }

    std::vector<float> logits((size_t) N * n_vocab);
    llama_batch_ext * batch = llama_batch_ext_init(ctx);

    auto run = [&](int p0, int n) -> bool {
        llama_batch_ext_clear(batch);
        for (int i = 0; i < n; i++) {
            int idx = llama_batch_ext_add_token(batch, 0, toks[p0 + i]);
            llama_pos pos = p0 + i;
            llama_batch_ext_set_pos(batch, idx, &pos);
            llama_batch_ext_set_output_logits(batch, idx, true);
        }
        st.active = dump_pos >= p0 && dump_pos < p0 + n && n == 1;
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch)) return false;
        st.active = false;
        for (int i = 0; i < n; i++) {
            const float * l = llama_get_logits_ith(ctx, i);
            memcpy(&logits[(size_t) (p0 + i) * n_vocab], l, sizeof(float) * n_vocab);
        }
        return true;
    };

    const int64_t t0 = ggml_time_us();
    if (!run(0, prefix)) { fprintf(stderr, "decode failed\n"); return 1; }
    for (int p = prefix; p < N; p++) {
        if (!run(p, 1)) { fprintf(stderr, "decode failed at %d\n", p); return 1; }
    }
    const double secs = (ggml_time_us() - t0) / 1e6;

    FILE * f = fopen(out_path.c_str(), "wb");
    if (!f) { fprintf(stderr, "cannot write %s\n", out_path.c_str()); return 1; }
    int32_t hdr[2] = { N, n_vocab };
    fwrite(hdr, sizeof(int32_t), 2, f);
    fwrite(toks.data(), sizeof(int32_t), N, f);
    fwrite(logits.data(), sizeof(float), logits.size(), f);
    fclose(f);
    if (st.index) fclose(st.index);
    fprintf(stderr, "llama_logits: %d tokens (prefix %d), %.1f s\n", N, prefix, secs);

    llama_batch_ext_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
