// ocu_cpu_loops.cpp -- LLVM pass plugin for the CPU path: kernels without
// synchronisation points get a "whole block" entry that runs every thread of
// a block as a plain loop over the thread index (docs/cpu-region-loops.md,
// stage 1).
//
// Input: a CPU-path translation unit (compiler/probe/ocu_cc.py --target cpu)
// compiled by Clang with -fpass-plugin. The language layer
// (runtime/include/ocu_cuda_lang_cpu.h) emits, per launch site, a module
// image {magic, version 1, count, entries} whose entries are {name, fn}; fn
// runs one GPU thread and reads its thread index from the per-thread
// variable ocu_cpu_tls_thread_idx (backends/cpu/ocu_cpu_kernel.h).
//
// For every image this pass:
//   1. checks each entry's call graph (after the front end, before any
//      optimisation): no block barrier or warp operation, no indirect call,
//      no inline assembly, no call that changes the floating-point control
//      state, and no external call other than the C and C++ runtime
//      libraries and the CPU backend's read-only device calls;
//   2. for an entry that passes, adds `<fn>.ocu_block(args, info)`: loops
//      z, y, x over the block (x innermost), and for each thread stores
//      threadIdx into ocu_cpu_tls_thread_idx (and the linear index into
//      info->current when the kernel calls ocu_cpu_thread_idx or
//      ocu_cpu_lane_id), then runs the thread's body. The body (the entry
//      and the three calls below it: thunk, launch lambda, kernel) is
//      inlined into the loop. With a thread order from the backend
//      (OCU_CPU_ORDER=random) a separate function visits the threads in
//      that order instead. An entry whose only obstacle is one call
//      through a pointer (the kernel passed as a value: cudaLaunchKernel,
//      launch helpers) gets a guarded block entry: it computes the pointer
//      first and declines (returns 1, nothing run) unless the pointer is one
//      of this module's kernels of that type that pass step 1 themselves;
//      the backend then uses fibers;
//   3. rewrites the image to version 2, whose entries are
//      {name, fn, block}; block is null for entries that failed step 1.
//      The fiber entry fn stays in every image.
//
// A second step, just before the loop vectoriser, deletes the per-thread
// stores of step 2 from the x loop when nothing left in the block entry can
// read them (all reads of threadIdx were inlined and forwarded, and no call
// that could read them remains): then the loop is the kernel body over x
// and can be vectorised.
//
// Why a loop is the same program: a kernel without synchronisation points
// runs each thread from start to end without a switch on the fiber path
// too (fixed order: 0, 1, 2, ...; random order: one shuffled pass), so the
// loop runs the same instructions in the same order. See the design note.
//
// OCU_CPU_LOOPS_REPORT=<file>: append one line per entry,
//   loop|fiber <TAB> kernel <TAB> reason <TAB> source file
//
// Written for this project; only LLVM's public C++ API is used.

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/InlineCost.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Demangle/Demangle.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalAlias.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/ADT/DenseMap.h"

#include <cstdlib>
#include <string>

using namespace llvm;

namespace {

constexpr uint64_t kImageMagic = 0x4f435543u; // "OCUC"
constexpr uint64_t kImageV1 = 1;
constexpr uint64_t kImageV2 = 2;
constexpr const char *kBlockAttr = "ocu-cpu-loop-block";
constexpr const char *kTlsTag = "ocu.loop.tid";
constexpr const char *kCurTag = "ocu.loop.current";
// Levels of calls inlined into the loop: entry, thunk, launch lambda, kernel.
constexpr int kInlineDepth = 4;
// Most kernels a guard on a kernel pointer compares against.
constexpr size_t kMaxTargets = 256;

// ---- what a kernel may call --------------------------------------------

bool isCut(StringRef n) { return n == "ocu_cpu_barrier" || n.starts_with("ocu_cpu_warp_"); }

// Calls that look up the running thread through the backend's state.
bool readsCurrent(StringRef n) { return n == "ocu_cpu_thread_idx" || n == "ocu_cpu_lane_id"; }

bool isReadOnlyBackendCall(StringRef n)
{
    return n == "ocu_cpu_dynamic_shared" || n == "ocu_cpu_block_idx" || n == "ocu_cpu_block_dim" ||
           n == "ocu_cpu_grid_dim" || n == "ocu_cpu_warp_size";
}

bool setsFpControl(StringRef n)
{
    return n == "fesetround" || n == "fesetenv" || n == "feupdateenv" || n == "feholdexcept" ||
           n == "fesetmode" || n == "fesetexcept" || n == "fesetexceptflag" || n == "feclearexcept" ||
           n == "feraiseexcept" || n == "fedisableexcept" || n == "feenableexcept" ||
           n.starts_with("_controlfp") || n == "_control87" || n == "__control87_2" ||
           n == "_fpreset";
}

bool setsFpControl(Intrinsic::ID id)
{
    switch (id) {
    case Intrinsic::set_rounding:
    case Intrinsic::set_fpenv:
    case Intrinsic::reset_fpenv:
    case Intrinsic::set_fpmode:
    case Intrinsic::reset_fpmode:
        return true;
    default:
        return false;
    }
}

// C math functions: base names with the double, float and long double
// spellings.
bool isMathName(StringRef n)
{
    static const char *const base[] = {
        "acos", "acosh", "asin", "asinh", "atan", "atan2", "atanh", "cbrt", "ceil", "copysign",
        "cos", "cosh", "erf", "erfc", "exp", "exp10", "exp2", "expm1", "fabs", "fdim", "floor",
        "fma", "fmax", "fmin", "fmod", "frexp", "hypot", "ilogb", "ldexp", "lgamma", "llrint",
        "llround", "log", "log10", "log1p", "log2", "logb", "lrint", "lround", "modf", "nan",
        "nearbyint", "nextafter", "nexttoward", "pow", "remainder", "remquo", "rint", "round",
        "scalbln", "scalbn", "sin", "sincos", "sinh", "sqrt", "tan", "tanh", "tgamma", "trunc",
        "j0", "j1", "jn", "y0", "y1", "yn", "significand", "gamma", "lgamma_r", "pow10"};
    StringRef s = n;
    s.consume_front("__");
    for (const char *b : base) {
        StringRef r = s;
        if (!r.consume_front(b)) continue;
        r.consume_back("_finite");
        if (r.empty() || r == "f" || r == "l") return true;
    }
    return s.starts_with("isnan") || s.starts_with("isinf") || s.starts_with("finite") ||
           s.starts_with("fpclassify") || s.starts_with("signbit") || s.starts_with("issignaling");
}

// C and C++ runtime functions: they neither synchronise threads nor look up
// the running GPU thread nor change the floating-point control state.
bool isRuntimeLibraryName(StringRef n)
{
    static const char *const exact[] = {
        "memcpy", "memmove", "memset", "memcmp", "memchr", "strlen", "strnlen", "strcmp",
        "strncmp", "strcpy", "strncpy", "strcat", "strchr", "strrchr", "strstr", "printf",
        "fprintf", "sprintf", "snprintf", "vprintf", "vfprintf", "vsprintf", "vsnprintf", "puts",
        "fputs", "putchar", "fputc", "putc", "fwrite", "fflush", "abort", "exit", "_exit",
        "malloc", "calloc", "realloc", "free", "aligned_alloc", "posix_memalign", "abs", "labs",
        "llabs", "div", "ldiv", "lldiv", "atoi", "atol", "atof", "strtol", "strtoul", "strtoll",
        "strtoull", "strtof", "strtod", "__errno_location", "__assert_fail", "__assert_rtn",
        "_assert", "__stack_chk_fail", "_Unwind_Resume", "__gxx_personality_v0", "clock",
        "time", "rand", "srand", "getenv"};
    for (const char *e : exact)
        if (n == e) return true;
    static const char *const prefix[] = {
        "__cxa_", "_ZSt", "_ZNSt", "_ZNKSt", "_ZN9__gnu_cxx", "_ZNK9__gnu_cxx", "_Znw", "_Zna",
        "_Zdl", "_Zda", "__printf_chk", "__fprintf_chk", "__sprintf_chk", "__snprintf_chk",
        "__vfprintf_chk", "__vprintf_chk", "__vsnprintf_chk", "__memcpy_chk", "__memmove_chk",
        "__memset_chk", "__strcpy_chk", "__isoc99_", "__isoc23_"};
    for (const char *p : prefix)
        if (n.starts_with(p)) return true;
    return isMathName(n);
}

// ---- call graph check -------------------------------------------------------

struct Verdict {
    std::string cut;    // first synchronisation point found
    std::string other;  // first other reason to refuse
    bool needsCurrent = false;
    SmallVector<CallBase *, 2> indirect; // calls through a pointer
    std::string indirectIn;              // the function of the first one
    bool ok() const { return cut.empty() && other.empty() && indirect.empty(); }
    std::string reason() const
    {
        if (!cut.empty()) return "cut: " + cut;
        if (!other.empty()) return other;
        return indirect.empty() ? std::string() : "indirect call in " + indirectIn;
    }
};

std::string shortName(StringRef mangled)
{
    std::string d = demangle(mangled.str());
    if (d.size() > 160) d = d.substr(0, 157) + "...";
    return d;
}

Function *directCallee(CallBase &cb)
{
    Value *v = cb.getCalledOperand()->stripPointerCasts();
    if (auto *ga = dyn_cast<GlobalAlias>(v)) {
        if (ga->isInterposable()) return nullptr;
        v = ga->getAliasee()->stripPointerCasts();
    }
    return dyn_cast<Function>(v);
}

Verdict check(Function &root)
{
    Verdict v;
    SmallPtrSet<Function *, 32> seen;
    SmallVector<Function *, 64> work{&root};
    auto refuse = [&](std::string r) {
        if (v.other.empty()) v.other = std::move(r);
    };
    while (!work.empty()) {
        Function *f = work.pop_back_val();
        if (!seen.insert(f).second) continue;
        StringRef n = f->getName();
        if (f->isDeclaration()) {
            if (isCut(n)) {
                if (v.cut.empty()) v.cut = n.str();
            } else if (readsCurrent(n)) {
                v.needsCurrent = true;
            } else if (isReadOnlyBackendCall(n)) {
            } else if (n.starts_with("ocu_cpu_")) {
                refuse("backend call " + n.str());
            } else if (setsFpControl(n)) {
                refuse("changes floating-point control state: " + n.str());
            } else if (!isRuntimeLibraryName(n)) {
                refuse("external call " + shortName(n));
            }
            continue;
        }
        // A definition another object may replace with a different body.
        if (f->isInterposable()) {
            refuse("replaceable definition " + shortName(n));
            continue;
        }
        for (Instruction &i : instructions(*f)) {
            auto *cb = dyn_cast<CallBase>(&i);
            if (cb == nullptr) continue;
            if (cb->isInlineAsm()) {
                refuse("inline assembly in " + shortName(n));
                continue;
            }
            Function *callee = directCallee(*cb);
            if (callee == nullptr) {
                if (v.indirect.empty()) v.indirectIn = shortName(n);
                v.indirect.push_back(cb);
                continue;
            }
            if (callee->isIntrinsic()) {
                if (setsFpControl(callee->getIntrinsicID()))
                    refuse("changes floating-point control state: " + callee->getName().str());
                continue;
            }
            work.push_back(callee);
        }
    }
    return v;
}

// The kernel an entry runs: the first function below the entry that is not
// part of the launch plumbing (thunk, launch lambda), for the report.
std::string kernelName(Function &entry)
{
    SmallVector<std::pair<Function *, int>, 16> work{{&entry, 0}};
    SmallPtrSet<Function *, 16> seen;
    for (size_t k = 0; k < work.size(); ++k) {
        auto [f, depth] = work[k];
        if (!seen.insert(f).second || f->isDeclaration()) continue;
        std::string d = demangle(f->getName().str());
        bool plumbing = d.find("ocu_cpu::") != std::string::npos ||
                        d.find("lambda") != std::string::npos ||
                        d.find("$_") != std::string::npos || d.rfind("std::", 0) == 0;
        if (!plumbing) return shortName(f->getName());
        if (depth >= kInlineDepth) continue;
        for (Instruction &i : instructions(*f))
            if (auto *cb = dyn_cast<CallBase>(&i))
                if (Function *c = directCallee(*cb); c != nullptr && !c->isIntrinsic())
                    work.push_back({c, depth + 1});
    }
    return shortName(entry.getName());
}

// ---- the whole-block entry -------------------------------------------------

GlobalVariable *findTid(Module &m)
{
    for (const char *n : {"ocu_cpu_tls_thread_idx", "\1ocu_cpu_tls_thread_idx"})
        if (GlobalVariable *g = m.getGlobalVariable(n, true)) return g;
    return nullptr;
}

void inlineChain(CallBase *cb, int depth)
{
    Function *callee = cb->getCalledFunction();
    if (callee == nullptr || callee->isDeclaration() || callee->isIntrinsic() ||
        !isInlineViable(*callee).isSuccess())
        return;
    InlineFunctionInfo ifi;
    if (!InlineFunction(*cb, ifi).isSuccess() || depth <= 1) return;
    SmallVector<CallBase *, 8> next(ifi.InlinedCallSites.begin(), ifi.InlinedCallSites.end());
    for (CallBase *c : next) inlineChain(c, depth - 1);
}

struct Builder {
    Module &m;
    LLVMContext &ctx;
    Type *i32, *ptr;
    StructType *info; // ocu_cpu_block_info
    GlobalVariable *tid;

    explicit Builder(Module &mod)
        : m(mod), ctx(mod.getContext()), i32(Type::getInt32Ty(ctx)),
          ptr(PointerType::getUnqual(ctx)), tid(findTid(mod))
    {
        // {order, tids, current, dim.x, dim.y, dim.z, nthreads}
        info = StructType::get(ctx, {ptr, ptr, ptr, i32, i32, i32, i32});
    }

    Function *makeFunction(Function &entry, StringRef suffix, Type *ret = nullptr,
                           ArrayRef<Type *> params = {})
    {
        SmallVector<Type *, 2> ps(params.begin(), params.end());
        if (ps.empty()) ps = {ptr, ptr};
        auto *ty = FunctionType::get(ret ? ret : Type::getVoidTy(ctx), ps, false);
        Function *f = Function::Create(ty, GlobalValue::InternalLinkage, entry.getName() + suffix, m);
        // Same target and floating-point attributes as the body.
        AttributeList al = entry.getAttributes();
        f->setAttributes(AttributeList::get(ctx, al.getFnAttrs(), AttributeSet(), {}));
        f->removeFnAttr(Attribute::NoInline);
        f->removeFnAttr(Attribute::OptimizeNone);
        f->removeFnAttr(Attribute::AlwaysInline);
        f->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);
        return f;
    }

    // Per thread: threadIdx into the per-thread variable, the linear index
    // into *current when asked, then the body.
    CallInst *emitThread(IRBuilder<> &b, Function &entry, Value *args, Value *cur, Value *x,
                         Value *y, Value *z, Value *linear, bool needsCurrent)
    {
        MDNode *none = MDNode::get(ctx, {});
        if (tid != nullptr) {
            Value *base = b.CreateThreadLocalAddress(tid);
            Value *xyz[3] = {x, y, z};
            for (int k = 0; k < 3; ++k) {
                Value *p = b.CreateConstInBoundsGEP1_32(b.getInt8Ty(), base, 4 * k);
                b.CreateAlignedStore(xyz[k], p, Align(4))->setMetadata(kTlsTag, none);
            }
        }
        if (needsCurrent) b.CreateAlignedStore(linear, cur, Align(4))->setMetadata(kCurTag, none);
        return b.CreateCall(entry.getFunctionType(), &entry, {args});
    }

    // Threads in the order the backend gives (random order).
    Function *makeOrdered(Function &entry, bool needsCurrent)
    {
        Function *f = makeFunction(entry, ".ocu_ordered");
        f->addFnAttr(Attribute::NoInline);
        Argument *args = f->getArg(0), *in = f->getArg(1);
        BasicBlock *bb0 = BasicBlock::Create(ctx, "entry", f);
        BasicBlock *body = BasicBlock::Create(ctx, "thread", f);
        BasicBlock *done = BasicBlock::Create(ctx, "done", f);
        IRBuilder<> b(bb0);
        Value *order = b.CreateLoad(ptr, b.CreateStructGEP(info, in, 0), "order");
        Value *tids = b.CreateLoad(ptr, b.CreateStructGEP(info, in, 1), "tids");
        Value *cur = b.CreateLoad(ptr, b.CreateStructGEP(info, in, 2), "current");
        Value *n = b.CreateLoad(i32, b.CreateStructGEP(info, in, 6), "n");
        b.CreateBr(body);
        b.SetInsertPoint(body);
        PHINode *k = b.CreatePHI(i32, 2, "k");
        k->addIncoming(b.getInt32(0), bb0);
        Value *i = b.CreateLoad(i32, b.CreateInBoundsGEP(i32, order, k), "i");
        Value *t = b.CreateInBoundsGEP(i32, tids, b.CreateMul(b.CreateZExt(i, b.getInt64Ty()),
                                                             b.getInt64(3)));
        Value *x = b.CreateLoad(i32, t);
        Value *y = b.CreateLoad(i32, b.CreateConstInBoundsGEP1_32(i32, t, 1));
        Value *z = b.CreateLoad(i32, b.CreateConstInBoundsGEP1_32(i32, t, 2));
        CallInst *c = emitThread(b, entry, args, cur, x, y, z, i, needsCurrent);
        c->addFnAttr(Attribute::NoInline);
        Value *k1 = b.CreateAdd(k, b.getInt32(1), "k.next", true, true);
        k->addIncoming(k1, body);
        b.CreateCondBr(b.CreateICmpULT(k1, n), body, done);
        b.SetInsertPoint(done);
        b.CreateRetVoid();
        return f;
    }

    // ---- entries that call the kernel through a pointer -------------------
    //
    // A launch through a function pointer (cudaLaunchKernel, a launch helper
    // that takes the kernel as a parameter) reaches the kernel by an
    // indirect call. The whole-block entry then first computes that pointer
    // (`<entry>.ocu_target`) and runs the loop only when it is one of the
    // kernels of this module with the same type that passed the check;
    // otherwise it declines (returns 1) before running any thread and the
    // backend uses fibers.

    // The pointer the entry calls through: the entry's plumbing (entry,
    // thunk, launch lambda) inlined, the indirect call replaced by returning
    // its target. Null when that plumbing does anything before the call but
    // read memory and write its own locals.
    Function *makeTarget(Function &entry, CallBase &through)
    {
        Function *f = makeFunction(entry, ".ocu_target", ptr, {ptr});
        BasicBlock *bb = BasicBlock::Create(ctx, "entry", f);
        IRBuilder<> b(bb);
        CallInst *c = b.CreateCall(entry.getFunctionType(), &entry, {f->getArg(0)});
        b.CreateRet(ConstantPointerNull::get(cast<PointerType>(ptr)));
        inlineChain(c, kInlineDepth - 1);
        SmallVector<CallBase *, 2> ind;
        for (Instruction &i : instructions(*f))
            if (auto *cb = dyn_cast<CallBase>(&i); cb && !cb->isInlineAsm() && !directCallee(*cb))
                ind.push_back(cb);
        auto fail = [&] {
            f->eraseFromParent();
            return nullptr;
        };
        if (ind.size() != 1 || ind[0]->getFunctionType() != through.getFunctionType()) return fail();
        CallBase *ic = ind[0];
        Value *fp = ic->getCalledOperand();
        BasicBlock *at = ic->getParent();
        at->splitBasicBlock(ic);
        at->getTerminator()->eraseFromParent();
        ReturnInst::Create(ctx, fp, at);
        removeUnreachableBlocks(*f);
        for (Instruction &i : instructions(*f)) {
            if (auto *cb = dyn_cast<CallBase>(&i)) {
                Function *callee = directCallee(*cb);
                if (callee == nullptr || !callee->isIntrinsic()) return fail();
                if (auto *mi = dyn_cast<AnyMemIntrinsic>(cb)) {
                    if (!isa<AllocaInst>(getUnderlyingObject(mi->getRawDest()))) return fail();
                } else if (cb->mayWriteToMemory() && !isa<LifetimeIntrinsic>(cb)) {
                    return fail();
                }
            } else if (auto *s = dyn_cast<StoreInst>(&i)) {
                if (!isa<AllocaInst>(getUnderlyingObject(s->getPointerOperand()))) return fail();
            } else if (i.mayWriteToMemory()) {
                return fail();
            }
        }
        return f;
    }

    DenseMap<Function *, std::optional<Verdict>> checked;

    // Kernels of this module an indirect call of this type may reach and
    // that pass the check themselves.
    SmallVector<Function *, 16> targets(FunctionType *ty, bool &needsCurrent)
    {
        SmallVector<Function *, 16> out;
        for (Function &f : m) {
            if (f.isDeclaration() || f.getFunctionType() != ty || f.hasFnAttribute(kBlockAttr) ||
                f.getName().contains(".ocu_") || !f.hasAddressTaken())
                continue;
            std::optional<Verdict> &v = checked[&f];
            if (!v) v = check(f);
            if (!v->ok()) continue;
            needsCurrent |= v->needsCurrent;
            out.push_back(&f);
        }
        return out;
    }

    // for z, for y, for x (all at least 1): the fixed order 0, 1, 2, ...
    // Returns 0; 1 when a guard (see above) declines.
    Function *makeBlock(Function &entry, bool needsCurrent, Function *target = nullptr,
                        ArrayRef<Function *> allowed = {})
    {
        Function *ordered = makeOrdered(entry, needsCurrent);
        Function *f = makeFunction(entry, ".ocu_block", i32);
        f->addFnAttr(kBlockAttr);
        Argument *args = f->getArg(0), *in = f->getArg(1);
        BasicBlock *bb0 = BasicBlock::Create(ctx, "entry", f);
        if (target != nullptr) {
            BasicBlock *guard = bb0;
            bb0 = BasicBlock::Create(ctx, "start", f);
            BasicBlock *no = BasicBlock::Create(ctx, "decline", f);
            IRBuilder<> b(guard);
            Value *fp = b.CreateCall(target->getFunctionType(), target, {args}, "target");
            Value *known = b.getFalse();
            for (Function *k : allowed) known = b.CreateOr(known, b.CreateICmpEQ(fp, k));
            b.CreateCondBr(known, bb0, no);
            b.SetInsertPoint(no);
            b.CreateRet(b.getInt32(1));
        }
        BasicBlock *shuffled = BasicBlock::Create(ctx, "ordered", f);
        BasicBlock *zh = BasicBlock::Create(ctx, "z", f);
        BasicBlock *yh = BasicBlock::Create(ctx, "y", f);
        BasicBlock *xb = BasicBlock::Create(ctx, "x", f);
        BasicBlock *yl = BasicBlock::Create(ctx, "y.next", f);
        BasicBlock *zl = BasicBlock::Create(ctx, "z.next", f);
        BasicBlock *done = BasicBlock::Create(ctx, "done", f);
        IRBuilder<> b(bb0);
        Value *order = b.CreateLoad(ptr, b.CreateStructGEP(info, in, 0), "order");
        Value *cur = b.CreateLoad(ptr, b.CreateStructGEP(info, in, 2), "current");
        Value *dx = b.CreateLoad(i32, b.CreateStructGEP(info, in, 3), "dx");
        Value *dy = b.CreateLoad(i32, b.CreateStructGEP(info, in, 4), "dy");
        Value *dz = b.CreateLoad(i32, b.CreateStructGEP(info, in, 5), "dz");
        b.CreateCondBr(b.CreateIsNull(order), zh, shuffled);

        b.SetInsertPoint(shuffled);
        b.CreateCall(ordered->getFunctionType(), ordered, {args, in});
        b.CreateRet(b.getInt32(0));

        b.SetInsertPoint(zh);
        PHINode *z = b.CreatePHI(i32, 2, "tz");
        z->addIncoming(b.getInt32(0), bb0);
        b.CreateBr(yh);

        b.SetInsertPoint(yh);
        PHINode *y = b.CreatePHI(i32, 2, "ty");
        y->addIncoming(b.getInt32(0), zh);
        Value *row = b.CreateMul(b.CreateAdd(b.CreateMul(z, dy, "", true, true), y, "", true, true),
                                 dx, "row", true, true);
        b.CreateBr(xb);

        b.SetInsertPoint(xb);
        PHINode *x = b.CreatePHI(i32, 2, "tx");
        x->addIncoming(b.getInt32(0), yh);
        Value *linear = b.CreateAdd(row, x, "linear", true, true);
        CallInst *c = emitThread(b, entry, args, cur, x, y, z, linear, needsCurrent);
        Value *x1 = b.CreateAdd(x, b.getInt32(1), "tx.next", true, true);
        x->addIncoming(x1, xb);
        b.CreateCondBr(b.CreateICmpULT(x1, dx), xb, yl);

        b.SetInsertPoint(yl);
        Value *y1 = b.CreateAdd(y, b.getInt32(1), "ty.next", true, true);
        y->addIncoming(y1, yl);
        b.CreateCondBr(b.CreateICmpULT(y1, dy), yh, zl);

        b.SetInsertPoint(zl);
        Value *z1 = b.CreateAdd(z, b.getInt32(1), "tz.next", true, true);
        z->addIncoming(z1, zl);
        b.CreateCondBr(b.CreateICmpULT(z1, dz), zh, done);

        b.SetInsertPoint(done);
        b.CreateRet(b.getInt32(0));

        inlineChain(c, kInlineDepth);
        return f;
    }
};

// ---- the image -------------------------------------------------------------

struct Entry {
    Constant *name;
    Function *fn;
};

bool parseImage(GlobalVariable &g, SmallVectorImpl<Entry> &out)
{
    if (!g.hasInitializer()) return false;
    auto *cs = dyn_cast<ConstantStruct>(g.getInitializer());
    if (cs == nullptr || cs->getNumOperands() != 4) return false;
    auto *magic = dyn_cast<ConstantInt>(cs->getOperand(0));
    auto *version = dyn_cast<ConstantInt>(cs->getOperand(1));
    auto *count = dyn_cast<ConstantInt>(cs->getOperand(2));
    if (magic == nullptr || version == nullptr || count == nullptr ||
        magic->getZExtValue() != kImageMagic || version->getZExtValue() != kImageV1)
        return false;
    auto *table = dyn_cast<GlobalVariable>(cs->getOperand(3)->stripPointerCasts());
    if (table == nullptr || !table->hasInitializer() || table->isInterposable()) return false;
    auto add = [&](Constant *e) {
        auto *s = dyn_cast<ConstantStruct>(e);
        if (s == nullptr || s->getNumOperands() != 2) return false;
        auto *fn = dyn_cast<Function>(s->getOperand(1)->stripPointerCasts());
        if (fn == nullptr || fn->isDeclaration()) return false;
        out.push_back({s->getOperand(0), fn});
        return true;
    };
    Constant *init = table->getInitializer();
    if (auto *arr = dyn_cast<ConstantArray>(init)) {
        for (Use &u : arr->operands())
            if (!add(cast<Constant>(u.get()))) return false;
    } else if (!add(init)) {
        return false;
    }
    return out.size() == count->getZExtValue();
}

void report(Module &m, const std::string &lines)
{
    const char *path = std::getenv("OCU_CPU_LOOPS_REPORT");
    if (path == nullptr || *path == '\0' || lines.empty()) return;
    std::error_code ec;
    raw_fd_ostream os(path, ec, sys::fs::OF_Append | sys::fs::OF_Text);
    if (!ec) os << lines; // one write: lines from parallel compiles stay whole
    (void)m;
}

struct BlockFormPass : PassInfoMixin<BlockFormPass> {
    PreservedAnalyses run(Module &m, ModuleAnalysisManager &)
    {
        SmallVector<GlobalVariable *, 16> images;
        for (GlobalVariable &g : m.globals()) images.push_back(&g);
        Builder build(m);
        std::string lines;
        bool changed = false;
        for (GlobalVariable *g : images) {
            SmallVector<Entry, 4> entries;
            if (!parseImage(*g, entries)) continue;
            SmallVector<Function *, 4> blocks;
            bool any = false;
            for (Entry &e : entries) {
                Verdict v = check(*e.fn);
                Function *blk = nullptr;
                std::string why = v.reason();
                if (v.ok()) {
                    blk = build.makeBlock(*e.fn, v.needsCurrent);
                    why = "-";
                } else if (v.cut.empty() && v.other.empty() && v.indirect.size() == 1) {
                    bool cur = v.needsCurrent;
                    SmallVector<Function *, 16> allowed =
                        build.targets(v.indirect[0]->getFunctionType(), cur);
                    Function *target = allowed.empty() || allowed.size() > kMaxTargets
                                           ? nullptr
                                           : build.makeTarget(*e.fn, *v.indirect[0]);
                    if (target != nullptr) {
                        blk = build.makeBlock(*e.fn, cur, target, allowed);
                        why = "through a pointer, " + std::to_string(allowed.size()) +
                              " kernel(s) accepted";
                    } else {
                        why += allowed.empty() ? " (no kernel of that type passes)"
                               : allowed.size() > kMaxTargets ? " (too many kernels of that type)"
                                                              : " (pointer not found before the call)";
                    }
                }
                blocks.push_back(blk);
                any |= blk != nullptr;
                lines += (blk ? "loop\t" : "fiber\t") + kernelName(*e.fn) + "\t" + why + "\t" +
                         m.getSourceFileName() + "\n";
            }
            if (!any) continue;
            // Version 2: {name, fn, block} per entry.
            LLVMContext &ctx = m.getContext();
            auto *ptr = PointerType::getUnqual(ctx);
            auto *ety = StructType::get(ctx, {ptr, ptr, ptr});
            SmallVector<Constant *, 4> rows;
            for (size_t k = 0; k < entries.size(); ++k)
                rows.push_back(ConstantStruct::get(
                    ety, {entries[k].name, entries[k].fn,
                          blocks[k] ? cast<Constant>(blocks[k]) : ConstantPointerNull::get(ptr)}));
            auto *aty = ArrayType::get(ety, rows.size());
            auto *table = new GlobalVariable(m, aty, true, GlobalValue::InternalLinkage,
                                             ConstantArray::get(aty, rows), g->getName() + ".ocu2");
            table->setAlignment(m.getDataLayout().getPointerABIAlignment(0));
            auto *cs = cast<ConstantStruct>(g->getInitializer());
            auto *i32 = Type::getInt32Ty(ctx);
            g->setInitializer(ConstantStruct::get(
                cs->getType(), {cs->getOperand(0), ConstantInt::get(i32, kImageV2),
                                cs->getOperand(2), table}));
            changed = true;
        }
        report(m, lines);
        return changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
    }
};

// ---- before the vectoriser: drop index stores nobody reads -----------------

bool readsNothingOfOurs(CallBase &cb)
{
    Function *c = directCallee(cb);
    if (c == nullptr || cb.isInlineAsm()) return false;
    if (c->isIntrinsic()) return true; // memory intrinsics are checked by use below
    if (c->hasFnAttribute(kBlockAttr) || c->getName().ends_with(".ocu_ordered")) return true;
    return c->isDeclaration() && isRuntimeLibraryName(c->getName());
}

struct DropIndexStoresPass : PassInfoMixin<DropIndexStoresPass> {
    PreservedAnalyses run(Function &f, FunctionAnalysisManager &)
    {
        if (!f.hasFnAttribute(kBlockAttr)) return PreservedAnalyses::all();
        SmallVector<StoreInst *, 8> tidStores, curStores;
        bool calls = false;
        for (Instruction &i : instructions(f)) {
            if (auto *s = dyn_cast<StoreInst>(&i)) {
                if (s->getMetadata(kTlsTag)) tidStores.push_back(s);
                if (s->getMetadata(kCurTag)) curStores.push_back(s);
            } else if (auto *cb = dyn_cast<CallBase>(&i)) {
                calls |= !readsNothingOfOurs(*cb);
            }
        }
        if (calls) return PreservedAnalyses::all();
        // Every use of the thread-index variable's address in this function
        // must be one of our stores (or an address computation for them).
        bool tidRead = false;
        Module &m = *f.getParent();
        if (GlobalVariable *tid = findTid(m)) {
            SmallVector<Value *, 16> addrs;
            for (Instruction &i : instructions(f))
                if (auto *ii = dyn_cast<IntrinsicInst>(&i))
                    if (ii->getIntrinsicID() == Intrinsic::threadlocal_address &&
                        ii->getArgOperand(0)->stripPointerCasts() == tid)
                        addrs.push_back(ii);
            for (User *u : tid->users())
                if (auto *i = dyn_cast<Instruction>(u); i && i->getFunction() == &f &&
                                                       !isa<IntrinsicInst>(i))
                    addrs.push_back(i);
            for (size_t k = 0; k < addrs.size() && !tidRead; ++k) {
                for (User *u : addrs[k]->users()) {
                    if (auto *s = dyn_cast<StoreInst>(u)) {
                        if (s->getMetadata(kTlsTag) && s->getPointerOperand() == addrs[k]) continue;
                        tidRead = true;
                    } else if (isa<GetElementPtrInst>(u) || isa<BitCastInst>(u)) {
                        addrs.push_back(u);
                    } else {
                        tidRead = true;
                    }
                    if (tidRead) break;
                }
            }
        }
        bool changed = false;
        if (!tidRead)
            for (StoreInst *s : tidStores) s->eraseFromParent(), changed = true;
        for (StoreInst *s : curStores) s->eraseFromParent(), changed = true; // no call reads it
        return changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
    }
};

} // namespace

extern "C" LLVM_ATTRIBUTE_WEAK LLVM_ATTRIBUTE_VISIBILITY_DEFAULT PassPluginLibraryInfo
llvmGetPassPluginInfo()
{
    return {LLVM_PLUGIN_API_VERSION, "OcuCpuLoops", "1", [](PassBuilder &pb) {
                pb.registerPipelineStartEPCallback([](ModulePassManager &mpm, OptimizationLevel l) {
                    if (l != OptimizationLevel::O0) mpm.addPass(BlockFormPass());
                });
                pb.registerVectorizerStartEPCallback(
                    [](FunctionPassManager &fpm, OptimizationLevel) {
                        fpm.addPass(DropIndexStoresPass());
                    });
                pb.registerPipelineParsingCallback(
                    [](StringRef name, ModulePassManager &mpm, ArrayRef<PassBuilder::PipelineElement>) {
                        if (name != "ocu-cpu-loops") return false;
                        mpm.addPass(BlockFormPass());
                        return true;
                    });
                pb.registerPipelineParsingCallback(
                    [](StringRef name, FunctionPassManager &fpm, ArrayRef<PassBuilder::PipelineElement>) {
                        if (name != "ocu-cpu-loops-drop") return false;
                        fpm.addPass(DropIndexStoresPass());
                        return true;
                    });
            }};
}
