import unittest

from cpu_rewrite import LAMBDA_ARG, RewriteError, rewrite


def rw(src):
    return rewrite(src)[0]


def launch(cfg, callee, args=""):
    lam = f"[](auto &&...{LAMBDA_ARG}) {{ {callee}({LAMBDA_ARG}...); }}"
    sep = ", " if args.strip() else ""
    return f"::ocu_cpu::launch(::ocu_cpu::cfg({cfg}), {lam}{sep}{args})"


class LaunchTests(unittest.TestCase):
    def test_simple(self):
        self.assertEqual(rw("k<<<1, 2>>>(a, b);"), launch("1, 2", "k", "a, b") + ";")

    def test_no_args(self):
        self.assertEqual(rw("k<<<g, b>>>();"), launch("g, b", "k") + ";")

    def test_template_and_namespace(self):
        self.assertEqual(rw("ns::k<float, 4><<<g, b>>>(x);"),
                         launch("g, b", "ns::k<float, 4>", "x") + ";")
        self.assertEqual(rw("::k<<<g, b>>>(x);"), launch("g, b", "::k", "x") + ";")
        self.assertEqual(rw("a::b::k<T<int>><<<g, b>>>(x);"),
                         launch("g, b", "a::b::k<T<int>>", "x") + ";")

    def test_shifts_and_comparisons_inside_template_parens(self):
        # review finding 6: '>' inside parentheses is not a template bracket
        self.assertEqual(rw("k<(N>>1)><<<g, b>>>(x);"),
                         launch("g, b", "k<(N>>1)>", "x") + ";")
        self.assertEqual(rw("k<(A>B)><<<g, b>>>(x);"),
                         launch("g, b", "k<(A>B)>", "x") + ";")

    def test_nested_parens_and_shifts_in_config(self):
        self.assertEqual(rw("k<<<dim3((n >> 1), 1), f(a, b), s>>>(x);"),
                         launch("dim3((n >> 1), 1), f(a, b), s", "k", "x") + ";")

    def test_keeps_line_count(self):
        src = "k\n  <<<g,\n b>>>\n (x,\n y);\nint after;"
        out = rw(src)
        self.assertEqual(out.count("\n"), src.count("\n"))
        self.assertTrue(out.endswith("int after;"))

    def test_comments_between_tokens(self):
        # review finding 6
        self.assertEqual(rw("k /*x*/ <<<1, 1>>> /*y*/ (a);"),
                         launch("1, 1", "k", "a") + ";")
        self.assertEqual(rw("k // why\n<<<1, 1>>>(a);"),
                         launch("1, 1", "k", "a") + "\n;")

    def test_launch_inside_define(self):
        # review finding 6: launches in macro bodies are rewritten, and the
        # macro's line continuations survive
        out = rw("#define LAUNCH(g) k<<<g, 256>>>(p)\nLAUNCH(4);\n")
        self.assertTrue(out.startswith("#define LAUNCH(g) " + launch("g, 256", "k", "p")))
        out = rw("#define L2(g) \\\n  k<<<g, \\\n 256>>>(p)\nL2(1);\n")
        self.assertNotIn("<<<", out)
        self.assertEqual(out.count("\n"), 4)
        for line in out.split("\n")[:2]:
            self.assertTrue(line.endswith("\\"), line)

    def test_other_directives_untouched(self):
        src = '#include "k<<<1,1>>>(x).h"\n#if A<<<B\n#endif\n'
        self.assertEqual(rw(src), src)

    def test_comments_strings_untouched(self):
        src = ('// k<<<1,1>>>(x);\n/* k<<<1,1>>>(x); */\n'
               'const char *s = "k<<<1,1>>>(x)";\n'
               "const char *r = R\"d(k<<<1,1>>>(x))d\";\n"
               "int big = 1'000'000;\nint hex = 0x1'F;\n")
        self.assertEqual(rw(src), src)

    def test_prefixed_char_literals(self):
        # review finding 6: L'x' and u8'a' are literals, not digit separators
        out = rw("wchar_t c = L'x'; char d = u8'a'; k<<<1,1>>>(c);")
        self.assertIn(launch("1,1", "k", "c"), out)
        self.assertIn("L'x'", out)
        self.assertEqual(rw("char q = L'<'; int z = 0;"), "char q = L'<'; int z = 0;")

    def test_null_argument_becomes_nullptr(self):
        self.assertEqual(rw("k<<<1,1>>>(NULL, n);"), launch("1,1", "k", "nullptr, n") + ";")
        self.assertEqual(rw("k<<<1,1>>>(f(NULL), n);"), launch("1,1", "k", "f(NULL), n") + ";")

    def test_two_launches(self):
        out, stats = rewrite("a<<<1,1>>>(x); b<<<2,2>>>(y);")
        self.assertEqual(stats["launches"], 2)
        self.assertNotIn("<<<", out)

    def test_errors(self):
        with self.assertRaises(RewriteError):
            rw("k<<<1, 1>>> ;")
        with self.assertRaises(RewriteError):
            rw("k<<<1, 1;")
        with self.assertRaises(RewriteError):
            rw("<<<1, 1>>>(x);")


class ExternSharedTests(unittest.TestCase):
    def test_basic(self):
        out, stats = rewrite("extern __shared__ float s[];")
        self.assertEqual(out, "static ::ocu_cpu::dyn_shared<float> s;")
        self.assertEqual(stats["extern_shared"], 1)

    def test_alignment_dropped_qualifiers_kept(self):
        # review finding 6
        self.assertEqual(rw("extern __shared__ __align__(16) unsigned int s[];"),
                         "static ::ocu_cpu::dyn_shared<unsigned int> s;")
        self.assertEqual(rw("extern __shared__ float __align__(32) s[];"),
                         "static ::ocu_cpu::dyn_shared<float> s;")
        self.assertEqual(rw("extern volatile __shared__ float s[];"),
                         "static ::ocu_cpu::dyn_shared<volatile float> s;")
        self.assertEqual(rw("extern __shared__ volatile int s[];"),
                         "static ::ocu_cpu::dyn_shared<volatile int> s;")

    def test_keeps_lines(self):
        src = "extern __shared__\n  float s[];\nint x;"
        self.assertEqual(rw(src).count("\n"), 2)

    def test_in_comment_untouched(self):
        src = "// extern __shared__ float s[];\n"
        self.assertEqual(rw(src), src)


class PreprocessedTests(unittest.TestCase):
    """Whole preprocessed translation units, as ocu_cc.py rewrites them."""

    def test_spaced_chevrons(self):
        self.assertEqual(rw("k << < g, b >> > (x);"), launch(" g, b ", "k", "x") + ";")
        self.assertEqual(rw("k<< <1,1>> >();"), launch("1,1", "k") + ";")

    def test_shift_is_not_a_launch(self):
        src = "int a = b << c; v<v<int>> w; f(a >> b > c);"
        self.assertEqual(rw(src), src)

    def test_literal_zero_argument(self):
        self.assertEqual(rw("k<<<1,1>>>(p, 0);"),
                         launch("1,1", "k", "p, ::ocu_cpu::literal_zero{}") + ";")
        self.assertEqual(rw("k<<<1,1>>>(10, x0);"), launch("1,1", "k", "10, x0") + ";")

    def test_null_between_line_markers(self):
        src = 'k<<<1,1>>>(\n# 3 "a.cu" 3 4\n  __null\n# 3 "a.cu"\n  , 0);'
        out = rewrite(src, preprocessed=True)[0]
        self.assertIn("nullptr", out)
        self.assertIn("::ocu_cpu::literal_zero{}", out)
        self.assertIn('# 3 "a.cu" 3 4', out)  # markers kept
        self.assertEqual(out.count("\n"), src.count("\n"))

    def test_shared_marker(self):
        src = "__global__ void f() { extern __ocu_shared__ float s[]; __ocu_shared__ int t[4]; }"
        out = rewrite(src, preprocessed=True)[0]
        self.assertIn("static ::ocu_cpu::dyn_shared<float> s;", out)
        self.assertIn("static thread_local int t[4];", out)
        self.assertNotIn("__ocu_shared__", out)

    def test_many_launches_in_a_large_unit(self):
        body = "".join(f"k{i}<<<1, {i}>>>(p);\n" for i in range(200))
        filler = "// filler line with 'quotes' and \"strings\"\n" * 20000
        out = rw(filler + body)
        self.assertNotIn("<<<", out)
        self.assertEqual(out.count("::ocu_cpu::launch("), 200)


if __name__ == "__main__":
    unittest.main()
