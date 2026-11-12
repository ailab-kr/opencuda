import unittest

from cpu_rewrite import RewriteError, rewrite


def rw(src):
    return rewrite(src)[0]


class LaunchTests(unittest.TestCase):
    def test_simple(self):
        self.assertEqual(rw("k<<<1, 2>>>(a, b);"),
                         "::ocu_cpu::launch(k, ::ocu_cpu::cfg(1, 2), a, b);")

    def test_no_args(self):
        self.assertEqual(rw("k<<<g, b>>>();"),
                         "::ocu_cpu::launch(k, ::ocu_cpu::cfg(g, b));")

    def test_template_and_namespace(self):
        self.assertEqual(rw("ns::k<float, 4><<<g, b>>>(x);"),
                         "::ocu_cpu::launch(ns::k<float, 4>, ::ocu_cpu::cfg(g, b), x);")
        self.assertEqual(rw("::k<<<g, b>>>(x);"),
                         "::ocu_cpu::launch(::k, ::ocu_cpu::cfg(g, b), x);")
        self.assertEqual(rw("a::b::k<T<int>><<<g, b>>>(x);"),
                         "::ocu_cpu::launch(a::b::k<T<int>>, ::ocu_cpu::cfg(g, b), x);")

    def test_nested_parens_and_shifts_in_config(self):
        self.assertEqual(rw("k<<<dim3((n >> 1), 1), f(a, b), s>>>(x);"),
                         "::ocu_cpu::launch(k, ::ocu_cpu::cfg(dim3((n >> 1), 1), f(a, b), s), x);")

    def test_keeps_line_count(self):
        src = "k\n  <<<g,\n b>>>\n (x,\n y);\nint after;"
        out = rw(src)
        self.assertEqual(out.count("\n"), src.count("\n"))
        self.assertTrue(out.endswith("int after;"))

    def test_comments_strings_preprocessor_untouched(self):
        src = ('// k<<<1,1>>>(x);\n/* k<<<1,1>>>(x); */\n'
               'const char *s = "k<<<1,1>>>(x)";\n'
               "const char *r = R\"d(k<<<1,1>>>(x))d\";\n"
               '#define L k<<<1,1>>>(x)\n'
               "int big = 1'000'000;\n")
        self.assertEqual(rw(src), src)

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
        self.assertEqual(out, "float *s = (float *)::ocu_cpu::dynamic_shared();")
        self.assertEqual(stats["extern_shared"], 1)

    def test_aligned_and_unsigned(self):
        self.assertEqual(rw("extern __shared__ __align__(16) unsigned int s[];"),
                         "unsigned int *s = (unsigned int *)::ocu_cpu::dynamic_shared();")

    def test_in_comment_untouched(self):
        src = "// extern __shared__ float s[];\n"
        self.assertEqual(rw(src), src)


if __name__ == "__main__":
    unittest.main()
