import cupy
from cupyx.scipy.signal import abcd_normalize

from pytest import raises as assert_raises


def assert_equal(actual, desired):
    try:
        assert bool((actual == desired).all())
    except AttributeError:
        assert actual == desired


class Test_abcd_normalize:
    def setup_method(self):
        self.A = cupy.array([[1.0, 2.0], [3.0, 4.0]])
        self.B = cupy.array([[-1.0], [5.0]])
        self.C = cupy.array([[4.0, 5.0]])
        self.D = cupy.array([[2.5]])

    def test_no_matrix_fails(self):
        assert_raises(ValueError, abcd_normalize)

    def test_A_nosquare_fails(self):
        assert_raises(ValueError, abcd_normalize, [1, -1],
                      self.B, self.C, self.D)

    def test_AB_mismatch_fails(self):
        assert_raises(ValueError, abcd_normalize, self.A, [-1, 5],
                      self.C, self.D)

    def test_AC_mismatch_fails(self):
        assert_raises(ValueError, abcd_normalize, self.A, self.B,
                      [[4.0], [5.0]], self.D)

    def test_CD_mismatch_fails(self):
        assert_raises(ValueError, abcd_normalize, self.A, self.B,
                      self.C, [2.5, 0])

    def test_BD_mismatch_fails(self):
        assert_raises(ValueError, abcd_normalize, self.A, [-1, 5],
                      self.C, self.D)

    def test_normalized_matrices_unchanged(self):
        A, B, C, D = abcd_normalize(self.A, self.B, self.C, self.D)
        assert_equal(A, self.A)
        assert_equal(B, self.B)
        assert_equal(C, self.C)
        assert_equal(D, self.D)

    def test_shapes(self):
        A, B, C, D = abcd_normalize(self.A, self.B, [1, 0], 0)
        assert_equal(A.shape[0], A.shape[1])
        assert_equal(A.shape[0], B.shape[0])
        assert_equal(A.shape[0], C.shape[1])
        assert_equal(C.shape[0], D.shape[0])
        assert_equal(B.shape[1], D.shape[1])

    def test_zero_dimension_is_not_none1(self):
        B_ = cupy.zeros((2, 0))
        D_ = cupy.zeros((0, 0))
        A, B, C, D = abcd_normalize(A=self.A, B=B_, D=D_)
        assert_equal(A, self.A)
        assert_equal(B, B_)
        assert_equal(D, D_)
        assert_equal(C.shape[0], D_.shape[0])
        assert_equal(C.shape[1], self.A.shape[0])

    def test_zero_dimension_is_not_none2(self):
        B_ = cupy.zeros((2, 0))
        C_ = cupy.zeros((0, 2))
        A, B, C, D = abcd_normalize(A=self.A, B=B_, C=C_)
        assert_equal(A, self.A)
        assert_equal(B, B_)
        assert_equal(C, C_)
        assert_equal(D.shape[0], C_.shape[0])
        assert_equal(D.shape[1], B_.shape[1])

    def test_missing_A(self):
        A, B, C, D = abcd_normalize(B=self.B, C=self.C, D=self.D)
        assert_equal(A.shape[0], A.shape[1])
        assert_equal(A.shape[0], B.shape[0])
        assert_equal(A.shape, (self.B.shape[0], self.B.shape[0]))

    def test_missing_B(self):
        A, B, C, D = abcd_normalize(A=self.A, C=self.C, D=self.D)
        assert_equal(B.shape[0], A.shape[0])
        assert_equal(B.shape[1], D.shape[1])
        assert_equal(B.shape, (self.A.shape[0], self.D.shape[1]))

    def test_missing_C(self):
        A, B, C, D = abcd_normalize(A=self.A, B=self.B, D=self.D)
        assert_equal(C.shape[0], D.shape[0])
        assert_equal(C.shape[1], A.shape[0])
        assert_equal(C.shape, (self.D.shape[0], self.A.shape[0]))

    def test_missing_D(self):
        A, B, C, D = abcd_normalize(A=self.A, B=self.B, C=self.C)
        assert_equal(D.shape[0], C.shape[0])
        assert_equal(D.shape[1], B.shape[1])
        assert_equal(D.shape, (self.C.shape[0], self.B.shape[1]))

    def test_missing_AB(self):
        A, B, C, D = abcd_normalize(C=self.C, D=self.D)
        assert_equal(A.shape[0], A.shape[1])
        assert_equal(A.shape[0], B.shape[0])
        assert_equal(B.shape[1], D.shape[1])
        assert_equal(A.shape, (self.C.shape[1], self.C.shape[1]))
        assert_equal(B.shape, (self.C.shape[1], self.D.shape[1]))

    def test_missing_AC(self):
        A, B, C, D = abcd_normalize(B=self.B, D=self.D)
        assert_equal(A.shape[0], A.shape[1])
        assert_equal(A.shape[0], B.shape[0])
        assert_equal(C.shape[0], D.shape[0])
        assert_equal(C.shape[1], A.shape[0])
        assert_equal(A.shape, (self.B.shape[0], self.B.shape[0]))
        assert_equal(C.shape, (self.D.shape[0], self.B.shape[0]))

    def test_missing_AD(self):
        A, B, C, D = abcd_normalize(B=self.B, C=self.C)
        assert_equal(A.shape[0], A.shape[1])
        assert_equal(A.shape[0], B.shape[0])
        assert_equal(D.shape[0], C.shape[0])
        assert_equal(D.shape[1], B.shape[1])
        assert_equal(A.shape, (self.B.shape[0], self.B.shape[0]))
        assert_equal(D.shape, (self.C.shape[0], self.B.shape[1]))

    def test_missing_BC(self):
        A, B, C, D = abcd_normalize(A=self.A, D=self.D)
        assert_equal(B.shape[0], A.shape[0])
        assert_equal(B.shape[1], D.shape[1])
        assert_equal(C.shape[0], D.shape[0])
        assert_equal(C.shape[1], A.shape[0])
        assert_equal(B.shape, (self.A.shape[0], self.D.shape[1]))
        assert_equal(C.shape, (self.D.shape[0], self.A.shape[0]))

    def test_missing_ABC_fails(self):
        assert_raises(ValueError, abcd_normalize, D=self.D)

    def test_missing_BD_fails(self):
        assert_raises(ValueError, abcd_normalize, A=self.A, C=self.C)

    def test_missing_CD_fails(self):
        assert_raises(ValueError, abcd_normalize, A=self.A, B=self.B)
