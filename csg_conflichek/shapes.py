from abc import ABC, abstractmethod
import numpy as np


class SignedShape:
    def __init__(self, sign: int, shape: "Shape"):
        if sign not in (-1, 1):
            raise ValueError("sign must be +1 or -1")
        self.sign = sign
        self.shape = shape


class Shape(ABC):
    @abstractmethod
    def is_positive(self, px0, py0, pz0):
        pass

    # this is an abstract method,
    # class Shape does not mean to define its usage
    def __neg__(self):
        return SignedShape(-1, self)

    def __pos__(self):
        return SignedShape(1, self)


# Planes
class Plane(Shape):  # Plane: Ax+By+Cz=D
    def __init__(self, a, b, c, d):
        if a == 0 and b == 0 and c == 0:
            raise ValueError("plane normal must be nonzero")
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def is_positive(self, px0, py0, pz0):
        # Strict positive side: points on the plane are on the
        # negative side.
        return (px0 * self.a + py0 * self.b + pz0 * self.c) > self.d


class XPlane(Plane):  # XPlane: Ax=D
    def __init__(self, a, d):
        super().__init__(a, 0, 0, d)
        if a == 0:
            raise ValueError("there should be an A in Ax=D")
        self.x0 = self.d / self.a

    def is_positive(self, px0, py0, pz0):
        return self.a * px0 > self.d


class YPlane(Plane):  # YPlane: By=D
    def __init__(self, b, d):
        super().__init__(0, b, 0, d)
        if b == 0:
            raise ValueError("there should be a B in By=D")
        self.y0 = self.d / self.b

    def is_positive(self, px0, py0, pz0):
        return self.b * py0 > self.d


class ZPlane(Plane):  # ZPlane: Cz=D
    def __init__(self, c, d):
        super().__init__(0, 0, c, d)
        if c == 0:
            raise ValueError("there should be a C in Cz=D")
        self.z0 = self.d / self.c

    def is_positive(self, px0, py0, pz0):
        return self.c * pz0 > self.d


class Sphere(Shape):  # Sphere: (x-x0)^2+(y-y0)^2+(z-z0)^2=r^2
    def __init__(self, x0, y0, z0, r):
        if r <= 0:
            raise ValueError("sphere radius must be positive")
        self.x0 = x0
        self.y0 = y0
        self.z0 = z0
        self.r = r
        self.r_sq = r**2

    def is_positive(self, px0, py0, pz0):
        # Positive sense is outside the sphere.
        return ((self.x0 - px0) ** 2 + (self.y0 - py0) ** 2 + (self.z0 - pz0) ** 2) > self.r_sq


# Spheres
class OSphere(Sphere):  # OSphere: x^2+y^2+z^2=r^2
    def __init__(self, r):
        super().__init__(0, 0, 0, r)

    def is_positive(self, px0, py0, pz0):
        return (px0**2 + py0**2 + pz0**2) > self.r_sq


class XSphere(Sphere):  # XSphere: (x-x0)^2+y^2+z^2=r^2
    def __init__(self, x0, r):
        super().__init__(x0, 0, 0, r)

    def is_positive(self, px0, py0, pz0):
        return ((self.x0 - px0) ** 2 + py0**2 + pz0**2) > self.r_sq


class YSphere(Sphere):
    def __init__(self, y0, r):
        super().__init__(0, y0, 0, r)

    def is_positive(self, px0, py0, pz0):
        return (px0**2 + (self.y0 - py0) ** 2 + pz0**2) > self.r_sq


class ZSphere(Sphere):
    def __init__(self, z0, r):
        super().__init__(0, 0, z0, r)

    def is_positive(self, px0, py0, pz0):
        return (px0**2 + py0**2 + (self.z0 - pz0) ** 2) > self.r_sq


# Cylinders
class XCylinder(Shape):
    def __init__(self, y0, z0, r):
        if r <= 0:
            raise ValueError("cylinder radius must be positive")
        self.y0 = y0
        self.z0 = z0
        self.r = r
        self.r_sq = r**2

    def is_positive(self, px0, py0, pz0):
        return ((self.y0 - py0) ** 2 + (self.z0 - pz0) ** 2) > self.r_sq


class YCylinder(Shape):
    def __init__(self, x0, z0, r):
        if r <= 0:
            raise ValueError("cylinder radius must be positive")
        self.x0 = x0
        self.z0 = z0
        self.r = r
        self.r_sq = r**2

    def is_positive(self, px0, py0, pz0):
        return ((self.x0 - px0) ** 2 + (self.z0 - pz0) ** 2) > self.r_sq


class ZCylinder(Shape):
    def __init__(self, x0, y0, r):
        if r <= 0:
            raise ValueError("cylinder radius must be positive")
        self.x0 = x0
        self.y0 = y0
        self.r = r
        self.r_sq = r**2

    def is_positive(self, px0, py0, pz0):
        return ((self.x0 - px0) ** 2 + (self.y0 - py0) ** 2) > self.r_sq


# MCNP-style axis-centered cylinders
class CX(XCylinder):
    def __init__(self, r):
        super().__init__(0, 0, r)


class CY(YCylinder):
    def __init__(self, r):
        super().__init__(0, 0, r)


class CZ(ZCylinder):
    def __init__(self, r):
        super().__init__(0, 0, r)


# Cones, K/X, K/Y, K/Z
# KX, KY, KZ are just special K/*
class Cone(Shape):
    def __init__(self, x0, y0, z0, t_sq, k=1):
        if t_sq <= 0:
            raise ValueError("t_sq must be positive")
        if k not in (-1, 1):
            raise ValueError("k must be +1 or -1")
        self.x0 = x0
        self.y0 = y0
        self.z0 = z0
        self.t_sq = t_sq
        self.k = k
        self.t = t_sq**0.5

    def _outside(self, u, rad_sq):
        # MCNP selects one nappe with k: k*u >= 0 is the valid sheet.
        # Points on the other nappe are outside the cone.
        cone_sq = (self.t * u) ** 2
        return (self.k * u < 0) | ((self.k * u >= 0) & (rad_sq > cone_sq))

    @abstractmethod
    def is_positive(self, px0, py0, pz0):
        pass


class XCone(Cone):
    def __init__(self, x0, y0, z0, t_sq, k=1):
        super().__init__(x0, y0, z0, t_sq, k)

    def is_positive(self, px0, py0, pz0):
        u = px0 - self.x0
        rad_sq = (py0 - self.y0) ** 2 + (pz0 - self.z0) ** 2
        return self._outside(u, rad_sq)


class YCone(Cone):
    def __init__(self, x0, y0, z0, t_sq, k=1):
        super().__init__(x0, y0, z0, t_sq, k)

    def is_positive(self, px0, py0, pz0):
        u = py0 - self.y0
        rad_sq = (px0 - self.x0) ** 2 + (pz0 - self.z0) ** 2
        return self._outside(u, rad_sq)


class ZCone(Cone):
    def __init__(self, x0, y0, z0, t_sq, k=1):
        super().__init__(x0, y0, z0, t_sq, k)

    def is_positive(self, px0, py0, pz0):
        u = pz0 - self.z0
        rad_sq = (px0 - self.x0) ** 2 + (py0 - self.y0) ** 2
        return self._outside(u, rad_sq)


# MCNP-style axis-centered cones
class KX(XCone):
    def __init__(self, x0, t_sq, k=1):
        super().__init__(x0, 0, 0, t_sq, k)


class KY(YCone):
    def __init__(self, y0, t_sq, k=1):
        super().__init__(0, y0, 0, t_sq, k)


class KZ(ZCone):
    def __init__(self, z0, t_sq, k=1):
        super().__init__(0, 0, z0, t_sq, k)


# SQ: ellipsoid, hyperboloid, paraboloid or cylinder with axes
# parallel to x, y, or z.
# A*(x-x0)^2 + B*(y-y0)^2 + C*(z-z0)^2
# + 2*D*(x-x0) + 2*E*(y-y0) + 2*F*(z-z0) + G = 0
class SQ(Shape):
    def __init__(self, A, B, C, D, E, F, G, x0, y0, z0):
        if all(value == 0 for value in (A, B, C, D, E, F, G)):
            raise ValueError("SQ coefficients must not all be zero")
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.E = E
        self.F = F
        self.G = G
        self.x0 = x0
        self.y0 = y0
        self.z0 = z0

    def is_positive(self, px0, py0, pz0):
        dx = px0 - self.x0
        dy = py0 - self.y0
        dz = pz0 - self.z0
        return (
            self.A * dx**2
            + self.B * dy**2
            + self.C * dz**2
            + 2 * self.D * dx
            + 2 * self.E * dy
            + 2 * self.F * dz
            + self.G
            > 0
        )


# Backward-compatible alias: the old name described only one of the
# surfaces covered by SQ.
Ellipse = SQ


# universal func
class General(Shape):
    def __init__(self, A, B, C, D, E, F, G, H, J, K):
        if all(value == 0 for value in (A, B, C, D, E, F, G, H, J, K)):
            raise ValueError("GQ coefficients must not all be zero")
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.E = E
        self.F = F
        self.G = G
        self.H = H
        self.J = J
        self.K = K

    def is_positive(self, px0, py0, pz0):
        return (
            self.A * px0**2
            + self.B * py0**2
            + self.C * pz0**2
            + self.D * px0 * py0
            + self.E * py0 * pz0
            + self.F * px0 * pz0
            + self.G * px0
            + self.H * py0
            + self.J * pz0
            + self.K
            > 0
        )


# Torus
class Torus(Shape):
    def __init__(self, x0, y0, z0, A, B, C):
        if A <= 0 or B <= 0 or C <= 0:
            raise ValueError("torus radii must be positive")
        self.x0 = x0
        self.y0 = y0
        self.z0 = z0
        self.A = A
        self.B = B
        self.C = C

    @abstractmethod
    def is_positive(self, px0, py0, pz0):
        pass


class XTorus(Torus):
    def __init__(self, x0, y0, z0, A, B, C):
        super().__init__(x0, y0, z0, A, B, C)

    def is_positive(self, px0, py0, pz0):
        dx_sq = (self.x0 - px0) ** 2
        dy_sq = (self.y0 - py0) ** 2
        dz_sq = (self.z0 - pz0) ** 2
        B_sq = self.B**2
        C_sq = self.C**2
        return dx_sq / B_sq + (np.sqrt(dy_sq + dz_sq) - self.A) ** 2 / C_sq - 1 > 0


class YTorus(Torus):
    def __init__(self, x0, y0, z0, A, B, C):
        super().__init__(x0, y0, z0, A, B, C)

    def is_positive(self, px0, py0, pz0):
        dx_sq = (self.x0 - px0) ** 2
        dy_sq = (self.y0 - py0) ** 2
        dz_sq = (self.z0 - pz0) ** 2
        B_sq = self.B**2
        C_sq = self.C**2
        return dy_sq / B_sq + (np.sqrt(dx_sq + dz_sq) - self.A) ** 2 / C_sq - 1 > 0


class ZTorus(Torus):
    def __init__(self, x0, y0, z0, A, B, C):
        super().__init__(x0, y0, z0, A, B, C)

    def is_positive(self, px0, py0, pz0):
        dx_sq = (self.x0 - px0) ** 2
        dy_sq = (self.y0 - py0) ** 2
        dz_sq = (self.z0 - pz0) ** 2
        B_sq = self.B**2
        C_sq = self.C**2
        return dz_sq / B_sq + (np.sqrt(dx_sq + dy_sq) - self.A) ** 2 / C_sq - 1 > 0


# Surfaces expressed by equations are OVER.
