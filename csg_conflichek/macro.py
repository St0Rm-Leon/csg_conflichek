import math

from . import shapes as sp


_ORTHO_TOL = 1e-6


def _is_orthogonal(ax, ay, az, bx, by, bz):
    a_sq = ax**2 + ay**2 + az**2
    b_sq = bx**2 + by**2 + bz**2
    if a_sq == 0 or b_sq == 0:
        return False
    dot = ax * bx + ay * by + az * bz
    return abs(dot) <= _ORTHO_TOL * math.sqrt(a_sq * b_sq)


# RPP Rectangular Solid
class RPP(sp.Shape):
    def __init__(self, x0, x1, y0, y1, z0, z1):
        if not (x0 < x1 and y0 < y1 and z0 < z1):
            raise ValueError("RPP bounds must satisfy x0 < x1, y0 < y1, z0 < z1")
        self.x0 = x0
        self.x1 = x1
        self.y0 = y0
        self.y1 = y1
        self.z0 = z0
        self.z1 = z1

    def is_positive(self, px0, py0, pz0):
        return (
            (px0 < self.x0)
            | (px0 > self.x1)
            | (py0 < self.y0)
            | (py0 > self.y1)
            | (pz0 < self.z0)
            | (pz0 > self.z1)
        )


# RCC Cylinder
class RCC(sp.Shape):
    def __init__(self, x0, y0, z0, Hx, Hy, Hz, R):
        if R <= 0:
            raise ValueError("RCC radius must be positive")
        self.x0 = x0
        self.y0 = y0
        self.z0 = z0
        self.Hx = Hx
        self.Hy = Hy
        self.Hz = Hz
        self.R = R
        self.H_sq = Hx**2 + Hy**2 + Hz**2
        if self.H_sq == 0:
            raise ValueError("RCC height vector must be nonzero")
        self.threshold = self.H_sq * self.R**2

    def is_positive(self, px0, py0, pz0):
        dx = px0 - self.x0
        dy = py0 - self.y0
        dz = pz0 - self.z0
        cross_sq = (self.Hy * dz - self.Hz * dy) ** 2
        cross_sq += (self.Hz * dx - self.Hx * dz) ** 2
        cross_sq += (self.Hx * dy - self.Hy * dx) ** 2
        dot = dx * self.Hx + dy * self.Hy + dz * self.Hz
        return (dot < 0) | (dot > self.H_sq) | (cross_sq > self.threshold)


class BOX(sp.Shape):
    def __init__(self, x0, y0, z0, ax, ay, az, bx, by, bz, cx, cy, cz):
        self.x0 = x0
        self.y0 = y0
        self.z0 = z0
        self.ax = ax
        self.ay = ay
        self.az = az
        self.bx = bx
        self.by = by
        self.bz = bz
        self.cx = cx
        self.cy = cy
        self.cz = cz
        self.a_sq = ax**2 + ay**2 + az**2
        self.b_sq = bx**2 + by**2 + bz**2
        self.c_sq = cx**2 + cy**2 + cz**2
        if self.a_sq == 0 or self.b_sq == 0 or self.c_sq == 0:
            raise ValueError("BOX side vectors must be nonzero")
        if not (
            _is_orthogonal(ax, ay, az, bx, by, bz)
            and _is_orthogonal(ax, ay, az, cx, cy, cz)
            and _is_orthogonal(bx, by, bz, cx, cy, cz)
        ):
            raise ValueError("BOX side vectors must be mutually orthogonal")

    def is_positive(self, px0, py0, pz0):
        dx = px0 - self.x0
        dy = py0 - self.y0
        dz = pz0 - self.z0
        a_dot = dx * self.ax + dy * self.ay + dz * self.az
        b_dot = dx * self.bx + dy * self.by + dz * self.bz
        c_dot = dx * self.cx + dy * self.cy + dz * self.cz
        return (
            (a_dot < 0)
            | (a_dot > self.a_sq)
            | (b_dot < 0)
            | (b_dot > self.b_sq)
            | (c_dot < 0)
            | (c_dot > self.c_sq)
        )


class HEX(sp.Shape):
    def __init__(self, x0, y0, z0, hx, hy, hz, ax, ay, az, bx, by, bz, cx, cy, cz):
        self.x0 = x0
        self.y0 = y0
        self.z0 = z0
        self.hx = hx
        self.hy = hy
        self.hz = hz
        self.ax = ax
        self.ay = ay
        self.az = az
        self.bx = bx
        self.by = by
        self.bz = bz
        self.cx = cx
        self.cy = cy
        self.cz = cz
        self.h_sq = hx**2 + hy**2 + hz**2
        self.a_sq = ax**2 + ay**2 + az**2
        self.b_sq = bx**2 + by**2 + bz**2
        self.c_sq = cx**2 + cy**2 + cz**2
        if self.h_sq == 0:
            raise ValueError("height vector must be nonzero")
        if self.a_sq == 0 or self.b_sq == 0 or self.c_sq == 0:
            raise ValueError("facet vectors must be nonzero")
        if not (
            _is_orthogonal(hx, hy, hz, ax, ay, az)
            and _is_orthogonal(hx, hy, hz, bx, by, bz)
            and _is_orthogonal(hx, hy, hz, cx, cy, cz)
        ):
            raise ValueError("HEX facet vectors must be perpendicular to h")
        # MCNP HEX/RHP supplies all three facet vectors. If a
        # regular-hex shorthand is needed, the parser can expand it
        # before constructing HEX.

    def is_positive(self, px0, py0, pz0):
        dx = px0 - self.x0
        dy = py0 - self.y0
        dz = pz0 - self.z0
        h_dot = dx * self.hx + dy * self.hy + dz * self.hz
        a_dot = dx * self.ax + dy * self.ay + dz * self.az
        b_dot = dx * self.bx + dy * self.by + dz * self.bz
        c_dot = dx * self.cx + dy * self.cy + dz * self.cz
        return (
            (h_dot < 0)
            | (h_dot > self.h_sq)
            | (a_dot < -self.a_sq)
            | (a_dot > self.a_sq)
            | (b_dot < -self.b_sq)
            | (b_dot > self.b_sq)
            | (c_dot < -self.c_sq)
            | (c_dot > self.c_sq)
        )


RHP = HEX


class REC(sp.Shape):
    def __init__(self, x0, y0, z0, hx, hy, hz, ax, ay, az, bx, by=None, bz=None):
        self.x0 = x0
        self.y0 = y0
        self.z0 = z0
        self.hx = hx
        self.hy = hy
        self.hz = hz
        self.ax = ax
        self.ay = ay
        self.az = az
        self.h_sq = hx**2 + hy**2 + hz**2
        self.a_sq = ax**2 + ay**2 + az**2
        self.bx = bx
        self.b_sq = bx**2
        self.is_b_vector = False
        if (by is None) != (bz is None):
            raise ValueError("by and bz must be provided together")
        if by is not None:
            self.by = by
            self.bz = bz
            self.b_sq += by**2 + bz**2
            self.is_b_vector = True
            # is b a vector?
        if self.h_sq == 0:
            raise ValueError("REC height vector must be nonzero")
        if self.a_sq == 0:
            raise ValueError("REC major axis vector must be nonzero")
        if not _is_orthogonal(hx, hy, hz, ax, ay, az):
            raise ValueError("REC major axis vector must be perpendicular to h")
        if self.is_b_vector:
            if self.b_sq == 0:
                raise ValueError("REC minor axis vector must be nonzero")
            if not (
                _is_orthogonal(hx, hy, hz, bx, by, bz) and _is_orthogonal(ax, ay, az, bx, by, bz)
            ):
                raise ValueError("REC minor axis vector must be perpendicular to h and a")
        elif bx <= 0:
            raise ValueError("REC minor radius must be positive")
        self.b_over_a_sq = self.b_sq / self.a_sq
        # Note:
        # if by,bz are None (DEFAULT)
        # self.bx is minor radius of the elliptic

    def is_positive(self, px0, py0, pz0):
        dx = px0 - self.x0
        dy = py0 - self.y0
        dz = pz0 - self.z0
        d_sq = dx**2 + dy**2 + dz**2
        h_dot = dx * self.hx + dy * self.hy + dz * self.hz
        a_dot = dx * self.ax + dy * self.ay + dz * self.az
        if self.is_b_vector:
            b_dot = dx * self.bx + dy * self.by + dz * self.bz
            b_dot_sq = b_dot**2
        else:
            b_dot_sq = (d_sq - (h_dot**2) / self.h_sq - (a_dot**2) / self.a_sq) * self.b_sq
        return (
            (h_dot < 0)
            | (h_dot > self.h_sq)
            | (
                (self.b_over_a_sq * a_dot**2 + b_dot_sq / self.b_over_a_sq)
                > (self.a_sq * self.b_sq)
            )
        )


class TRC(sp.Shape):
    def __init__(self, x0, y0, z0, hx, hy, hz, r1, r2):
        if r1 <= 0 or r2 <= 0:
            raise ValueError("TRC radii must be positive")
        self.x0 = x0
        self.y0 = y0
        self.z0 = z0
        self.hx = hx
        self.hy = hy
        self.hz = hz
        self.r1 = r1
        self.r2 = r2
        self.h_sq = hx**2 + hy**2 + hz**2
        if self.h_sq == 0:
            raise ValueError("TRC height vector must be nonzero")

    def is_positive(self, px0, py0, pz0):
        dx = px0 - self.x0
        dy = py0 - self.y0
        dz = pz0 - self.z0
        h_dot = dx * self.hx + dy * self.hy + dz * self.hz
        d_sq = dx**2 + dy**2 + dz**2
        radial_sq = d_sq - h_dot**2 / self.h_sq
        t = h_dot / self.h_sq
        radius = self.r1 + (self.r2 - self.r1) * t
        return (h_dot < 0) | (h_dot > self.h_sq) | (radial_sq > radius**2)


class ELL(sp.Shape):
    def __init__(self, x1, y1, z1, x2, y2, z2, radius):
        if radius == 0:
            raise ValueError("ELL radius must be nonzero")

        if radius > 0:
            dx = x2 - x1
            dy = y2 - y1
            dz = z2 - z1
            focus_norm = math.sqrt(dx**2 + dy**2 + dz**2)
            center = (
                (x1 + x2) / 2.0,
                (y1 + y2) / 2.0,
                (z1 + z2) / 2.0,
            )
            major = float(radius)
            focus_distance = focus_norm / 2.0
            if focus_distance > major:
                raise ValueError("ELL foci are outside the major radius")
            if focus_norm == 0:
                axis = (1.0, 0.0, 0.0)
                minor = major
            else:
                axis = (dx / focus_norm, dy / focus_norm, dz / focus_norm)
                minor = math.sqrt(major**2 - focus_distance**2)
                if minor == 0:
                    raise ValueError("ELL has zero minor radius")
        else:
            cx, cy, cz = x1, y1, z1
            ax, ay, az = x2, y2, z2
            axis_norm = math.sqrt(ax**2 + ay**2 + az**2)
            if axis_norm == 0:
                raise ValueError("ELL major axis vector must be nonzero")
            center = (cx, cy, cz)
            axis = (ax / axis_norm, ay / axis_norm, az / axis_norm)
            major = axis_norm
            minor = abs(radius)
            if minor == 0:
                raise ValueError("ELL minor radius must be nonzero")

        self.x0, self.y0, self.z0 = center
        self.ax, self.ay, self.az = axis
        self.a = major
        self.b = minor

    def is_positive(self, px0, py0, pz0):
        dx = px0 - self.x0
        dy = py0 - self.y0
        dz = pz0 - self.z0
        u = dx * self.ax + dy * self.ay + dz * self.az
        radial_sq = dx**2 + dy**2 + dz**2 - u**2
        return (u / self.a) ** 2 + radial_sq / self.b**2 > 1


class ARB(sp.Shape):
    def __init__(self, *args):
        if len(args) != 30:
            raise ValueError("ARB expects 8 vertices and 6 face entries")

        coords = [float(value) for value in args[:24]]
        vertices = [tuple(coords[index : index + 3]) for index in range(0, 24, 3)]
        face_entries = [int(value) for value in args[24:]]

        used = set()
        planes = []

        for entry in face_entries:
            if entry == 0:
                continue
            digits = [int(character) for character in str(entry)]
            if len(digits) not in (3, 4):
                raise ValueError("ARB face entries must have 3 or 4 digits")

            indices = [digit - 1 for digit in digits if digit]
            if len(indices) < 3 or any(index < 0 or index >= 8 for index in indices):
                raise ValueError("invalid ARB face vertex index")
            used.update(indices)

            p0 = vertices[indices[0]]
            p1 = vertices[indices[1]]
            p2 = vertices[indices[2]]
            ux = p1[0] - p0[0]
            uy = p1[1] - p0[1]
            uz = p1[2] - p0[2]
            vx = p2[0] - p0[0]
            vy = p2[1] - p0[1]
            vz = p2[2] - p0[2]
            nx = uy * vz - uz * vy
            ny = uz * vx - ux * vz
            nz = ux * vy - uy * vx
            norm = math.sqrt(nx * nx + ny * ny + nz * nz)
            if norm == 0:
                raise ValueError("ARB face has zero area")
            planes.append((p0, (nx, ny, nz)))

        if len(used) < 4:
            raise ValueError("ARB needs at least 4 used vertices")

        centroid = (
            sum(vertices[index][0] for index in used) / len(used),
            sum(vertices[index][1] for index in used) / len(used),
            sum(vertices[index][2] for index in used) / len(used),
        )

        oriented_planes = []
        for p0, normal in planes:
            value = (
                normal[0] * (centroid[0] - p0[0])
                + normal[1] * (centroid[1] - p0[1])
                + normal[2] * (centroid[2] - p0[2])
            )
            if value > 0:
                normal = tuple(-component for component in normal)
            oriented_planes.append((p0, normal))

        self.vertices = tuple(vertices)
        self.faces = tuple(face_entries)
        self.planes = tuple(oriented_planes)

    def is_positive(self, px0, py0, pz0):
        result = None
        for p0, normal in self.planes:
            value = (
                normal[0] * (px0 - p0[0]) + normal[1] * (py0 - p0[1]) + normal[2] * (pz0 - p0[2])
            )
            outside = value > 0
            result = outside if result is None else (result | outside)
        return result


class WED(sp.Shape):
    def __init__(self, x0, y0, z0, ax, ay, az, bx, by, bz, hx, hy, hz):
        self.x0 = x0
        self.y0 = y0
        self.z0 = z0
        self.ax = ax
        self.ay = ay
        self.az = az
        self.bx = bx
        self.by = by
        self.bz = bz
        self.hx = hx
        self.hy = hy
        self.hz = hz
        self.a_sq = ax**2 + ay**2 + az**2
        self.b_sq = bx**2 + by**2 + bz**2
        self.h_sq = hx**2 + hy**2 + hz**2
        if self.a_sq == 0 or self.b_sq == 0 or self.h_sq == 0:
            raise ValueError("WED vectors must be nonzero")

    def is_positive(self, px0, py0, pz0):
        dx = px0 - self.x0
        dy = py0 - self.y0
        dz = pz0 - self.z0
        h_dot = dx * self.hx + dy * self.hy + dz * self.hz
        a_dot = dx * self.ax + dy * self.ay + dz * self.az
        b_dot = dx * self.bx + dy * self.by + dz * self.bz
        alpha = a_dot / self.a_sq
        beta = b_dot / self.b_sq
        return (h_dot < 0) | (h_dot > self.h_sq) | (alpha < 0) | (beta < 0) | (alpha + beta > 1)
