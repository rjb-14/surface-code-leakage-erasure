"""Reference spatial checks shared by X-prepared stability experiments."""

from .surface_code import Coord, SurfaceCodeLayout


def reference_x_stability_geometry(d):
    """Return data coordinates and regular checks for a validated even d."""
    corners = set() if d == 2 else {
        Coord(1, 1), Coord(2*d-1, 1),
        Coord(1, 2*d-1), Coord(2*d-1, 2*d-1),
    }
    data = {
        Coord(x, y) for x in range(1, 2*d, 2) for y in range(1, 2*d, 2)
    } - corners
    checks = {}
    for x in range(0, 2*d+1, 2):
        for y in range(0, 2*d+1, 2):
            a = Coord(x, y)
            support = frozenset(a + delta for delta in SurfaceCodeLayout.Z_DELTAS) & data
            if d == 2:
                if a not in {Coord(2, 0), Coord(0, 2), Coord(2, 2),
                             Coord(4, 2), Coord(2, 4)}:
                    continue
                check_type = "X" if a == Coord(2, 2) else "Z"
            else:
                interior = 0 < x < 2*d and 0 < y < 2*d
                odd = ((x-y)//2) % 2 == 1
                if len(support) < 2 or (not interior and odd):
                    continue
                check_type = "X" if interior and odd else "Z"
            checks[a] = (check_type, support)
    return frozenset(data), checks
