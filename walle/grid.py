"""Small grid helpers. Coordinates are (x, y) with y growing downward; arrays are [y, x]."""
from collections import deque

import numpy as np

DIRS = {"N": (0, -1), "E": (1, 0), "S": (0, 1), "W": (-1, 0)}


def bfs_dist(blocked: np.ndarray, sources) -> np.ndarray:
    """Shortest 4-connected step distance from any source; inf where unreachable."""
    H, W = blocked.shape
    blk = blocked.tolist()
    dist = [[-1] * W for _ in range(H)]
    dq = deque()
    for x, y in sources:
        if 0 <= x < W and 0 <= y < H and not blk[y][x] and dist[y][x] < 0:
            dist[y][x] = 0
            dq.append((x, y))
    while dq:
        x, y = dq.popleft()
        d = dist[y][x] + 1
        for dx, dy in DIRS.values():
            nx, ny = x + dx, y + dy
            if 0 <= nx < W and 0 <= ny < H and not blk[ny][nx] and dist[ny][nx] < 0:
                dist[ny][nx] = d
                dq.append((nx, ny))
    out = np.array(dist, dtype=float)
    out[out < 0] = np.inf
    return out


def cheb(a, b) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))
