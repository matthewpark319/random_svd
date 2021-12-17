def start_end_index(total_pts, num_procs, rank):
    pts_per_rank = int((total_pts + num_procs - 1) / num_procs)
    return rank * pts_per_rank, min((rank + 1) * pts_per_rank, total_pts)

def dot(v1, v2):
    assert len(v1) == len(v2)
    return sum(v1[i] * v2[i] for i in range(len(v1)))
