"""Standalone REID evaluation on GT crops (independent of the tracker).

Pick the best REID model AND tune cluster-management/identity-assignment params using
sklearn clustering metrics (Fowlkes-Mallows / Silhouette / Calinski-Harabasz) before
plugging anything into the live system.
"""
