"""rover: single-process Jetson rover assistant (no ROS2).

The brain is one process on a 6-core board, but numpy/OpenCV/torch each start an
OpenMP worker pool that *spin-waits*; left alone they burned ~5 cores at idle
(measured 517% -> 134% after this cap). Set the caps before any of those
libraries is imported, and let the operator override via the environment.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
# Moonshine uses ONNX Runtime's own pool; single-thread mode cut its CPU ~4x
# (117% -> 30%) with equal/better p90 and slightly more YOLO headroom (item 3).
os.environ.setdefault("MOONSHINE_ORT_SINGLE_THREAD", "1")
