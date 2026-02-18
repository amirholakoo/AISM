import os
import cv2
import torch


def apply_performance_settings():
    """Configure threading and math libraries for better CPU performance."""
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"
    os.environ["OMP_DYNAMIC"] = "FALSE"
    os.environ["KMP_AFFINITY"] = "granularity=fine,compact,1,0"

    cv2.setNumThreads(4)
    cv2.setUseOptimized(True)

    torch.set_num_threads(4)
import os
import cv2
import torch


def configure_runtime():
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"
    os.environ["OMP_DYNAMIC"] = "FALSE"
    os.environ["KMP_AFFINITY"] = "granularity=fine,compact,1,0"
    cv2.setNumThreads(4)
    cv2.setUseOptimized(True)
    torch.set_num_threads(4)

