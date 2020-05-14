import tensorflow as tf
import numpy as np

from network import network


def test_network_build():
    """
    check whether the network is built sucessfully or not 
    """
    x = np.float32(np.random.random((3, 128, 128, 3)))

    blazeface_extractor = network((128, 128, 3))
    feature = blazeface_extractor(x)
    print(feature)
    assert feature[0].shape == (3, 16, 16, 96)
    assert feature[1].shape == (3, 8, 8, 96)
    assert feature[2].shape == (3, 4, 4, 96)
    assert feature[3].shape == (3, 2, 2, 96)
    print("TEST PASSED")

if __name__ == "__main__":
    test_network_build()
