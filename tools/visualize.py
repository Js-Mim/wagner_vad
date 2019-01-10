# -*- coding: utf-8 -*-
__author__ = 'S.I. Mimilakis'
__copyright__ = 'MacSeNet'

# imports
from visdom import Visdom
import numpy as np

viz = Visdom()


def init_visdom():
    viz.close()
    window = viz.line(X=np.arange(0, 1), Y=np.reshape(0, 1))
    windowB = viz.line(X=np.arange(0, 1), Y=np.reshape(0, 1))

    return window, windowB

# EOF
