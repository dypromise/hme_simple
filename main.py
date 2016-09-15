import numpy as np
import hme


X = []
Y = []
hme_ = hme.hme(X,Y)
hme_.para_estimation(X,Y)
x_new = []
y_out = hme_.prediction(x_new)
