import numpy as np

mb = 300    # kg
mw = 60     # kg
bs = 1000   # N/m/s
ks = 16000  # N/m
kt = 190000 # N/m

A = np.array([[0, 1, 0, 0],
              [-ks/mb, -bs/mb, ks/mb, bs/mb],
              [0, 0, 0, 1],
              [ks/mw, bs/mw, (-ks-kt)/mw, -bs/mw]])

B = np.array([[0, 0],
              [0, 1/mb],
              [0, 0],
              [kt/mw, -1/mw]])

B1 = np.array([[0],
               [0],
               [0],
               [kt/mw]])

B2 = np.array([[0],
               [1/mb],
               [0],
               [-1/mw]])

C = np.array([[1, 0, 0, 0],
              [-ks/mb, -bs/mb, ks/mb, bs/mb]])

C1 = C

C2 = C1

D = np.array([[0, 0],
              [0, 1/mb]])

D11 = np.array([[0],
                [0]])

D12 = np.array([[0],
                [1/mb]])

D21 = D11

D22 = D12
