import time
import numpy as np
import matplotlib.pyplot as pl


from cpymad.madx import Madx
import pysixtracklib as pyst


# prepare madx
mad = Madx()
mad.options.echo = False
mad.call(file="SPS_Q20_thin.seq")
mad.use(sequence='sps')
twiss = mad.twiss()
q1mad = twiss.summary['q1']
q2mad = twiss.summary['q2']
print(q1mad, q2mad)

# Build elements for SixTrackLib
elements = pyst.Elements.from_mad(mad.sequence.sps)


# Tracking

npart = int(2e4)
print("CPU")
particles = pyst.Particles.from_ref(npart, p0c=26e9)
particles.x=np.linspace(0,0.001,npart)
job = pyst.TrackJob(elements, particles, arch='cpu')
for nturns in range(10,50,10):
  start=time.time()
  job.track(nturns)
  print(f"turn {nturns:4d} - {time.time() - start:10.6f} s")

print("First OpenCL device")
particles = pyst.Particles.from_ref(npart, p0c=6e9)
particles.x=np.linspace(0,0.001,npart)
job = pyst.TrackJob(elements.cbuffer, particles, device="opencl:0.0")
for nturns in range(10,50,10):
  start=time.time()
  job.track(nturns)
  job.collect()
  print(f"turn {nturns:4d} - {time.time() - start:10.6f} s")


