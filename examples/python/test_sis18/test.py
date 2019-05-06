import numpy as np
import time

# Machine setup

from cpymad.madx import Madx
import pysixtracklib as pyst

mad = Madx()
mad.options.echo = False

mad.call(file="fodo.madx")
mad.command.beam(particle='proton', energy='6')
mad.use(sequence="FODO")
mad.twiss()

mad.command.select(flag="makethin", class_="quadrupole", slice='8')
mad.command.select(flag="makethin", class_="sbend", slice='8')
mad.command.makethin(makedipedge=False, style="teapot", sequence="fodo")

mad.twiss()

sis18 = mad.sequence.FODO

elements = pyst.Elements.from_mad(sis18)

# Tracking

npart = int(1e3)
print("CPU")
particles = pyst.Particles.from_ref(npart, p0c=6e9)
particles.x=np.linspace(0,0.001,npart)
job = pyst.TrackJob(elements, particles, arch='cpu')
for nturns in range(1,5,1):
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

