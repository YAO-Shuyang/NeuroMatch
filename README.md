Tracking neurons across multiple days/weeks/months have become a crucial way to investigate the long-term evolution of neural representation and dynamics. Large-scale imaging has brought the field to a thriving future, which enables us to detect hundreds to thousands of neurons simultaneously and chronically. However, tracking the same neuron across distinct recording sessions are very difficult because of multiple reasons: 1) noisy data quality, 2) distortion of the brain tissue, 3) limitations of algorithms to extract calcium signals (e.g., [CNMF-E](https://doi.org/10.7554/eLife.28728), [suit2p](https://doi.org/10.1101/061507)), 4) drift of the focal plane with time elapsing, 5) and the cumulative movement of re-attached objective lens which alters the field-of-view(FOV), etc. (more discussion is available: [Tasci & Schnitzer, 2022](https://purl.stanford.edu/rt839xk2428 "Iterative cell extraction and registration for analysis of time-lapse neural calcium imaging datasets"))

[CellReg](https://doi.org/10.1016/j.celrep.2017.10.013 "Tracking the Same Neurons across Multiple Days in Ca2+ Imaging Data") is an excellent software developed by Ziv's Lab, aiming at solving this tough question, and they achieved success to some degree. However, CellReg is not perfect, with a great number (15-24%, see [Gonzalez et al., Science, 2019](https://www.science.org/doi/10.1126/science.aav9199) ) of neurons mis-matched during registration. Moreover, the results of CellReg alter when choosing distinct reference sessions, creating inconsistent and conflicted results (showcasing below).

![CellReg-registered neuron pair (set session A as reference)](https://github.com/YAO-Shuyang/NeuroMatch/blob/main/doc/image/cellpairs_ref1.png) 

This is a well-registered cell group which only missed on one day. When choosing other reference sessions, the results altered, with the previously lost neuron being identified.

![CellReg-registered neuron pair (set session B as reference)](https://github.com/YAO-Shuyang/NeuroMatch/blob/main/doc/image/cellpairs_ref2.png) 

![CellReg-registered neuron pair (set session B as reference)](https://github.com/YAO-Shuyang/NeuroMatch/blob/main/doc/image/cellpairs_ref3.png)

This phenomenon is likely a natural consequence of the greedy algorithm applied by CellReg, which is not likely to reach a globally optimized registration in most cases. However, challenges are always accompanying with opportunities. By taking this 'mildly distinct results', it may be possible to re-match these neurons based on these distinct results to obtain a better result. That's our aim.

This package is not yet finished.