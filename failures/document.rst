Deblender Failures
==================
The deblender sometimes produces results where the footprint of an object extends
out into the surrounding objects. The flux in this extended footprint is very low,
but the distribution of this flux and the surrounding noises causes miscalculations
in the shape parameters.

.. image:: document_plots/deblend_failure.png
  Footprint extending beyond apparent source

Multiple Aparent minima
=======================
In some cases it seems that there are two apparent shapes that the various
algorithms converge too as shown in the following plot

.. image:: document_plots/4572.png
  Shapes as measured by different routines

.. image:: document_plots/4572-xx.png
  Marginalized likelyhood for the xx moment

.. image:: document_plots/4572-yy.png
  Marginalized likelyhood for the yy moment

.. image:: document_plots/4572-xy.png
  Marginalized likelyhood for the xy moment

However when looking at the likelihood space as measured by Monte-Carlo sampling
with an elliptical Gaussian, it becomes clear that the results of shape
determining routines does not necessarily follow minimums in likelihood space.

Possible miss estimation on high snr
====================================
In some high signal to noise cases, an elliptical fit seems to agree best with
simple shape, where HSM and SdssShape seem to agree with each other. This can
be seen in the following plot.

.. image:: document_plots/5656.png
  Shapes as determined on a high signal to noise object

SdssShape and HSM produce moments of approximately 12.6, 9.5, and -0.01 for xx,
yy, and xy respectively. SimpleShape gives values of 6.3, 4.1, and 0.1 while
fitting a Gaussian gives values of 6.9, 4.8, .003. The following posterior
distributions show that these values seem to be favored in the likelihood space.

.. image:: document_plots/5656-xx.png
  Marginalized likelyhood for the xx moment

.. image:: document_plots/5656-yy.png
  Marginalized likelyhood for the yy moment

.. image:: document_plots/5656-xx.png
  Marginalized likelyhood for the xy moment

Another case where this happens is show below.

.. image:: document_plots/13927.png
  Shapes on a high signal to noise object

SdssShape and HSM moments are approximately 16.5, 24.3, and 8.1, where as
SimpleShape gets moments of 6.5, 6.4, and 1.2 and Fitting an Elliptical gets
moments of 4.2, 6, and 0.1. Like the above example SdssShape and HSM moments
do not fall within the likelihood space as determined by the Elliptical.

Elliptical Agreement for low SNR
================================
In contrast to the situation above, some low signal to noise sources have good
agreements between adaptive moment algorithms and fitting an elliptical Gaussian.
Below are a few such examples.

.. image::document_plots/6849.png

.. image::document_plots/31027.png

.. image::document_plots/27600.png

.. image::document_plots/24697.png

Simple shape in the case does not agree with the other algorithms, most likely
due to the levels of noise throwing off the measurements.

Agreement in high SNR
=====================
Unlike the above cases, sometimes with high signal to noise objects there is
great agreement between an elliptical fit, HSM, and SdssShape, and disagreement
with SimpleShape.

.. image:: document_plots/22982.png

Failures in adaptive moments
============================
Sometimes both adaptive moments fail, as seen below

.. image:: document_plots/6057.png

This may be related to the above issue where the moments do not correspond to
a place in the likelihood space of an elliptical Gaussian. It is also possible
that this failure is related to the deblender failure where there are odd pixels
on one side of the object only. I have verified that the failure mode of shape
HSM is that the algorithm reaches a maximum number of iterations. SdssShape
failed with a flag unweightedBad, meaning both weighted and unweighted moments
were invalid. This implies that the near by deblending issues has caused a
problem, however it is unclear as SimpleShape was able to determine moments.
Looking at the marginalized likelihoods, the parameters determined by simple shape seem to
be within the uncertainty bounds as determined by the MCMC chain.
