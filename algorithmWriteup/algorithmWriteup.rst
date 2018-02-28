Construction of a likelyhood function for the moments of an image
#################################################################

.. _weighted-moments:

Weighted Moments
----------------

Define the weighted moments as measured from the image. Let :math:`r_i` be a
vector :math:`[x_i,y_i]`.


.. math::
    :label: zeroth_raw_moment

    R_0 = \sum_i z(r_i)w(r_i)

.. math::
    :label: first_raw_moment

    R_1 = \sum_i z(r_i)w(r_i)r_i

.. math::
    :label: second_raw_moment

    R_2 = \sum_i z(r_i)W(r_i)r_ir_i^T


Weight Function
"""""""""""""""

We will define the weight function used in calculating moments to be an
elliptical Gaussian:

.. math::
    w(r_i) = \frac{e^{-\frac{1}{2}r_i^TC^{-1}_2r_i}}{C_0}

    C_0 = \int_{-\infty}^{\infty}e^{-\frac{1}{2}r^TC^{-1}_2r}d^kr

Debiasing
"""""""""
The weight function used in calculating the moments is useful for suppressing
extranious contributions from noise, but it also biases the measurement. To
correct for this, we can calculate an approximate debiasing factor by assuming
the source :math:`z(r_i)` is itself a Gaussian with moments :math:`Q_i` of the
form:

.. math::
    z(r_i) = \frac{e^{-\frac{1}{2}(r_i - Q_1)^T Q_2^{-1}(r_i-Q_1)}}{z_0}

    z_0(Q_2) = \int_{-\infty}^{\infty}e^{-1\frac{1}{2}r^TQ^{-1}_2r}d^kr

Zeroth Moment
"""""""""""""

With this assumption the calculation of :math:`R_0` becomes:

.. math::
    R_0 = \frac{1}{W_0}\frac{Q_0}{z_0(Q_2)}\int_{-\infty}^{\infty}
          e^{-\frac{1}{2}(r-Q_1)^TQ_2^{-1}(r-Q_1)} e^{-\frac{1}{2}(r-C_1)^TC_2^{-1}
          (r-C_1)} d^kr

Using the Matrix cook book, we recognize the integral can be re-expressed as:

.. math::
    R_0 = Q_0N\int_{-\infty}^{\infty}e^{-\frac{1}{2}(r-\alpha)^T\beta^-1
                                           (r-\alpha)}d^kr

Where:

.. math::

    N = \frac{e^{\frac{-1}{2}(Q_1 - C_1)^T(Q_2 + C_2)^{-1}(Q_1 - C_1)}}
             {\sqrt{\det(2\pi(Q_2 + C_2))}}

    \alpha = (Q_2^{-1} + C_2^{-1})^{-1}(Q_2^{-1}Q_1 + C_2^{-1}C_1)

    \beta = (Q_2^{-1} + C_2^{-1})^{-1}

If we make the assumption that :math:`\beta_x\beta_y - \beta_{xy}^2 > 0` the
above integral evaluates to:

.. math::
    R_0 = 2\pi Q_0 \det(\beta)N

First Moment
""""""""""""
Following the procedure for the zeroth moment, the first moment is simply:

.. math::

    R_1 = 2\pi Q_0\det(\beta)N\alpha

Second Moment
"""""""""""""
Likewise the second moment is:

.. math::

    R_2 = 2\pi Q_0\det(\beta)N[\beta + \alpha\alpha^T]

.. _uncertanties-in-moments:

Uncertanties in Moments
-----------------------

Because the data used to calculate moments is uncertain, the moments themselves
have uncertanty. Also, because each of the moments is calculated using the same
underlying data, the uncertanty in each moment will be correlated every other
moment.

To calculate the uncertanties in the moments, we will express the transformation
from pixel space to moment space as the following linear algebra equation,

.. math::

    \vec{R} = A\vec{Z}

where :math:`\vec{R}` is the vector of moments, :math:`A` is a number of pixels
by number of moments matrix of the coefficients in calculation moments, and
:math:`\vec{Z}` is the vector of pixels in the image.

The A matrix can be visualized with the first row contaning the value of the
weight function :math:`W(r_i)` in each column. The second row would be the value
of the weight function times the position in x, the third the value of the
weight funciton times y, so on and so fourth.

The matrix :math:`A` can then be used to transfrom the diagonal matrix of image
pixel uncertainties denoated by :math:`\Sigma_Z` according to the standard
transformation:

.. math::
    \Sigma_R = A\Sigma_ZA^T

Likelihood of Moments
---------------------

We seek to find what true moments, :math:`\vec{Q}` are maximumly likely given
moments :math:`\vec{R}` measured on a noisy image. We can express this as the
probability of moments :math:`\vec{Q}` given :math:`\vec{R}`, i.e.
:math:`P(\vec{Q}|\vec{R})`. Bayes formula can be used to re-express this in the
following way:

.. math::
    P(\vec{Q}|\vec{R}) \propto P(\vec{R}|\vec{Q})*P(\vec{Q})

We can then determine the maximumly probable moments, :math:`\vec{Q}` by finding
the arguments of :math:`\vec{Q}` that maximize the likelyhood of the right hand
side of the inequality.

Probability of Measured Moments Given Real Moments
""""""""""""""""""""""""""""""""""""""""""""""""""

The first step in maximizing the likelyhood is finding an expression for
:math:`P(\vec{R}|\vec{Q})`. This expression says that an object that has real
moments :math:`\vec{Q}` will produce moments :math:`\vec{R}` when measured from
and image that contains noise. This relation can be expressed as a Gaussian
random variable distributed about a function of the vector :math:`\vec{Q}`:

.. math::
    P(\vec{R}|\vec{Q}) = \frac{1}{a}e^{\frac{-1}{2}(\vec{R}-\vec{f(\vec{Q})})^T
                                       \Sigma_R^{-1}(\vec{R}-\vec{f(\vec{Q})})}

In this equation :math:`a` is the normalization constant for a Gaussian and is
does not contribute to in finding the maximum likelyhood. :math:`\vec{R}` is
the vector of moments measured from the image,

.. math::
    \vec{R} = <R_0, R_{1x}, R_{1y}, R_{2x}, R_{2y}, R_{2xy}>

The mean of the Gaussian is a vector of functions where each component
is the expression used to calcuate the corresponding weighted moment given
:math:`\vec{Q}` as an input. These are the expressions derived in
weighted-moments_.

Finally, the convariance of this Gaussian :math:`\Sigma_R` is as derived in the
uncertanty of moments section uncertanties-in-moments_.

Probability of True moments
"""""""""""""""""""""""""""

The probability of "true" moments :math:`P(\vec{Q})` can be interpreted as a
prior probability distribution on the parameters :math:`\vec{Q}`. When
constructing this prior we must take into account that objects measured in an
image are not heterogenious, that is objects may either be a star or a galaxy.
The form the prior takes will differ depending on which type of object is being
measured. We can express a single prior by making it a linear combination of
both types weighted by the probability of the object being either type:

.. math::
    P(\vec{Q}) = P(\vec{Q})_{gal}[1-P(*)] + P(\vec{Q})_{star}P(*)

where :math:`P(*)` is the probability of an object being a star. This
probabiliy can be tuned to different areas of the sky, and is most likely to
be a funciton of flux. At this point we will not specify a specific form.

Specifying :math:`P(\vec{Q})` then becomes an exercise in specifying
:math:`P(\vec{Q})_{gal}` and :math:`P(\vec{Q})_{star}`.

The form of :math:`P(\vec{Q})_{star}` is strait forward, as stars should always
have moments which correspond to the PSF, making :math:`P(\vec{Q})_{star}` a
delta function in parameters space, at the location corresponsing to the PSF.
Using a delta funciton can cause issues when evalutating the likelyhood with
numeric solvers, as the delta function will contain infinite derivatives. We
therefore adopt a delta function convolved by a narrow Gaussian which serves
to "fuzz" the delta function in parameter space. The width of this Gaussian
can be tuned at evaluation time.

The last peice needed to specifiy :math:`P(\vec{Q})` is an expression for the
distribution of true moments for Galaxies. This is hard to determine in an
analytically, as it relies on knowing "true" information about the properties
of the universe, which are also what we are attempting to determine. As such,
we adopt a distribution that is constructed from past measurements of galaxy
moments. This distribution will serve as a prior on what parameters are likely
and help regularize the search though the entire space of possible moments. For
the pourposes of this document, we use a Gaussian Mixture model with XX
components fit to shape parameters determined from the HSC survey. By using
a Guassian mixture model we have an analytic model that makes it possible to
evaluate first and second derivatives.
