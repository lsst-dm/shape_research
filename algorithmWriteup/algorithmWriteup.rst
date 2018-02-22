Construction of a likelyhood function for the moments of an image
#################################################################

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
