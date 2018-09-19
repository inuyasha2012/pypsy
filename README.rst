.. image:: https://img.shields.io/travis/inuyasha2012/pypsy.svg
        :target: https://travis-ci.org/inuyasha2012/pypsy

.. image:: https://coveralls.io/repos/github/inuyasha2012/pypsy/badge.svg?branch=master
        :target: https://coveralls.io/github/inuyasha2012/pypsy?branch=master

pypsy
=====

`中文 <./README_ZH.rst>`_

`DINA Model and Parameter Estimation: A
   Didactic <http://www.stat.cmu.edu/~brian/PIER-methods/For%202013-03-04/Readings/de%20la%20Torre-dina-est-115-30-jebs.pdf>`

psychometrics package, including structural equation model, confirmatory
factor analysis, unidimensional item response theory, multidimensional
item response theory, cognitive diagnosis model, factor analysis and
adaptive testing. The package is still a doll. will be finished in
future.

unidimensional item response theory
-----------------------------------

models
~~~~~~

-  binary response data IRT (two parameters, three parameters).

-  grade respone data IRT (GRM model)

Parameter estimation algorithm
------------------------------

-  EM algorithm (2PL, GRM)

-  MCMC algorithm (3PL）

--------------

Multidimensional item response theory (full information item factor analysis)
-----------------------------------------------------------------------------

Parameter estimation algorithm
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The initial value
^^^^^^^^^^^^^^^^^

The approximate polychoric correlation is calculated, and the slope
initial value is obtained by factor analysis of the polychoric
correlation matrix.

EM algorithm
^^^^^^^^^^^^

-  E step uses GH integral.

-  M step uses Newton algorithm (sparse matrix is divided into non
   sparse matrix).

Factor rotation
^^^^^^^^^^^^^^^

Gradient projection algorithm

The shortcomings
~~~~~~~~~~~~~~~~

GH integrals can only estimate low dimensional parameters.

--------------

Cognitive diagnosis model
-------------------------

models
~~~~~~

-  Dina

-  ho-dina

parameter estimation algorithms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  EM algorithm

-  MCMC algorithm

-  maximum likelihood estimation (only for estimating skill parameters
   of subjects)

--------------

Structural equation model
-------------------------

-  contains three parameter estimation methods(ULS, ML and GLS).

-  based on gradient descent

--------------

Confirmatory factor analysis
----------------------------

-  can be used for continuous data, binary data and ordered data.

-  based on gradient descent

-  binary and ordered data based on Polychoric correlation matrix.

--------------

Factor analysis
---------------

For the time being, only for the calculation of full information item
factor analysis, it is very simple.

The algorithm
~~~~~~~~~~~~~

principal component analysis

The rotation algorithm
~~~~~~~~~~~~~~~~~~~~~~

gradient projection

--------------

Adaptive test
-------------

model
~~~~~

Thurston IRT model (multidimensional item response theory model for
personality test)

Algorithm
~~~~~~~~~

Maximum information method for multidimensional item response theory

Require
-------

-  numpy

-  progressbar2

How to use it
-------------

See demo in detail

TODO LIST
---------

-  theta parameterization of CCFA

-  parameter estimation of structural equation models for multivariate
   data

-  Bayesin knowledge tracing (Bayesian knowledge tracking)

-  multidimensional item response theory (full information item factor
   analysis)

-  high dimensional computing algorithm (adaptive integral, etc.)

-  various item response models

-  cognitive diagnosis model

-  G-DINA model

-  Q matrix correlation algorithm

-  Factor analysis

-  maximum likelihood estimation

-  various factor rotation algorithms

-  adaptive

-  adaptive cognitive diagnosis

-  other adaption model

-  standard error and P value

-  code annotation, testing and documentation.

Reference
---------

-  `DINA Model and Parameter Estimation: A
   Didactic <http://www.stat.cmu.edu/~brian/PIER-methods/For%202013-03-04/Readings/de%20la%20Torre-dina-est-115-30-jebs.pdf>`__
-  `Higher-order latent trait models for cognitive
   diagnosis <http://www.aliquote.org/pub/delatorre2004.pdf>`__
-  `Full-Information Item Factor
   Analysis. <http://conservancy.umn.edu/bitstream/11299/104282/1/v12n3p261.pdf>`__
-  `Multidimensional adaptive
   testing <http://media.metrik.de/uploads/incoming/pub/Literatur/1996_Multidimensional%20adaptive%20testing.pdf>`__
-  `Derivative free gradient projection algorithms for rotation <https://cloudfront.escholarship.org/dist/prd/content/qt9938p4wc/qt9938p4wc.pdf>`__
