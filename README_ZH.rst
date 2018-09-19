pypsy
=====

自编心理测量库，包含结构方程模型，验证性因子分析，单维项目反应理论，多维项目反应理论，认知诊断，因子分析和自适应测验等等，还在整理中，仅供学习

单维项目反应理论
----------------

支持模型
~~~~~~~~

-  二级计分IRT（双参数，三参数）
-  多级计分IRT（GRM模型）

参数估计算法
~~~~~~~~~~~~

-  EM算法（双参数，GRM）
-  MCMC算法（三参数）

--------------

多维项目反应理论（全息项目因子分析）
------------------------------------

参数估计算法
~~~~~~~~~~~~

初值
^^^^

计算近似polychoric correlation, 对这个相关矩阵进行因子分析，获得斜率初值

EM算法
^^^^^^

-  E步用GH积分
-  M步用牛顿算法（把稀疏矩阵拆成不稀疏的矩阵计算）

因子旋转
^^^^^^^^

基于梯度投影算法

缺点
~~~~

GH积分只能计算低维度的参数估计

--------------

认知诊断
--------

支持两种模型
~~~~~~~~~~~~

-  dina
-  ho-dina

支持三种参数估计算法
~~~~~~~~~~~~~~~~~~~~

-  EM算法
-  MCMC算法
-  极大似然估计（仅限估计被试技能掌握参数）

--------------

结构方程模型
------------

-  包含ULS, ML, GLS三种参数估计方法
-  基于梯度下降

--------------

验证性因子分析
--------------

-  支持连续数据、二分数据和有序数据
-  基于梯度下降
-  二分数据和有序数据基于Polychoric相关矩阵

--------------

因子分析
--------

暂时只为计算全息项目因子分析而存在，很简单的实现

算法
~~~~

主成分分析

旋转算法
~~~~~~~~

梯度投影

--------------

自适应测验
----------

支持模型
~~~~~~~~

瑟斯顿IRT模型（用于人格测验的多维项目反应理论模型）

抽题算法
~~~~~~~~

多维项目反应理论的最大信息法

--------------

require
-------

-  numpy
-  progressbar2

--------------

使用方法
--------

install
~~~~~~~
::

    pip install psy

详见demo

TODO LIST
---------

-  CCFA的theta参数化
-  多样化数据的结构方程模型参数估计
-  贝叶斯知识追踪(Bayesin knowledge tracing)
-  多维项目反应理论（全息项目因子分析）

   -  高维度计算算法（自适应积分等）
   -  各类项目反应模型

-  认知诊断

   -  G-DINA模型
   -  Q矩阵相关算法

-  因子分析

   -  极大似然估计
   -  各类因子旋转算法

-  自适应

   -  自适应认知诊断
   -  其他自适应

-  标准误、P值
-  代码注释、测试和文档

参考文献
--------

-  `DINA Model and Parameter Estimation: A
   Didactic <http://www.stat.cmu.edu/~brian/PIER-methods/For%202013-03-04/Readings/de%20la%20Torre-dina-est-115-30-jebs.pdf>`__
-  `Higher-order latent trait models for cognitive
   diagnosis <http://www.aliquote.org/pub/delatorre2004.pdf>`__
-  `Full-Information Item Factor
   Analysis. <http://conservancy.umn.edu/bitstream/11299/104282/1/v12n3p261.pdf>`__
-  `Multidimensional adaptive
   testing <http://media.metrik.de/uploads/incoming/pub/Literatur/1996_Multidimensional%20adaptive%20testing.pdf>`__
-  `Derivative free gradient projection algorithms for rotation <https://cloudfront.escholarship.org/dist/prd/content/qt9938p4wc/qt9938p4wc.pdf>`__
