# pypsy

psychometrics package, including structural equation model, confirmatory factor analysis, unidimensional item response theory, multidimensional item response theory, cognitive diagnosis model, factor analysis and adaptive testing. The package is still a doll. will be finished in future.


## unidimensional item response theory

### models

* binary response data IRT (two parameters, three parameters).

* grade respone data IRT (GRM model)


## Parameter estimation algorithm

* EM algorithm (2PL, GRM)

* MCMC algorithm (3PL）


* * *


## Multidimensional item response theory (full information item factor analysis)


### Parameter estimation algorithm

#### The initial value

The approximate polychoric correlation is calculated, and the slope initial value is obtained by factor analysis of the polychoric correlation matrix.


#### EM algorithm

* E step uses GH integral.

* M step uses Newton algorithm (sparse matrix is divided into non sparse matrix).


#### Factor rotation

Gradient projection algorithm


### The shortcomings

GH integrals can only estimate low dimensional parameters.


* * *


## Cognitive diagnosis model

### models

* Dina

* ho-dina


### parameter estimation algorithms

* EM algorithm

* MCMC algorithm

* maximum likelihood estimation (only for estimating skill parameters of subjects)


* * *


## Structural equation model

* contains three parameter estimation methods(ULS, ML and GLS).

* based on gradient descent


* * *


## Confirmatory factor analysis

* can be used for continuous data, binary data and ordered data.

* based on gradient descent

* binary and ordered data based on Polychoric correlation matrix.


* * *


## Factor analysis

For the time being, only for the calculation of full information item factor analysis, it is very simple.


### The algorithm

principal component analysis


### The rotation algorithm

gradient projection


* * *


## Adaptive test (bug, need to be repaired)

### model

Thurston IRT model (multidimensional item response theory model for personality test)


### Algorithm

Maximum information method for multidimensional item response theory


## Require

* numpy

* progressbar2


## How to use it

See demo in detail


## TODO LIST

* theta parameterization of CCFA

* parameter estimation of structural equation models for multivariate data

* Bayesin knowledge tracing (Bayesian knowledge tracking)

* multidimensional item response theory (full information item factor analysis)

   * high dimensional computing algorithm (adaptive integral, etc.)

   * various item response models

* cognitive diagnosis model

   * G-DINA model

   * Q matrix correlation algorithm

* Factor analysis

   * maximum likelihood estimation

   * various factor rotation algorithms

* adaptive

   * adaptive cognitive diagnosis

   * other adaption model

* standard error and P value

* code annotation, testing and documentation.


## Reference

* [DINA Model and Parameter Estimation: A Didactic](http://www.stat.cmu.edu/~brian/PIER-methods/For%202013-03-04/Readings/de%20la%20Torre-dina-est-115-30-jebs.pdf)
* [Higher-order latent trait models for cognitive diagnosis](http://www.aliquote.org/pub/delatorre2004.pdf)
* [Full-Information Item Factor Analysis.](http://conservancy.umn.edu/bitstream/11299/104282/1/v12n3p261.pdf)
* [Multidimensional adaptive testing](http://media.metrik.de/uploads/incoming/pub/Literatur/1996_Multidimensional%20adaptive%20testing.pdf)
* [Derivative free gradient projection algorithms for rotation]()

# pypsy
自编心理测量库，包含结构方程模型，验证性因子分析，单维项目反应理论，多维项目反应理论，认知诊断，因子分析和自适应测验等等，还在整理中，仅供学习

## 单维项目反应理论
### 支持模型
* 二级计分IRT（双参数，三参数）
* 多级计分IRT（GRM模型）

### 参数估计算法
* EM算法（双参数，GRM）
* MCMC算法（三参数）

* * *

## 多维项目反应理论（全息项目因子分析）

### 参数估计算法
#### 初值
计算近似polychoric correlation, 对这个相关矩阵进行因子分析，获得斜率初值

#### EM算法
* E步用GH积分
* M步用牛顿算法（把稀疏矩阵拆成不稀疏的矩阵计算）

#### 因子旋转
基于梯度投影算法

### 缺点
GH积分只能计算低维度的参数估计

* * *

## 认知诊断
### 支持两种模型
* dina
* ho-dina

### 支持三种参数估计算法
* EM算法
* MCMC算法
* 极大似然估计（仅限估计被试技能掌握参数）

* * *

## 结构方程模型
* 包含ULS, ML, GLS三种参数估计方法
* 基于梯度下降

* * *

## 验证性因子分析
* 支持连续数据、二分数据和有序数据
* 基于梯度下降
* 二分数据和有序数据基于Polychoric相关矩阵

* * *

## 因子分析
暂时只为计算全息项目因子分析而存在，很简单的实现

### 算法
主成分分析

### 旋转算法
梯度投影

* * *

## 自适应测验（有bug，待修复）
### 支持模型
瑟斯顿IRT模型（用于人格测验的多维项目反应理论模型）

### 抽题算法
多维项目反应理论的最大信息法

## require
* numpy
* progressbar2

## 使用方法
详见demo

## TODO LIST
* CCFA的theta参数化
* 多样化数据的结构方程模型参数估计
* 贝叶斯知识追踪(Bayesin knowledge tracing)
* 多维项目反应理论（全息项目因子分析）
    * 高维度计算算法（自适应积分等）
    * 各类项目反应模型
* 认知诊断
    * G-DINA模型
    * Q矩阵相关算法
* 因子分析
    * 极大似然估计
    * 各类因子旋转算法
* 自适应
    * 自适应认知诊断
    * 其他自适应
* 标准误、P值
* 代码注释、测试和文档

## 参考文献
* [DINA Model and Parameter Estimation: A Didactic](http://www.stat.cmu.edu/~brian/PIER-methods/For%202013-03-04/Readings/de%20la%20Torre-dina-est-115-30-jebs.pdf)
* [Higher-order latent trait models for cognitive diagnosis](http://www.aliquote.org/pub/delatorre2004.pdf)
* [Full-Information Item Factor Analysis.](http://conservancy.umn.edu/bitstream/11299/104282/1/v12n3p261.pdf)
* [Multidimensional adaptive testing](http://media.metrik.de/uploads/incoming/pub/Literatur/1996_Multidimensional%20adaptive%20testing.pdf)
* [Derivative free gradient projection algorithms for rotation]()
