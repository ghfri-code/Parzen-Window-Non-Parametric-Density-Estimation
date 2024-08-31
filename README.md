# Non Parametric Density Estimation
For this project, we generate a dataset for three classes each with 500 samples from three Gaussian distribution described below:

$$ class1:\quad\mu = \binom{2}{5} \qquad 
\sum =
\begin{pmatrix}
2 & 0 
\\
0 & 2
\end{pmatrix}
$$

$$ class2:\quad\mu = \binom{8}{1} \qquad 
\sum =
\begin{pmatrix}
3 & 1
\\
1 & 3
\end{pmatrix}
$$

$$ class3:\quad\mu = \binom{5}{3} \qquad 
\sum =
\begin{pmatrix}
2 & 1
\\
1 & 2
\end{pmatrix}
$$

Use generated data and estimate the density without pre-assuming a model for the distribution which is done by a non-parametric estimation.
Implement the Parzen Window PDF estimation methods using h=0.09,0.3,0.6. Estimate P(X) and Plot the true and estimated PDF.
### True Density 3D
![true density 3d](https://github.com/Ghafarian-code/Parzen-Window-Non-Parametric-Density-Estimation/blob/main/images/Figure_2.png)
### Parzen Window Density 3D
![Parzen Window density 3d](https://github.com/Ghafarian-code/Parzen-Window-Non-Parametric-Density-Estimation/blob/main/images/Figure_4.png)

It is clear that with smaller 'h', we have discontinuities and it causes no samples to be included in some cases. The larger 'h' is, the smoother the density function is and the classes are more separated.                                   
