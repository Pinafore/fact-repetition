# Improve Human Learning with Representation + Spaced Repetition

## Variations of Half-life regression (HLR)

short-hand for each record \begin{align}<\cdot>&=<\Delta,x,P[\text{recall}]\in[0,1]>\\&=<\Delta,x,y\in\{0,1\}>\end{align}

Regression against recall probability $$l_\text{recall}(<\cdot>;\theta)=(p-f_\theta(x,\Delta))^2$$

Regression against back-solved half-life $$l_\text{half-life}(<\cdot>;\theta)=(\frac{-\Delta}{\log_2{p}}-f_\theta(x,\Delta))^2$$

Binary recall classification $$l_\text{binary}(<\cdot>;\theta)=\text{xent}(f_\theta(x,\Delta),y)$$

Assume that half-life increases exponentially with each repeated exposure, with a linear approximator, you get $f_\theta(x,\Delta)=2^{\theta\cdot x}$. Use this parameterization with regression against both recall probability and back-solved half-life, you get Settles' formulation:
$$l(<\cdot>; \theta) = (p - 2^{-\frac{\Delta}{2^{\theta\cdot x}}})^2 + \alpha(\frac{-\Delta}{\log_2(p)} - 2^{\theta\cdot x})^2 + \lambda |\theta|_2^2$$
Note that this formulation incorporates two heuristics
1. the memory strength follows an exponential forgetting curve, hence the half-life
2. half-life increases exponentially with number of repetitions

Doesn't this mean memory strength is a linear function of number of repetitions?

There are several knobs to be tweaked in the general formulation:
- labels: derived $p$, binary $y$
- linear, nn
- on/off: exponentially increased half-life
- on/off: loss wrt back-solved half-life (from derived $p$)
