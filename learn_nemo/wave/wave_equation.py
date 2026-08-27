from sympy import Symbol, Function, Number
from physicsnemo.sym.eq.pde import PDE

# 定义 1D 波动方程 PDE: d2u/dt2 - c^2 * d2u/dx2 = 0
class WaveEquation1D(PDE):
    """自定义一维波动方程"""
    def __init__(self, c=1.0):
        self.dim = 2

        t = Symbol('t')
        x = Symbol('x')
        # 这种写法可以扩充自变量
        input_var = {"x":x, "t":t}
        u = Function('u')(*input_var)  # *对字典进行解包

        self.equations = {
            'wave_equation': u.diff(t, 2) - c**2 * u.diff(x, 2),
        }


c = 1.0  # 波速
wave_equation = WaveEquation1D(c=c)
print("1D 波动方程 PDE:", wave_equation.equations)