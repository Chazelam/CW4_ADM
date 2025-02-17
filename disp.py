a1 = 0.112
a2 = 5.38**(-4)
b1 = 1.06

c1 = 7.37
c2 = 4.29*10**3
d1 = -0.0957
d2 = -0.6

c3, c4 = 0.33, 1.5
def dy(x):
    return (c3*x)/((1+c4*10**(-4)*x)**(1/2))

def F(x):
    if z0 <= 0.1:
        return np.log(c1*(x**d1)*(1 + (c2*x**d2)**(-1)))
    else:
        return np.log(c1*(x**d1)*((1 + c2*x**d2)**(-1)))

def g(x):
    return (a1*x**b1)/(1+a2*x**d2)

vg = 0

def dz:
    return (F(x)*g(x))/((1 + vg**2)**(1/2))