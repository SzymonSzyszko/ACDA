import scipy.integrate as integrate
import scipy.special as special

class Sect:
    def __init__(self, c, y):
        self.c = c
        self.y = y



sect1 = Sect(0.7, 0)
sect2 = Sect(0.5, 0.1)
sect3 = Sect(0.1, 1.5)
sect4 = Sect(0.05, 1.8)

slices =[]
slices.append([sect1, sect2])
slices.append([sect2, sect3])
slices.append([sect3, sect4])

integral = 0
area = 0
for slice in slices:
    c_r = slice[0].c
    c_t = slice[1].c
    y_r = slice[0].y
    y_t = slice[1].y
    a_coeff = (c_r - c_t) / (y_r - y_t)
    b_coeff = c_r - a_coeff * y_r
    chord_sqr = lambda y: (a_coeff * y + b_coeff) ** 2
    d_area = (c_r + c_t) * (y_t - y_r) / 2.
    integral = integral + integrate.quad(chord_sqr , y_r, y_t)[0]
    area = area + d_area








mac_slice = (1/area) * integral
print(mac_slice)
