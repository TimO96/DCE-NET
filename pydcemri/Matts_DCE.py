from matplotlib.pylab import *
import numpy as np

def enhance(signal,delay=20,sds=1,percentage=10):
    stds=np.std(signal[:,:delay-5],axis=1)
    means=np.mean(signal[:,:delay-5],axis=1)
    cutoff=np.array(means+sds*stds)
    selects = signal[:,delay+5:] < np.repeat(np.expand_dims(cutoff,1),np.shape(signal)[1]-delay-5,axis=1)
    selected = np.sum(selects,1) < (percentage/100*(np.shape(signal)[1]-delay-5))
    return selected

def aif(Hct = 0.4):
    aif = {'ab': (0.8959 / (1. - Hct)), 'mb': 25.5671, 'ae': (1.3354 / (1. - Hct)), 'me': 0.2130, 't0': 0.0}
    tb = 2 * np.pi / aif['mb']
    aif['ab'] = aif['ab'] / tb
    sc = tb * SpecialCosineExp(tb * aif['me'], tb * aif['mb'])
    aif['ae'] = aif['ae'] / aif['ab'] / sc
    return aif

def aif_conversion(Hct, hp):
    aif = {'ab': (hp.aif.a1 / (1. - Hct)), 'mb': hp.aif.m1, 'ae': (hp.aif.a2 / (1. - Hct)), 'me': hp.aif.m2, 't0': 0.0}
    tb = 2 * np.pi / aif['mb']
    aif['ab'] = aif['ab'] / tb
    sc = tb * SpecialCosineExp(tb * aif['me'], tb * aif['mb'])
    aif['ae'] = aif['ae'] / aif['ab'] / sc
    return aif

def Cosine4AIF_ExtKety(t,aif,ke,dt,ve,vp):
    # offset time array
    t = t - aif['t0'] - dt

    cpBolus = aif['ab']*CosineBolus(t,aif['mb'])
    cpWashout = aif['ab']*aif['ae']*ConvBolusExp(t,aif['mb'],aif['me'])
    ceBolus = ke*aif['ab']*ConvBolusExp(t,aif['mb'],ke)
    ceWashout = ke*aif['ab']*aif['ae']*ConvBolusExpExp(t,aif['mb'],aif['me'],ke)

    cp = cpBolus + cpWashout
    ce = ceBolus + ceWashout

    ct = zeros(shape(t))
    ct[t > 0] = vp * cp[t > 0] + ve * ce[t > 0]

    return ct

def CosineBolus(t,m):
    z = array(m * t)
    I1 = (z < 0.2) & (z > 0)
    I2 = (z >= 0.2) & (z < (2 * pi))

    y = zeros(shape(t))
    if any(I1):
        z2 = square(z[I1])
        y[I1] = multiply(z2 / (1 * 2), multiply(1 - z2 / (3 * 4), multiply(1 - z2 / (5 * 6), multiply(1 - z2 / (7 * 8), (1 - z2 / (9 * 10))))))

    y[I2] = 1 - cos(z[I2])

    return y

def ConvBolusExp(t,m,k):
    tB = 2 * pi / m
    tB=tB
    t=array(t)
    I1 = (t > 0) & (t < tB)
    I2 = t >= tB

    y = zeros(shape(t))

    y[I1] = multiply(t[I1],SpecialCosineExp(k*t[I1], m*t[I1]))
    y[I2] = tB*SpecialCosineExp(k*tB, m*tB)*exp(-k*(t[I2]-tB))

    return y


def ConvBolusExpExp(t,m,k1,k2):
    tol = 1e-4

    tT = tol / abs(k2 - k1)
    tT = array(tT)
    Ig = (t > 0) & (t < tT)
    Ie = t >= tT
    y = zeros(shape(t))

    y[Ig] = ConvBolusGamma(t[Ig], m, 0.5 * (k1 + k2))
    y1 = ConvBolusExp(t[Ie], m, k1)
    y2 = ConvBolusExp(t[Ie], m, k2)
    y[Ie] = (y1 - y2) / (k2 - k1)

    return y

def ConvBolusGamma(t,m,k):
    tB = 2 * pi / m
    tB=array(tB)
    y = zeros(shape(t))
    I1 = (t > 0) & (t < tB)
    I2 = t >= tB

    ce = SpecialCosineExp(k * tB, m * tB)
    cg = SpecialCosineGamma(k * tB, m * tB)

    y[I1] = square(t[I1]) * SpecialCosineGamma(k * t[I1], m * t[I1])
    y[I2] = tB * multiply(((t[I2] - tB) * ce + tB * cg), exp(-k * (t[I2] - tB)))

    return y

def SpecialCosineGamma(x, y):
    ex = 1
    ey = 0.8

    I0  = (x>ex)  & (y>ey)
    Ix  = (x<=ex) & (y>ey)
    Iy  = (x>ex)  & (y<=ey)
    Ixy = (x<=ex) & (y<=ey)

    expTerm=zeros(shape(x))
    trigTerm=zeros(shape(x))
    f=zeros(shape(x))

    I0y = I0 | Iy
    expTerm[I0y] = multiply(3+divide(square(y[I0y]), square(x[I0y])), (1 - exp(-x[I0y]))) \
                   - multiply(divide(square(y[I0y]) + square(x[I0y]),x[I0y]), exp(-x[I0y]))
    yE2 = square(y[Ix])
    xE = x[Ix]

    if any(Ix):
        expTerm[Ix] = yE2/2 - 0 - multiply(xE/1, (yE2/3 - 2 - multiply(xE/2, (yE2/4 - 1- multiply(xE/3, (yE2/5- multiply(xE/4, (yE2/6 + 1- multiply(xE/5,(yE2/7 + 2 - multiply(xE / 6, (yE2 / 8 + 3- multiply(xE/7,(yE2/9 + 4 - multiply(xE/8,(yE2/10 + 5- multiply(xE/9,(yE2/11 + 6- multiply(xE/10,(yE2/12 + 7- multiply(xE/11,(yE2/13 + 8 - multiply(xE / 12, (yE2 / 14 + 9 -multiply(xE / 13, (yE2 / 15 + 10 - multiply(xE / 14, (yE2 / 16 + 11 - multiply(xE / 15, (yE2 / 17 + 12 - multiply(xE / 16, (yE2 / 18 + 13 - multiply(xE / 17, (yE2 / 19 + 14 - multiply(xE / 18, (yE2 / 20 + 15 - multiply(xE / 19, (yE2 / 21 + 16- multiply(xE / 20, (yE2 / 22 + 17))))))))))))))))))))))))))))))))))))))))

    I0x = (I0 | Ix)
    trigTerm[I0x] = multiply((square(x[I0x]) - square(y[I0x])), (1 - cos(y[I0x]))) - multiply(multiply(2 * x[I0x],y[I0x]), sin(y[I0x]))
    yT2 = square(y[Iy])
    xT2 = square(x[Iy])
    xT = 4 * x[Iy]

    if any(Iy):
        trigTerm[Iy] = multiply(yT2 / (1 * 2), (xT2 - 1 * xT + 0 - multiply(yT2 / (3 * 4), (xT2 - 2 * xT + 2 * 2 * 3 - multiply( yT2 / (5 * 6), (xT2 - 3 * xT + 2 * 3 * 5 - multiply( yT2 / (7 * 8), (xT2 - 4 * xT + 2 * 4 * 7 - multiply(yT2 / (9 * 10), (xT2 - 5 * xT + 2 * 5 * 9 - multiply(yT2 / (11 * 12), (xT2 - 6 * xT + 2 * 6 * 11 - multiply( yT2 / (13 * 14), (xT2 - 7 * xT + 2 * 7 * 13 - multiply( yT2 / (15 * 16), (xT2 - 8 * xT + 2 * 8 * 15 - multiply(yT2 / (17 * 18), (xT2 - 9 * xT + 2 * 9 * 17 - multiply(yT2 / (19 * 20), (xT2 - 10 * xT + 2 * 10 * 19))))))))))))))))))))

    I0xy = I0 | Ix | Iy
    f[I0xy] = divide((trigTerm[I0xy] + multiply(square(y[I0xy]), expTerm[I0xy])), square(square(y[I0xy]) + square(x[I0xy])))
    yH2 = square(y[Ixy])
    xH = x[Ixy]

    if any(Ixy):
        f[Ixy] = multiply(yH2 / (2 * 3 * 4), (1 - multiply(xH / 5, (2 - multiply(xH / 6, (3 - multiply(xH / 7, (4 - multiply(xH / 8, (5 - multiply(xH / 9, (6 - multiply(xH / 10, (7 - multiply(xH / 11, (8 - multiply(xH / 12, (9 - multiply(xH / 13, (10 - multiply(xH / 14, (11 - multiply(xH / 15, (12 - multiply(xH / 16, (13 - multiply(xH / 17, (14 - multiply(xH / 18, (15 - multiply(xH / 19, (16 - xH / 20)))))))))))))))))))))))))))))) - multiply(yH2 / (5 * 6), (1 - multiply(xH / 7, (2 - multiply(xH / 8, (3 - multiply(xH / 9, (4 - multiply(xH / 10, (5 - multiply(xH / 11, (6 - multiply(xH / 12, (7 - multiply(xH / 13, (8 - multiply(xH / 14, (9 - multiply(xH / 15, (10 - multiply(xH / 16, ( 11 - multiply(xH / 17, (12 - multiply(xH / 18, ( 13 - multiply(xH / 19, (14 - multiply(xH / 20, (15 - multiply(xH / 21, (16 - xH / 22)))))))))))))))))))))))))))))) - multiply(yH2  / (7 * 8), (1 - multiply(xH / 9, (2 - multiply(xH / 10, ( 3 - multiply(xH / 11, (4 - multiply(xH / 12, (5 - multiply(xH / 13, (6 - multiply(xH / 14, (7 - multiply(xH / 15, (8 - multiply(xH / 16, (9 - multiply(xH / 17, (10 - multiply(xH / 18, (11 - multiply(xH / 19, (12 - multiply(xH / 20, ( 13 - multiply(xH / 21, (14 - multiply(xH / 22, (15 - multiply(xH / 23, (16 - xH / 24)))))))))))))))))))))))))))))) - multiply(yH2  / (9 * 10), (1 - multiply(xH / 11, ( 2 - multiply(xH / 12, (3 - multiply(xH / 13, (4 - multiply(xH / 14, (5 - multiply(xH / 15, (6 - multiply(xH / 16, (7 - multiply(xH / 17, (8 - multiply(xH / 18, (9 - multiply(xH / 19, (10 - multiply(xH / 20, (11 - multiply(xH / 21, (12 - multiply(xH / 22, (13 - multiply(xH / 23, (14 - multiply(xH / 24, (15 - multiply(xH / 25, (16 - xH / 26)))))))))))))))))))))))))))))) - multiply(yH2  / (11 * 12), (1 - multiply(xH / 13, (2 - multiply(xH / 14, (3 - multiply(xH / 15, (4 - multiply(xH / 16, (5 - multiply(xH / 17, ( 6 - multiply(xH / 18, (7 - multiply(xH / 19, (8 - multiply(xH / 20, (9 - multiply(xH / 21, (10 - multiply(xH / 22, (11 - multiply(xH / 23, (12 - multiply(xH / 24, (13 - multiply(xH / 25, (14 - multiply(xH / 26, (15 - multiply(xH / 27, (16 - xH / 28)))))))))))))))))))))))))))))) - multiply(yH2  / (13 * 14), (1 - multiply(xH / 15, (2 - multiply(xH / 16, (3 - multiply(xH / 17, (4 - multiply(xH / 18, (5 - multiply(xH / 19, (6 - multiply(xH / 20, (7 - multiply(xH / 21, (8 - multiply(xH / 22, (9 - multiply(xH / 23, (10 - multiply(xH / 24, (11 - multiply(xH / 25, (12 - multiply(xH / 26, (13 - multiply(xH / 27, (14 - multiply(xH / 28, (15 - multiply(xH / 29, (16 - xH / 30)))))))))))))))))))))))))))))) - multiply(yH2  / (15 * 16), (1 - multiply(xH / 17, (2 - multiply(xH / 18, (3 - multiply(xH / 19, (4 - multiply(xH / 20, (5 - multiply(xH / 21, (6 - multiply(xH / 22, (7 - multiply(xH / 23, (8 - multiply(xH / 24, (9 - multiply(xH / 25, (10 - multiply(xH / 26, (11 - multiply(xH / 27, (12 - multiply(xH / 28, (13 - multiply(xH / 29, (14 - multiply(xH / 30, (15 - multiply(xH / 31, (16 - xH / 32))))))))))))))))))))))))))))))))))))))))))))

    return f

def SpecialCosineExp(x,y):
    ex = 0.6
    ey = 0.45
    x=array(x)
    y=array(y)
    I0  = (x>ex)  & (y>ey)
    Ix  = (x<=ex) & (y>ey)
    Iy  = (x>ex)  & (y<=ey)
    Ixy = (x<=ex) & (y<=ey)

    expTerm=zeros(shape(x))
    trigTerm=zeros(shape(x))
    f=zeros(shape(x))

    I0y = I0 | Iy
    expTerm[I0y] = divide((1 - exp(-x[I0y])), x[I0y])
    xH = x[Ix]
    if any(Ix):
        expTerm[Ix] = 1 - multiply(xH / 2, (1 - multiply(xH / 3, (1 - multiply(xH / 4, (1 - multiply(xH / 5, (1 - multiply(xH / 6, (1 - multiply(xH / 7, (1 - multiply(xH / 8, (1 - multiply(xH / 9, (1 - multiply(xH / 10, (1 - multiply(xH / 11, (1 - multiply(xH / 12, (1 - multiply(xH / 13, (1 - multiply(xH / 14, (1 - xH / 15))))))))))))))))))))))))))

    I0x = I0 | Ix
    trigTerm[I0x] = multiply(x[I0x], (1 - cos(y[I0x]))) - multiply(y[I0x], sin(y[I0x]))
    yH2 = square(y[Iy])
    xH = x[Iy]

    if any(Iy):
        trigTerm[Iy] = multiply(yH2 / (1 * 2), (xH - 2 - multiply(yH2 / (3 * 4), (xH - 4 - multiply(yH2 / (5 * 6), (xH - 6 - multiply(yH2 / (7 * 8), (xH - 8 - multiply(yH2 / (9 * 10), (xH - 10 - multiply(yH2 / (11 * 12), (xH - 12 - multiply(yH2 / (13 * 14), (xH - 14))))))))))))))

    I0xy = I0 | Ix | Iy
    f[I0xy] = divide((trigTerm[I0xy] + multiply(square(y[I0xy]), expTerm[I0xy])), (square(x[I0xy]) + square(y[I0xy])))

    xH = x[Ixy]
    yH2 = square(y[Ixy])

    if any(Ixy):
        f[Ixy] = multiply(yH2 / (2 * 3), (1 - multiply(xH / 4, (1 - multiply(xH / 5, (1 - multiply(xH / 6, (1 - multiply(xH / 7, (1 - multiply(xH / 8, (1 - multiply(xH / 9, (1 - multiply(xH / 10, (1 - multiply(xH / 11, (1 - multiply(xH / 12, (1 - multiply(xH / 13, (1 - multiply(xH / 14, (1 - xH / 15))))))))))))))))))))) - multiply(yH2 / (4 * 5), (1 - multiply(xH / 6, (1 - multiply(xH / 7, (1 - multiply(xH / 8, (1 - multiply(xH / 9, (1 - multiply(xH / 10, (1 - multiply(xH / 11, (1 - multiply(xH / 12, (1 - multiply(xH / 13, (1 - multiply(xH / 14, (1 - multiply(xH / 15, (1 - multiply(xH / 16, (1 - xH / 17))))))))))))))))))))) - multiply(yH2 / (6 * 7), (1 - multiply(xH / 8, (1 - multiply(xH / 9, (1 - multiply(xH / 10, (1 - multiply(xH / 11, (1 - multiply(xH / 12, (1 - multiply(xH / 13, (1 - multiply(xH / 14, (1 - multiply(xH / 15, (1 - multiply(xH / 16, (1 - multiply(xH / 17, (1 - multiply(xH / 18, (1 - xH / 19))))))))))))))))))))) - multiply(yH2 / (8 * 9), (1 - multiply(xH / 10, (1 - multiply(xH / 11, ( 1 - multiply(xH / 12, (1 - multiply(xH / 13, (1 - multiply(xH / 14, (1 - multiply(xH / 15, (1 - multiply(xH / 16, (1 - multiply(xH / 17, (1 - multiply(xH / 18, (1 - multiply(xH / 19, (1 - multiply(xH / 20, (1 - xH / 21))))))))))))))))))))) - multiply(yH2 / (10 * 11), ( 1 - multiply(xH / 12, (1 - multiply(xH / 13, (1 - multiply(xH / 14, (1 - multiply(xH / 15, (1 - multiply(xH / 16, (1 - multiply(xH / 17, (1 - multiply(xH / 18, (1 - multiply(xH / 19, (1 - multiply(xH / 20, (1 - multiply(xH / 21, (1 - multiply(xH / 22, (1 - xH / 23))))))))))))))))))))) - multiply(yH2 / (12 * 13), (1 - multiply(xH / 14, (1 - multiply(xH / 15, (1 - multiply(xH / 16, (1 - multiply(xH / 17, (1 - multiply(xH / 18, (1 - multiply(xH / 19, (1 - multiply(xH / 20, (1 - multiply(xH / 21, (1 - multiply(xH / 22, (1 - multiply(xH / 23, (1 - multiply(xH / 24, (1 - xH / 25)))))))))))))))))))))))))))))))))))))))

    return f