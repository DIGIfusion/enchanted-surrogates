from scipy import interpolate
import numpy as np


def interp(xin,yin,xnew):
    """
    xin: x variable input
    yin: y variable input
    xnew: new x grid on which to interpolate
    yout: new y interpolated on xnew
    """

    #splrep returns a knots and coefficients for cubic spline
    rho_tck = interpolate.splrep(xin,yin)
    #Use these knots and coefficients to get new y
    yout = interpolate.splev(xnew,rho_tck,der=0)

    return yout

def interp_lin(xin,yin,xnew):
    """
    xin: x variable input
    yin: y variable input
    xnew: new x grid on which to interpolate
    yout: new y interpolated on xnew
    """

    if xnew[0] < xin[0]:
       low_index = np.argmin(abs(xnew-xin[0]))
       if xnew[low_index] < xin[0]:
           low_index += 1
    else:
       low_index = 0
    if xnew[-1] > xin[-1]:
       high_index = np.argmin(abs(xnew-xin[-1]))
       if xnew[high_index] > xin[0]:
           high_index -= 1
    else:
        high_index = -1

    ynew = interpolate.interp1d(xin,yin)
    yout = np.zeros(len(xnew))
    yout[low_index:high_index] = ynew(xnew[low_index:high_index])

    return yout


def full_interp(func_xin,xin,xconv,yconv,yout):
    """
    Takes function func_xin on grid xin and outputs the function on yout grid
    func_xin: function to interpolate
    xin: grid corresponding to func_xin
    xconv: xgrid for conversion
    yconv: ygrid for conversion
    yout: output grid
    """

    #If necessary truncate func_xin onto correct range
    if xin[0] < xconv[0]:
        low_index = np.argmin(abs(xconv-xin[0]))
    else:
        low_index = 0
    if xin[-1] > xconv[-1]:
        high_index = np.argmin(abs(xconv-xin[-1]))
    else:
        high_index = -1

    if high_index == -1:
        func_xin = func_xin[low_index:]
    else:
        func_xin = func_xin[low_index:high_index]

    func_xconv = interp(xin,func_xin,xconv)
    func_yout = interp(yconv,func_xconv,yout)

    return func_yout

def full_interp_lin(func_xin,xin,xconv,yconv,yout):
    """
    Takes function func_xin on grid xin and outputs the function on yout grid
    func_xin: function to interpolate
    xin: grid corresponding to func_xin
    xconv: xgrid for conversion
    yconv: ygrid for conversion
    yout: output grid
    """

    func_xconv = interp_lin(xin,func_xin,xconv)
    func_yout = interp_lin(yconv,func_xconv,yout)

    return func_yout


