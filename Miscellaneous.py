__version__ = "1.0"

import numpy as np
from numpy import array as arr
from matplotlib.cm import get_cmap
from pandas import DataFrame

import IPython
import textwrap
import os
import ExpFile as exp


def printExperimentFileInfo(loc="J:/Data repository/New Data Repository/2020/February/February 20/Raw Data/", includeNotes=False):
    files = os.listdir(loc)
    def sortF(fn):
        try:
            return int(fn[5:-3])
        except ValueError:
            return -1
    sortedFiles = sorted(files, key = sortF)
    for filename in sortedFiles:
        if filename[:4] == "data":
            try:
                title, notes, lev = exp.getAnnotation(loc+filename, useBase=False)
                print(filename[5:-3], title)
                if includeNotes:
                    wrapper = textwrap.TextWrapper(initial_indent="\t", subsequent_indent='\t')
                    print(wrapper.fill(notes))
                    
            except RuntimeError:
                print(filename[5:-3], "FILE NOT ANNOTATED")
            except OSError:
                print(filename[5:-3], "Bad File")
                
                
def loopProgress(inc, total):
    IPython.display.clear_output(wait=True)
    print(round_sig_str(inc/total*100), '% Complete...')
    
def rebin(arr, new_shape):
    shape = (new_shape[0], arr.shape[0] // new_shape[0],
             new_shape[1], arr.shape[1] // new_shape[1])
    return arr.reshape(shape).mean(-1).mean(1)

def ratioToDb(ratio):
    print('Assuming input is ratio of powers.')
    return 10*np.log10(ratio)

def revList(l):
    return list(reversed(l))


def femtowattDetectorConversion(volts):
    """
    based on the newport model 2151 femtowatt detector, for 780-850 or so nm light, for DC gain.
    :return: power in W
    """
    # factor of 0.5 comes from approximate responsivity at this wavelength range.
    # 0.5 A/W photodetector sensitivity
    # 2*10**10 dc gain.
    convFactor = 0.5 * 2 * 10**10
    return volts / convFactor


def prefix(num):
    """
    Convert number to nearest numbers with SI prefixes.
    :param num: the number to convert
    """
    # determine which range it lies in, r1/r2 means reduction 1 or reduction 2
    divisors = [1e-24 * pow(10, 3 * x) for x in range(17)]
    prefixes = list(reversed(['Yotta (Y)', 'Zetta (Z)', 'Exa (E)', 'Peta (P)', 'Tera (T)', 'Giga (G)', 'Mega (M)',
                              'Kilo (K)', '', 'Milli (m)', 'Micro (mu)', 'Nano (n)', 'Pico (p)', 'Femto (f)',
                              'Atto (a)', 'Zepto (z)', 'Yocto (y)']))
    exp = np.floor(np.log10(np.abs(num)))
    if exp < 0:
        exp -= 3
    expIndex = int(exp / 3) + 8
    expIndex = 0 if expIndex < 0 else expIndex
    expIndex = len(prefixes)-1 if expIndex >= len(prefixes) else expIndex
    r1 = prefixes[expIndex]
    num1 = num / divisors[expIndex]
    if expIndex != len(prefixes):
        r2 = prefixes[expIndex + 1]
        num2 = num / divisors[expIndex + 1]
    else:
        num2 = None
    retStr = str(num1) + ' ' + r1
    if num2 is not None:
        retStr += '\nor\n' + str(num2) + ' ' + r2
    return retStr


def what(obj, callingLocals=locals()):
    """
    quick function to print name of input and value.
    If not for the default-Valued callingLocals, the function would always
    get the name as "obj", which is not what I want.

    :param obj: the object to print info for
    :param callingLocals: don't use, always should be locals().
    """
    name = "name not found"
    for k, v in list(callingLocals.items()):
        if v is obj:
            name = k
    if type(obj) == float:
        print(name, "=", "{:,}".format(obj))
    else:
        print(name, "=", obj)


def transpose(l):
    """
    Transpose a list.
    :param l: the list to be transposed
    :return: the tranposed list
    """
    return list(map(list, zip(*l)))


def getStats(data, printStats=False):
    """
    get some basic statistics about the input data, in the form of a pandas dataframe.
    :param data: the data to analyze
    :param printStats: an option to print the results
    :return: the dataframe containing the statistics
    """
    data = list(data)
    d = DataFrame()
    d['Avg'] = [np.mean(data)]
    d['len'] = [len(data)]
    d['min'] = [min(data)]
    d['max'] = [max(data)]
    d['std'] = [np.std(data)]
    d = d.transpose()
    d.columns = ['Stats']
    d = d.transpose()
    if printStats:
        print(d)
    return d


def getMarkers():
    return ['o','v','<','>','^','*','x','+','D']


def getColors(num, rgb=False, cmStr='nipy_spectral'):
    """
    Get an array of colors, typically to use for plotting.

    :param rgb: an option to return the colors as an rgb tuple instead of a hex.
    :param num: number of colors to get
    :return: the array of colors, hex or rgb (see above)
    """
    #rowSize = 8
    #maps = ['Greys', 'Blues', 'Greens', 'Reds', 'Oranges', 'copper', 'Purples', 'pink']
    #cmapRGB = get_cmap('brg', num-1)
    #c = []
    #for m in maps:
    #    cmapRGB = get_cmap(m, rowSize+1)
    #    for i in range(rowSize):
    #        c.append(cmapRGB(i+1)[:-1])
    cmapRGB = get_cmap(cmStr, num+1)
    c = [cmapRGB(i)[:-1] for i in range(num+1)][:]
    c = c[1:]
    
    if rgb:
        return c
    # the negative of the first color
    c2 = [tuple(arr((1, 1, 1)) - arr(color)) for color in c]
    c = ['#%02x%02x%02x' % tuple(int(255 * color[i]) for i in range(len(color))) for color in c]
    c2 = ['#%02x%02x%02x' % tuple(int(255 * color[i]) for i in range(len(color))) for color in c2]
    return c, c2


def round_sig(x, sig=3):
    """
    round a float to some number of significant digits
    :param x: the numebr to round
    :param sig: the number of significant digits to use in the rounding
    :return the rounded number, as a float.
    """
    if np.isnan(x):
        x = 0
    try:
        return round(x, sig-int(np.floor(np.log10(abs(x)+2*np.finfo(float).eps)))-1)
    except ValueError:
        print(abs(x))


def getExp(val):
    if val == 0:
        return 0
    return np.floor(np.log10(np.abs(val)))


def round_sig_str(x, sig=3):
    """
    round a float to some number of significant digits
    :param x: the numebr to round
    :param sig: the number of significant digits to use in the rounding
    :return the rounded number, as a string.
    """
    if sig<=0:
        return "0"
    if np.isnan(x):
        x = 0
    try:
        res = np.floor(np.log10(abs(x)+2*np.finfo(float).eps))
        if res == np.inf:
            res = 0
        num = round(x, sig-int(res)-1)
        decimals = sig-getExp(num)-1
        if decimals == float('inf'):
            decimals = 3
        if decimals <= 0:
            decimals = 0
        result = ("{0:."+str(int(decimals))+"f}").format(num)
        # make sure result has the correct number of significant digits given the precision.
        return result
    except ValueError:
        print(abs(x))


def errString(val, err, precision=None):
    """
    takes the input value and error and makes a nice error string. e.g.
    inputs of
    1.423, 0.086, 3 gives
    1.42(9)
    :param val:
    :param err:
    :param precision:
    :return:
    """
    if np.isinf(err) or np.isnan(err):
        err = 0
    if np.isinf(val) or np.isnan(val):
        return "?(?)"
    if err == 0:
        if precision is None:
            precision = 3
        return round_sig_str(val, precision) + '(0)'
    valE = getExp(val)
    # determine number of values of err to show.
    errE = getExp(err)
    if np.isinf(errE):
        errE = 0;
        #return  round_sig_str(val, precision) +'(?)'
    if precision is None:
        # determine first significant digit of error and use one more than that. 
        precision = int(valE - errE + 2)
    if np.isinf(valE):
        return "?(?)"
    try:
        num = int(errE-valE+precision)
        if num < 0:
            num = 0
        expFactor = -errE + num - 1
    except ValueError:
        print('bad number!')
        num = 0
        expFactor=0
    if expFactor <= 0:
        expFactor = 0
    errNum = int(round(err*10**expFactor))
    result = round_sig_str(val, precision) + '(' + round_sig_str(errNum, num) + ')'
    return result


def dblErrString(val, err1, err2, precision=None):
    """
    takes the input value and error and makes a nice error string with two error measures, 
    e.g. an ensemble std and an mean error, in the format num(err1)(err2) e.g. 0.514(40)(3)
    :return:
    """
    if np.isinf(val) or np.isnan(val):
        return "?(?)"
    if err1 == 0 or err2 == 0:
        if precision is None:
            precision = 3
        return round_sig_str(val, precision) + '(0)(0)'
    valE = getExp(val)
    # determine number of values of err to show.
    errE1 = getExp(err1)
    errE2 = getExp(err2)
    errE_m = min((errE1, errE2))
    if errE_m == float("NaN"):
        errE_m = 0
    if valE == float('Inf') or valE == float('-Inf') or np.isnan(valE):
        return "?(?)"
    if precision is None:
        # determine first significant digit of error and use one more than that. 
        precision = int(valE - errE_m + 2)
    if np.isinf(errE_m):
        return  round_sig_str(val, precision) +'(?)'
    try:
        num = int(errE_m-valE+precision)
        num1 = int(errE1-valE+precision)
        num2 = int(errE2-valE+precision)
        if num < 0:
            num = 0
        expFactor = -errE_m + num - 1
    except ValueError:
        print('bad number!')
        num = 0
        expFactor=0
    if expFactor <= 0:
        expFactor = 0
    errNum1 = int(round(err1*10**expFactor))
    errNum2 = int(round(err2*10**expFactor))
    result = round_sig_str(val, precision) + '(' + round_sig_str(errNum1, num1) + ')' + '(' + round_sig_str(errNum2, num2) + ')'
    return result

def asymErrString(val, err1, err2, precision=None):
    """
    takes the input value with asymmetric error values makes a nice error string, 
    e.g. r'$0.542^((3))_((20))$'
    The string is meant to be rendered using latex. 
    :return:
    """
    if np.isinf(val) or np.isnan(val):
        return "?(?)"
    if err1 == 0 or err2 == 0:
        if precision is None:
            precision = 3
        return round_sig_str(val, precision) + '(0)(0)'
    valE = getExp(val)
    # determine number of values of err to show.
    errE1 = getExp(err1)
    errE2 = getExp(err2)
    errE_m = min((errE1, errE2))
    if errE_m == float("NaN"):
        errE_m = 0
    if valE == float('Inf') or valE == float('-Inf') or np.isnan(valE):
        return "?(?)"
    if precision is None:
        # determine first significant digit of error and use one more than that. 
        precision = int(valE - errE_m + 2)
    if np.isinf(errE_m):
        return  round_sig_str(val, precision) +'(?)'
    try:
        num = int(errE_m-valE+precision)
        num1 = int(errE1-valE+precision)
        num2 = int(errE2-valE+precision)
        if num < 0:
            num = 0
        expFactor = -errE_m + num - 1
    except ValueError:
        print('bad number!')
        num = 0
        expFactor=0
    if expFactor <= 0:
        expFactor = 0
    errNum1 = int(round(err1*10**expFactor))
    errNum2 = int(round(err2*10**expFactor))
    result = r'$'+round_sig_str(val, precision) + '^{(' + round_sig_str(errNum1, num1) + ')}' + '_{(' + round_sig_str(errNum2, num2) + ')}$'
    return result


def dblAsymErrString(val, err_L1, err_U1, err_L2, err_U2, precision=None):
    """
    takes the input value with asymmetric error values makes a nice error string, 
    e.g. r'$0.542^((3))_((20))$'
    The string is meant to be rendered using latex. 
    :return:
    """
    if np.isinf(val) or np.isnan(val):
        return "?(?)"
    if err_L1 == 0 or err_U1 == 0 or err_L2 == 0 or err_U2 == 0:
        if precision is None:
            precision = 3
        return round_sig_str(val, precision) + '(0)(0)'
    valE = getExp(val)
    # determine number of values of err to show.
    errLE1 = getExp(err_L1)
    errLE2 = getExp(err_L2)
    errUE1 = getExp(err_U1)
    errUE2 = getExp(err_U2)
    errE_m = min((errLE1, errLE2, errUE1, errUE2))
    if errE_m == float("NaN"):
        errE_m = 0
    if valE == float('Inf') or valE == float('-Inf') or np.isnan(valE):
        return "?(?)"
    if precision is None:
        # determine first significant digit of error and use one more than that. 
        precision = int(valE - errE_m + 2)
    if np.isinf(errE_m):
        return  round_sig_str(val, precision) +'(?)'
    try:
        num = int(errE_m-valE+precision)
        numL1 = int(errLE1-valE+precision)
        numL2 = int(errLE2-valE+precision)
        numU1 = int(errUE1-valE+precision)
        numU2 = int(errUE2-valE+precision)
        if num < 0:
            num = 0
        expFactor = -errE_m + num - 1
    except ValueError:
        print('bad number!')
        num = 0
        expFactor=0
    if expFactor <= 0:
        expFactor = 0
    errNumL1 = int(round(err_L1*10**expFactor))
    errNumL2 = int(round(err_L2*10**expFactor))
    errNumU1 = int(round(err_U1*10**expFactor))
    errNumU2 = int(round(err_U2*10**expFactor))
    result = (r'$'+round_sig_str(val, precision) + '^{(' + round_sig_str(errNumU1, numU1) + ')('+round_sig_str(errNumU2, numU2)+')}' 
              + '_{(' + round_sig_str(errNumL1, numL1) + ')('+ round_sig_str(errNumL2, numL2) + ')}$')
    return result

def reportProgress(num, total):
    print( round_sig_str(num/total*100) + '%                     ',  end='\r' )
    #IPython.display.clear_output(wait=True)
