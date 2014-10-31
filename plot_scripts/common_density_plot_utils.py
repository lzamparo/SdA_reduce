# coding: utf-8
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt

from scipy.stats import gaussian_kde
import husl

def husl_gen():
    '''Generate random set of HUSL colors, one dark, one light'''
    hue = np.random.randint(0, 360)
    saturation, lightness = np.random.randint(0, 100, 2)
    husl_dark = husl.husl_to_hex(hue, saturation, lightness/3)
    husl_light = husl.husl_to_hex(hue, saturation, lightness)
    return husl_dark, husl_light

def rstyle(ax): 
    '''Styles x,y axes to appear like ggplot2
    Must be called after all plot and axis manipulation operations have been 
    carried out (needs to know final tick spacing)
    '''
    #Set the style of the major and minor grid lines, filled blocks
    ax.grid(True, 'major', color='w', linestyle='-', linewidth=1.4)
    ax.grid(True, 'minor', color='0.99', linestyle='-', linewidth=0.7)
    ax.patch.set_facecolor('0.90')
    ax.set_axisbelow(True)
    
    #Set minor tick spacing to 1/2 of the major ticks
    ax.xaxis.set_minor_locator((pylab.MultipleLocator((plt.xticks()[0][1]
                                -plt.xticks()[0][0]) / 2.0 )))
    ax.yaxis.set_minor_locator((pylab.MultipleLocator((plt.yticks()[0][1]
                                -plt.yticks()[0][0]) / 2.0 )))
    
    #Remove axis border
    for child in ax.get_children():
        if isinstance(child, matplotlib.spines.Spine):
            child.set_alpha(0)
       
    #Restyle the tick lines
    for line in ax.get_xticklines() + ax.get_yticklines():
        line.set_markersize(5)
        line.set_color("gray")
        line.set_markeredgewidth(1.4)
    
    #Remove the minor tick lines    
    for line in (ax.xaxis.get_ticklines(minor=True) + 
                 ax.yaxis.get_ticklines(minor=True)):
        line.set_markersize(0)
    
    #Only show bottom left ticks, pointing out of axis
    plt.rcParams['xtick.direction'] = 'out'
    plt.rcParams['ytick.direction'] = 'out'
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

def rfill_between(ax, x_range, dist, label, **kwargs):
    '''Create a distribution fill with default parameters to resemble ggplot2
    kwargs can be passed to change other parameters
    '''
    husl_dark_hex, husl_light_hex = husl_gen()
    defaults = {'color': husl_dark_hex,
                'facecolor': husl_light_hex,
                'linewidth' : 2.0, 
                'alpha': 0.1}
                
    for x,y in defaults.iteritems():
        kwargs.setdefault(x, y)
    
    ax.plot(x_range, dist, label=label,  antialiased=True)       
    return ax.fill_between(x_range, dist, **kwargs)

def make_x_axis(vals,granularity=500):
    ''' Take an Series of a DataFrame, return an appropriately scaled x-axis sampled at granularity points for plotting '''
    return np.linspace(vals.min(), vals.max(), granularity)

def make_kde(vals):
    ''' Return a scipy.stats.gaussian_kde object, but shrink the default bandwidth '''
    gkde = gaussian_kde(vals)
    gkde.silverman_factor()
    return gkde