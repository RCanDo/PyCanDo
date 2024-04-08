import matplotlib.pyplot as plt


def set_style(style='', style_default=path_wd+'/black.mplstyle', c='', c_default='y'):
    '''
    c : color;

    style : mpl.style
    '''
    global path_wd

    if c=='': c=c_default
    if style=='': style=style_default  ##'dark_background'

    if style in [ 'dark_background', path_wd+'/black.mplstyle' ]:
        c = c.replace('b','y').replace('k','w')

    plt.style.use(style)

    return style, c
