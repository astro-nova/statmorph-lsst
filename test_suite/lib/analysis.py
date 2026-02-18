import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import ndimage as ndi
from scipy.stats import gaussian_kde
from sklearn.feature_selection import mutual_info_regression
from skimage import transform as T

# Suppress SyntaxWarning: invalid escape sequence '\m'
warnings.simplefilter("ignore", SyntaxWarning)

##### Step 1: Finding the baselines ########
def _find_baseline_row(rows, res_col, base_res, snr_col, base_snr):
    """Given a group of rows, find which of these matches best the baseline resolution and SNR. 
    Args:
        rows (pd.DataFrame): DataFrame containing the rows to search through
        res_col (str): Name of the column containing resolution values
        res_value (float): Target resolution value to match
        snr_col (str): Name of the column containing SNR values
        snr_value (float): Target SNR value to match
    Returns:
        pd.Series: The row that best matches the target resolution and SNR
    """

    # Define a good enough match. 0.3dex is a factor of 2 in terms of effective resolution
    res_diff_lim = 30 if res_col == 'pxscale_pc' else 0.3
    snr_diff_lim = 0.3 # dex in terms of log snr/px

    # If the baseline is surface brightness, just do a simple match on sblim 
    galaxy = rows.iloc[0].galaxy
    if snr_col == 'sblim':
        # Find everything with matching SB
        subset = rows[rows.sblim == base_snr]
        if len(subset) == 0:
            print(f"No matching SB rows found for {galaxy}.")
            return -1
            
        # Now find resolution matches
        closest_idx = (subset[res_col]-base_res).abs().idxmin()
        diff = np.abs(subset.loc[closest_idx, res_col] - base_res)

        # Warn if resolution is too far
        if diff > res_diff_lim:
            print(f"Warning: No close resolution match found for {galaxy} (closest is {subset.loc[closest_idx, res_col]:0.2} , target is {base_res:0.2f}).")
            return -1
            

    # If matching on SNR, we need to do a fuzzy match on resolution and SNR. Still do this in the same order
    elif snr_col == 'log_snr':
            # Find everything with matching SB
            subset = rows[np.abs(rows.log_snr - base_snr) <= snr_diff_lim]
            if len(subset) == 0:
                print(f"No matching SNR rows found for {galaxy}.")
                return -1
                
            
            # Now find resolution matches
            closest_idx = (subset[res_col]-base_res).abs().idxmin()
            diff = np.abs(subset.loc[closest_idx, res_col] - base_res)

            # Warn if resolution is too far
            if diff > res_diff_lim:
                print(f"Warning: No close resolution match found for {galaxy} (closest is {subset.loc[closest_idx, res_col]:0.2f}, target is {base_res:0.2f}).")
                return -1
                
    return closest_idx

def get_baseline(df_raw, res_col, base_res, snr_col, base_snr, param):
    """Given a DataFrame with multiple rows per galaxy, find the baseline row for each galaxy.
    Args:
        df (pd.DataFrame): DataFrame containing the rows to search through
        res_col (str): Name of the column containing resolution values
        base_res (float): Target resolution value to match
        snr_col (str): Name of the column containing SNR values
        base_snr (float): Target SNR value to match
    Returns:
        pd.DataFrame: DataFrame containing the baseline row for each galaxy
    """
    # Convert radius and centroid columns to kpc
    df = df_raw.copy()
    if 'base_rp_kpc' in df.columns:
        df.drop(columns=['base_rp_kpc'], inplace=True)
    if f'base_{param}' in df.columns:
        df.drop(columns=[f'base_{param}', f'd_{param}', f'f_{param}'], inplace=True)

    radii = ['r20','r50','r80','rmax_circ','rmax_ellip','rhalf_circ','rhalf_ellip','sersic_rhalf','rpetro_circ','rpetro_ellip']
    centers = ['xc_centroid', 'yc_centroid', 'xc_asymmetry', 'yc_asymmetry', 'sersic_xc', 'sersic_yc']
    for col in radii+centers:
        if f'{col}_kpc' in df.columns: break
        df[f'{col}_kpc'] = df[col]*df['pxscale_pc']/1000

    # Get all the baselines
    fun = lambda x: _find_baseline_row(x, res_col, base_res, snr_col, base_snr)
    baselines = df.groupby(by='galaxy').apply(fun)

    # Galaxies without a baseline
    no_match = baselines[baselines == -1].index.values
    baselines = baselines[baselines != -1]

    # Assign baseline flag
    df['baseline'] = df.apply(lambda x: x.name in baselines.values, axis=1)
    df = df[~df.galaxy.isin(no_match)]

    # For every row, find the baseline galaxy row, and fetch the parameter value at the baseline. Assign to all rows of that galaxy
    # Also store the baseline Petrosian radius
    base_params =  df.loc[baselines.values][['galaxy', param, 'rpetro_circ_kpc']]
    base_params = base_params.rename(columns={param:f'base_{param}', 'rpetro_circ_kpc':'base_rp_kpc'})
    df = pd.merge(df, base_params, left_on='galaxy', right_on='galaxy', how='left')

    # Calculate effective resolution for every row
    df['nres'] = 1000*df.base_rp_kpc/df.pxscale_pc
    df['log_nres'] = np.log10(df['nres'])

    # Calculate errors
    df[f'd_{param}'] = df[param] - df[f'base_{param}']
    df[f'f_{param}'] = df[param] / df[f'base_{param}']

    return df, no_match


##### Step 2: Mutual info regression #######
def get_important_features(df, param, paramlabel,
                           cols=['log_nres','pxscale_pc','sblim','log_snr'],
                           discrete=[False, False, False, False]):

    # Baseline parameter and random as extra columns
    cols = [f'base_{param}'] + cols + ['random']
    discrete = [True] + discrete + [False]

    # Random number column for reference
    df['random'] = np.random.rand(len(df))

    # Run mutual information regression
    info = mutual_info_regression(df[cols], df[param], n_neighbors=25, discrete_features=discrete)

    # Make a plot
    labels = [f'Base {paramlabel}',  r'$\mathcal{R}_{\mathrm{eff}}$', '$\mathcal{R}$', '$\mu_0$', r'$\langle$SNR$\rangle$', 'Random']
    fig = plt.figure(figsize=(4,2.5))
    ax = plt.axes()
    ax.plot(labels, info, 'ko--', lw=0.5, ms=3)
    ax.tick_params('x', rotation=15)
    ax.set_ylabel('Importance')
    ax.set_yscale('log')

    # Return the two most important features
    snr_col = cols[3+np.argmax(info[3:5])]
    res_col = cols[1+np.argmax(info[1:3])]
    df.drop(columns=['random'], inplace=True)

    # Get new baseline values
    base_snr = df[df.baseline][snr_col].median()
    base_res = df[df.baseline][res_col].median()

    return snr_col, base_snr, res_col, base_res, fig

##### Step 3: Plot the bias grid #######
def plot_bias_grid(df, paramlabel, snr_col, base_snr, res_col, base_res, nbins=11, min_rows=20,
                   vmin=None, vmax=None):

    ##################### Calculating the grid #####################
    # Generate bins of different resolutions and SNR
    resmin, resmax = df[res_col].quantile(q=[0.005, 0.95]).values
    snrmin, snrmax = df[snr_col].quantile(q=[0.02, 0.95]).values

    res_bins = np.linspace(resmin, resmax, nbins)
    snr_bins = np.linspace(snrmin, snrmax, nbins)

    # Within each bin, calculate the 16, 50, 84 percentiles of the parameter
    Qs = np.nan*np.ones((3,len(res_bins)-1, len(snr_bins)-1))
    Sigmas = np.nan*np.ones((len(res_bins)-1, len(snr_bins)-1))            
    for i, res in enumerate(res_bins[:-1]):
        for j, snr in enumerate(snr_bins[:-1]):
            # Select rows in this bin
            rows = df[ 
                (df[res_col] >= res) & (df[res_col] < res_bins[i+1]) & 
                (df[snr_col] >= snr) & (df[snr_col] < snr_bins[j+1])         
            ]
            # If not enough rows, return nan
            if len(rows) > 20:
                qs = np.quantile(rows['err'], q=[0.16,0.5,0.84])
                Qs[:,j,i] = qs
                Sigmas[j,i] = np.std(rows['err'])
            else:
                Qs[:,j,i] = [-np.nan, np.nan, np.nan]
                Sigmas[j,i] = np.nan

    # Recalculate the grid based on bin centers
    dres = np.diff(res_bins)[0]
    dsnr = np.diff(snr_bins)[0]
    res_cent = res_bins[:-1] + dres
    snr_cent = snr_bins[:-1] + dsnr
    Res, Snr = np.meshgrid(res_cent, snr_cent)
    nbins -= 1

    # Calculate "sigma" - the difference between 84th and 16th percentiles
    dQ = (Qs[2]-Qs[0])/2
    dQ[dQ == 0] = np.nan


    ##################### Generating the plot #####################
    # Calculating the colormap
    if vmin is None: vmin = np.nanquantile(Qs[1].flatten(), 0.15)
    if vmax is None: vmax = np.nanquantile(Qs[1].flatten(), 0.85)
    dv = (vmax-vmin)
    vmin -= 0.1*dv
    vmax += 0.1*dv
    if (vmin<0) and (vmax>0):
        midpoint = (0-vmin)/(vmax-vmin)
    else:
        midpoint = 0.5
    colors = [(0,'#F45B69'),  (midpoint, '#fff'), (1,'#06BCC1')]
    my_cmap = mpl.colors.LinearSegmentedColormap.from_list('my_cmap', colors, N=100)


    fig = plt.figure(figsize=(4,3.5))
    ax = plt.axes()

    ###### Plotting the bias as a colormesh ######
    smoothed = ndi.gaussian_filter(Qs[1],2,truncate=2)
    im1 = plt.pcolormesh(Res, Snr, Qs[1], cmap=my_cmap, vmin=vmin, vmax=vmax, edgecolor='face', lw=0.5, shading='gouraud')
    
    ###### Plotting the uncertainty as contours ######
    # Upscale dQ for smoother contours
    scale = 2
    smoothed_dq = T.rescale(np.nan_to_num(dQ, nan=0), scale, order=0)
    smoothed_dq = ndi.gaussian_filter(smoothed_dq,1.5,truncate=3, cval=1)
    res_bins_big = np.linspace(resmin, resmax, nbins*scale)
    snr_bins_big = np.linspace(snrmin, snrmax, nbins*scale)
    Res_big, Snr_big = np.meshgrid(res_bins_big, snr_bins_big)

    # Plot the smoothed contours
    cmap_dq = plt.get_cmap('gray_r')
    dqmin = np.nanquantile(smoothed_dq, 0.1)
    dqmax = np.nanquantile(smoothed_dq, 0.9)
    dq_range = dqmax - dqmin
    dq_levels = np.arange(dqmin, dqmax, dq_range/5)
    contour = plt.contour(Res_big, Snr_big, smoothed_dq, linewidths=0.3, cmap=cmap_dq, vmin=dqmin-dq_range, vmax=dqmax,
                        levels=dq_levels, linestyles='-', alpha=1, extend='both')


    ####### Polising the plot ######
    plt.clabel(contour, inline=True, fmt=r'$\sigma$=%0.2f', fontsize=7, inline_spacing=3)
    plt.scatter(base_res, base_snr, color='k', marker='+', s=200, lw=0.7)
    plt.axhline(base_snr, color='k', lw=0.5, alpha=0.2)
    plt.axvline(base_res, color='k', lw=0.5, alpha=0.2)

    # Labels
    res_label = r'$\mathcal{R}_{\mathrm{eff}}$' if res_col == 'log_nres' else r'log $\mathcal{R}$ [pc/px]'    
    snr_label = r'$\log \langle$SNR$\rangle$' if snr_col == 'log_snr' else r'$\mu_0$ [mag/arcsec$^2$]'
    plt.xlabel(f'log {res_label}')
    plt.ylabel(f'{snr_label}')
    plt.ylim(snrmin+dsnr, snrmax)
    plt.xlim(1.01*resmin+dres, 0.98*resmax)

    # Colorbar
    cbar1=plt.colorbar(im1, aspect=40, pad=0)
    cbar1.ax.annotate(r'$\Delta$', xy=(2,1), xycoords='axes fraction', va='top', ha='left')
    cbar1.ax.xaxis.set_label_position('top') 
    cbar1.ax.yaxis.set_ticks_position('right')

    # Resolution in linear scale rather than log
    ax2 = ax.twiny()
    xticks = [5,10,25,50,100] if res_col == 'log_nres' else [30,100,300,500,1000]
    xticks_og = np.log10(np.array(xticks))
    ax2.set_xticks(xticks_og)
    ax2.set_xticklabels(xticks)
    ax2.set_xlabel(res_label, labelpad=7)
    ax2.set_xlim(ax.get_xlim())

    # Final label of the bias parameter
    ax.text(0.05, 0.95,  rf"$\Delta${paramlabel}", transform=ax.transAxes,
                ha="left", va="top", size=12, bbox=dict(boxstyle='square,pad=0.4', fc=(1,1,1,1), alpha=1, ec='k', lw=0.5), zorder=30)
    return fig

##### Step 4: Plot an example for two galaxies #######
def plot_example(df, param, paramlabel, sb_low, sb_high, galaxy1=None, galaxy2=None):

    # Pick two galaxies spanning two ends of the baseline distribution
    baselines = df[df.baseline]
    if galaxy1 is None:
        # Pick one with a low value of the baseline parameter 
        qlow = baselines[param].quantile(0.2)
        low_subset = baselines[baselines[param] <= qlow]
        galaxy1 = low_subset.sample(n=1).iloc[0].galaxy
    if galaxy2 is None:
        # Pick one with a high value of the baseline parameter
        qhigh = baselines[param].quantile(0.8)
        high_subset = baselines[baselines[param] >= qhigh]
        galaxy2 = high_subset.sample(n=1).iloc[0].galaxy
    galaxies = [galaxy1, galaxy2]
                        

    fig = plt.figure(figsize=(4, 3))
    ax = plt.axes()
    colors = ['#a26','#98c']
    ys = []
    xs = []
    for galaxy, color in zip(galaxies, colors):

        subset = df[df.galaxy == galaxy]
        baseline = subset[subset.baseline].iloc[0]
        res_subset = subset[subset.sblim == sb_high].sort_values(by='log_nres')
        plt.plot(res_subset.log_nres, res_subset[param], '.-', color=color, lw=0.5)

        res_subset2 = subset[subset.sblim == sb_low].sort_values(by='log_nres')
        plt.plot(res_subset2.log_nres, res_subset2[param], '--', color=color, lw=0.5, marker='s', mfc='none', ms=3)

        ys.append(res_subset.iloc[-1][param])
        xs.append(res_subset.dropna(subset=[param])['log_nres'].max())

    # Legend
    custom_lines = [mpl.lines.Line2D([0], [0], color='k', lw=0.5, ls='-'),
                    mpl.lines.Line2D([0], [0], color='k', lw=0.5, ls=':')]
    plt.legend(custom_lines, [fr'$\mu_0$={sb_high}',fr'$\mu_0$={sb_low}'], frameon=False, loc='upper left')

    # Labels
    plt.xlabel(r'$\log \mathcal{R}_{\mathrm{eff}}$')
    plt.ylabel(paramlabel)
    plt.axvline(baseline.log_nres, color='#999', lw=0.5)
    ann_y = 0.5*(ax.get_ylim()[1]-ax.get_ylim()[0]) + ax.get_ylim()[0]
    plt.annotate('baseline', xy=(baseline.log_nres, ann_y), color='#444', rotation=90, ha='right', va='center')

    # Annotate galaxies
    plt.annotate(galaxies[0].upper(), xy=(xs[0], ys[0]), color=colors[0], ha='left', va='center')
    plt.annotate(galaxies[1].upper(), xy=(xs[1], ys[1]), color=colors[1], ha='left', va='center')

    ax.set_xlim(ax.get_xlim()[0], np.max(xs)+0.5)
    return fig


##### Step 5: Julia analysis #######
def prep_julia_analysis(df, param, res_col, snr_col, csv_outpath='.'):
    
    # Grab the columns we want
    julia_res_col = 'nres' if res_col == 'log_nres' else res_col # Symbolic regression can take the log, if needed
    julia_snr_col = 'sn_per_pixel' if snr_col == 'log_snr' else 'sblim0'
    fit_data = df[['galaxy', f'base_{param}', param, julia_res_col, julia_snr_col]].copy()

    # Add random noise to the columns
    fit_data[param] = fit_data[param] + np.random.normal(loc=0, scale=0.01*np.abs(fit_data[param]))
    fit_data[f'base_{param}'] = fit_data[f'base_{param}'] + np.random.normal(loc=0, scale=0.01*np.abs(fit_data[f'base_{param}']))

    # Plot the dataset we are fitting
    fig = plt.figure(figsize=(6,3))
    plt.scatter(fit_data[f'base_{param}'], fit_data[param], s=1, alpha=0.1, c=fit_data[julia_res_col], 
                vmin=fit_data[julia_res_col].quantile(0.1), vmax=fit_data[julia_res_col].quantile(0.9))
    plt.xlabel(f'Baseline {param}')
    plt.ylabel(param)
    cbar = plt.colorbar()
    cbar.set_label(julia_res_col)

    # Save the data
    fit_data = fit_data.rename(columns={julia_res_col:'res', julia_snr_col:'snr'})
    fit_data.to_csv(f'{csv_outpath}', index=False)

    df[f'{param}_noisy'] = fit_data[param]
    df[f'base_{param}_noisy'] = fit_data[f'base_{param}']
    return df, fig

def julia_str_to_func(str):
    """Convert a Julia expression string to a Python function. Where an example string is "f = log10(#1) * (0.5398675057524245 + (-0.07107564733327766 / #2))"
    Possible operators are:     binary_operators=(+, -, *, /),
    unary_operators=(exp, tanh, log10, inv, sqrt),
    """

    str = str.replace(' ', '')

    # Remove the "f = " part
    str_expressions = str.split(';')
    expressions = []
    for expr in str_expressions:

        expr = expr.split('=')[1].strip()
        # Replace Julia syntax with Python syntax
        expr = expr.replace('#1', 'x')
        expr = expr.replace('#2', 'y')
        expr = expr.replace('log10', 'np.log10')
        expr = expr.replace('exp', 'np.exp')
        expr = expr.replace('sqrt', 'np.sqrt')
        expr = expr.replace('tanh', 'np.tanh')
        expr = expr.replace('inv', '1/')
        expressions.append(expr)

    def f(x, y):
        return eval(expressions[0])
    def g(x, y):
        return eval(expressions[1])
    
    func = lambda base, x, y: base * f(x, y) + g(x, y)
    return func
    
def load_sr_results(julia_cat_path):
    """Load the Julia symbolic regression results from the output path and assign a score to each fit."""
    julia_cat = pd.read_csv(julia_cat_path)

    # Assign scores to each row based on loss and complexity
    scores = []
    for i, row in julia_cat.iterrows():
        if i == 0: 
            score=0
        else:
            score=-np.log10(row.Loss / julia_cat.iloc[i-1].Loss) / (row.Complexity - julia_cat.iloc[i-1].Complexity)
            score*=100
        scores.append(score)
    julia_cat['Score'] = scores
    julia_cat['best'] = julia_cat.Score == julia_cat.Score.max()
    return julia_cat

def plot_correction(df, param, paramlabel, reslim=5, snrlim=2, xlims=None, noisy=True):
    """Makes a two-panel plot showing the parameter and corrected parameter distributions as a function of the baseline value."""

    # Define axis limits based on the parameter distribution
    if xlims is not None:
        xmin, xmax = xlims
    else:
        xmin, xmax = df[param].quantile(q=[0.01,0.99]).values


    # Colormap
    colors = ['#fff','#faac9b','#e56b6f','#b56576','#6d597a']
    cmap = mpl.colors.LinearSegmentedColormap.from_list('my_cmap2', colors, N=256)

    # Plot idea: KDE of the base/measured parameter space, overlaid contours from the same plot for non-corrected
    # With points or in the corner, overplot points with low resolution or low SNR
    fig, axs = plt.subplots(1, 2, figsize=(7.5,3.75), sharey=True)

    if noisy:
        yparams = [f'{param}_noisy', f'{param}_noisy_corr']
    else:
        yparams = [f'{param}', f'{param}_corr']
    ylabels = [f'{paramlabel}', f'Corrected {paramlabel}']
    for ax, yparam, ylabel in zip(axs, yparams, ylabels):

        # If no correction, we skip the right panel
        if ax == axs[1] and df[yparam].isna().all():
            ax.annotate('No correction applied', xy=(0.5, 0.5), xycoords='axes fraction', ha='center', va='center')
            continue

        plot_data = df[ (df[yparam] >= xmin) & (df[yparam] <= xmax)].dropna(subset=[f'base_{param}', yparam, 'nres', 'snr_px'])
        
        # Smooth the data with a Gaussian KDE
        kde = gaussian_kde(plot_data[[f'base_{param}', yparam]].sample(n=5000).values.T, bw_method=0.15)

        # Define grid for evaluation
        npoints=100
        xs = np.linspace(xmin, xmax, npoints)
        ys = np.linspace(xmin, xmax, npoints)
        Xs, Ys = np.meshgrid(xs, ys)
        pts = np.vstack([Xs.ravel(), Ys.ravel()])
        
        # Evaluate KDE on the grid
        density = kde(pts).reshape(npoints, npoints) 

        # Plot the KDE
        vmin, vmax = np.nanquantile(density.flatten(), [0.1, 0.99])
        im=ax.pcolormesh(Xs,Ys, density, cmap=cmap, vmin=vmin, vmax=vmax)

        ###### Contour overlay of poor resolution ########
        bad_res_points = plot_data[plot_data.nres < reslim]
        arr = bad_res_points[[f'base_{param}', yparam]].dropna().values.T
        kde_bad = gaussian_kde(bad_res_points[[f'base_{param}', yparam]].values.T, bw_method=0.2)
        density_bad = kde_bad(pts).reshape(npoints, npoints)
        ax.contour(Xs,Ys, density_bad, colors='k', linewidths=0.5)

        ###### Contour overlay of poor SNR ########
        bad_snr_points = plot_data[plot_data.snr_px < snrlim]
        try:
            kde_bad = gaussian_kde(bad_snr_points[[f'base_{param}', yparam]].values.T, bw_method=0.2)
        except:
            return bad_snr_points
        density_bad = kde_bad(pts).reshape(npoints, npoints)
        ax.contour(Xs,Ys, density_bad, colors='w', linewidths=0.5)

        # Set axis limit and draw a 1:1 line
        ax.plot([xmin, xmax],[xmin, xmax], color='k', lw=0.5)
        ax.set_xlim(xmin, xmax)
        ax.set_ylabel(ylabel)
        ax.set_xlabel(f'Base {paramlabel}')
        
    # Adjust the plot
    axs[0].set_ylim(xmin, xmax)
    axs[1].tick_params(left=False, labelleft=False, right=True, labelright=True)
    axs[1].yaxis.set_label_position("right")
    plt.subplots_adjust(wspace=0.02)

    # Add a legend
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    reslabel = r'$\mathcal{R}_{\mathrm{eff}}$'
    snr_label = r'$\langle$SNR$\rangle$'
    legend_elements = [Line2D([0], [0], color='k', lw=0.5, label=f'{reslabel} $<$ {reslim}'),
                    Line2D([0], [0], color='#aaa', lw=0.5, label=f'{snr_label} $<$ {snrlim}')]
    axs[0].legend(handles=legend_elements, loc='upper left', frameon=False)

    # Add a colorbar
    cax = fig.add_axes([0, 0.96, 1, 0.02])
    cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
    cbar.ax.xaxis.set_ticks_position("top")
    cbar.ax.xaxis.set_label_position("top")
    cbar.set_label('PDF', labelpad=-5)
    plt.subplots_adjust(left=0, right=1, top=0.95)

    return fig

def symbolic_correction(df, param, res_col, snr_col, corr_fun=None, julia_cat=None, row_idx=None):
    """Apply the Julia symbolic regression correction to the DataFrame. Either applies the correction from row `row_idx`, 
    or the simplest correction with Score>1 if `row_idx` is None.
    Args:
        df (pd.DataFrame): DataFrame containing the rows to correct
        param (str): Name of the parameter to correct
        res_col (str): Name of the column containing resolution values used in Julia
        snr_col (str): Name of the column containing SNR values used in Julia
        corr_fun (function): Function to use for correction. If None, uses the Julia symbolic regression results.
        julia_cat (pd.DataFrame): DataFrame containing the Julia symbolic regression results.
        row_idx (int): Index of the row in `julia_cat` to use for correction. If None, uses the simplest correction with Score>1.
    Returns:
        pd.DataFrame: DataFrame with an additional column for the corrected parameter"""

    # Load the Julia symbolic regression results if not provided
    if corr_fun is None:
        if row_idx is None:
            row = julia_cat[(julia_cat.Score > 1)].sort_values(by='Complexity').iloc[0]
        else:
            row = julia_cat.iloc[row_idx]

    # Julia doesn't do logs of columns
    julia_res_col = 'nres' if res_col == 'log_nres' else res_col # Symbolic regression can take the log, if needed
    julia_snr_col = 'sn_per_pixel' if snr_col == 'log_snr' else 'sblim0'

    # Make a copy of the dataframe to no overwrite anything
    df_corr = df.copy()

    # Parse the equation
    if corr_fun is None:
        corr_fun = julia_str_to_func(row.Equation)

    # Apply the correction
    df_corr[f'{param}_corr'] = df_corr.apply(lambda x: corr_fun(x[f'base_{param}'], x[julia_res_col], x[julia_snr_col]), axis=1)
    df_corr[f'{param}_noisy_corr'] = df_corr.apply(lambda x: corr_fun(x[f'base_{param}_noisy'], x[julia_res_col], x[julia_snr_col]), axis=1)
    return df_corr
