""" This code is for plot making.
Re-coded after a previous file lost.

Created on: 09/12/2022

@author: katse
"""

def fitmap(gridx, gridy, grid, title, period, colour_map, cmin, cm, cmax, name, path):
    
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy
    import matplotlib.colors as colors
    import cartopy.io.img_tiles as cimgt
    import matplotlib.ticker as mticker
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    from matplotlib_scalebar.scalebar import ScaleBar
    
    
    #We use only PROMICE and GCNet stations
    stations_names = ['Peterman_ELA', 'KPC_L', 'KPC_U', 'EGP', 'NASA_E', 'SCO_L', 'SCO_U', 'Summit',
                      'NASA_SE', 'South Dome', 'KAN_L', 'KAN_M', 'KAN_U', 'DYE-2', 'JAR_1', 'Swiss Camp', 'CP1', 
                      'UPE_L', 'UPE_U', 'NASA_U', 'GITS', 'NEEM']
    
    stations_coords = [[-244306, -1047222], [391168, -1023448], [374565, -1038474], [245519, -1545750], [422939, -1578126],
                       [605565, -1843811], [586431, -1830173], [215048, -1889006], [112747, -2580907], [9445, -2960973], 
                       [-216931, -2504048], [-168343, -2510999], [-89205, -2522571], [-57831, -2582094], [-184047, -2236749],
                       [-168891, -2230133], [-76703, -2200166], [-301483, -1841957], [-278493, -1846178], [-138429, -1756197],
                       [-386548, -1344511],[-138963, -1351904]]


    stations_labels = [[-244306, -1017222], [301168, -1003448], [334565, -1108474], [245519, -1515750], [422939, -1548126],
                       [445565, -1803811], [606431, -1890173], [215048, -1859006], [102747, -2565907], [-129445, -2930973], 
                       [-356931, -2484048], [-178343, -2490999], [-70205, -2552571], [-107831, -2652094], [-229047, -2306749],
                       [-148891, -2260133], [-76703, -2170166], [-321483, -1811957], [-248493, -1856178], [-138429, -1726197],
                       [-386548, -1314511],[-138963, -1321904]]

    
    stations_elev = [924, 370, 870, 2660, 2614, 460, 970, 3199, 2373, 2901, 670, 
                     1270, 1840, 2099, 932, 1176, 2022, 220, 940, 2334, 1869, 2454]
    
    cmap=plt.cm.get_cmap(colour_map)
    norm = colors.Normalize(cmin,cmax) #TwoSlopeNorm(0, vmin=-3, vmax =1)
    
    plt.rcParams['font.sans-serif'] = "Helvetica"
    #plt.rcParams["figure.figsize"] = (6.8, 8)
    ax = plt.axes(projection=ccrs.NorthPolarStereo(central_longitude=-45.0, globe=None))
    plt.gcf().set_size_inches(6.8, 8)

    sea = cimgt.GoogleTiles(style = 'satellite')
    #topo = cimgt.MapboxTiles('pk.eyJ1Ijoia21zZWphbiIsImEiOiJjbGUwNG1kaDYwOWt1M25wNWsydWhjN200In0.Vw-DuMfDepMWLw9jgP_5xw', 
                             #'outdoors-v12')
    
    ax.add_image(sea, 5, alpha = 1, zorder = 0)
    #ax.add_image(topo, 5, zorder = 1)


    ax.set_extent([-26.5, -58, 58.8, 83.5])
    mapp = ax.pcolormesh(gridx, gridy, grid, norm = norm, cmap = cmap, transform = ccrs.epsg(3413), zorder=2)
    
    #ax.coastlines(resolution='50m', linewidth=0.5)
    
    #ax.add_feature(cartopy.feature.OCEAN, zorder=0)
    #ax.add_feature(cartopy.feature.LAND, zorder=1)
    gl = ax.gridlines(draw_labels=dict(top= 'x', left= 'y'), linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    
    gl.bottom_labels = False
    #gl.left_labels = True
    gl.right_labels = False
    #gl.top_labels = True
    gl.rotate_labels=False
    gl.labels_bbox_style = dict(alpha = 0.5, facecolor = 'white', edgecolor= 'none', boxstyle='round', pad = 0.1)
    gl.xpadding = -5
    gl.ypadding = -5
    gl.xlocator = mticker.FixedLocator([-75, -60, -45, -30, -15, 0])
    gl.xformatter = cartopy.mpl.ticker.LongitudeFormatter(direction_label=True, auto_hide=False)
    gl.ylocator = mticker.FixedLocator([60, 65, 70, 75, 80])
    gl.yformatter = cartopy.mpl.ticker.LatitudeFormatter(direction_label=True, auto_hide=False)
    gl.xlabel_style = dict(size= 8, color= 'black') #bbox= dict(alpha = 0.5, facecolor = 'white', edgecolor='black', boxstyle='round')
    gl.ylabel_style = dict(size= 8, color= 'black')

    for artist in gl.top_label_artists:
        artist.set_visible(True).draw()
    for artist in gl.left_label_artists:
        artist.set_visible(True)


    for i,n in enumerate(stations_names):
        
        ax.plot(stations_coords[i][0], stations_coords[i][1], 'ko', markersize=1.1, transform=ccrs.epsg(3413), zorder=3)
        ax.text(stations_labels[i][0], stations_labels[i][1], stations_names[i], fontsize=8, transform=ccrs.epsg(3413), zorder=4)
    
    cbaxes = inset_axes(ax, width="20%", height="2%", bbox_to_anchor = (1400,600,3000,3000), loc=8) #bbox_to_anchor = (0,0,1,1)
    cb = plt.colorbar(mapp, cax=cbaxes, ticks=[cmin, cm, cmax], orientation='horizontal', pad = 0.03)
    cb.ax.tick_params(labelsize=9, labelcolor = 'white', pad = -0.1, bottom = False, labeltop = True, labelbottom = False)
    cb.outline.set_color('w')
    cbaxes.set_title(title, color = 'white', fontsize=12, fontweight = 'semibold', pad = 10)

    
    ax.add_artist(ScaleBar(0.001, units = 'km', location = 3, frameon = False, color = 'white', font_properties = {'size': 9}, fixed_units = 'km'))
    ax.text(-775000, -3250000, period, color = 'white', fontsize=12, fontweight = 'semibold')
    
    plt.savefig(path+name + '.jpg', dpi = 600, bbox_inches='tight')
    plt.clf()

def massmap4(gridx, gridy, grids, titles, mgridx, mgridy, mask, tgrid, extend, period, stations_names, stations_coords, stations_labels, colour_map, cmin, cm, cmax, colort, colorb, px, py, zone, name, path):
    
    import numpy as np
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy
    import matplotlib.colors as colors
    import cartopy.io.img_tiles as cimgt
    import matplotlib.ticker as mticker
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    from matplotlib_scalebar.scalebar import ScaleBar
    import matplotlib.patches as mpatches    
    
    
    cmap=plt.cm.get_cmap(colour_map)
    norm = colors.Normalize(cmin,cmax) #TwoSlopeNorm(0, vmin=-3, vmax =1)
    colour_mapm = colors.LinearSegmentedColormap.from_list('', ['magenta', 'cyan', 'indigo', 'slategrey', 'lavender'])
    cmapm=plt.cm.get_cmap(colour_mapm)
    normm = colors.Normalize(1,5) #TwoSlopeNorm(0, vmin=-3, vmax =1)

    
    plt.rcParams['font.sans-serif'] = "Helvetica"


    fig, axes = plt.subplots(2, 2, subplot_kw={'projection': ccrs.NorthPolarStereo(central_longitude=-45.0, globe=None)})
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.1, hspace=0.1)
    
    locs = [(0,0), (0,1), (1,0), (1,1)]
    for i,ax in np.ndenumerate(axes):

        plt.gcf().set_size_inches(6, 8)
        
        for p,k in enumerate(locs):
            if i == k:
                j = p

        sea = cimgt.GoogleTiles(style = 'satellite')
        
        ax.add_image(sea, 5, alpha = 1, zorder = 0)
        ax.set_extent(extend)
        mappm = ax.pcolormesh(mgridx, mgridy, mask, norm = normm, cmap = cmapm, alpha = 0.7, transform = ccrs.epsg(3413), zorder=1)
        mapp = ax.pcolormesh(gridx, gridy, grids[j], norm = norm, cmap = cmap, alpha = 1, transform = ccrs.epsg(3413), zorder=2)
    
        if zone == 'wet':
            cont = ax.contour(mgridx, mgridy, mask, [1], colors = ['magenta'], linewidths = 0.8, transform = ccrs.epsg(3413), zorder=3)    
            cont = ax.contour(mgridx, mgridy, mask, [3], colors = ['indigo'], linewidths = 0.8, transform = ccrs.epsg(3413), zorder=3)    
            cont = ax.contour(mgridx, mgridy, mask, [4], colors = ['slategrey'], linewidths = 0.8, transform = ccrs.epsg(3413), zorder=3)    
        elif zone == 'dynamic':    
            cont = ax.contour(mgridx, mgridy, mask, [2], colors = ['cyan'], linewidths = 0.8, transform = ccrs.epsg(3413), zorder=3)    
            cont = ax.contour(mgridx, mgridy, mask, [1], colors = ['magenta'], linewidths = 0.8, transform = ccrs.epsg(3413), zorder=3)    
            cont = ax.contour(mgridx, mgridy, mask, [4], colors = ['slategrey'], linewidths = 0.8, transform = ccrs.epsg(3413), zorder=3)    

        cont = ax.contour(gridx, gridy, tgrid, [1500, 1750, 2000], colors = ['white', 'white', 'white'], linewidths = 0.5, alpha = 0.8, transform = ccrs.epsg(3413), zorder=3)    
        ax.clabel(cont, cont.levels, inline=True, fontsize=6)

        gl = ax.gridlines(draw_labels=False, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
            
        for artist in gl.top_label_artists:
            artist.set_visible(True).draw()
        for artist in gl.left_label_artists:
            artist.set_visible(True)
    
    
        for i,n in enumerate(stations_names):
            
            ax.plot(stations_coords[i][0], stations_coords[i][1], 'ko', markersize=1.1, transform=ccrs.epsg(3413), zorder=4)
            ax.text(stations_labels[i][0], stations_labels[i][1], stations_names[i], fontsize=8, transform=ccrs.epsg(3413), zorder=5)
        
        cbaxes = inset_axes(ax, width="30%", height="4%", loc=3, borderpad=1) #bbox_to_anchor = (0,0,1,1)
        cb = plt.colorbar(mapp, cax=cbaxes, ticks=[cmin, cm, cmax], orientation='horizontal', pad = 0)
        cb.ax.tick_params(labelsize=7, labelcolor = colort, pad = 0, bottom = False, labeltop = True, labelbottom = False)
        cb.outline.set_color(colort)
        cbaxes.set_title(titles[j], color = colort, fontsize=7, fontweight = 'semibold', pad = 10)
    
        
        ax.add_artist(ScaleBar(0.001, units = 'km', location = 1, frameon = False, color = colorb, font_properties = {'size': 7}, fixed_units = 'km'))
        ax.text(px, py, period, color = colorb, fontsize=7, fontweight = 'semibold', horizontalalignment='right')

        ax.legend( [mpatches.Rectangle((0, 0), 0.1, 0.1, facecolor='magenta'), mpatches.Rectangle((0, 0), 0.1, 0.1, facecolor='cyan'), 
                    mpatches.Rectangle((0, 0), 0.1, 0.1, facecolor='indigo'), mpatches.Rectangle((0, 0), 0.1, 0.1, facecolor='slategrey'),
                    mpatches.Rectangle((0, 0), 0.1, 0.1, facecolor='lavender'), ], ['Ablation', 'Dynamic', 'Wet', 'Percolation', 'Dry'], loc=4, framealpha=1, fontsize = 6)    
    

    plt.savefig(path+name + '.jpg', dpi = 300, bbox_inches='tight')
    plt.clf()


def massmap6(gridx, gridy, grids, titles, mgridx, mgridy, mask, tgrid, extend, period, stations_names, stations_coords, stations_labels, colour_map, cmin, cm, cmax, colort, colorb, px, py, zone, name, path):
    
    import numpy as np
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy
    import matplotlib.colors as colors
    import cartopy.io.img_tiles as cimgt
    import matplotlib.ticker as mticker
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    from matplotlib_scalebar.scalebar import ScaleBar
    import matplotlib.patches as mpatches    
    
    
    cmap=plt.cm.get_cmap(colour_map)
    norm = colors.Normalize(cmin,cmax) #TwoSlopeNorm(0, vmin=-3, vmax =1)
    colour_mapm = colors.LinearSegmentedColormap.from_list('', ['magenta', 'cyan', 'indigo', 'slategrey', 'lavender'])
    cmapm=plt.cm.get_cmap(colour_mapm)
    normm = colors.Normalize(1,5) #TwoSlopeNorm(0, vmin=-3, vmax =1)

    
    plt.rcParams['font.sans-serif'] = "Helvetica"


    fig, axes = plt.subplots(3, 2, subplot_kw={'projection': ccrs.NorthPolarStereo(central_longitude=-45.0, globe=None)})
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=-0.5, hspace=0.1)
    
    locs = [(0,0), (0,1), (1,0), (1,1), (2,0), (2,1)]
    for i,ax in np.ndenumerate(axes):

        plt.gcf().set_size_inches(6, 8)
        
        for p,k in enumerate(locs):
            if i == k:
                j = p

        if i ==(2,0):
            cmap=plt.cm.get_cmap('Reds')
            cmin = 0
            cm = 0.1
            cmax = 0.2
            norm = colors.Normalize(cmin,cmax)
        elif i ==(2,1):
            cmap=plt.cm.get_cmap('Blues')
            cmin = 0
            cm = 0.1
            cmax = 0.2
            norm = colors.Normalize(cmin,cmax)


        sea = cimgt.GoogleTiles(style = 'satellite')
        
        ax.add_image(sea, 5, alpha = 1, zorder = 0)
        ax.set_extent(extend)
        mappm = ax.pcolormesh(mgridx, mgridy, mask, norm = normm, cmap = cmapm, alpha = 0.7, transform = ccrs.epsg(3413), zorder=1)
        mapp = ax.pcolormesh(gridx, gridy, grids[j], norm = norm, cmap = cmap, alpha = 1, transform = ccrs.epsg(3413), zorder=2)
    
        if zone == 'wet':
            cont = ax.contour(mgridx, mgridy, mask, [1], colors = ['magenta'], linewidths = 0.8, transform = ccrs.epsg(3413), zorder=3)    
            cont = ax.contour(mgridx, mgridy, mask, [3], colors = ['indigo'], linewidths = 0.8, transform = ccrs.epsg(3413), zorder=3)    
            cont = ax.contour(mgridx, mgridy, mask, [4], colors = ['slategrey'], linewidths = 0.8, transform = ccrs.epsg(3413), zorder=3)    
        elif zone == 'dynamic':    
            cont = ax.contour(mgridx, mgridy, mask, [2], colors = ['cyan'], linewidths = 0.8, transform = ccrs.epsg(3413), zorder=3)    
            cont = ax.contour(mgridx, mgridy, mask, [1], colors = ['magenta'], linewidths = 0.8, transform = ccrs.epsg(3413), zorder=3)    
            cont = ax.contour(mgridx, mgridy, mask, [4], colors = ['slategrey'], linewidths = 0.8, transform = ccrs.epsg(3413), zorder=3)    

        cont = ax.contour(gridx, gridy, tgrid, [1500, 1750, 2000], colors = ['white', 'white', 'white'], linewidths = 0.5, alpha = 0.8, transform = ccrs.epsg(3413), zorder=3)    
        ax.clabel(cont, cont.levels, inline=True, fontsize=5)

        gl = ax.gridlines(draw_labels=False, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
            
        for artist in gl.top_label_artists:
            artist.set_visible(True).draw()
        for artist in gl.left_label_artists:
            artist.set_visible(True)
    
    
        for i,n in enumerate(stations_names):
            
            ax.plot(stations_coords[i][0], stations_coords[i][1], 'ko', markersize=1.1, transform=ccrs.epsg(3413), zorder=4)
            ax.text(stations_labels[i][0], stations_labels[i][1], stations_names[i], fontsize=5, transform=ccrs.epsg(3413), zorder=5)
        
        cbaxes = inset_axes(ax, width="30%", height="4%", loc=3, borderpad=1) #bbox_to_anchor = (0,0,1,1)
        cb = plt.colorbar(mapp, cax=cbaxes, ticks=[cmin, cm, cmax], orientation='horizontal', pad = 0)
        cb.ax.tick_params(labelsize=5, labelcolor = colort, pad = 0, bottom = False, labeltop = True, labelbottom = False)
        cb.outline.set_color(colort)
        cbaxes.set_title(titles[j], color = colort, fontsize=6, fontweight = 'semibold', backgroundcolor= 'darkgrey', pad = 15)
    
        
        ax.add_artist(ScaleBar(0.001, units = 'km', location = 1, frameon = False, color = colorb, font_properties = {'size': 5}, fixed_units = 'km'))
        ax.text(px, py, period, color = colorb, fontsize=5, fontweight = 'semibold', horizontalalignment='right')

        ax.legend( [mpatches.Rectangle((0, 0), 0.1, 0.1, facecolor='magenta'), mpatches.Rectangle((0, 0), 0.1, 0.1, facecolor='cyan'), 
                    mpatches.Rectangle((0, 0), 0.1, 0.1, facecolor='indigo'), mpatches.Rectangle((0, 0), 0.1, 0.1, facecolor='slategrey'),
                    mpatches.Rectangle((0, 0), 0.1, 0.1, facecolor='lavender'), ], ['Ablation', 'Dynamic', 'Wet', 'Percolation', 'Dry'], loc=4, framealpha=1, fontsize = 5)    
    

    plt.savefig(path+name + '.jpg', dpi = 300, bbox_inches='tight')
    plt.clf()


def maskmap(gridx, gridy, grid, title, period, colour_map, cmin, cmax, name, path):
    
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy
    import matplotlib.colors as colors
    import cartopy.io.img_tiles as cimgt
    import matplotlib.ticker as mticker
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    from matplotlib_scalebar.scalebar import ScaleBar
    import matplotlib.patches as mpatches
    
    
    #We use only PROMICE and GCNet stations
    stations_names = ['Peterman_ELA', 'KPC_L', 'KPC_U', 'EGP', 'NASA_E', 'SCO_L', 'SCO_U', 'Summit',
                      'NASA_SE', 'South Dome', 'KAN_L', 'KAN_M', 'KAN_U', 'DYE-2', 'JAR_1', 'Swiss Camp', 'CP1', 
                      'UPE_L', 'UPE_U', 'NASA_U', 'GITS', 'NEEM']
    
    stations_coords = [[-244306, -1047222], [391168, -1023448], [374565, -1038474], [245519, -1545750], [422939, -1578126],
                       [605565, -1843811], [586431, -1830173], [215048, -1889006], [112747, -2580907], [9445, -2960973], 
                       [-216931, -2504048], [-168343, -2510999], [-89205, -2522571], [-57831, -2582094], [-184047, -2236749],
                       [-168891, -2230133], [-76703, -2200166], [-301483, -1841957], [-278493, -1846178], [-138429, -1756197],
                       [-386548, -1344511],[-138963, -1351904]]


    stations_labels = [[-244306, -1017222], [301168, -1003448], [334565, -1108474], [245519, -1515750], [422939, -1548126],
                       [445565, -1803811], [606431, -1890173], [215048, -1859006], [102747, -2565907], [-129445, -2930973], 
                       [-356931, -2484048], [-178343, -2490999], [-70205, -2552571], [-107831, -2652094], [-229047, -2306749],
                       [-148891, -2260133], [-76703, -2170166], [-321483, -1811957], [-248493, -1856178], [-138429, -1726197],
                       [-386548, -1314511],[-138963, -1321904]]

    
    stations_elev = [924, 370, 870, 2660, 2614, 460, 970, 3199, 2373, 2901, 670, 
                     1270, 1840, 2099, 932, 1176, 2022, 220, 940, 2334, 1869, 2454]
    
    cmap=plt.cm.get_cmap(colour_map)
    norm = colors.Normalize(cmin,cmax) #TwoSlopeNorm(0, vmin=-3, vmax =1)
    
    plt.rcParams['font.sans-serif'] = "Helvetica"
    #plt.rcParams["figure.figsize"] = (6.8, 8)
    ax = plt.axes(projection=ccrs.NorthPolarStereo(central_longitude=-45.0, globe=None))
    plt.gcf().set_size_inches(6.8, 8)

    sea = cimgt.GoogleTiles(style = 'satellite')
    #topo = cimgt.MapboxTiles('pk.eyJ1Ijoia21zZWphbiIsImEiOiJjbGUwNG1kaDYwOWt1M25wNWsydWhjN200In0.Vw-DuMfDepMWLw9jgP_5xw', 
                             #'outdoors-v12')
    
    ax.add_image(sea, 5, alpha = 0.9, zorder = 0)
    #ax.add_image(topo, 5, zorder = 1)


    ax.set_extent([-26.5, -58, 58.8, 83.5])
    mapp = ax.pcolormesh(gridx, gridy, grid, norm = norm, cmap = cmap, alpha = 0.7, transform = ccrs.epsg(3413), zorder=2)
    
    gl = ax.gridlines(draw_labels=dict(top= 'x', left= 'y'), linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    
    gl.bottom_labels = False
    #gl.left_labels = True
    gl.right_labels = False
    #gl.top_labels = True
    gl.rotate_labels=False
    gl.labels_bbox_style = dict(alpha = 0.5, facecolor = 'white', edgecolor= 'none', boxstyle='round', pad = 0.1)
    gl.xpadding = -5
    gl.ypadding = -5
    gl.xlocator = mticker.FixedLocator([-75, -60, -45, -30, -15, 0])
    gl.xformatter = cartopy.mpl.ticker.LongitudeFormatter(direction_label=True, auto_hide=False)
    gl.ylocator = mticker.FixedLocator([60, 65, 70, 75, 80])
    gl.yformatter = cartopy.mpl.ticker.LatitudeFormatter(direction_label=True, auto_hide=False)
    gl.xlabel_style = dict(size= 8, color= 'black') #bbox= dict(alpha = 0.5, facecolor = 'white', edgecolor='black', boxstyle='round')
    gl.ylabel_style = dict(size= 8, color= 'black')

    for artist in gl.top_label_artists:
        artist.set_visible(True).draw()
    for artist in gl.left_label_artists:
        artist.set_visible(True)

    for i,n in enumerate(stations_names):
        
        ax.plot(stations_coords[i][0], stations_coords[i][1], 'ko', markersize=1.1, transform=ccrs.epsg(3413), zorder=3)
        ax.text(stations_labels[i][0], stations_labels[i][1], stations_names[i], fontsize=8, transform=ccrs.epsg(3413), zorder=4)

    
    ax.add_artist(ScaleBar(0.001, units = 'km', location = 3, frameon = False, color = 'white', font_properties = {'size': 9}, fixed_units = 'km'))
    ax.text(-775000, -3250000, period, color = 'white', fontsize=12, fontweight = 'semibold')
    
    ax.legend( [mpatches.Rectangle((0, 0), 1, 1, facecolor='magenta'), mpatches.Rectangle((0, 0), 1, 1, facecolor='cyan')], ['Ablation', 'Dynamic'], loc=4, framealpha=1)    

    plt.savefig(path+name + '.jpg', dpi = 600, bbox_inches='tight')
    plt.clf()

def faciesmap(gridx, gridy, grid, title, period, cmin, cmax, name, path):
    
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy
    import matplotlib.colors as colors
    import cartopy.io.img_tiles as cimgt
    import matplotlib.ticker as mticker
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    from matplotlib_scalebar.scalebar import ScaleBar
    import matplotlib.patches as mpatches
    import matplotlib.colors
    
    
    #We use only PROMICE and GCNet stations
#    stations_names = ['EGP', 'Summit', 'South Dome', 'Swiss Camp', 'GITS']    
#    stations_coords = [[245519, -1545750], [215048, -1889006], [9445, -2960973], [-168891, -2230133], [-386548, -1344511]]
#    stations_labels = [[245519, -1515750], [215048, -1859006], [-129445, -2930973], [-148891, -2260133], [-386548, -1314511]]    
#    stations_elev = [2660, 3199, 2901, 1176, 1869]

    stations_names = ['Peterman_ELA', 'KPC_L', 'KPC_U', 'EGP', 'NASA_E', 'SCO_L', 'SCO_U', 'Summit',
                      'NASA_SE', 'South Dome', 'KAN_L', 'KAN_M', 'KAN_U', 'DYE-2', 'JAR_1', 'Swiss Camp', 'CP1', 
                      'UPE_L', 'UPE_U', 'NASA_U', 'GITS', 'NEEM']
    
    stations_coords = [[-244306, -1047222], [391168, -1023448], [374565, -1038474], [245519, -1545750], [422939, -1578126],
                       [605565, -1843811], [586431, -1830173], [215048, -1889006], [112747, -2580907], [9445, -2960973], 
                       [-216931, -2504048], [-168343, -2510999], [-89205, -2522571], [-57831, -2582094], [-184047, -2236749],
                       [-168891, -2230133], [-76703, -2200166], [-301483, -1841957], [-278493, -1846178], [-138429, -1756197],
                       [-386548, -1344511],[-138963, -1351904]]


    stations_labels = [[-244306, -1017222], [301168, -1003448], [334565, -1108474], [245519, -1515750], [422939, -1548126],
                       [445565, -1803811], [606431, -1890173], [215048, -1859006], [102747, -2565907], [-129445, -2930973], 
                       [-356931, -2484048], [-178343, -2490999], [-70205, -2552571], [-107831, -2652094], [-229047, -2306749],
                       [-148891, -2260133], [-76703, -2170166], [-321483, -1811957], [-248493, -1856178], [-138429, -1726197],
                       [-386548, -1314511],[-138963, -1321904]]

    
    stations_elev = [924, 370, 870, 2660, 2614, 460, 970, 3199, 2373, 2901, 670, 
                     1270, 1840, 2099, 932, 1176, 2022, 220, 940, 2334, 1869, 2454]

    
    colour_map = matplotlib.colors.LinearSegmentedColormap.from_list('', ['magenta', 'cyan', 'indigo', 'slategrey', 'lavender'])
    cmap=plt.cm.get_cmap(colour_map)
    norm = colors.Normalize(cmin,cmax) #TwoSlopeNorm(0, vmin=-3, vmax =1)
    
    plt.rcParams['font.sans-serif'] = "Helvetica"
    #plt.rcParams["figure.figsize"] = (6.8, 8)
    ax = plt.axes(projection=ccrs.NorthPolarStereo(central_longitude=-45.0, globe=None))
    plt.gcf().set_size_inches(6.8, 8)

    sea = cimgt.GoogleTiles(style = 'satellite')
    #topo = cimgt.MapboxTiles('pk.eyJ1Ijoia21zZWphbiIsImEiOiJjbGUwNG1kaDYwOWt1M25wNWsydWhjN200In0.Vw-DuMfDepMWLw9jgP_5xw', 
                             #'outdoors-v12')
    
    ax.add_image(sea, 5, alpha = 0.9, zorder = 0)
    #ax.add_image(topo, 5, zorder = 1)


    ax.set_extent([-26.5, -58, 58.8, 83.5])
    mapp = ax.pcolormesh(gridx, gridy, grid, norm = norm, cmap = cmap, alpha = 0.7, transform = ccrs.epsg(3413), zorder=2)
    
    gl = ax.gridlines(draw_labels=dict(top= 'x', left= 'y'), linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    
    gl.bottom_labels = False
    #gl.left_labels = True
    gl.right_labels = False
    #gl.top_labels = True
    gl.rotate_labels=False
    gl.labels_bbox_style = dict(alpha = 0.5, facecolor = 'white', edgecolor= 'none', boxstyle='round', pad = 0.1)
    gl.xpadding = -5
    gl.ypadding = -5
    gl.xlocator = mticker.FixedLocator([-75, -60, -45, -30, -15, 0])
    gl.xformatter = cartopy.mpl.ticker.LongitudeFormatter(direction_label=True, auto_hide=False)
    gl.ylocator = mticker.FixedLocator([60, 65, 70, 75, 80])
    gl.yformatter = cartopy.mpl.ticker.LatitudeFormatter(direction_label=True, auto_hide=False)
    gl.xlabel_style = dict(size= 8, color= 'black') #bbox= dict(alpha = 0.5, facecolor = 'white', edgecolor='black', boxstyle='round')
    gl.ylabel_style = dict(size= 8, color= 'black')

    for artist in gl.top_label_artists:
        artist.set_visible(True).draw()
    for artist in gl.left_label_artists:
        artist.set_visible(True)

    for i,n in enumerate(stations_names):
        
        ax.plot(stations_coords[i][0], stations_coords[i][1], 'ko', markersize=1.1, transform=ccrs.epsg(3413), zorder=3)
        ax.text(stations_labels[i][0], stations_labels[i][1], stations_names[i], fontsize=8, transform=ccrs.epsg(3413), zorder=4)

    
    ax.add_artist(ScaleBar(0.001, units = 'km', location = 3, frameon = False, color = 'white', font_properties = {'size': 9}, fixed_units = 'km'))
    ax.text(-775000, -3250000, period, color = 'white', fontsize=12, fontweight = 'semibold')
    
    ax.legend( [mpatches.Rectangle((0, 0), 1, 1, facecolor='magenta'), mpatches.Rectangle((0, 0), 1, 1, facecolor='cyan'), 
                mpatches.Rectangle((0, 0), 1, 1, facecolor='indigo'), mpatches.Rectangle((0, 0), 1, 1, facecolor='slategrey'),
                mpatches.Rectangle((0, 0), 1, 1, facecolor='lavender'), ], ['Ablation', 'Dynamic', 'Wet', 'Percolation', 'Dry'], loc=4, framealpha=1)    

    plt.savefig(path+name + '.jpg', dpi = 600, bbox_inches='tight')
    plt.clf()


def topomap(gridx, gridy, grid, title, period, colour_map, cmin, cmax, name, path):
    
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy
    import matplotlib.colors as colors
    import cartopy.io.img_tiles as cimgt
    import matplotlib.ticker as mticker
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    from matplotlib_scalebar.scalebar import ScaleBar
    import matplotlib.patches as mpatches
    
    #We use only PROMICE and GCNet stations
    stations_names = ['EGP', 'Summit', 'South Dome', 'Swiss Camp', 'GITS']
    
    stations_coords = [[245519, -1545750], [215048, -1889006], [9445, -2960973], [-168891, -2230133], [-386548, -1344511]]


    stations_labels = [[245519, -1515750], [215048, -1859006], [-129445, -2930973], [-148891, -2260133], [-386548, -1314511]]

    
    stations_elev = [2660, 3199, 2901, 1176, 1869]

        
    cmap=plt.cm.get_cmap(colour_map)
    norm = colors.Normalize(cmin,cmax) #TwoSlopeNorm(0, vmin=-3, vmax =1)
    
    plt.rcParams['font.sans-serif'] = "Helvetica"
    #plt.rcParams["figure.figsize"] = (6.8, 8)
    ax = plt.axes(projection=ccrs.NorthPolarStereo(central_longitude=-45.0, globe=None))
    plt.gcf().set_size_inches(6.8, 8)

    sea = cimgt.GoogleTiles(style = 'satellite')
    #topo = cimgt.MapboxTiles('pk.eyJ1Ijoia21zZWphbiIsImEiOiJjbGUwNG1kaDYwOWt1M25wNWsydWhjN200In0.Vw-DuMfDepMWLw9jgP_5xw', 
                             #'outdoors-v12')
    
    ax.add_image(sea, 5, alpha = 0.9, zorder = 0)
    #ax.add_image(topo, 5, zorder = 1)


    ax.set_extent([-26.5, -58, 58.8, 83.5])
    mapp = ax.pcolormesh(gridx, gridy, grid, norm = norm, cmap = cmap, alpha = 0.7, transform = ccrs.epsg(3413), zorder=2)
    
    gl = ax.gridlines(draw_labels=dict(top= 'x', left= 'y'), linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    
    gl.bottom_labels = False
    #gl.left_labels = True
    gl.right_labels = False
    #gl.top_labels = True
    gl.rotate_labels=False
    gl.labels_bbox_style = dict(alpha = 0.5, facecolor = 'white', edgecolor= 'none', boxstyle='round', pad = 0.1)
    gl.xpadding = -5
    gl.ypadding = -5
    gl.xlocator = mticker.FixedLocator([-75, -60, -45, -30, -15, 0])
    gl.xformatter = cartopy.mpl.ticker.LongitudeFormatter(direction_label=True, auto_hide=False)
    gl.ylocator = mticker.FixedLocator([60, 65, 70, 75, 80])
    gl.yformatter = cartopy.mpl.ticker.LatitudeFormatter(direction_label=True, auto_hide=False)
    gl.xlabel_style = dict(size= 8, color= 'black') #bbox= dict(alpha = 0.5, facecolor = 'white', edgecolor='black', boxstyle='round')
    gl.ylabel_style = dict(size= 8, color= 'black')

    for artist in gl.top_label_artists:
        artist.set_visible(True).draw()
    for artist in gl.left_label_artists:
        artist.set_visible(True)

    cont = ax.contour(gridx, gridy, grid, [1000, 1450, 2000, 3000], colors = ['black', 'red', 'black', 'black'], linewidths = 0.5, alpha = 0.6, transform = ccrs.epsg(3413), zorder=3)    
    ax.clabel(cont, cont.levels, inline=True, fontsize=6)

    cbaxes = inset_axes(ax, width="20%", height="2%", bbox_to_anchor = (1400,600,3000,3000), loc=8) #bbox_to_anchor = (0,0,1,1)
    cb = plt.colorbar(mapp, cax=cbaxes, ticks=[cmin, cmax], orientation='horizontal', pad = 0.03)
    cb.ax.tick_params(labelsize=9, labelcolor = 'white', pad = -0.1, bottom = False, labeltop = True, labelbottom = False)
    cb.outline.set_color('w')
    cbaxes.set_title(title, color = 'white', fontsize=12, fontweight = 'semibold', pad = 10)

    for i,n in enumerate(stations_names):
        
        ax.plot(stations_coords[i][0], stations_coords[i][1], 'ko', markersize=1.1, transform=ccrs.epsg(3413), zorder=4)
        ax.text(stations_labels[i][0], stations_labels[i][1], stations_names[i], fontsize=8, transform=ccrs.epsg(3413), zorder=5)


    ax.add_artist(ScaleBar(0.001, units = 'km', location = 3, frameon = False, color = 'white', font_properties = {'size': 9}, fixed_units = 'km'))


    plt.savefig(path+name + '.jpg', dpi = 600, bbox_inches='tight')
    plt.clf()

def slopemap(gridx, gridy, grid, title, period, colour_map, cmin, cmax, name, path):
    
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy
    import matplotlib.colors as colors
    import cartopy.io.img_tiles as cimgt
    import matplotlib.ticker as mticker
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    from matplotlib_scalebar.scalebar import ScaleBar
    import matplotlib.patches as mpatches
    
    #We use only PROMICE and GCNet stations
    stations_names = ['EGP', 'Summit', 'South Dome', 'Swiss Camp', 'GITS']
    
    stations_coords = [[245519, -1545750], [215048, -1889006], [9445, -2960973], [-168891, -2230133], [-386548, -1344511]]


    stations_labels = [[245519, -1515750], [215048, -1859006], [-129445, -2930973], [-148891, -2260133], [-386548, -1314511]]

    
    stations_elev = [2660, 3199, 2901, 1176, 1869]

        
    cmap=plt.cm.get_cmap(colour_map)
    norm = colors.Normalize(cmin,cmax) #TwoSlopeNorm(0, vmin=-3, vmax =1)
    
    plt.rcParams['font.sans-serif'] = "Helvetica"
    #plt.rcParams["figure.figsize"] = (6.8, 8)
    ax = plt.axes(projection=ccrs.NorthPolarStereo(central_longitude=-45.0, globe=None))
    plt.gcf().set_size_inches(6.8, 8)

    sea = cimgt.GoogleTiles(style = 'satellite')
    #topo = cimgt.MapboxTiles('pk.eyJ1Ijoia21zZWphbiIsImEiOiJjbGUwNG1kaDYwOWt1M25wNWsydWhjN200In0.Vw-DuMfDepMWLw9jgP_5xw', 
                             #'outdoors-v12')
    
    ax.add_image(sea, 5, alpha = 0.9, zorder = 0)
    #ax.add_image(topo, 5, zorder = 1)


    ax.set_extent([-26.5, -58, 58.8, 83.5])
    mapp = ax.pcolormesh(gridx, gridy, grid, norm = norm, cmap = cmap, alpha = 0.9, transform = ccrs.epsg(3413), zorder=2)
    
    gl = ax.gridlines(draw_labels=dict(top= 'x', left= 'y'), linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    
    gl.bottom_labels = False
    #gl.left_labels = True
    gl.right_labels = False
    #gl.top_labels = True
    gl.rotate_labels=False
    gl.labels_bbox_style = dict(alpha = 0.5, facecolor = 'white', edgecolor= 'none', boxstyle='round', pad = 0.1)
    gl.xpadding = -5
    gl.ypadding = -5
    gl.xlocator = mticker.FixedLocator([-75, -60, -45, -30, -15, 0])
    gl.xformatter = cartopy.mpl.ticker.LongitudeFormatter(direction_label=True, auto_hide=False)
    gl.ylocator = mticker.FixedLocator([60, 65, 70, 75, 80])
    gl.yformatter = cartopy.mpl.ticker.LatitudeFormatter(direction_label=True, auto_hide=False)
    gl.xlabel_style = dict(size= 8, color= 'black') #bbox= dict(alpha = 0.5, facecolor = 'white', edgecolor='black', boxstyle='round')
    gl.ylabel_style = dict(size= 8, color= 'black')

    for artist in gl.top_label_artists:
        artist.set_visible(True).draw()
    for artist in gl.left_label_artists:
        artist.set_visible(True)


    cbaxes = inset_axes(ax, width="20%", height="2%", bbox_to_anchor = (1400,600,3000,3000), loc=8) #bbox_to_anchor = (0,0,1,1)
    cb = plt.colorbar(mapp, cax=cbaxes, ticks=[cmin, cmax], orientation='horizontal', pad = 0.03)
    cb.ax.tick_params(labelsize=9, labelcolor = 'white', pad = -0.1, bottom = False, labeltop = True, labelbottom = False)
    cb.outline.set_color('w')
    cbaxes.set_title(title, color = 'white', fontsize=12, fontweight = 'semibold', pad = 10)

    for i,n in enumerate(stations_names):
        
        ax.plot(stations_coords[i][0], stations_coords[i][1], 'ko', markersize=1.1, transform=ccrs.epsg(3413), zorder=4)
        ax.text(stations_labels[i][0], stations_labels[i][1], stations_names[i], fontsize=8, transform=ccrs.epsg(3413), zorder=5)


    ax.add_artist(ScaleBar(0.001, units = 'km', location = 3, frameon = False, color = 'white', font_properties = {'size': 9}, fixed_units = 'km'))


    plt.savefig(path+name + '.jpg', dpi = 600, bbox_inches='tight')
    plt.clf()


def basinmap(gridx, gridy, grid, title, period, colour_map, cmin, cmax, name, path):
    
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy
    import matplotlib.colors as colors
    import cartopy.io.img_tiles as cimgt
    import matplotlib.ticker as mticker
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    from matplotlib_scalebar.scalebar import ScaleBar
    import matplotlib.patches as mpatches
    
    
    #We use only PROMICE and GCNet stations
    stations_names = ['Peterman_ELA', 'KPC_L', 'KPC_U', 'EGP', 'NASA_E', 'SCO_L', 'SCO_U', 'Summit',
                      'NASA_SE', 'South Dome', 'KAN_L', 'KAN_M', 'KAN_U', 'DYE-2', 'JAR_1', 'Swiss Camp', 'CP1', 
                      'UPE_L', 'UPE_U', 'NASA_U', 'GITS', 'NEEM']
    
    stations_coords = [[-244306, -1047222], [391168, -1023448], [374565, -1038474], [245519, -1545750], [422939, -1578126],
                       [605565, -1843811], [586431, -1830173], [215048, -1889006], [112747, -2580907], [9445, -2960973], 
                       [-216931, -2504048], [-168343, -2510999], [-89205, -2522571], [-57831, -2582094], [-184047, -2236749],
                       [-168891, -2230133], [-76703, -2200166], [-301483, -1841957], [-278493, -1846178], [-138429, -1756197],
                       [-386548, -1344511],[-138963, -1351904]]


    stations_labels = [[-244306, -1017222], [301168, -1003448], [334565, -1108474], [245519, -1515750], [422939, -1548126],
                       [445565, -1803811], [606431, -1890173], [215048, -1859006], [102747, -2565907], [-129445, -2930973], 
                       [-356931, -2484048], [-178343, -2490999], [-70205, -2552571], [-107831, -2652094], [-229047, -2306749],
                       [-148891, -2260133], [-76703, -2170166], [-321483, -1811957], [-248493, -1856178], [-138429, -1726197],
                       [-386548, -1314511],[-138963, -1321904]]

    
    stations_elev = [924, 370, 870, 2660, 2614, 460, 970, 3199, 2373, 2901, 670, 
                     1270, 1840, 2099, 932, 1176, 2022, 220, 940, 2334, 1869, 2454]
    
    cmap=plt.cm.get_cmap(colour_map)
    norm = colors.Normalize(cmin,cmax) #TwoSlopeNorm(0, vmin=-3, vmax =1)
    
    plt.rcParams['font.sans-serif'] = "Helvetica"
    #plt.rcParams["figure.figsize"] = (6.8, 8)
    ax = plt.axes(projection=ccrs.NorthPolarStereo(central_longitude=-45.0, globe=None))
    plt.gcf().set_size_inches(6.8, 8)

    sea = cimgt.GoogleTiles(style = 'satellite')
    #topo = cimgt.MapboxTiles('pk.eyJ1Ijoia21zZWphbiIsImEiOiJjbGUwNG1kaDYwOWt1M25wNWsydWhjN200In0.Vw-DuMfDepMWLw9jgP_5xw', 
                             #'outdoors-v12')
    
    ax.add_image(sea, 5, alpha = 0.9, zorder = 0)
    #ax.add_image(topo, 5, zorder = 1)


    ax.set_extent([-26.5, -58, 58.8, 83.5])
    mapp = ax.pcolormesh(gridx, gridy, grid, norm = norm, cmap = cmap, alpha = 0.8, transform = ccrs.epsg(3413), zorder=2)
    
    gl = ax.gridlines(draw_labels=dict(top= 'x', left= 'y'), linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    
    gl.bottom_labels = False
    #gl.left_labels = True
    gl.right_labels = False
    #gl.top_labels = True
    gl.rotate_labels=False
    gl.labels_bbox_style = dict(alpha = 0.5, facecolor = 'white', edgecolor= 'none', boxstyle='round', pad = 0.1)
    gl.xpadding = -5
    gl.ypadding = -5
    gl.xlocator = mticker.FixedLocator([-75, -60, -45, -30, -15, 0])
    gl.xformatter = cartopy.mpl.ticker.LongitudeFormatter(direction_label=True, auto_hide=False)
    gl.ylocator = mticker.FixedLocator([60, 65, 70, 75, 80])
    gl.yformatter = cartopy.mpl.ticker.LatitudeFormatter(direction_label=True, auto_hide=False)
    gl.xlabel_style = dict(size= 8, color= 'black') #bbox= dict(alpha = 0.5, facecolor = 'white', edgecolor='black', boxstyle='round')
    gl.ylabel_style = dict(size= 8, color= 'black')

    for artist in gl.top_label_artists:
        artist.set_visible(True).draw()
    for artist in gl.left_label_artists:
        artist.set_visible(True)

    for i,n in enumerate(stations_names):
        
        ax.plot(stations_coords[i][0], stations_coords[i][1], 'ko', markersize=1.1, transform=ccrs.epsg(3413), zorder=3)
        ax.text(stations_labels[i][0], stations_labels[i][1], stations_names[i], fontsize=8, transform=ccrs.epsg(3413), zorder=4)

    #plt.colorbar(mapp, pad = 0.03).ax.tick_params(labelsize=14)
    
    cols = [plt.cm.Dark2(i) for i in range(8)]
    
    handels = [mpatches.Rectangle((0, 0), 1, 1, facecolor=colors.to_hex(cols[0])), 
               mpatches.Rectangle((0, 0), 1, 1, facecolor=colors.to_hex(cols[1])),
               mpatches.Rectangle((0, 0), 1, 1, facecolor=colors.to_hex(cols[2])),
               mpatches.Rectangle((0, 0), 1, 1, facecolor=colors.to_hex(cols[3])),
               mpatches.Rectangle((0, 0), 1, 1, facecolor=colors.to_hex(cols[4])),
               mpatches.Rectangle((0, 0), 1, 1, facecolor=colors.to_hex(cols[5])),
               mpatches.Rectangle((0, 0), 1, 1, facecolor=colors.to_hex(cols[6]))]
    
    labels = ['NW', 'CW', 'SW', 'SE', 'CE', 'NE', 'NO']
    
    ax.add_artist(ScaleBar(0.001, units = 'km', location = 3, frameon = False, color = 'white', font_properties = {'size': 9}, fixed_units = 'km'))
    
    ax.legend(handels, labels, loc=4, framealpha=1)    

    plt.savefig(path+name + '.jpg', dpi = 600, bbox_inches='tight')
    plt.clf()



def timeseries_mass4(time, tseries1, terror1, label1, tseries2, terror2, label2, tseries3, terror3, label3, tseries4, terror4, label4, title, units, name, path):
    
    import matplotlib.pyplot as plt
    import numpy as np
    color = 'grey'
    color1 = 'purple'
    color2 = 'orange'
    color3 = 'red'
    color4 = 'mediumblue'
    
    te1b = np.array(tseries1)-np.array(terror1)
    te1t = np.array(tseries1)+np.array(terror1)
    te2b = np.array(tseries2)-np.array(terror2)
    te2t = np.array(tseries2)+np.array(terror2)
    te3b = np.array(tseries3)-np.array(terror3)
    te3t = np.array(tseries3)+np.array(terror3)
    te4b = np.array(tseries4)-np.array(terror4)
    te4t = np.array(tseries4)+np.array(terror4)

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex = 'all', sharey = 'all')

    plt.rcParams['font.sans-serif'] = "Helvetica"
    plt.gcf().set_size_inches(8, 6)

    plt.xlabel('time')
    fig.text( 0.0, 0.5, name + ' ' + units, va='center', rotation='vertical')
    
    plt.xlim(time[0], time[-1])
    plt.ylim(min([min(te1b), min(te2b), min(te3b), min(te4b)]) - (-max(te1b)/4), max([max(te1t), max(te2t), max(te3t), max(te4t)]) + (max(te1t))/4)
    

    ax1.plot(time, tseries2, marker = '*', fillstyle = 'full', markersize = 5, color = color, linewidth=0, alpha=0.8)
    ax1.plot(time, tseries2, color = color, linewidth=0.8, alpha=0.4)
    ax1.plot(time, tseries3, marker = 's', fillstyle = 'none', markersize = 3.5, color = color, linewidth=0, alpha=0.8)
    ax1.plot(time, tseries3, color = color, linewidth=0.8, alpha=0.4)
    ax1.plot(time, tseries4, marker = '1', fillstyle = 'full', markeredgewidth = 1.5, markersize = 5, color = color, linewidth=0, alpha=0.8)
    ax1.plot(time, tseries4, color = color, linewidth=0.8, alpha=0.4)
    l1, = ax1.plot(time, tseries1, marker = 'o', fillstyle = 'none', markersize = 4, color = color1, linewidth=0.8, alpha=0.8)
            
    ax1.fill_between(time, te1b, te1t, alpha=0.2, edgecolor=color1, facecolor=color1, antialiased=True, linewidth=0)

    ax1.legend([l1], [label1], loc = 'lower left', title_fontsize = 10)

    
    ax2.plot(time, tseries1, marker = 'o', fillstyle = 'none', markersize = 4, color = color, linewidth=0, alpha=0.8)
    ax2.plot(time, tseries1, color = color, linewidth=0.8, alpha=0.4)
    ax2.plot(time, tseries3, marker = 's', fillstyle = 'none', markersize = 3.5, color = color, linewidth=0, alpha=0.8)
    ax2.plot(time, tseries3, color = color, linewidth=0.8, alpha=0.4)
    ax2.plot(time, tseries4, marker = '1', fillstyle = 'full', markeredgewidth = 1.5, markersize = 5, color = color, linewidth=0, alpha=0.8)
    ax2.plot(time, tseries4, color = color, linewidth=0.8, alpha=0.4)
    l2, = ax2.plot(time, tseries2, marker = '*', fillstyle = 'full', markersize = 5, color = color2, linewidth=0.8, alpha=0.8)
            
    ax2.fill_between(time, te2b, te2t, alpha=0.2, edgecolor=color2, facecolor=color2, antialiased=True, linewidth=0)

    ax2.legend([l2], [label2], loc = 'lower left', title_fontsize = 10)


    ax3.plot(time, tseries1, marker = 'o', fillstyle = 'none', markersize = 4, color = color, linewidth=0, alpha=0.8)
    ax3.plot(time, tseries1, color = color, linewidth=0.8, alpha=0.4)
    ax3.plot(time, tseries2, marker = '*', fillstyle = 'full', markersize = 5, color = color, linewidth=0, alpha=0.8)
    ax3.plot(time, tseries2, color = color, linewidth=0.8, alpha=0.4)
    ax3.plot(time, tseries4, marker = '1', fillstyle = 'full', markeredgewidth = 1.5, markersize = 5, color = color, linewidth=0, alpha=0.8)
    ax3.plot(time, tseries4, color = color, linewidth=0.8, alpha=0.4)
    l3, = ax3.plot(time, tseries3, marker = 's', fillstyle = 'none', markersize = 3.5, color = color3, linewidth=0.8, alpha=0.8)
            
    ax3.fill_between(time, te3b, te3t, alpha=0.2, edgecolor=color3, facecolor=color3, antialiased=True, linewidth=0)

    ax3.legend([l3], [label3], loc = 'lower left', title_fontsize = 10)


    ax4.plot(time, tseries1, marker = 'o', fillstyle = 'none', markersize = 4, color = color, linewidth=0, alpha=0.8)
    ax4.plot(time, tseries1, color = color, linewidth=0.8, alpha=0.4)
    ax4.plot(time, tseries2, marker = '*', fillstyle = 'full', markersize = 5, color = color, linewidth=0, alpha=0.8)
    ax4.plot(time, tseries2, color = color, linewidth=0.8, alpha=0.4)
    ax4.plot(time, tseries3, marker = 's', fillstyle = 'none', markersize = 3.5, color = color, linewidth=0, alpha=0.8)
    ax4.plot(time, tseries3, color = color, linewidth=0.8, alpha=0.4)
    l4, = ax4.plot(time, tseries4, marker = '1', fillstyle = 'full', markeredgewidth = 1.5, markersize = 5, color = color4, linewidth=0.8, alpha=0.8)
            
    ax4.fill_between(time, te4b, te4t, alpha=0.2, edgecolor=color4, facecolor=color4, antialiased=True, linewidth=0)

    ax4.legend([l4], [label4], loc = 'lower left', title_fontsize = 10)
    
    plt.tight_layout()
    plt.savefig(path+name + '.jpg', dpi = 300, bbox_inches='tight')
    plt.clf()
    
    
def timeseries_mass5(time, tseries1, terror1, label1, tseries2, terror2, label2, tseries3, terror3, label3, tseries4, terror4, label4, gseries, gerror, glabel, title, units, name, path):
    
    import matplotlib.pyplot as plt
    import numpy as np
    color = 'grey'
    color1 = 'purple'
    color2 = 'orange'
    color3 = 'red'
    color4 = 'mediumblue'
    color5 = 'green'
    
    te1b = np.array(tseries1)-np.array(terror1)
    te1t = np.array(tseries1)+np.array(terror1)
    te2b = np.array(tseries2)-np.array(terror2)
    te2t = np.array(tseries2)+np.array(terror2)
    te3b = np.array(tseries3)-np.array(terror3)
    te3t = np.array(tseries3)+np.array(terror3)
    te4b = np.array(tseries4)-np.array(terror4)
    te4t = np.array(tseries4)+np.array(terror4)

    geb = gseries-gerror
    get = gseries+gerror
    
    
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, sharex = 'all', sharey = 'all')

    plt.rcParams['font.sans-serif'] = "Helvetica"
    plt.gcf().set_size_inches(8, 10)

    plt.xlabel('time')
    fig.text( 0.0, 0.5, name + ' ' + units, va='center', rotation='vertical')
    
    plt.xlim(time[0], time[-1])
    plt.ylim(min([min(te1b), min(te2b), min(te3b), min(te4b)]) - (-max(te1b)/4), max([max(te1t), max(te2t), max(te3t), max(te4t)]) + (max(te1t))/4)
    
    ax1.plot(time, gseries, marker = 'd', fillstyle = 'full', markersize = 4, color = color, linewidth=0.8, alpha=0.8 )
    ax1.plot(time, gseries, color = color, linewidth=0.8, alpha=0.4)
    ax1.plot(time, tseries2, marker = '*', fillstyle = 'full', markersize = 5, color = color, linewidth=0, alpha=0.8)
    ax1.plot(time, tseries2, color = color, linewidth=0.8, alpha=0.4)
    ax1.plot(time, tseries3, marker = 's', fillstyle = 'none', markersize = 3.5, color = color, linewidth=0, alpha=0.8)
    ax1.plot(time, tseries3, color = color, linewidth=0.8, alpha=0.4)
    ax1.plot(time, tseries4, marker = '1', fillstyle = 'full', markeredgewidth = 1.5, markersize = 5, color = color, linewidth=0, alpha=0.8)
    ax1.plot(time, tseries4, color = color, linewidth=0.8, alpha=0.4)
    l1, = ax1.plot(time, tseries1, marker = 'o', fillstyle = 'none', markersize = 4, color = color1, linewidth=0.8, alpha=0.8)
            
    ax1.fill_between(time, te1b, te1t, alpha=0.2, edgecolor=color1, facecolor=color1, antialiased=True, linewidth=0)

    ax1.legend([l1], [label1], loc = 'lower left', title_fontsize = 10)

    ax2.plot(time, gseries, marker = 'd', fillstyle = 'full', markersize = 4, color = color, linewidth=0.8, alpha=0.8 )
    ax2.plot(time, gseries, color = color, linewidth=0.8, alpha=0.4)    
    ax2.plot(time, tseries1, marker = 'o', fillstyle = 'none', markersize = 4, color = color, linewidth=0, alpha=0.8)
    ax2.plot(time, tseries1, color = color, linewidth=0.8, alpha=0.4)
    ax2.plot(time, tseries3, marker = 's', fillstyle = 'none', markersize = 3.5, color = color, linewidth=0, alpha=0.8)
    ax2.plot(time, tseries3, color = color, linewidth=0.8, alpha=0.4)
    ax2.plot(time, tseries4, marker = '1', fillstyle = 'full', markeredgewidth = 1.5, markersize = 5, color = color, linewidth=0, alpha=0.8)
    ax2.plot(time, tseries4, color = color, linewidth=0.8, alpha=0.4)
    l2, = ax2.plot(time, tseries2, marker = '*', fillstyle = 'full', markersize = 5, color = color2, linewidth=0.8, alpha=0.8)
            
    ax2.fill_between(time, te2b, te2t, alpha=0.2, edgecolor=color2, facecolor=color2, antialiased=True, linewidth=0)

    ax2.legend([l2], [label2], loc = 'lower left', title_fontsize = 10)

    ax3.plot(time, gseries, marker = 'd', fillstyle = 'full', markersize = 4, color = color, linewidth=0.8, alpha=0.8 )
    ax3.plot(time, gseries, color = color, linewidth=0.8, alpha=0.4)
    ax3.plot(time, tseries1, marker = 'o', fillstyle = 'none', markersize = 4, color = color, linewidth=0, alpha=0.8)
    ax3.plot(time, tseries1, color = color, linewidth=0.8, alpha=0.4)
    ax3.plot(time, tseries2, marker = '*', fillstyle = 'full', markersize = 5, color = color, linewidth=0, alpha=0.8)
    ax3.plot(time, tseries2, color = color, linewidth=0.8, alpha=0.4)
    ax3.plot(time, tseries4, marker = '1', fillstyle = 'full', markeredgewidth = 1.5, markersize = 5, color = color, linewidth=0, alpha=0.8)
    ax3.plot(time, tseries4, color = color, linewidth=0.8, alpha=0.4)
    l3, = ax3.plot(time, tseries3, marker = 's', fillstyle = 'none', markersize = 3.5, color = color3, linewidth=0.8, alpha=0.8)
            
    ax3.fill_between(time, te3b, te3t, alpha=0.2, edgecolor=color3, facecolor=color3, antialiased=True, linewidth=0)

    ax3.legend([l3], [label3], loc = 'lower left', title_fontsize = 10)

    ax4.plot(time, gseries, marker = 'd', fillstyle = 'full', markersize = 4, color = color, linewidth=0.8, alpha=0.8 )
    ax4.plot(time, gseries, color = color, linewidth=0.8, alpha=0.4)
    ax4.plot(time, tseries1, marker = 'o', fillstyle = 'none', markersize = 4, color = color, linewidth=0, alpha=0.8)
    ax4.plot(time, tseries1, color = color, linewidth=0.8, alpha=0.4)
    ax4.plot(time, tseries2, marker = '*', fillstyle = 'full', markersize = 5, color = color, linewidth=0, alpha=0.8)
    ax4.plot(time, tseries2, color = color, linewidth=0.8, alpha=0.4)
    ax4.plot(time, tseries3, marker = 's', fillstyle = 'none', markersize = 3.5, color = color, linewidth=0, alpha=0.8)
    ax4.plot(time, tseries3, color = color, linewidth=0.8, alpha=0.4)
    l4, = ax4.plot(time, tseries4, marker = '1', fillstyle = 'full', markeredgewidth = 1.5, markersize = 5, color = color4, linewidth=0.8, alpha=0.8)
            
    ax4.fill_between(time, te4b, te4t, alpha=0.2, edgecolor=color4, facecolor=color4, antialiased=True, linewidth=0)

    ax4.legend([l4], [label4], loc = 'lower left', title_fontsize = 10)
    
    ax5.plot(time, tseries4, marker = '1', fillstyle = 'full', markeredgewidth = 1.5, markersize = 5, color = color, linewidth=0.8, alpha=0.8)
    ax5.plot(time, tseries4, color = color, linewidth=0.8, alpha=0.4)
    ax5.plot(time, tseries1, marker = 'o', fillstyle = 'none', markersize = 4, color = color, linewidth=0, alpha=0.8)
    ax5.plot(time, tseries1, color = color, linewidth=0.8, alpha=0.4)
    ax5.plot(time, tseries2, marker = '*', fillstyle = 'full', markersize = 5, color = color, linewidth=0, alpha=0.8)
    ax5.plot(time, tseries2, color = color, linewidth=0.8, alpha=0.4)
    ax5.plot(time, tseries3, marker = 's', fillstyle = 'none', markersize = 3.5, color = color, linewidth=0, alpha=0.8)
    ax5.plot(time, tseries3, color = color, linewidth=0.8, alpha=0.4)
    l5, = ax5.plot(time, gseries, marker = 'd', fillstyle = 'full', markersize = 4, color = color5, linewidth=0.8, alpha=0.8 )
            
    ax5.fill_between(time, geb, get, alpha=0.2, edgecolor=color5, facecolor=color5, antialiased=True, linewidth=0)

    ax5.legend([l5], [glabel], loc = 'lower left', title_fontsize = 10)

        
    plt.tight_layout()
    plt.savefig(path+name + '.jpg', dpi = 300, bbox_inches='tight')
    plt.clf()


def timeseries_mass6(time, tseries1, terror1, label1, tseries2, terror2, label2, tseries3, terror3, label3, tseries4, terror4, label4, gseries, gerror, glabel, sseries, slabel, title, units, name, path):
    
    import matplotlib.pyplot as plt
    import numpy as np
    color = 'grey'
    color1 = 'purple'
    color2 = 'orange'
    color3 = 'red'
    color4 = 'mediumblue'
    color5 = 'green'
    color6 = 'teal'
    
    te1b = np.array(tseries1)-np.array(terror1)
    te1t = np.array(tseries1)+np.array(terror1)
    te2b = np.array(tseries2)-np.array(terror2)
    te2t = np.array(tseries2)+np.array(terror2)
    te3b = np.array(tseries3)-np.array(terror3)
    te3t = np.array(tseries3)+np.array(terror3)
    te4b = np.array(tseries4)-np.array(terror4)
    te4t = np.array(tseries4)+np.array(terror4)

    geb = gseries-gerror
    get = gseries+gerror
    
    
    fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, 1, sharex = 'all', sharey = 'all')

    plt.rcParams['font.sans-serif'] = "Helvetica"
    plt.gcf().set_size_inches(8, 13)

    plt.xlabel('time')
    fig.text( 0.0, 0.5, name + ' ' + units, va='center', rotation='vertical')
    
    plt.xlim(time[0], time[-1])
    plt.ylim(min([min(te1b), min(te2b), min(te3b), min(te4b)]) - (-max(te1b)/4), max([max(te1t), max(te2t), max(te3t), max(te4t)]) + (max(te1t))/4)

    ax1.plot(time, sseries, marker = 10, fillstyle = 'full', markersize = 4, color = color, linewidth=0.8, alpha=0.8 )    
    ax1.plot(time, gseries, marker = 'd', fillstyle = 'full', markersize = 4, color = color, linewidth=0.8, alpha=0.8 )
    ax1.plot(time, gseries, color = color, linewidth=0.8, alpha=0.4)
    ax1.plot(time, tseries2, marker = '*', fillstyle = 'full', markersize = 5, color = color, linewidth=0, alpha=0.8)
    ax1.plot(time, tseries2, color = color, linewidth=0.8, alpha=0.4)
    ax1.plot(time, tseries3, marker = 's', fillstyle = 'none', markersize = 3.5, color = color, linewidth=0, alpha=0.8)
    ax1.plot(time, tseries3, color = color, linewidth=0.8, alpha=0.4)
    ax1.plot(time, tseries4, marker = '1', fillstyle = 'full', markeredgewidth = 1.5, markersize = 5, color = color, linewidth=0, alpha=0.8)
    ax1.plot(time, tseries4, color = color, linewidth=0.8, alpha=0.4)
    l1, = ax1.plot(time, tseries1, marker = 'o', fillstyle = 'none', markersize = 4, color = color1, linewidth=0.8, alpha=0.8)
            
    ax1.fill_between(time, te1b, te1t, alpha=0.2, edgecolor=color1, facecolor=color1, antialiased=True, linewidth=0)

    ax1.legend([l1], [label1], loc = 'lower left', title_fontsize = 10)

    ax2.plot(time, sseries, marker = 10, fillstyle = 'full', markersize = 4, color = color, linewidth=0.8, alpha=0.8 )    
    ax2.plot(time, gseries, marker = 'd', fillstyle = 'full', markersize = 4, color = color, linewidth=0.8, alpha=0.8 )
    ax2.plot(time, gseries, color = color, linewidth=0.8, alpha=0.4)    
    ax2.plot(time, tseries1, marker = 'o', fillstyle = 'none', markersize = 4, color = color, linewidth=0, alpha=0.8)
    ax2.plot(time, tseries1, color = color, linewidth=0.8, alpha=0.4)
    ax2.plot(time, tseries3, marker = 's', fillstyle = 'none', markersize = 3.5, color = color, linewidth=0, alpha=0.8)
    ax2.plot(time, tseries3, color = color, linewidth=0.8, alpha=0.4)
    ax2.plot(time, tseries4, marker = '1', fillstyle = 'full', markeredgewidth = 1.5, markersize = 5, color = color, linewidth=0, alpha=0.8)
    ax2.plot(time, tseries4, color = color, linewidth=0.8, alpha=0.4)
    l2, = ax2.plot(time, tseries2, marker = '*', fillstyle = 'full', markersize = 5, color = color2, linewidth=0.8, alpha=0.8)
            
    ax2.fill_between(time, te2b, te2t, alpha=0.2, edgecolor=color2, facecolor=color2, antialiased=True, linewidth=0)

    ax2.legend([l2], [label2], loc = 'lower left', title_fontsize = 10)

    ax3.plot(time, sseries, marker = 10, fillstyle = 'full', markersize = 4, color = color, linewidth=0.8, alpha=0.8 )    
    ax3.plot(time, gseries, marker = 'd', fillstyle = 'full', markersize = 4, color = color, linewidth=0.8, alpha=0.8 )
    ax3.plot(time, gseries, color = color, linewidth=0.8, alpha=0.4)
    ax3.plot(time, tseries1, marker = 'o', fillstyle = 'none', markersize = 4, color = color, linewidth=0, alpha=0.8)
    ax3.plot(time, tseries1, color = color, linewidth=0.8, alpha=0.4)
    ax3.plot(time, tseries2, marker = '*', fillstyle = 'full', markersize = 5, color = color, linewidth=0, alpha=0.8)
    ax3.plot(time, tseries2, color = color, linewidth=0.8, alpha=0.4)
    ax3.plot(time, tseries4, marker = '1', fillstyle = 'full', markeredgewidth = 1.5, markersize = 5, color = color, linewidth=0, alpha=0.8)
    ax3.plot(time, tseries4, color = color, linewidth=0.8, alpha=0.4)
    l3, = ax3.plot(time, tseries3, marker = 's', fillstyle = 'none', markersize = 3.5, color = color3, linewidth=0.8, alpha=0.8)
            
    ax3.fill_between(time, te3b, te3t, alpha=0.2, edgecolor=color3, facecolor=color3, antialiased=True, linewidth=0)

    ax3.legend([l3], [label3], loc = 'lower left', title_fontsize = 10)

    ax4.plot(time, sseries, marker = 10, fillstyle = 'full', markersize = 4, color = color, linewidth=0.8, alpha=0.8 )    
    ax4.plot(time, gseries, marker = 'd', fillstyle = 'full', markersize = 4, color = color, linewidth=0.8, alpha=0.8 )
    ax4.plot(time, gseries, color = color, linewidth=0.8, alpha=0.4)
    ax4.plot(time, tseries1, marker = 'o', fillstyle = 'none', markersize = 4, color = color, linewidth=0, alpha=0.8)
    ax4.plot(time, tseries1, color = color, linewidth=0.8, alpha=0.4)
    ax4.plot(time, tseries2, marker = '*', fillstyle = 'full', markersize = 5, color = color, linewidth=0, alpha=0.8)
    ax4.plot(time, tseries2, color = color, linewidth=0.8, alpha=0.4)
    ax4.plot(time, tseries3, marker = 's', fillstyle = 'none', markersize = 3.5, color = color, linewidth=0, alpha=0.8)
    ax4.plot(time, tseries3, color = color, linewidth=0.8, alpha=0.4)
    l4, = ax4.plot(time, tseries4, marker = '1', fillstyle = 'full', markeredgewidth = 1.5, markersize = 5, color = color4, linewidth=0.8, alpha=0.8)
            
    ax4.fill_between(time, te4b, te4t, alpha=0.2, edgecolor=color4, facecolor=color4, antialiased=True, linewidth=0)

    ax4.legend([l4], [label4], loc = 'lower left', title_fontsize = 10)

    ax5.plot(time, sseries, marker = 10, fillstyle = 'full', markersize = 4, color = color, linewidth=0.8, alpha=0.8 )        
    ax5.plot(time, tseries4, marker = '1', fillstyle = 'full', markeredgewidth = 1.5, markersize = 5, color = color, linewidth=0.8, alpha=0.8)
    ax5.plot(time, tseries4, color = color, linewidth=0.8, alpha=0.4)
    ax5.plot(time, tseries1, marker = 'o', fillstyle = 'none', markersize = 4, color = color, linewidth=0, alpha=0.8)
    ax5.plot(time, tseries1, color = color, linewidth=0.8, alpha=0.4)
    ax5.plot(time, tseries2, marker = '*', fillstyle = 'full', markersize = 5, color = color, linewidth=0, alpha=0.8)
    ax5.plot(time, tseries2, color = color, linewidth=0.8, alpha=0.4)
    ax5.plot(time, tseries3, marker = 's', fillstyle = 'none', markersize = 3.5, color = color, linewidth=0, alpha=0.8)
    ax5.plot(time, tseries3, color = color, linewidth=0.8, alpha=0.4)
    l5, = ax5.plot(time, gseries, marker = 'd', fillstyle = 'full', markersize = 4, color = color5, linewidth=0.8, alpha=0.8 )
            
    ax5.fill_between(time, geb, get, alpha=0.2, edgecolor=color5, facecolor=color5, antialiased=True, linewidth=0)

    ax5.legend([l5], [glabel], loc = 'lower left', title_fontsize = 10)

    ax6.plot(time, gseries, marker = 'd', fillstyle = 'full', markersize = 4, color = color, linewidth=0.8, alpha=0.8 )
    ax6.plot(time, gseries, color = color, linewidth=0.8, alpha=0.4)        
    ax6.plot(time, tseries4, marker = '1', fillstyle = 'full', markeredgewidth = 1.5, markersize = 5, color = color, linewidth=0.8, alpha=0.8)
    ax6.plot(time, tseries4, color = color, linewidth=0.8, alpha=0.4)
    ax6.plot(time, tseries1, marker = 'o', fillstyle = 'none', markersize = 4, color = color, linewidth=0, alpha=0.8)
    ax6.plot(time, tseries1, color = color, linewidth=0.8, alpha=0.4)
    ax6.plot(time, tseries2, marker = '*', fillstyle = 'full', markersize = 5, color = color, linewidth=0, alpha=0.8)
    ax6.plot(time, tseries2, color = color, linewidth=0.8, alpha=0.4)
    ax6.plot(time, tseries3, marker = 's', fillstyle = 'none', markersize = 3.5, color = color, linewidth=0, alpha=0.8)
    ax6.plot(time, tseries3, color = color, linewidth=0.8, alpha=0.4)
    l6, = ax6.plot(time, sseries, marker = 10, fillstyle = 'full', markersize = 4, color = color6, linewidth=0.8, alpha=0.8 )

    ax6.legend([l6], [slabel], loc = 'lower left', title_fontsize = 10)

        
    plt.tight_layout()
    plt.savefig(path+name + '.jpg', dpi = 300, bbox_inches='tight')
    plt.clf()


def grace_series_output(datar, dataz, label1, label2, period_start, period_end, title, units, name, path):
    
    import matplotlib.pyplot as plt
    import numpy as np
    from firnpy import analise
    
    color1 = 'royalblue'
    color2 = 'darkorchid'    
    
    
    massr_fit, massr_errorfit = analise.fit_grace(datar['ref_mass'], period_start, period_end)
    massz_fit, massz_errorfit = analise.fit_grace(dataz['ref_mass'], period_start, period_end)
    
    f = open(path + name + '_rates_' + period_start + '_' + period_end + '.txt', 'w')
    f.write('Study period: ' + period_start + '-' + period_end)
    f.write('\n')

    f.write('GRACE total mass vhange for GIS (Rignot): ' + str(np.round(massr_fit, 2)) + ' +/- ' + str(np.round(massr_errorfit, 2)) + ' Gt/yr')
    f.write('\n')
    f.write('GRACE total mass vhange for GIS (Zwally): ' + str(np.round(massz_fit, 2)) + ' +/- ' + str(np.round(massz_errorfit, 2)) + ' Gt/yr')
    f.write('\n')

    f.close()

    plt.rcParams['font.sans-serif'] = "Helvetica"
    
    plt.title(title)
    
    plt.plot(datar['time'], datar['ref_mass'], color = color1 )
    plt.plot(dataz['time'], dataz['ref_mass'], color = color2 )
    
    plt.gcf().set_size_inches(8, 6)
    
    plt.legend([label1, label2])
    
    plt.fill_between(datar['time'], datar['ref_mass']-datar['std'], datar['ref_mass']+datar['std'], alpha=0.2, edgecolor=color1, facecolor=color1, antialiased=True, linewidth=0)
    plt.fill_between(dataz['time'], dataz['ref_mass']-dataz['std'], dataz['ref_mass']+dataz['std'], alpha=0.2, edgecolor=color1, facecolor=color1, antialiased=True, linewidth=0)

    
    plt.xlabel('time')
    plt.ylabel(name + ' ' + units)
    
    plt.savefig(path+name + '_' + period_start + '_' + period_end + '.jpg', dpi = 300, bbox_inches='tight')
    plt.clf()



def grace_series(dataz, label1, period_start, period_end, title, units, name, path):
    
    import matplotlib.pyplot as plt
    import numpy as np
    from firnpy import analise
    
    color1 = 'green'    
    
    te1b = dataz['ref_mass']-dataz['std']
    te1t = dataz['ref_mass']+dataz['std']
    
    
    plt.rcParams['font.sans-serif'] = "Helvetica"
    
    plt.plot(dataz['time'], dataz['ref_mass'], marker = 'o', fillstyle = 'none', markersize = 4, color = color1, linewidth=0.8, alpha=0.8 )
    
    plt.gcf().set_size_inches(8, 2.5)
    
    plt.legend([label1], title = title, fontsize = 8)
    
    plt.fill_between(dataz['time'], te1b, te1t, alpha=0.1, edgecolor=color1, facecolor=color1, antialiased=True, linewidth=0)
    plt.xlim(dataz['time'][0], dataz['time'][-1])
    plt.ylim(min(te1b) - 200, max(te1t) + 200)

    plt.xlabel('time')
    plt.ylabel(name + ' ' + units)
    
    
    plt.savefig(path+name + '_' + period_start + '_' + period_end + '.jpg', dpi = 300, bbox_inches='tight')
    plt.clf()

def grace_noseason_series(time, series, error, label1, period_start, period_end, title, units, name, path):
    
    import matplotlib.pyplot as plt
    import numpy as np
    from firnpy import analise
    
    color1 = 'green'    
    
    te1b = series-error
    te1t = series+error
    
    
    plt.rcParams['font.sans-serif'] = "Helvetica"
    
    #plt.plot(time, series, marker = 'o', fillstyle = 'none', markersize = 4, color = color1, linewidth=0.8, alpha=0.8 )
    plt.plot(time, series, color = color1, linewidth=0.8, alpha=0.8 )
    
    plt.gcf().set_size_inches(8, 2.5)
    
    plt.legend([label1], title = title, fontsize = 8)
    
    plt.fill_between(time, te1b, te1t, alpha=0.1, edgecolor=color1, facecolor=color1, antialiased=True, linewidth=0)
    plt.xlim(time[0], time[-1])
    plt.ylim(min(te1b) - 200, max(te1t) + 200)

    plt.xlabel('time')
    plt.ylabel(name + ' ' + units)
    
    
    plt.savefig(path+name + '_' + period_start + '_' + period_end + '.jpg', dpi = 300, bbox_inches='tight')
    plt.clf()


def smb_series(time, timeseries, label1, period_start, period_end, title, units, name, path):
    
    import matplotlib.pyplot as plt
    import numpy as np
    from firnpy import analise
    
    color1 = 'teal'    
    
    #te1b = timeseries-error
    #te1t = timeseries+error

    
    plt.rcParams['font.sans-serif'] = "Helvetica"
    
    plt.plot(time, timeseries, marker = 'o', fillstyle = 'none', markersize = 4, color = color1, linewidth=0.8, alpha=0.8 )
    
    plt.gcf().set_size_inches(8, 2.5)
    
    plt.legend([label1], title = title, fontsize = 8)
    
    #plt.fill_between(time, te1b, te1t, alpha=0.1, edgecolor=color1, facecolor=color1, antialiased=True, linewidth=0)
    plt.xlim(time[0], time[-1])
    #plt.ylim(min(te1b) - 200, max(te1t) + 200)

    plt.xlabel('time')
    plt.ylabel(name + ' ' + units)
    
    
    plt.savefig(path+name + '_' + period_start + '_' + period_end + '.jpg', dpi = 300, bbox_inches='tight')
    plt.clf()



def grace_rmse(massseries, graceseries, label1, period_start, period_end, title, units, name, path, rmseval, rval):
    
    import matplotlib.pyplot as plt
    import numpy as np
    from firnpy import analise
        
    
    plt.rcParams['font.sans-serif'] = "Helvetica"
    
    plt.scatter(massseries, graceseries, c = 'black', marker = 'o', alpha=0.6, zorder = 1)
    
    plt.xlim(-300, 500)
    plt.ylim(-300, 500)

    ax = plt.gca()
    
    plt.gcf().set_size_inches(6, 6)
    
    #plt.legend([label1], title = title, fontsize = 8)
    
    plt.plot(ax.get_xlim(), ax.get_ylim(), ls="-", c='black', zorder = 0)

    plt.xlabel('GRACE' + ' (' + units + ')')
    plt.ylabel(name + ' (' + units + ')')
    
    plt.text(-200, 300, 'RMSE = ' + str(rmseval) + ' ' + units, fontfamily = 'Arial', fontsize = 12, fontweight = 'extra bold')
    plt.text(-200, 400, 'R = ' + str(rval), fontfamily = 'Arial', fontsize = 12, fontweight = 'extra bold')
    
    plt.savefig(path+name + '_vs_GRACE_' + period_start + '_' + period_end + '.jpg', dpi = 300, bbox_inches='tight')
    plt.clf()


def elev_rmse(fdmseries, csseries, period_start, period_end, units, name, path, rmseval, rval):
    
    import matplotlib.pyplot as plt
    import numpy as np
    from firnpy import analise
        
    
    plt.rcParams['font.sans-serif'] = "Helvetica"
    
    plt.scatter(fdmseries, csseries, c = 'black', marker = 'o', alpha=0.6, zorder = 1)
    
    plt.xlim(-1, 0.4)
    plt.ylim(-1, 0.4)

    ax = plt.gca()
    
    plt.gcf().set_size_inches(6, 6)
        
    plt.plot(ax.get_xlim(), ax.get_ylim(), ls="-", c='black', zorder = 0)

    plt.xlabel('CryoSat-2' + ' (' + units + ')')
    plt.ylabel('IMAU_FDM' + ' (' + units + ')')
    
    plt.text(-0.3, -0.7, 'RMSE = ' + str(rmseval) + ' ' + units, fontfamily = 'Arial', fontsize = 12, fontweight = 'extra bold')
    plt.text(-0.3, -0.8, 'R = ' + str(rval), fontfamily = 'Arial', fontsize = 12, fontweight = 'extra bold')
    
    plt.savefig(path+ 'IMAU-FDM_vs_CyroSat2_' + name + period_start + '_' + period_end + '.jpg', dpi = 300, bbox_inches='tight')
    plt.clf()

    
    
def stations_massseries(time, gridx, gridy, grids1, egrids1, grids2, egrids2, grids3, egrids3, grids4, egrids4, grids5, grids6, egrids6, units, title, path): # units can be 'Gt' or 'kg'
    
    import matplotlib.pyplot as plt
    import numpy as np
    from firnpy import analise
    
    if units == 'Gt':
        scale = 10**-6 # because 10**-12 to go to Gt but 10**6 to go to m2 from km2
        #scale = 10**-12
        exp = ''
    elif units == 'kg':
        scale = 10**6 # because 10**6 to go to m2 from km2
        #scale = 1
        exp = ''
    
    color1 = 'purple'
    color2 = 'orange'
    color3 = 'red'
    color4 = 'mediumblue'
        
    #We use only PROMICE and GCNet stations
    stations_names = ['Peterman_ELA', 'KPC_L', 'KPC_U', 'EGP', 'NASA_E', 'SCO_L', 'SCO_U', 'Summit',
                      'NASA_SE', 'South Dome', 'KAN_L', 'KAN_M', 'KAN_U', 'DYE-2', 'JAR_1', 'Swiss Camp', 'CP1', 
                      'UPE_L', 'UPE_U', 'NASA_U', 'GITS', 'NEEM']
    
    stations_coords = [[-244306, -1047222], [391168, -1023448], [374565, -1038474], [245519, -1545750], [422939, -1578126],
                       [605565, -1843811], [586431, -1830173], [215048, -1889006], [112747, -2580907], [9445, -2960973], 
                       [-216931, -2504048], [-168343, -2510999], [-89205, -2522571], [-57831, -2582094], [-184047, -2236749],
                       [-168891, -2230133], [-76703, -2200166], [-301483, -1841957], [-278493, -1846178], [-138429, -1756197],
                       [-386548, -1344511],[-138963, -1351904]]
    
    stations_elev = [924, 370, 870, 2660, 2614, 460, 970, 3199, 2373, 2901, 670, 
                     1270, 1840, 2099, 932, 1176, 2022, 220, 940, 2334, 1869, 2454]
    

    #plt.rcParams["figure.figsize"] = (8, 6)
    
    rho_i = 917 #[kg/m3]
    rho_w = 997 #[kg/m3]
    station_area = 10**2 #km2
    
    f = open(path + 'stations_rates.txt', 'w')
    f.write('Stations')
    f.write('\n')
    
    for i,r in enumerate(stations_names):
                
        name = 'station_' + stations_names[i]
        
        f.write(name)
        f.write('\n')
        

        
        #AO
        series1 = analise.station(grids1, gridx, gridy, stations_coords[i], np.sqrt(station_area)) * rho_i * scale * station_area
        error1 = analise.statione_CS(egrids1, gridx, gridy, stations_coords[i], np.sqrt(station_area), 3) * rho_i * scale * station_area
        
        series1_fit, series1_errorfit = analise.fit_series(series1)
        
        f.write('Altimetry Only: ' + str(round(series1_fit, 4)) + exp + ' +/- ' + str(round(series1_errorfit,4)) + exp + ' ' + units + '/yr')
        f.write('\n')
        
        #K
        series2 = analise.station(grids1*grids2, gridx, gridy, stations_coords[i], np.sqrt(station_area)) * scale * station_area
        error2 = np.sqrt((analise.statione_CS(egrids1, gridx, gridy, stations_coords[i], np.sqrt(station_area), 3) * rho_i)**2 + (analise.station(grids1*egrids2, gridx, gridy, stations_coords[i], np.sqrt(station_area)))**2) * scale * station_area 
        
        series2_fit, series2_errorfit = analise.fit_series(series2)
        
        f.write('Kapelsberger: ' + str(round(series2_fit, 4)) + exp + ' +/- ' + str(round(series2_errorfit, 4)) + exp + ' ' + units + '/yr')
        f.write('\n')
        
        #MCM
        gm = (grids1 + grids6) * grids3 #[kg/m2/yr]
        series3 = analise.station(gm, gridx, gridy, stations_coords[i], np.sqrt(station_area)) * scale * station_area
        e1 = np.sqrt(analise.statione_CS(egrids1*grids3, gridx, gridy, stations_coords[i], np.sqrt(station_area), 3)**2 + analise.station(np.array([egrids6*grids3 for g in egrids1]), gridx, gridy, stations_coords[i], np.sqrt(station_area))**2)
        e2 =  analise.station((grids1 + grids6)* egrids3, gridx, gridy, stations_coords[i], np.sqrt(station_area))
        error3 = np.sqrt(e1**2 + e2**2) * scale * station_area

        series3_fit, series3_errorfit = analise.fit_series(series3)
        
        f.write('McMillan: ' + str(round(series3_fit, 4)) +  exp + ' +/- ' + str(round(series3_errorfit, 4)) + exp + ' ' + units  + '/yr')
        f.write('\n')

        #IMAU
        gi = ((grids1 - grids4) * rho_i) + (grids5 * rho_w) #[kg/m2/yr]
        series4 = analise.station(gi, gridx, gridy, stations_coords[i], np.sqrt(station_area)) * scale * station_area
        error4 = np.sqrt(analise.statione_CS(egrids1, gridx, gridy, stations_coords[i], np.sqrt(station_area), 3)**2 + analise.station(egrids4, gridx, gridy, stations_coords[i], np.sqrt(station_area))**2) * rho_i * scale * station_area
        
        series4_fit, series4_errorfit = analise.fit_series(series4)
        
        f.write('IMAU: ' + str(round(series4_fit, 4)) + exp + ' +/- ' + str(round(series4_errorfit, 4)) + exp + ' ' + units + '/yr')
        f.write('\n')
        
        
        te1b = np.array(series1)-np.array(error1)
        te1t = np.array(series1)+np.array(error1)
        te2b = np.array(series2)-np.array(error2)
        te2t = np.array(series2)+np.array(error2)
        te3b = np.array(series3)-np.array(error3)
        te3t = np.array(series3)+np.array(error3)
        te4b = np.array(series4)-np.array(error4)
        te4t = np.array(series4)+np.array(error4)

        
        plt.rcParams['font.sans-serif'] = "Helvetica"
    
        plt.plot(time, series1, marker = 'o', fillstyle = 'none', markersize = 4, color = color1, linewidth=0.8, alpha=0.8 )
        plt.plot(time, series2, marker = '*', fillstyle = 'full', markersize = 5, color = color2, linewidth=0.8, alpha=0.8)
        plt.plot(time, series3, marker = 's', fillstyle = 'none', markersize = 3.5, color = color3, linewidth=0.8, alpha=0.8)
        plt.plot(time, series4, marker = '1', fillstyle = 'full', markersize = 5, color = color4, linewidth=0.8, alpha=0.8)
        
        plt.gcf().set_size_inches(8, 2.5)
        
        plt.legend(['Altimetry Only', 'Kappelsberger', 'McMillan', 'IMAU'], title = title, fontsize = 8)
        
        plt.fill_between(time, te1b, te1t, alpha=0.1, edgecolor=color1, facecolor=color1, antialiased=True, linewidth=0)
        plt.fill_between(time, te2b, te2t, alpha=0.1, edgecolor=color2, facecolor=color2, antialiased=True, linewidth=0)
        plt.fill_between(time, te3b, te3t, alpha=0.1, edgecolor=color3, facecolor=color3, antialiased=True, linewidth=0)
        plt.fill_between(time, te4b, te4t, alpha=0.1, edgecolor=color4, facecolor=color4, antialiased=True, linewidth=0)
    
        plt.xlabel('time')
        plt.ylabel(name + ' ' + exp + ' ' + units + '/m2')
        
        
        plt.xlim(time[0], time[-1])
        plt.ylim(min([min(te1b), min(te2b), min(te3b), min(te4b)]) - (-max(te1b)/4), max([max(te1t), max(te2t), max(te3t), max(te4t)]) + (max(te1t))/4)

        #ax = plt.gca()

        plt.savefig(path+name +'_mass' + '.jpg', dpi = 300, bbox_inches='tight')
        plt.clf()

    f.close()

def basins_massseries(bgrid, time, gridx, gridy, grids1, egrids1, grids2, egrids2, grids3, egrids3, grids4, egrids4, grids5, grids6, egrids6, units, title, path): # units can be 'Gt' or 'kg'
    
    import matplotlib.pyplot as plt
    import numpy as np
    from firnpy import analise
    import pandas as pd
    
    if units == 'Gt':
        scale = 10**-6 # because 10**-12 to go to Gt but 10**6 to go to m2 from km2
        #scale = 10**-12
        exp = ''
    elif units == 'kg':
        scale = 10**6 # because 10**6 to go to m2 from km2
        #scale = 1
        exp = ''

    
    color1 = 'purple'
    color2 = 'orange'
    color3 = 'red'
    color4 = 'mediumblue'

    
    #Basins
    basins = [1, 2, 3, 4, 5, 6, 7]
    basin_labels = ['NW', 'CW', 'SW', 'SE', 'CE', 'NE', 'NO']
    methods = ['AO', 'K', 'MCM', 'IMAU']
    colors = ['purple', 'orange', 'red', 'mediumblue']

    #plt.rcParams["figure.figsize"] = (8, 6)
    
    rho_i = 917 #[kg/m3]
    rho_w = 997 #[kg/m3]
    cell_size = 1.5**2 #m2
    
    
    f = open(path + 'basins_rates.txt', 'w')
    f.write('Basins')
    f.write('\n')

    basin =[]
    method =[]
    rate =[]
    error =[]
    color = []
    for i,b in enumerate(basins):
        
        
        name = 'basin_' + basin_labels[i]

        f.write(name)
        f.write('\n')
        
        basin_area = bgrid[bgrid==b].size * cell_size
        
        
        #AO
        series1 = analise.basin(grids1, gridx, gridy, bgrid, b) * rho_i * scale * basin_area
        error1 = analise.basine_CS(egrids1, gridx, gridy, bgrid, b, 3) * rho_i * scale * basin_area
        
        series1_fit, series1_errorfit = analise.fit_series(series1)

        f.write('Altimetry Only: ' + str(round(series1_fit, 4)) +  exp + ' +/- ' + str(round(series1_errorfit,4)) + exp + ' ' + units + '/yr')
        f.write('\n')

        
        #K
        series2 = analise.basin(grids1*grids2, gridx, gridy, bgrid, b) * scale * basin_area
        error2 = np.sqrt((analise.basine_CS(egrids1, gridx, gridy, bgrid, b, 3) * rho_i)**2 + (analise.basin(grids1*egrids2, gridx, gridy, bgrid, b))**2) * scale * basin_area 

        series2_fit, series2_errorfit = analise.fit_series(series2)
        
        f.write('Kapelsberger: ' + str(round(series2_fit, 4)) +  exp + ' +/- ' + str(round(series2_errorfit, 4)) + exp + ' ' + units + '/yr')
        f.write('\n')

        #MCM
        gm = (grids1 + grids6) * grids3 #[kg/m2/yr]
        series3 = analise.basin(gm, gridx, gridy, bgrid, b) * scale * basin_area
        e1 = np.sqrt(analise.basine_CS(egrids1*grids3, gridx, gridy, bgrid, b, 3)**2 + analise.basin(np.array([egrids6*grids3 for g in egrids1]), gridx, gridy, bgrid, b)**2)
        e2 =  analise.basin((grids1 + grids6)* egrids3, gridx, gridy, bgrid, b)
        error3 = np.sqrt(e1**2 + e2**2) * scale #* basin_area

        series3_fit, series3_errorfit = analise.fit_series(series3)
        
        f.write('McMillan: ' + str(round(series3_fit, 4)) +  exp + ' +/- ' + str(round(series3_errorfit, 4)) + exp + ' ' + units + '/yr')
        f.write('\n')
        

        #IMAU
        gi = ((grids1 - grids4) * rho_i) + (grids5 * rho_w) #[kg/m2/yr]
        series4 = analise.basin(gi, gridx, gridy, bgrid, b) * scale * basin_area
        error4 = np.sqrt(analise.basine_CS(egrids1, gridx, gridy, bgrid, b, 3)**2 + analise.basin(egrids4, gridx, gridy, bgrid, b)**2) * rho_i * scale * basin_area

        series4_fit, series4_errorfit = analise.fit_series(series4)
        
        f.write('IMAU: ' + str(round(series4_fit, 4)) +  exp + ' +/- ' + str(round(series4_errorfit, 4)) + exp + ' ' + units + '/yr')
        f.write('\n')
        
        seriesf = [round(series1_fit, 4), round(series2_fit, 4), round(series3_fit, 4), round(series4_fit, 4)]
        errorsf = [round(series1_errorfit, 4), round(series2_errorfit, 4), round(series3_errorfit, 4), round(series4_errorfit, 4)]
        
        for j,m in enumerate(methods):
            basin.extend([basin_labels[i]])
            method.extend([m])
            rate.extend([seriesf[j]])
            error.extend([errorsf[j]])
            color.extend([colors[j]])

        
        te1b = np.array(series1)-np.array(error1)
        te1t = np.array(series1)+np.array(error1)
        te2b = np.array(series2)-np.array(error2)
        te2t = np.array(series2)+np.array(error2)
        te3b = np.array(series3)-np.array(error3)
        te3t = np.array(series3)+np.array(error3)
        te4b = np.array(series4)-np.array(error4)
        te4t = np.array(series4)+np.array(error4)

        
        plt.rcParams['font.sans-serif'] = "Helvetica"
    
        plt.plot(time, series1, marker = 'o', fillstyle = 'none', markersize = 4, color = color1, linewidth=0.8, alpha=0.8 )
        plt.plot(time, series2, marker = '*', fillstyle = 'full', markersize = 5, color = color2, linewidth=0.8, alpha=0.8)
        plt.plot(time, series3, marker = 's', fillstyle = 'none', markersize = 3.5, color = color3, linewidth=0.8, alpha=0.8)
        plt.plot(time, series4, marker = '1', fillstyle = 'full', markersize = 5, color = color4, linewidth=0.8, alpha=0.8)
        
        plt.gcf().set_size_inches(8, 2.5)
        
        plt.legend(['Altimetry Only', 'Kappelsberger', 'McMillan', 'IMAU'], title = title, fontsize = 8)
        
        plt.fill_between(time, te1b, te1t, alpha=0.1, edgecolor=color1, facecolor=color1, antialiased=True, linewidth=0)
        plt.fill_between(time, te2b, te2t, alpha=0.1, edgecolor=color2, facecolor=color2, antialiased=True, linewidth=0)
        plt.fill_between(time, te3b, te3t, alpha=0.1, edgecolor=color3, facecolor=color3, antialiased=True, linewidth=0)
        plt.fill_between(time, te4b, te4t, alpha=0.1, edgecolor=color4, facecolor=color4, antialiased=True, linewidth=0)
    
        plt.xlabel('time')
        plt.ylabel(name + ' ' + exp + ' ' + units + '/m2')
        
        plt.xlim(time[0], time[-1])
        plt.ylim(min([min(te1b), min(te2b), min(te3b), min(te4b)]) - (-max(te1b)/4), max([max(te1t), max(te2t), max(te3t), max(te4t)]) + (max(te1t))/4)

        plt.savefig(path+name +'_mass' + '.jpg', dpi = 300, bbox_inches='tight')
        plt.clf()

    f.close()

    plt.rcParams['font.sans-serif'] = "Helvetica"
    plt.rcParams['legend.numpoints'] = 4
    
    fig, ax = plt.subplots()
    
    for k,r in enumerate(rate):
        
        ax.scatter(basin[k], rate[k], label= method[k],s = 100, c= color[k], marker = 's', alpha = 0.6)
        ax.errorbar(basin[k], rate[k], yerr= error[k], fmt= '', ecolor='black')
    
    
    plt.gcf().set_size_inches(8, 6)
    ax.legend(method[:4])
    
    plt.xlabel('Basins')
    plt.ylabel('Mass change rate [Gt/yr]')
    
    plt.savefig(path+ 'Basins_mass_permethod' + '.jpg', dpi = 300, bbox_inches='tight')
    plt.clf()

def basinsgrace_massseries(bgrid, time, gridx, gridy, grids1, egrids1, grids2, egrids2, grids3, egrids3, grids4, egrids4, grids5, grids6, egrids6, grace_rates, grace_error, units, title, path): # units can be 'Gt' or 'kg'
    
    import matplotlib.pyplot as plt
    import numpy as np
    from firnpy import analise
    import pandas as pd
    
    if units == 'Gt':
        scale = 10**-6 # because 10**-12 to go to Gt but 10**6 to go to m2 from km2
        #scale = 10**-12
        exp = ''
    elif units == 'kg':
        scale = 10**6 # because 10**6 to go to m2 from km2
        #scale = 1
        exp = ''

    
    color1 = 'purple'
    color2 = 'orange'
    color3 = 'red'
    color4 = 'mediumblue'

    
    #Basins
    basins = [1, 2, 3, 4, 5, 6, 7, 8]
    basin_labels = ['GIS01', 'GIS02', 'GIS03', 'GIS04', 'GIS05', 'GIS06', 'GIS07', 'GIS08']
    methods = ['AO', 'K', 'MCM', 'IMAU', 'GRACE']
    colors = ['purple', 'orange', 'red', 'mediumblue', 'green']

    
    rho_i = 917 #[kg/m3]
    rho_w = 997 #[kg/m3]
    cell_size = 1.5**2 #m2
    
    
    f = open(path + 'basinsgrace_rates.txt', 'w')
    f.write('Basins')
    f.write('\n')

    basin =[]
    method =[]
    rate =[]
    error =[]
    color = []
    for i,b in enumerate(basins):
        
        
        name = 'basin_' + basin_labels[i]

        f.write(name)
        f.write('\n')
        
        basin_area = bgrid[bgrid==b].size * cell_size
        
        
        #AO
        series1 = analise.basin(grids1, gridx, gridy, bgrid, b) * rho_i * scale * basin_area
        error1 = analise.basine_CS(egrids1, gridx, gridy, bgrid, b, 3) * rho_i * scale * basin_area
        
        series1_fit, series1_errorfit = analise.fit_series(series1)

        f.write('Altimetry Only: ' + str(round(series1_fit, 4)) +  exp + ' +/- ' + str(round(series1_errorfit,4)) + exp + ' ' + units + '/yr')
        f.write('\n')

        
        #K
        series2 = analise.basin(grids1*grids2, gridx, gridy, bgrid, b) * scale * basin_area
        error2 = np.sqrt((analise.basine_CS(egrids1, gridx, gridy, bgrid, b, 3) * rho_i)**2 + (analise.basin(grids1*egrids2, gridx, gridy, bgrid, b))**2) * scale * basin_area 

        series2_fit, series2_errorfit = analise.fit_series(series2)
        
        f.write('Kapelsberger: ' + str(round(series2_fit, 4)) +  exp + ' +/- ' + str(round(series2_errorfit, 4)) + exp + ' ' + units + '/yr')
        f.write('\n')

        #MCM
        gm = (grids1 + grids6) * grids3 #[kg/m2/yr]
        series3 = analise.basin(gm, gridx, gridy, bgrid, b) * scale * basin_area
        e1 = np.sqrt(analise.basine_CS(egrids1*grids3, gridx, gridy, bgrid, b, 3)**2 + analise.basin(np.array([egrids6*grids3 for g in egrids1]), gridx, gridy, bgrid, b)**2)
        e2 =  analise.basin((grids1 + grids6)* egrids3, gridx, gridy, bgrid, b)
        error3 = np.sqrt(e1**2 + e2**2) * scale #* basin_area

        series3_fit, series3_errorfit = analise.fit_series(series3)
        
        f.write('McMillan: ' + str(round(series3_fit, 4)) +  exp + ' +/- ' + str(round(series3_errorfit, 4)) + exp + ' ' + units + '/yr')
        f.write('\n')
        

        #IMAU
        gi = ((grids1 - grids4) * rho_i) + (grids5 * rho_w) #[kg/m2/yr]
        series4 = analise.basin(gi, gridx, gridy, bgrid, b) * scale * basin_area
        error4 = np.sqrt(analise.basine_CS(egrids1, gridx, gridy, bgrid, b, 3)**2 + analise.basin(egrids4, gridx, gridy, bgrid, b)**2) * rho_i * scale * basin_area

        series4_fit, series4_errorfit = analise.fit_series(series4)
        
        f.write('IMAU: ' + str(round(series4_fit, 4)) +  exp + ' +/- ' + str(round(series4_errorfit, 4)) + exp + ' ' + units + '/yr')
        f.write('\n')
        
        #GRACE
        
        
        seriesf = [round(series1_fit, 4), round(series2_fit, 4), round(series3_fit, 4), round(series4_fit, 4), grace_rates[i]]
        errorsf = [round(series1_errorfit, 4), round(series2_errorfit, 4), round(series3_errorfit, 4), round(series4_errorfit, 4), grace_error[i]]
        
        for j,m in enumerate(methods):
            basin.extend([basin_labels[i]])
            method.extend([m])
            rate.extend([seriesf[j]])
            error.extend([errorsf[j]])
            color.extend([colors[j]])

        
        te1b = np.array(series1)-np.array(error1)
        te1t = np.array(series1)+np.array(error1)
        te2b = np.array(series2)-np.array(error2)
        te2t = np.array(series2)+np.array(error2)
        te3b = np.array(series3)-np.array(error3)
        te3t = np.array(series3)+np.array(error3)
        te4b = np.array(series4)-np.array(error4)
        te4t = np.array(series4)+np.array(error4)

        
        plt.rcParams['font.sans-serif'] = "Helvetica"
    
        plt.plot(time, series1, marker = 'o', fillstyle = 'none', markersize = 4, color = color1, linewidth=0.8, alpha=0.8 )
        plt.plot(time, series2, marker = '*', fillstyle = 'full', markersize = 5, color = color2, linewidth=0.8, alpha=0.8)
        plt.plot(time, series3, marker = 's', fillstyle = 'none', markersize = 3.5, color = color3, linewidth=0.8, alpha=0.8)
        plt.plot(time, series4, marker = '1', fillstyle = 'full', markersize = 5, color = color4, linewidth=0.8, alpha=0.8)
        
        plt.gcf().set_size_inches(8, 2.5)
        
        plt.legend(['Altimetry Only', 'Kappelsberger', 'McMillan', 'IMAU'], title = title, fontsize = 8)
        
        plt.fill_between(time, te1b, te1t, alpha=0.1, edgecolor=color1, facecolor=color1, antialiased=True, linewidth=0)
        plt.fill_between(time, te2b, te2t, alpha=0.1, edgecolor=color2, facecolor=color2, antialiased=True, linewidth=0)
        plt.fill_between(time, te3b, te3t, alpha=0.1, edgecolor=color3, facecolor=color3, antialiased=True, linewidth=0)
        plt.fill_between(time, te4b, te4t, alpha=0.1, edgecolor=color4, facecolor=color4, antialiased=True, linewidth=0)
    
        plt.xlabel('time')
        plt.ylabel(name + ' ' + exp + ' ' + units + '/m2')
        
        plt.xlim(time[0], time[-1])
        plt.ylim(min([min(te1b), min(te2b), min(te3b), min(te4b)]) - (-max(te1b)/4), max([max(te1t), max(te2t), max(te3t), max(te4t)]) + (max(te1t))/4)

        plt.savefig(path+name +'_mass' + '.jpg', dpi = 300, bbox_inches='tight')
        plt.clf()

    f.close()
    

    
    plt.rcParams['font.sans-serif'] = "Helvetica"
    plt.rcParams['legend.numpoints'] = 4
    
    fig, ax = plt.subplots()
    
    for k,r in enumerate(rate):
        
        ax.scatter(basin[k], rate[k], label= method[k],s = 100, c= color[k], marker = 's', alpha = 0.6)
        ax.errorbar(basin[k], rate[k], yerr= error[k], fmt= '', ecolor='black')
    
    
    plt.gcf().set_size_inches(8, 6)
    ax.legend(method[:4])
    
    plt.xlabel('Basins')
    plt.ylabel('Mass change rate [Gt/yr]')
    
    plt.savefig(path+ 'Basinsgrace_mass_permethod' + '.jpg', dpi = 300, bbox_inches='tight')
    plt.clf()



def mask_massseries(mgrid, time, gridx, gridy, grids1, egrids1, grids2, egrids2, grids3, egrids3, grids4, egrids4, grids5, grids6, egrids6, units, title, name, path): # units can be 'Gt' or 'kg'
    
    import matplotlib.pyplot as plt
    import numpy as np
    from firnpy import analise
    
    if units == 'Gt':
        scale = 10**-6 # because 10**-12 to go to Gt but 10**6 to go to m2 from km2
        exp = ''
    elif units == 'kg':
        scale = 10**6 # because 10**6 to go to m2 from km2 
        exp = ''

    
    color1 = 'purple'
    color2 = 'orange'
    color3 = 'red'
    color4 = 'mediumblue'

        
    
    rho_i = 917 #[kg/m3]
    rho_w = 997 #[kg/m3]
    cell_size = 1.5**2 #m2
    
    
    f = open(path + name + '_rates.txt', 'w')
    f.write(name)
    f.write('\n')


    f.write(name)
    f.write('\n')
    
    mask_area = mgrid[mgrid==1].size * cell_size
    
    
    #AO
    series1 = analise.basin(grids1, gridx, gridy, mgrid, 1) * rho_i * scale * mask_area
    error1 = analise.basine_CS(egrids1, gridx, gridy, mgrid, 1, 3) * rho_i * scale * mask_area
    
    series1_fit, series1_errorfit = analise.fit_series(series1)

    f.write('Altimetry Only: ' + str(round(series1_fit, 4)) +  exp + ' +/- ' + str(round(series1_errorfit,4)) + exp + ' ' + units + '/yr')
    f.write('\n')

    
    #K
    series2 = analise.basin(grids1*grids2, gridx, gridy, mgrid, 1) * scale * mask_area
    error2 = np.sqrt((analise.basine_CS(egrids1, gridx, gridy, mgrid, 1, 3) * rho_i)**2 + (analise.basin(grids1*egrids2, gridx, gridy, mgrid, 1))**2) * scale * mask_area 

    series2_fit, series2_errorfit = analise.fit_series(series2)
    
    f.write('Kapelsberger: ' + str(round(series2_fit, 4)) +  exp + ' +/- ' + str(round(series2_errorfit, 4)) + exp + ' ' + units + '/yr')
    f.write('\n')
    
    #MCM
    gm = (grids1 + grids6) * grids3 #[kg/m2/yr]
    series3 = analise.basin(gm, gridx, gridy, mgrid, 1) * scale * mask_area
    e1 = np.sqrt(analise.basine_CS(egrids1*grids3, gridx, gridy, mgrid, 1, 3)**2 + analise.basin(np.array([egrids6*grids3 for g in egrids1]), gridx, gridy, mgrid, 1)**2)
    e2 =  analise.basin((grids1 + grids6)* egrids3, gridx, gridy, mgrid, 1)
    error3 = np.sqrt(e1**2 + e2**2) * scale * mask_area

    series3_fit, series3_errorfit = analise.fit_series(series3)
    
    f.write('McMillan: ' + str(round(series3_fit, 4)) +  exp + ' +/- ' + str(round(series3_errorfit, 4)) + exp + ' ' + units + '/yr')
    f.write('\n')


    #IMAU
    gi = ((grids1 - grids4) * rho_i) + (grids5 * rho_w) #[kg/m2/yr]
    series4 = analise.basin(gi, gridx, gridy, mgrid, 1) * scale * mask_area
    error4 = np.sqrt(analise.basine_CS(egrids1, gridx, gridy, mgrid, 1, 3)**2 + analise.basin(egrids4, gridx, gridy, mgrid, 1)**2) * rho_i * scale * mask_area

    series4_fit, series4_errorfit = analise.fit_series(series4)
    
    f.write('IMAU: ' + str(round(series4_fit, 4)) +  exp + ' +/- ' + str(round(series4_errorfit, 4)) + exp + ' ' + units + '/yr')
    f.write('\n')


    te1b = np.array(series1)-np.array(error1)
    te1t = np.array(series1)+np.array(error1)
    te2b = np.array(series2)-np.array(error2)
    te2t = np.array(series2)+np.array(error2)
    te3b = np.array(series3)-np.array(error3)
    te3t = np.array(series3)+np.array(error3)
    te4b = np.array(series4)-np.array(error4)
    te4t = np.array(series4)+np.array(error4)

    
    plt.rcParams['font.sans-serif'] = "Helvetica"

    plt.plot(time, series1, marker = 'o', fillstyle = 'none', markersize = 4, color = color1, linewidth=0.8, alpha=0.8 )
    plt.plot(time, series2, marker = '*', fillstyle = 'full', markersize = 5, color = color2, linewidth=0.8, alpha=0.8)
    plt.plot(time, series3, marker = 's', fillstyle = 'none', markersize = 3.5, color = color3, linewidth=0.8, alpha=0.8)
    plt.plot(time, series4, marker = '1', fillstyle = 'full', markersize = 5, color = color4, linewidth=0.8, alpha=0.8)
    
    plt.gcf().set_size_inches(8, 2.5)
    
    plt.legend(['Altimetry Only', 'Kappelsberger', 'McMillan', 'IMAU'], title = title, fontsize = 8)
    
    plt.fill_between(time, te1b, te1t, alpha=0.1, edgecolor=color1, facecolor=color1, antialiased=True, linewidth=0)
    plt.fill_between(time, te2b, te2t, alpha=0.1, edgecolor=color2, facecolor=color2, antialiased=True, linewidth=0)
    plt.fill_between(time, te3b, te3t, alpha=0.1, edgecolor=color3, facecolor=color3, antialiased=True, linewidth=0)
    plt.fill_between(time, te4b, te4t, alpha=0.1, edgecolor=color4, facecolor=color4, antialiased=True, linewidth=0)

    plt.xlabel('time')
    plt.ylabel(name + ' ' + exp + ' ' + units)
    
    plt.xlim(time[0], time[-1])
    plt.ylim(min([min(te1b), min(te2b), min(te3b), min(te4b)]) - (-max(te1b)/4), max([max(te1t), max(te2t), max(te3t), max(te4t)]) + (max(te1t))/4)

    plt.savefig(path+name +'_mass' + '.jpg', dpi = 300, bbox_inches='tight')
    plt.clf()

    f.close()
    
    rates = [series1_fit, series2_fit, series3_fit, series4_fit]
    errors = [series1_errorfit, series2_errorfit, series3_errorfit, series4_errorfit]
    
    return rates, errors


def timeseries_elev4(time, tseries1, terror1, label1, tseries2, terror2, label2, tseries3, terror3, label3, tseries4, terror4, label4, title, units, name, path):
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    
    color1 = 'blue'
    color2 = 'magenta'
    color3 = 'dimgrey'
    color4 = 'red'
    
    te1b = np.array(tseries1)-np.array(terror1)
    te1t = np.array(tseries1)+np.array(terror1)
    te2b = np.array(tseries2)-np.array(terror2)
    te2t = np.array(tseries2)+np.array(terror2)
    te3b = np.array(tseries3)-np.array(terror3)
    te3t = np.array(tseries3)+np.array(terror3)
    te4b = np.array(tseries4)-np.array(terror4)
    te4t = np.array(tseries4)+np.array(terror4)


    plt.rcParams['font.sans-serif'] = "Helvetica"
    plt.gcf().set_size_inches(8, 6)

    plt.xlabel('time')
    plt.ylabel(name + ' ' + units)
    
    plt.xlim(time[0], time[-1])
    plt.ylim(min([min(te1b), min(te2b), min(te3b), min(te4b)]) - (-max(te1b)/4), max([max(te1t), max(te2t), max(te3t), max(te4t)]) + (max(te1t))/4)
    

    plt.plot(time, tseries1, color = color1, linewidth=0.8, alpha=0.8)
    plt.plot(time, tseries2, color = color2, linewidth=0.8, alpha=0.8)
    plt.plot(time, tseries3, color = color3, linewidth=0.8, alpha=0.8)
    plt.plot(time, tseries4, color = color4, linewidth=0.8, alpha=0.8)
            
    plt.fill_between(time, te1b, te1t, alpha=0.2, edgecolor=color1, facecolor=color1, antialiased=True, linewidth=0)
    plt.fill_between(time, te2b, te2t, alpha=0.2, edgecolor=color2, facecolor=color2, antialiased=True, linewidth=0)
    plt.fill_between(time, te3b, te3t, alpha=0.2, edgecolor=color3, facecolor=color3, antialiased=True, linewidth=0)
    plt.fill_between(time, te4b, te4t, alpha=0.2, edgecolor=color4, facecolor=color4, antialiased=True, linewidth=0)

    plt.legend([label1, label2, label3, label4], loc = 'lower left', title_fontsize = 10)

    
    plt.tight_layout()
    plt.savefig(path+name + '.jpg', dpi = 300, bbox_inches='tight')
    plt.clf()



def mask_elevseries(mgrid, time, gridx, gridy, grids1, egrids1, grids2, egrids2, grids3, egrids3, grids4, units, title, name, path):
    
    import matplotlib.pyplot as plt
    import numpy as np
    from firnpy import analise
    
    
    color1 = 'blue'
    color2 = 'magenta'
    color3 = 'dimgrey'
    color4 = 'red'


    f = open(path + name + '_elevation_rates.txt', 'w')
    f.write(name)
    f.write('\n')


    f.write(name)
    f.write('\n')
        
    
    #CS2
    series1 = analise.basin(grids1, gridx, gridy, mgrid, 1) 
    error1 = analise.basine_CS(egrids1, gridx, gridy, mgrid, 1, 3)
    
    series1_fit, series1_errorfit = analise.fit_series(series1)

    f.write('CryoSat-2: ' + str(round(series1_fit, 4)) + ' +/- ' + str(round(series1_errorfit,4)) + ' ' + units + '/yr')
    f.write('\n')
    f.write('CryoSat-2 Total: ' + str(round(series1[-1], 4)))
    f.write('\n')

    
    #FDM
    series2 = analise.basin(grids2, gridx, gridy, mgrid, 1)
    error2 = analise.basin(egrids2, gridx, gridy, mgrid, 1)

    series2_fit, series2_errorfit = analise.fit_series(series2)
    
    f.write('IMAU-FDM: ' + str(round(series2_fit, 4)) + ' +/- ' + str(round(series2_errorfit, 4)) + ' ' + units + '/yr')
    f.write('\n')
    f.write('IMAU-FDM Total: ' + str(round(series2[-1], 4)))
    f.write('\n')

    #Residuals
    series3 = [series1[i] - series2[i] for i,g in enumerate(grids1)]
    error3 = [np.sqrt(error1[i]**2 + error2[i]**2) for i,g in enumerate(grids1)]

    series3_fit, series3_errorfit = analise.fit_series(series3)
    
    f.write('Residuals: ' + str(round(series3_fit, 4)) + ' +/- ' + str(round(series3_errorfit, 4)) + ' ' + units + '/yr')
    f.write('\n')
    f.write('Residuals Total: ' + str(round(series3[-1], 4)))
    f.write('\n')


    #Compaction
    series4 = analise.basin(grids3, gridx, gridy, mgrid, 1)
    error4 = analise.basin(egrids3, gridx, gridy, mgrid, 1)

    series4_fit, series4_errorfit = analise.fit_series(series4)
    
    f.write('Compaction: ' + str(round(series4_fit, 4)) + ' +/- ' + str(round(series4_errorfit, 4)) + ' ' + units + '/yr')
    f.write('\n')
    f.write('Compaction Total: ' + str(round(series4[-1], 4)))
    f.write('\n')


    #SMB
    series5 = analise.basin(grids4, gridx, gridy, mgrid, 1)

    series5_fit, series5_errorfit = analise.fit_series(series5)
    
    f.write('SMB: ' + str(round(series5_fit, 4)) + ' +/- ' + str(round(series5_errorfit, 4)) + ' ' + 'm w.e. /yr')
    f.write('\n')
    f.write('SMB Total: ' + str(round(series5[-1], 4)))
    f.write('\n')



    te1b = np.array(series1)-np.array(error1)
    te1t = np.array(series1)+np.array(error1)
    te2b = np.array(series2)-np.array(error2)
    te2t = np.array(series2)+np.array(error2)
    te3b = np.array(series3)-np.array(error3)
    te3t = np.array(series3)+np.array(error3)
    te4b = np.array(series4)-np.array(error4)
    te4t = np.array(series4)+np.array(error4)

    
    plt.rcParams['font.sans-serif'] = "Helvetica"

    plt.plot(time, series1, color = color1, linewidth=0.8, alpha=0.8)
    plt.plot(time, series2, color = color2, linewidth=0.8, alpha=0.8)
    plt.plot(time, series3, color = color3, linewidth=0.8, alpha=0.8)
    plt.plot(time, series4, color = color4, linewidth=0.8, alpha=0.8)
    
    plt.gcf().set_size_inches(8, 2.5)
    
    plt.legend(['CryoSat-2', 'IMAU-FDM', 'Residuals', 'Compaction'], title = title, fontsize = 8)
    
    plt.fill_between(time, te1b, te1t, alpha=0.1, edgecolor=color1, facecolor=color1, antialiased=True, linewidth=0)
    plt.fill_between(time, te2b, te2t, alpha=0.1, edgecolor=color2, facecolor=color2, antialiased=True, linewidth=0)
    plt.fill_between(time, te3b, te3t, alpha=0.1, edgecolor=color3, facecolor=color3, antialiased=True, linewidth=0)
    plt.fill_between(time, te4b, te4t, alpha=0.1, edgecolor=color4, facecolor=color4, antialiased=True, linewidth=0)

    plt.xlabel('time')
    plt.ylabel(name + ' ' + units)
    
    plt.xlim(time[0], time[-1])
    plt.ylim(min([min(te1b), min(te2b), min(te3b), min(te4b)]) - (-max(te1b)/4), max([max(te1t), max(te2t), max(te3t), max(te4t)]) + (max(te1t))/4)

    plt.savefig(path+name +'_elevation' + '.jpg', dpi = 300, bbox_inches='tight')
    plt.clf()

    f.close()



def basins_elevseries(bgrid, time, gridx, gridy, grids1, egrids1, grids2, egrids2, grids3, egrids3, grids4, units, title, path):
    
    import matplotlib.pyplot as plt
    import numpy as np
    from firnpy import analise
    
    
    color1 = 'blue'
    color2 = 'magenta'
    color3 = 'dimgrey'
    color4 = 'red'

    
    #Basins
    basins = [1, 2, 3, 4, 5, 6, 7]
    basin_labels = ['NW', 'CW', 'SW', 'SE', 'CE', 'NE', 'NO']
    
    
    
    f = open(path + 'basins_elevation_rates.txt', 'w')
    f.write('Basins')
    f.write('\n')

    
    for i,b in enumerate(basins):
        
        
        name = 'basin_' + basin_labels[i]

        f.write(name)
        f.write('\n')
        
        #CS2
        series1 = analise.basin(grids1, gridx, gridy, bgrid, b) 
        error1 = analise.basine_CS(egrids1, gridx, gridy, bgrid, b, 3)
        
        series1_fit, series1_errorfit = analise.fit_series(series1)
    
        f.write('CryoSat-2: ' + str(round(series1_fit, 4)) + ' +/- ' + str(round(series1_errorfit,4)) + ' ' + units + '/yr')
        f.write('\n')
    
        
        #FDM
        series2 = analise.basin(grids2, gridx, gridy, bgrid, b)
        error2 = analise.basin(egrids2, gridx, gridy, bgrid, b)
    
        series2_fit, series2_errorfit = analise.fit_series(series2)
        
        f.write('IMAU-FDM: ' + str(round(series2_fit, 4)) + ' +/- ' + str(round(series2_errorfit, 4)) + ' ' + units + '/yr')
        f.write('\n')
        
        #Residuals
        series3 = [series1[i] - series2[i] for i,g in enumerate(grids1)]
        error3 = [np.sqrt(error1[i]**2 + error2[i]**2) for i,g in enumerate(grids1)]
    
        series3_fit, series3_errorfit = analise.fit_series(series3)
        
        f.write('Residuals: ' + str(round(series3_fit, 4)) + ' +/- ' + str(round(series3_errorfit, 4)) + ' ' + units + '/yr')
        f.write('\n')
    
    
        #Compaction
        series4 = analise.basin(grids3, gridx, gridy, bgrid, b)
        error4 = analise.basin(egrids3, gridx, gridy, bgrid, b)
    
        series4_fit, series4_errorfit = analise.fit_series(series4)
        
        f.write('Compaction: ' + str(round(series4_fit, 4)) + ' +/- ' + str(round(series4_errorfit, 4)) + ' ' + units + '/yr')
        f.write('\n')

        #SMB
        series5 = analise.basin(grids4, gridx, gridy, bgrid, b)
    
        series5_fit, series5_errorfit = analise.fit_series(series5)
        
        f.write('SMB: ' + str(round(series5_fit, 4)) + ' +/- ' + str(round(series5_errorfit, 4)) + ' ' + 'm w.e. /yr')
        f.write('\n')
    

        te1b = np.array(series1)-np.array(error1)
        te1t = np.array(series1)+np.array(error1)
        te2b = np.array(series2)-np.array(error2)
        te2t = np.array(series2)+np.array(error2)
        te3b = np.array(series3)-np.array(error3)
        te3t = np.array(series3)+np.array(error3)
        te4b = np.array(series4)-np.array(error4)
        te4t = np.array(series4)+np.array(error4)

        
        plt.rcParams['font.sans-serif'] = "Helvetica"
    
        plt.plot(time, series1, color = color1, linewidth=0.8, alpha=0.8)
        plt.plot(time, series2, color = color2, linewidth=0.8, alpha=0.8)
        plt.plot(time, series3, color = color3, linewidth=0.8, alpha=0.8)
        plt.plot(time, series4, color = color4, linewidth=0.8, alpha=0.8)
        
        plt.gcf().set_size_inches(8, 2.5)
        
        plt.legend(['CryoSat-2', 'IMAU-FDM', 'Residuals', 'Compaction'], title = title, fontsize = 8)
        
        plt.fill_between(time, te1b, te1t, alpha=0.1, edgecolor=color1, facecolor=color1, antialiased=True, linewidth=0)
        plt.fill_between(time, te2b, te2t, alpha=0.1, edgecolor=color2, facecolor=color2, antialiased=True, linewidth=0)
        plt.fill_between(time, te3b, te3t, alpha=0.1, edgecolor=color3, facecolor=color3, antialiased=True, linewidth=0)
        plt.fill_between(time, te4b, te4t, alpha=0.1, edgecolor=color4, facecolor=color4, antialiased=True, linewidth=0)
    
        plt.xlabel('time')
        plt.ylabel(name + ' ' + units)
        
        plt.xlim(time[0], time[-1])
        plt.ylim(min([min(te1b), min(te2b), min(te3b), min(te4b)]) - (-max(te1b)/4), max([max(te1t), max(te2t), max(te3t), max(te4t)]) + (max(te1t))/4)

        plt.savefig(path+name +'_elevation' + '.jpg', dpi = 300, bbox_inches='tight')
        plt.clf()

    f.close()



def stations_elevseries(time, gridx, gridy, grids1, egrids1, grids2, egrids2, grids3, egrids3, grids4, units, title, path): # units can be 'Gt' or 'kg'
    
    import matplotlib.pyplot as plt
    import numpy as np
    from firnpy import analise
    
    
    color1 = 'blue'
    color2 = 'magenta'
    color3 = 'dimgrey'
    color4 = 'red'

    
    #We use only PROMICE and GCNet stations
    stations_names = ['Peterman_ELA', 'KPC_L', 'KPC_U', 'EGP', 'NASA_E', 'SCO_L', 'SCO_U', 'Summit',
                      'NASA_SE', 'South Dome', 'KAN_L', 'KAN_M', 'KAN_U', 'DYE-2', 'JAR_1', 'Swiss Camp', 'CP1', 
                      'UPE_L', 'UPE_U', 'NASA_U', 'GITS', 'NEEM']
    
    stations_coords = [[-244306, -1047222], [391168, -1023448], [374565, -1038474], [245519, -1545750], [422939, -1578126],
                       [605565, -1843811], [586431, -1830173], [215048, -1889006], [112747, -2580907], [9445, -2960973], 
                       [-216931, -2504048], [-168343, -2510999], [-89205, -2522571], [-57831, -2582094], [-184047, -2236749],
                       [-168891, -2230133], [-76703, -2200166], [-301483, -1841957], [-278493, -1846178], [-138429, -1756197],
                       [-386548, -1344511],[-138963, -1351904]]
    
    stations_elev = [924, 370, 870, 2660, 2614, 460, 970, 3199, 2373, 2901, 670, 
                     1270, 1840, 2099, 932, 1176, 2022, 220, 940, 2334, 1869, 2454]
    
    
    station_area = 10**2 #km2
    
    f = open(path + 'stations_elevation_rates.txt', 'w')
    f.write('Stations')
    f.write('\n')
    
    for i,r in enumerate(stations_names):
                
        name = 'station_' + stations_names[i]
        
        f.write(name)
        f.write('\n')
        
        
        #CS2
        series1 = analise.station(grids1, gridx, gridy, stations_coords[i], np.sqrt(station_area)) 
        error1 = analise.statione_CS(egrids1, gridx, gridy, stations_coords[i], np.sqrt(station_area), 3)
        
        series1_fit, series1_errorfit = analise.fit_series(series1)
    
        f.write('CryoSat-2: ' + str(round(series1_fit, 4)) + ' +/- ' + str(round(series1_errorfit,4)) + ' ' + units + '/yr')
        f.write('\n')
    
        
        #FDM
        series2 = analise.station(grids2, gridx, gridy, stations_coords[i], np.sqrt(station_area))
        error2 = analise.station(egrids2, gridx, gridy, stations_coords[i], np.sqrt(station_area))
    
        series2_fit, series2_errorfit = analise.fit_series(series2)
        
        f.write('IMAU-FDM: ' + str(round(series2_fit, 4)) + ' +/- ' + str(round(series2_errorfit, 4)) + ' ' + units + '/yr')
        f.write('\n')
        
        #Residuals
        series3 = [series1[i] - series2[i] for i,g in enumerate(grids1)]
        error3 = [np.sqrt(error1[i]**2 + error2[i]**2) for i,g in enumerate(grids1)]
    
        series3_fit, series3_errorfit = analise.fit_series(series3)
        
        f.write('Residuals: ' + str(round(series3_fit, 4)) + ' +/- ' + str(round(series3_errorfit, 4)) + ' ' + units + '/yr')
        f.write('\n')
    
    
        #Compaction
        series4 = analise.station(grids3, gridx, gridy, stations_coords[i], np.sqrt(station_area))
        error4 = analise.station(egrids3, gridx, gridy, stations_coords[i], np.sqrt(station_area))
    
        series4_fit, series4_errorfit = analise.fit_series(series4)
        
        f.write('Compaction: ' + str(round(series4_fit, 4)) + ' +/- ' + str(round(series4_errorfit, 4)) + ' ' + units + '/yr')
        f.write('\n')

        #SMB
        series5 = analise.station(grids4, gridx, gridy, stations_coords[i], np.sqrt(station_area))
    
        series5_fit, series5_errorfit = analise.fit_series(series5)
        
        f.write('SMB: ' + str(round(series5_fit, 4)) + ' +/- ' + str(round(series5_errorfit, 4)) + ' ' + 'm w.e. /yr')
        f.write('\n')
        
        
        te1b = np.array(series1)-np.array(error1)
        te1t = np.array(series1)+np.array(error1)
        te2b = np.array(series2)-np.array(error2)
        te2t = np.array(series2)+np.array(error2)
        te3b = np.array(series3)-np.array(error3)
        te3t = np.array(series3)+np.array(error3)
        te4b = np.array(series4)-np.array(error4)
        te4t = np.array(series4)+np.array(error4)

        
        plt.rcParams['font.sans-serif'] = "Helvetica"
    
        plt.plot(time, series1, color = color1, linewidth=0.8, alpha=0.8 )
        plt.plot(time, series2, color = color2, linewidth=0.8, alpha=0.8)
        plt.plot(time, series3, color = color3, linewidth=0.8, alpha=0.8)
        plt.plot(time, series4, color = color4, linewidth=0.8, alpha=0.8)
        
        plt.gcf().set_size_inches(8, 2.5)
        
        plt.legend(['CryoSat-2', 'IMAU-FDM', 'Residuals', 'Compaction'], title = title, fontsize = 8)
        
        plt.fill_between(time, te1b, te1t, alpha=0.1, edgecolor=color1, facecolor=color1, antialiased=True, linewidth=0)
        plt.fill_between(time, te2b, te2t, alpha=0.1, edgecolor=color2, facecolor=color2, antialiased=True, linewidth=0)
        plt.fill_between(time, te3b, te3t, alpha=0.1, edgecolor=color3, facecolor=color3, antialiased=True, linewidth=0)
        plt.fill_between(time, te4b, te4t, alpha=0.1, edgecolor=color4, facecolor=color4, antialiased=True, linewidth=0)
    
        plt.xlabel('time')
        plt.ylabel(name + ' ' + units)
        
        plt.xlim(time[0], time[-1])
        plt.ylim(min([min(te1b), min(te2b), min(te3b), min(te4b)]) - (max(te1b)/4), max([max(te1t), max(te2t), max(te3t), max(te4t)]) + (max(te1t)/4))

        plt.savefig(path+name +'_elevation' + '.jpg', dpi = 300, bbox_inches='tight')
        plt.clf()

    f.close()


def nilsson_elevseries(time, gridx, gridy, grids1, egrids1, units, title, path):
    
    import matplotlib.pyplot as plt
    import numpy as np
    from firnpy import analise
    
    
    color1 = 'blue'

    
    #We use only PROMICE and GCNet stations
    stations_names = ['NEEM']
    
    stations_coords = [[-138963, -1351904]]
    
    stations_elev = [2454]
    
    
    station_area = 25**2 #km2
    
    f = open(path + 'nilsson_neem_elevation_rates.txt', 'w')
    f.write('Stations')
    f.write('\n')
    
    for i,r in enumerate(stations_names):
                
        name = 'station_' + stations_names[i]
        
        f.write(name)
        f.write('\n')
        
        
        #CS2
        series1 = analise.station(grids1, gridx, gridy, stations_coords[i], np.sqrt(station_area)) 
        error1 = analise.statione_CS(egrids1, gridx, gridy, stations_coords[i], np.sqrt(station_area), 3)
        
        series1_fit, series1_errorfit = analise.fit_series(series1)
    
        f.write('CryoSat-2: ' + str(round(series1_fit, 4)) + ' +/- ' + str(round(series1_errorfit,4)) + ' ' + units + '/yr')
        f.write('\n')
    
        
        te1b = np.array(series1)-np.array(error1)
        te1t = np.array(series1)+np.array(error1)

        
        plt.rcParams['font.sans-serif'] = "Helvetica"
    
        plt.plot(time, series1, marker = 'o', fillstyle = 'none', markersize = 4, color = color1, linewidth=0.8, alpha=0.8 )
        plt.plot(time[17:19], series1[17:19], marker = 'o', fillstyle = 'none', markersize = 4, color = 'red', linewidth=0.8, alpha=0.8 )
 
        
        plt.gcf().set_size_inches(8, 2.5)
        
        plt.legend(['CryoSat-2', 'June-July'], title = title, fontsize = 8)
        
        plt.fill_between(time, te1b, te1t, alpha=0.1, edgecolor=color1, facecolor=color1, antialiased=True, linewidth=0)
    
        plt.xlabel('time')
        plt.ylabel(name + ' ' + units)
        
        plt.xlim(time[0], time[-1])
        plt.ylim(min(te1b) - (max(te1b)/4), max(te1t) + (max(te1t)/4))
        
        plt.text(time[18], -0.05, str(round(series1[18]-series1[17],2)) + ' ' + units, color = 'red', fontfamily = 'Arial', fontsize = 8)
        
        plt.savefig(path+ 'NEEM' + name +'_elevation' + '.jpg', dpi = 300, bbox_inches='tight')
        plt.clf()

    f.close()
