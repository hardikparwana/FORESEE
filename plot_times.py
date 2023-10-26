
import matplotlib
import matplotlib.pyplot as plt

font = {'family' : 'normal',
        # 'weight' : 'bold',
        'size'   : 20}

matplotlib.rc('font', **font)

f, (ax2, ax) = plt.subplots(2, 1, sharex=True)

bar_labels1 = ['MC 500 H1', 'MC 5000 H1', 'MC 50000 H1', 'EC Layer - UT H1', 'EC Layer - GenUT H1', 'E layer - UT H1', 'E layer - UT H2', 'E layer - UT H3', 'E layer - UT H4', 'E layer - UT H5', 'E layer - UT H6' ]
bar_counts1 = [0.007509589, 0.073462725, 0.732954264, 0.000451763,0.000496546, 0.00041794, 0.001773357, 0.00844645, 0.04363536, 0.2262163162, 1.7395920   ]

bar_labels2 = ['MC 500 H1', 'MC 5000 H1', 'MC 50000 H1', 'EC Layer - UT H1', 'EC Layer - GenUT H1', 'E layer - UT H1', 'E layer - UT H2', 'E layer - UT H3', 'E layer - UT H4', 'E layer - UT H5', 'E layer - UT H6' ]
bar_counts2 = [0.007509589, 0.073462725, 0.732954264, 0.000451763,0.000496546, 0.00041794, 0.001773357, 0.00844645, 0.04363536, 0.2262163162, 1.7395920   ]
bar_colors = ['b', 'b', 'b', 'y', 'salmon', 'y', 'y', 'y', 'y', 'y', 'y']
ax.bar( bar_labels1, bar_counts1, color=bar_colors )
ax2.bar( bar_labels2, bar_counts2, color=bar_colors )

ax.spines['top'].set_visible(False)
ax2.spines['bottom'].set_visible(False)

ax.set_ylim(0.0,0.08 )
ax2.set_ylim(0.2, 1.75)

ax2.xaxis.tick_top()
ax2.tick_params(labeltop=False) 
ax.xaxis.tick_bottom()

ax.set_ylabel('time (s)')
ax2.set_ylabel('time (s)')

d = .015  # how big to make the diagonal lines in axes coordinates
# arguments to pass to plot, just so we don't keep repeating them
kwargs = dict(transform=ax2.transAxes, color='k', clip_on=False)
ax2.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
ax2.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

kwargs.update(transform=ax.transAxes)  # switch to the bottom axes
ax.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
ax.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

dx = 5/72.; dy = 0/72. 
offset = matplotlib.transforms.ScaledTranslation(dx, dy, f.dpi_scale_trans)
  
for label in ax.get_xticklabels():
  label.set_horizontalalignment("right")
  label.set_rotation(90)
  label.set_ha('right')
  label.set_y(1.2)
  
  label.set_transform(label.get_transform() + offset)
#   label.set_x(-2.0)
  

plt.show()