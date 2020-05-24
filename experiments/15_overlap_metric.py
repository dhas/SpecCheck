from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

out_dir     = Path('_15_overlap_metric/')
out_dir.mkdir(exist_ok=True)

lower_xplot_limit_orange = -0.25
upper_xplot_limit_orange = 7.5
lower_xplot_limit_blue = 1
upper_xplot_limit_blue = 5

y_lim_lower = -0.25
y_lim_upper = 1.6

alpha_orange = 0.5
alpha_blue = 0.5

text_size = 14


fig, ax = plt.subplots(1, 1, figsize=(12, 8))
plot_x1 = np.linspace(lower_xplot_limit_orange, upper_xplot_limit_orange,800)
plot_y1 = 1*np.exp(-(plot_x1-2)**2) + 0.8*np.exp(-(plot_x1-3)**2) + 0.8*np.exp(-(plot_x1-5)**2)

plot_x2 = np.linspace(lower_xplot_limit_blue, upper_xplot_limit_blue,200)
plot_y2 = 0.3*beta.pdf((plot_x2-1)/4,3,2)# - 1 + np.cos(plot_x2/10)**2#0.2*np.power((plot_x2-2), 1/2) - 0.255*(plot_x2-4)

plot_x2_before = np.linspace(lower_xplot_limit_orange, lower_xplot_limit_blue, 300)
plot_x2_after = np.linspace(upper_xplot_limit_blue, upper_xplot_limit_orange, 300)
plot_x_orange_color = np.concatenate((plot_x2_before, plot_x2, plot_x2_after))
plot_y_orange_color_lower = np.array(300*[0] + plot_y2.tolist() + 300*[0])
plot_y_orange_color_upper = 1*np.exp(-(plot_x_orange_color-2)**2) + 0.8*np.exp(-(plot_x_orange_color-3)**2) + 0.8*np.exp(-(plot_x_orange_color-5)**2)


ax.plot(plot_x1,plot_y1, 'tab:orange')
ax.plot([lower_xplot_limit_orange -0.5, upper_xplot_limit_orange + 0.5], 2*[0], 'k')

ax.fill_between(plot_x_orange_color, plot_y_orange_color_lower, plot_y_orange_color_upper, facecolor = 'tab:orange', alpha = alpha_orange)
ax.plot(plot_x2,plot_y2, 'b')
ax.fill_between(plot_x2, plot_x2.shape[0]*[0], plot_y2, facecolor = 'b', alpha = alpha_blue)

ax.set_ylim(y_lim_lower, y_lim_upper)

ax.axvline(lower_xplot_limit_orange,0.12,0.15, color = 'k')
ax.axvline(upper_xplot_limit_orange,0.12,0.15, color = 'k')
ax.axvline(lower_xplot_limit_blue,0.12,0.15, color = 'k')
ax.axvline(upper_xplot_limit_blue,0.12,0.15, color = 'k')
ax.annotate('', (lower_xplot_limit_orange, -0.1), (lower_xplot_limit_blue, -0.1), arrowprops={'arrowstyle':'<->'})

lower_text_mean = (lower_xplot_limit_orange + lower_xplot_limit_blue)/2
ax.text(lower_text_mean, -0.2, '$l_1$', size = text_size)
ax.annotate('', (lower_xplot_limit_blue, -0.1), (upper_xplot_limit_blue, -0.1), arrowprops={'arrowstyle':'<->'})
middle_text_mean = (lower_xplot_limit_blue + upper_xplot_limit_blue)/2
ax.text(middle_text_mean, -0.2, '$l_2$', size = text_size)
ax.annotate('', (upper_xplot_limit_blue, -0.1), (upper_xplot_limit_orange, -0.1), arrowprops={'arrowstyle':'<->'})
upper_text_mean = (upper_xplot_limit_blue + upper_xplot_limit_orange)/2
ax.text(upper_text_mean, -0.2, '$l_3$', size = text_size)
ax.axis('off')

fig.savefig(out_dir/'1_overlap_metric.png')


