import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Arc
import seaborn as sns
from scipy import stats
from subprocess import check_output


def draw_court(ax=None, color='black', lw=2, outer_lines=False):
    if ax is None:
        ax = plt.gca()
        ax.set_ylim([-50, 500])

    # Create the various parts of an NBA basketball court
    # Create the basketball hoop
    # Diameter of a hoop is 18" so it has a radius of 9", which is a value
    # 7.5 in our coordinate system
    hoop = Circle((0, 0), radius=7.5, linewidth=lw, color=color, fill=False)

    # Create backboard
    backboard = Rectangle((-30, -7.5), 60, -1, linewidth=lw, color=color)

    # The paint
    # Create the outer box 0f the paint, width=16ft, height=19ft
    outer_box = Rectangle((-80, -47.5), 160, 190, linewidth=lw, color=color,
                          fill=False)
    # Create the inner box of the paint, widt=12ft, height=19ft
    inner_box = Rectangle((-60, -47.5), 120, 190, linewidth=lw, color=color,
                          fill=False)

    # Create free throw top arc
    top_free_throw = Arc((0, 142.5), 120, 120, theta1=0, theta2=180,
                         linewidth=lw, color=color, fill=False)
    # Create free throw bottom arc
    bottom_free_throw = Arc((0, 142.5), 120, 120, theta1=180, theta2=0,
                            linewidth=lw, color=color, linestyle='dashed')
    # Restricted Zone, it is an arc with 4ft radius from center of the hoop
    restricted = Arc((0, 0), 80, 80, theta1=0, theta2=180, linewidth=lw,
                     color=color)

    # Three point line
    # Create the side 3pt lines, they are 14ft long before they begin to arc
    corner_three_a = Rectangle((-220, -47.5), 0, 140, linewidth=lw,
                               color=color)
    corner_three_b = Rectangle((220, -47.5), 0, 140, linewidth=lw, color=color)
    # 3pt arc - center of arc will be the hoop, arc is 23'9" away from hoop
    # I just played around with the theta values until they lined up with the
    # threes
    three_arc = Arc((0, 0), 475, 475, theta1=22, theta2=158, linewidth=lw,
                    color=color)

    # Center Court
    center_outer_arc = Arc((0, 422.5), 120, 120, theta1=180, theta2=0,
                           linewidth=lw, color=color)
    center_inner_arc = Arc((0, 422.5), 40, 40, theta1=180, theta2=0,
                           linewidth=lw, color=color)

    # List of the court elements to be plotted onto the axes
    court_elements = [hoop, backboard, outer_box, inner_box, top_free_throw,
                      bottom_free_throw, restricted, corner_three_a,
                      corner_three_b, three_arc, center_outer_arc,
                      center_inner_arc]

    if outer_lines:
        # Draw the half court line, baseline and side out bound lines
        outer_lines = Rectangle((-250, -47.5), 500, 470, linewidth=lw,
                                color=color, fill=False)
        court_elements.append(outer_lines)

    # Add the court elements onto the axes
    for element in court_elements:
        ax.add_patch(element)

    return ax


df = pd.read_csv('../input/data.csv')
list(df)  # check output
print("total de arremessos = %d" % len(df))
# Court visualization of misses and shots
court_scale = 7
alpha = 0.05
plt.figure(figsize=(2 * court_scale, court_scale * (42.0 / 50.0)))

# shots hit
plt.subplot(121)
h = df.loc[df.shot_made_flag == 1]
print("total de acertos = %d" % len(h))
plt.scatter(h.loc_x, h.loc_y, color='green', alpha=alpha)
plt.title('Shots made')
draw_court(outer_lines=True)

# shots miss
plt.subplot(122)
h = df.loc[df.shot_made_flag == 0]
print("total de erros = %d" % len(h))
plt.scatter(h.loc_x, h.loc_y, color='red', alpha=alpha)
plt.title('Shots missed')
draw_court(outer_lines=True)
# plt.savefig('charts/shots_made_and_missed.png')
plt.show()

joint_shot_chart = sns.jointplot(x="loc_x", y="loc_y", data=df, kind="kde", n_levels=50)

ax = joint_shot_chart.ax_joint

ax.set_xlim(-250, 250)
ax.set_ylim(422.5, -47.5)
draw_court(ax)

# Get rid of axis labels and tick marks
ax.set_xlabel('')
ax.set_ylabel('')
ax.tick_params(labelbottom='off', labelleft='off')

for season, group in df.groupby('season'):
    plt.figure(figsize=(2 * court_scale, court_scale * (42.0 / 50.0)))

    # hit
    plt.subplot(121)
    h = group.loc[df.shot_made_flag == 1]
    plt.scatter(h.loc_x, h.loc_y, color='green', alpha=alpha)
    plt.title('Shots made ' + season)
    draw_court(outer_lines=True)

    # miss
    plt.subplot(122)
    h = group.loc[df.shot_made_flag == 0]
    plt.scatter(h.loc_x, h.loc_y, color='red', alpha=alpha)
    plt.title('Shots missed ' + season)
    draw_court(outer_lines=True)
