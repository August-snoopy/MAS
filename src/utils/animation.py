from datasets.h36m_eval import H36MEval
from config import config
from utils.visualization import plot_animation

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation


def create_pose(ax, plots, vals, pred=True, update=False):
    # h36m 32 joints(full) 改这里，父子关系
    connect = [
        (1, 2), (2, 3), (3, 4), (4, 5),
        (6, 7), (7, 8), (8, 9), (9, 10),
        (0, 1), (0, 6),
        (6, 17), (17, 18), (18, 19), (19, 20), (20, 21), (21, 22),
        (1, 25), (25, 26), (26, 27), (27, 28), (28, 29), (29, 30),
        (24, 25), (24, 17),
        (24, 14), (14, 15)
    ]
    LR = [
        False, True, True, True, True,
        True, False, False, False, False,
        False, True, True, True, True,
        True, True, False, False, False,
        False, False, False, False, True,
        False, True, True, True, True,
        True, True
    ]
    #
    # connect = [
    #     (0, 1), (1, 2), (2, 3),
    #     (0, 6), #(3, 6),
    #     (6, 7), (7, 8), #(6, 9),
    #     (0, 12), (12, 13), #(9, 12),
    #   #  (12, 13), (12, 14),
    #     (13, 14),
    #     #(13, 16), (12, 16), (14, 17), (12, 17),
    #     (14, 15), (13, 17),
    #     (17, 18), (18, 19), (13, 25), (25, 26),
    #   #  (22, 20), (23, 21),# wrists
    # (26, 27)]
    #
    #
    #
    # LR = np.array([
    #     True, True, True, True,
    #     False, False, False, False, False, False, False, False, False,
    #     True, True, True])

    # Start and endpoints of our representation
    I = np.array([touple[0] for touple in connect])
    J = np.array([touple[1] for touple in connect])
    # Left / right indicator
    LR = np.array([LR[a] or LR[b] for a, b in connect])
    if pred:
        lcolor = "#DE9F83"
        rcolor = "#DE9F83"
        # rcolor = "#4794BC"

    else:
        lcolor = "#383838"
        rcolor = "#383838"
        # rcolor = "#AFAC9D"

    for i in np.arange(len(I)):
        x = np.array([vals[I[i], 0], vals[J[i], 0]])
        z = np.array([vals[I[i], 1], vals[J[i], 1]])
        y = np.array([vals[I[i], 2], vals[J[i], 2]])
        if not update:

            if i == 0:
                plots.append(ax.plot(x, y, z, lw=2, linestyle='--', c=lcolor if LR[i] else rcolor,
                                     label=['GT' if not pred else 'Pred']))
            else:
                plots.append(ax.plot(x, y, z, lw=2, linestyle='--', c=lcolor if LR[i] else rcolor))

        elif update:
            plots[i][0].set_xdata(x)
            plots[i][0].set_ydata(y)
            plots[i][0].set_3d_properties(z)
            plots[i][0].set_color(lcolor if LR[i] else rcolor)

    return plots



def update(num, data_gt, data_pred, plots_gt, plots_pred, fig, ax):
    gt_vals = data_gt[num]
    pred_vals = data_pred[num]
    plots_gt = create_pose(ax, plots_gt, gt_vals, pred=False, update=True)
    plots_pred = create_pose(ax, plots_pred, pred_vals, pred=True, update=True)

    r = 0.7
    xroot, zroot, yroot = gt_vals[0, 0], gt_vals[0, 1], gt_vals[0, 2]
    ax.set_xlim3d([-r + xroot, r + xroot])
    ax.set_ylim3d([-r + yroot, r + yroot])
    ax.set_zlim3d([-r + zroot, r + zroot])
    # ax.set_title('pose at time frame: '+str(num))
    # ax.set_aspect('equal')

    return plots_gt, plots_pred


def visualize(pred, gt, output_n, action):
    import random

    data_pred = torch.squeeze(pred, 0).cpu().data.numpy()# in meters
    data_gt = torch.squeeze(gt, 0).cpu().data.numpy()

    i = random.randint(1, 128)

    data_pred = data_pred[i, ...]
    data_gt = data_gt[i, ...]

    # print (data_gt.shape,data_pred.shape)

    fig = plt.figure()
    ax = Axes3D(fig)
    fig.add_axes(ax)

    ax.view_init(elev=-0, azim=-45)
    ax.axis('off')
    ax.grid(None)
    vals = np.zeros((32, 3))  # or joints_to_consider
    gt_plots = []
    pred_plots = []

    gt_plots = create_pose(ax, gt_plots, vals, pred=False, update=False)
    pred_plots = create_pose(ax, pred_plots, vals, pred=True, update=False)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend(loc='lower left')

    ax.set_xlim3d([-1, 1.5])
    ax.set_xlabel('X')

    ax.set_ylim3d([-1, 1.5])
    ax.set_ylabel('Y')

    ax.set_zlim3d([0.0, 1.5])
    ax.set_zlabel('Z')
    ax.set_title(' for action : ' + str(action) + ' for ' + str(output_n) + ' frames')

    line_anim = animation.FuncAnimation(fig, update, output_n, fargs=(data_gt, data_pred, gt_plots, pred_plots,
                                                                      fig, ax), interval=70, blit=False)
    # line_anim = animation.FuncAnimation(fig, update, output_n, fargs=(data_gt, gt_plots,
    #                                                                   fig, ax), interval=70, blit=False)
    plt.show()

    line_anim.save('./visualizations/pred{}/human_viz{}.gif'.format(25, i), writer='pillow')


if __name__ == "__main__":

    pred = motion_pred  # (batch, seq_len, joint_nums, feature)
    gt = motion_target
    while cnt < 10:
        visualize(pred, gt, config.motion.h36m_target_length, args.action)
        # print('aaa', motion_pred[0, ...].squeeze(0).shape)
        # pose_visual = plot_animation(motion_pred[0, ...].squeeze(0), motion_target[0, ...].squeeze(0), 'aaa')
        # pose_visual.plot()
        cnt += 1
