import numpy as np


def calculate_thetas_smooth(pos, win_size):
    pad = int(np.floor(win_size/2))

    thetas = []

    for idx in range(pad):
        a = pos[idx]
        b = pos[idx+1]

        dy = b[1] - a[1]
        dx = b[0] - a[0]
        if dx == 0:
            theta = 0 if dy > 0 else np.pi
        else:
            theta = np.arctan(dy/dx)

        if dx > 0 : theta -= np.pi/2
        elif dx < 0 : theta += np.pi/2

        thetas.append(theta)

    for p in np.lib.stride_tricks.sliding_window_view(pos, win_size, axis=0):
        dy = np.mean(np.diff(p[1, :]))
        dx = np.mean(np.diff(p[0, :]))
        
        if dx == 0:
            theta = 0 if dy > 0 else np.pi
        else:
            theta = np.arctan(dy/dx)

        if dx > 0 : theta -= np.pi/2
        elif dx < 0 : theta += np.pi/2

        thetas.append(theta)

    for idx in range(len(pos)-pad, len(pos)-1):
        a = pos[idx]
        b = pos[idx+1]

        dy = b[1] - a[1]
        dx = b[0] - a[0]
        if dx == 0:
            theta = 0 if dy > 0 else np.pi
        else:
            theta = np.arctan(dy/dx)

        if dx > 0 : theta -= np.pi/2
        elif dx < 0 : theta += np.pi/2

        thetas.append(theta)
    
    return thetas

def calculate_rot_velocity(thetas):
    rot_velocity = np.zeros_like(thetas, dtype=np.float32)
    for idx in range(1, len(thetas)):
        diff = thetas[idx] - thetas[idx-1]
        if diff > np.pi:
            diff -= 2*np.pi
        elif diff < -np.pi:
            diff += 2*np.pi
        rot_velocity[idx] = diff
    return rot_velocity
