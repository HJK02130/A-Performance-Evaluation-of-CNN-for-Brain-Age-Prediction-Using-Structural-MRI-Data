import numpy



def find_closest(value, array):
    array = numpy.asarray(array)
    idx = (numpy.abs(array - value)).argmin()
    return array[idx]


def frame_drop(scan, frame_keep_style="random", frame_keep_fraction=1, frame_dim=1, impute="drop"):
    if frame_keep_fraction >= 1:
        return scan

    n = scan.shape[frame_dim]
    if frame_keep_style == "random":
        frames_to_keep = int(numpy.floor(n * frame_keep_fraction))
        indices = numpy.random.permutation(numpy.arange(n))[:frames_to_keep]
    elif frame_keep_style == "ordered":
        k = int(numpy.ceil(1 / frame_keep_fraction))
        indices = numpy.arange(0, n, k)
        # pick every k'th frame
    else:
        raise Exception("Wrong frame drop style")

    if impute == "zeros":
        if frame_dim == 1:
            t = numpy.zeros((1, n, 1, 1))
            t[:, indices] = 1
            return scan * t
        if frame_dim == 2:
            t = numpy.zeros((1, 1, n, 1))
            t[:, :, indices] = 1
            return scan * t
        if frame_dim == 3:
            t = numpy.zeros((1, 1, 1, n))
            t[:, :, :, indices] = 1
            return scan * t
    elif impute == "fill":
        # fill with nearest available frame
        if frame_dim == 1:
            for i in range(scan.shape[1]):
                if i in indices:
                    pass
                scan[:, i, :, :] = scan[:, find_closest(i, indices), :, :]
            return scan

        if frame_dim == 2:
            for i in range(scan.shape[1]):
                if i in indices:
                    pass
                scan[:, :, i, :] = scan[:, :, find_closest(i, indices), :]
            return scan

        if frame_dim == 3:
            for i in range(scan.shape[1]):
                if i in indices:
                    pass
                scan[:, :, :, i] = scan[:, :, :, find_closest(i, indices)]
            return scan
    elif impute == "noise":
        noise = numpy.random.uniform(high=scan.max(), low=scan.min(), size=scan.shape)

        if frame_dim == 1:
            t = numpy.zeros((1, n, 1, 1))
            t[:, indices] = 1
            return scan + noise * (1 - t)
        if frame_dim == 2:
            t = numpy.zeros((1, 1, n, 1))
            t[:, :, indices] = 1
            return scan + noise * (1 - t)
        if frame_dim == 3:
            t = numpy.zeros((1, 1, 1, n))
            t[:, :, :, indices] = 1
            return scan + noise * (1 - t)
    else: # drop
        if frame_dim == 1:
            return scan[:, indices, :, :]
        if frame_dim == 2:
            return scan[:, :, indices, :]
        if frame_dim == 3:
            return scan[:, :, :, indices]
    return scan




def gaussian_noise(scan, sigma=0.0):
    return scan + sigma * numpy.random.randn(*scan.shape)


def intensity_scaling(scan):
    scale = 2 ** (2 * numpy.random.rand() - 1)  # 2**(-1,1)
    return scan * scale
