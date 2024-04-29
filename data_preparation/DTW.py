import pywt
from scipy.interpolate import CubicSpline

def DWT(segments, n_level_DWT):
    new_segments = [[[] for i in range(len(segments[0]))] for i in range(n_level_DWT*len(segments))]

    for i_segment in range(len(segments[0])):
        for i_channel in range(len(segments)):

            segment = segments[i_channel][i_segment]
            waveletname = 'db3'

            data = segment

            for i_level in range(n_level_DWT):

                (data, coeff_d) = pywt.dwt(data, waveletname)

                x = np.arange(0, len(data))
                y = data

                # Interpolation
                temp = CubicSpline(x, y)
                xnew = np.arange(0, len(data), len(data)/len(segment))
                ynew = temp(xnew)
                if  ( len(ynew) % 10 == 1 ):
                    ynew = ynew[:(len(ynew)-1)]


                new_segments[i_channel*n_level_DWT+i_level][i_segment] = ynew
    return new_segments