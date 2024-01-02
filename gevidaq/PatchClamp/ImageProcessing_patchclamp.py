# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 11:52:09 2021

@author: TvdrBurgt
"""

import numpy as np
from scipy import cluster
from skimage import feature, filters, transform


class PatchClampImageProcessing:
    @staticmethod
    def makeGaussian(size=(2048, 2048), mu=(1024, 1024), sigma=(512, 512)):
        """
        This function returns a normalized Gaussian distribution in 2D with a
        user specified center position and standarddeviation.
        """

        x = np.arange(0, size[0], 1, float)
        y = np.arange(0, size[1], 1, float)

        def gauss(x, mu, sigma):
            return np.exp((-((x - mu) ** 2)) / (2 * sigma**2))

        xs = gauss(x, mu[0], sigma[0])
        ys = gauss(y, mu[1], sigma[1])

        xgrid = np.tile(xs, (size[1], 1))
        ygrid = np.tile(ys, (size[0], 1)).transpose()

        window = np.multiply(xgrid, ygrid)

        return window / np.sum(window)

    @staticmethod
    def detectPipettetip(img_a, img_b, diameter, orientation, plotflag=False):
        """
        Tip detection algorithm
        input parameters:
            img_a       = previous image of pipette tip
            img_b       = current image of pipette tip
            diameter    = diameter of pipette tip in pixels
            orientation = pipette orientation in degrees, clockwise calculated
                          from the horizontal pointing to the right (0 degrees)
            plotflag    = if True it will generate figures
        output parameters:
            xpos        = x position of the pipette tip
            ypos        = y position of the pipette tip
        """
        # Image normalization
        img_a = img_a / np.sum(img_a)
        img_b = img_b / np.sum(img_b)

        # Gaussian blur
        left_blurred = filters.gaussian(img_a, 1)
        right_blurred = filters.gaussian(img_b, 1)

        # Image subtraction
        img_blurred = left_blurred - right_blurred

        # Canny edge detection
        edges = feature.canny(
            img_blurred,
            sigma=3,
            low_threshold=0.99,
            high_threshold=0,
            use_quantiles=True,
        )

        # Hough transform
        angle_range = np.linspace(0, np.pi, 500) + np.deg2rad(orientation)
        hspace, angles, distances = transform.hough_line(edges, angle_range)

        # Find Hough peaks
        _, Tpeaks, Rpeaks = transform.hough_line_peaks(
            hspace, angles, distances, num_peaks=10, threshold=0
        )

        # Cluster peaks
        idx_lowT = np.argmin(Tpeaks)
        idx_highT = np.argmax(Tpeaks)
        initial_clusters = np.array(
            [
                [Tpeaks[idx_lowT], Rpeaks[idx_lowT]],
                [Tpeaks[idx_highT], Rpeaks[idx_highT]],
            ]
        )
        data = np.transpose(np.vstack([Tpeaks, Rpeaks]))
        centroids, labels = cluster.vq.kmeans2(
            data, k=initial_clusters, iter=10, minit="matrix"
        )
        centroid1, centroid2 = centroids

        # Find intersection between
        # X1*cos(T1)+Y1*sin(T1)=R1 and X2*cos(T2)+Y2*sin(T2)=R2
        if centroid1[0] > centroid2[0]:
            angle1, dist1 = centroid1
            angle2, dist2 = centroid2
        else:
            angle1, dist1 = centroid2
            angle2, dist2 = centroid1

        LHS = np.array(
            [
                [np.cos(angle1), np.sin(angle1)],
                [np.cos(angle2), np.sin(angle2)],
            ]
        )
        RHS = np.array([dist1, dist2])
        xpos, ypos = np.linalg.solve(LHS, RHS)

        # Bias correction
        H = diameter / (2 * np.tan((angle1 - angle2) / 2))
        alpha = (angle1 + angle2) / 2 - np.pi / 2
        xpos = xpos - H * np.cos(alpha)
        ypos = ypos - H * np.sin(alpha)

        return xpos, ypos

    @staticmethod
    def comp_variance_of_Laplacian(img):
        """
        Computes a sharpness score that is minimal for sharp images.
        """
        # average images
        img_average = filters.gaussian(img, 4)

        # calculate laplacian
        img_laplace = filters.laplace(img_average, 3)

        # calculate variance
        score = np.var(img_laplace)

        return score
