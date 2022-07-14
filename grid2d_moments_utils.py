""" Contains functions useful for regridding 2D data, and calculating
moments of 2D data. This code accompanies example scripts that
demonstrate the use of these functions for calculating moments
diagnostics in different contexts, such as for use with characterizing
the geometry of the stratospheric polar vortex.

Author(s): Zachary D. Lawrence
First created: Mar 2017
Last updated: Jul 2022
"""

import numpy as np
import scipy.spatial.qhull as qhull


def _parse_bool_cond(bool_cond):
    r""" A private function for determining the proper numpy boolean
    relation function based on a user-inputted string.

    Parameters
    ----------
    bool_cond : string
        A string that specifies one of the standard boolean relations.
        Function supports C / Fortran style string tokens

    Returns
    -------
    A handle to a numpy relational operator function

    """
    valid_strs = ['gt', 'ge', 'lt', 'le', 'eq', 'ne', '>', '>=',
                  '<', '<=', '==', '!=', '/=']
    if not str(bool_cond).lower() in valid_strs:
        msg = 'bool_cond must be one of {}'.format(str(valid_strs))
        raise ValueError(msg)

    bool_funcs = {
        'gt': np.greater,
        '>' : np.greater,
        'ge': np.greater_equal,
        '>=': np.greater_equal,
        'lt': np.less,
        '<' : np.less,
        'le': np.less_equal,
        '<=': np.less_equal,
        'eq': np.equal,
        '==': np.equal,
        'ne': np.not_equal,
        '!=': np.not_equal,
        '/=': np.not_equal
    }

    return bool_funcs[bool_cond]


def get_interp_params(in_coords, out_coords):
    """ Calculates the interpolation vertices and weights for (linear)
    regridding from a set of input coordinates, to a set of output coordinates

    Parameters
    ----------
    in_coords : 2D np.array
        The input coordinates of a set of data. Assumed to be 2D with the
        first dim corresponding to the number of data points, and the second
        dim corresponding to the number of dimensions
    out_coords : 2D np.array
        Desired output coordinates of the regridded data. Assumed to be 2D
        with the first dim corresponding to the number of data points, and
        the second dim corresponding to the number of dimensions

    Returns
    -------
    vertices, weights: tuple of np.arrays
        The vertices and weights necessary for doing the regrid interpolations

    References
    ----------
    Based on the stackoverflow answer at
    https://stackoverflow.com/a/20930910

    """
    d = in_coords.shape[1]

    # get tessellation of input coordinates
    tri = qhull.Delaunay(in_coords)

    # get simplices from input coordinates that contain the outpout coordinates
    simplex = tri.find_simplex(out_coords)

    # take the vertices that encompass the output coordinates
    vertices = np.take(tri.simplices, simplex, axis=0)

    # get barycentric coordinates of output coordinates with respect to the
    # encompassing simplices
    temp = np.take(tri.transform, simplex, axis=0)
    delta = out_coords - temp[:, d]
    bary = np.einsum('njk,nk->nj', temp[:, :d, :], delta)

    # compute the interpolation weights
    weights = np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))

    return (vertices, weights)


def apply_interp(data, vtx, wts, fill_value=np.nan):
    """ Performs linear interpolation regridding using pre-calculated
    simplex vertices and interpolations weights (i.e., provided by
    get_interp_params)

    Parameters
    ----------
    data : np.array
        Input data to be interpolated. Should have shape consistent with
        the vertices determined from the Delaunay tesselation
    vtx : np.array
        The vertices determined from Delaunay tesselation of data's input grid
        to the desired output grid
    wts : np.array
        The interpolation weights determined from barycentric coordinates

    kwargs
    ------
    fill_value : numeric
       The value to fill in spots where the interpolation results in
       baddata (negative weights). Defaults to nan.

    Returns
    -------
    interped: np.array
        The input data regridded based on the given vertices and weights

    References
    ----------
    Based on the stackoverflow answer at
    https://stackoverflow.com/a/20930910

    """
    interped = np.sum(np.take(data, vtx, axis=-1)*wts, axis=-1)
    interped[...,np.any(wts < 0, axis=1)] = fill_value
    return interped


def gen_regular_grid_from_irreg(in_coords, nx, ny):
    """A convenience function for creating a regular 2D grid from the input
    coordinates. Uses the min and max of the x and y coordinates defined
    by in_coords to define a regularly spaced grid between them.

    Parameters
    ----------
    in_coords : 2d np.array
        Assumed to be 2D with the "x-coordinate" corresponding to
        in_coords[:,0] and the "y-coordinate" corresponding to in_coords[:,1]

    nx : int
        number of points in x direction of output grid

    ny : int
        number of points in y direction of output grid

    Returns
    -------
    out_coords : 2D array of the regular cartesian coordinates.
        The coordinates are returned stacked in the "proper" manner expected
        for regridding to a different set of cartesian coordinates (i.e., in
        the same manner that in_coords was expected).

    """
    xreg = np.linspace(in_coords[:,0].min(),in_coords[:,0].max(), nx)
    yreg = np.linspace(in_coords[:,1].min(),in_coords[:,1].max(), ny)
    xreg,yreg = np.meshgrid(xreg,yreg)
    out_coords = np.vstack((xreg.ravel(), yreg.ravel())).T

    return out_coords


def calc_2d_moments(data, x, y, edge=0, isolation='gt', mom_type='wtd'):
    """ Calculates 2D moment diagnostics for a set of data with a defined set
    of regular input coordinates. The moments are calculated for the region
    defined by the edge and isolation kwargs, which establishes the
    outer boundary in the 2D grid.

    For example, edge = 0.0 and isolation='gt' (the defaults) will calculate
    the 2D moments of only the data with values greater than 0.

    Parameters
    ----------
    data : 1d or 2d np.array
        The data used  for calculating the moments

    x : 1d or 2d np.array
        The array specifying the "x" coordinates of the data.
        Must be same shape as data and y

    y : 1d or 2d np.array
        The array specifying the "y" coordinates of the data.
        Must be same shape as data and x

    edge : numeric, optional
        The value that will be used for determining the boundary containing
        the points that will be used for calculating the 2D moments. Defaults
        to 0.

    isolation : string, optional
        A token string that specifies the boolean relationship defining
        the region/points used in the moments calculations. For example,
        "gt" specifies that the function will find the 2D moments of the
        region only where data > edge

    mom_type : string, optional
        A token string that specifies the type of moment calculation. Can
        be either "wtd" for a data-weighted calculation, or "geom" for
        the geometric calculation. These calculations are akin to finding
        the center of mass for an object with spatially varying density
        (in the case of "wtd"), or for an object with uniform density (in
        the case of "geom").

    Returns
    -------
    moments : A python dictionary of floats (or np.nans) containing the
              2D moments. xc and yc are the coordinates of the centroid;
              the Mpq variables are the (p+q)th order raw moments, while the
              Jpq variables are the (p+q)th order central moments. angle,
              asrat (aspect ratio) and ek (excess kurtosis) all specify the
              elliptical diagnostics

    Notes
    -----
    The moment diagnostics provided by this function are those consistent with
    the Matthewman et al (2009) paper listed in the references below, but note
    that these diagnostics are applicable to more general applications than
    just characterizing the stratospheric polar vortex. These diagnostics
    are, in essence, "image moments"
    (see https://en.wikipedia.org/wiki/Image_moment), which means they are
    applicable for any datasets that can be usefully thought of (or
    represented) as images.

    References
    ----------
    Matthewman, N. J., Esler, J. G., Charlton-Perez, A. J., & Polvani, L. M.
        (2009). A New Look at Stratospheric Sudden Warmings. Part III:
        Polar Vortex Evolution and Vertical Structure, Journal of Climate,
        22(6), 1566-1585
    Link: https://journals.ametsoc.org/view/journals/clim/22/6/2008jcli2365.1.xml

    """
    bool_func = _parse_bool_cond(isolation)

    if (data.ndim > 2):
        msg = "The data array must be 2D or less"
        raise ValueError(msg)

    if (data.shape != x.shape) or (data.shape != y.shape) or \
       (x.shape != y.shape):
        msg = "Shapes of data and coordinates must all match"
        raise ValueError(msg)

    if (mom_type not in ['wtd', 'geom']):
        msg = "mom_type must be one of 'wtd' or 'geom'"
        raise ValueError(msg)

    # Get indices where inside region satisfies the boolean condition
    ins = np.where(bool_func(data, edge))

    # Get indices where outside region satisfies the boolean condition
    outs = np.where(np.logical_not(bool_func(data, edge)))

    # If we're finding moments with density (data) weighting, then
    # make the modified data such that the outside region equals
    # 0, and the inside region is the "excess" over the edge value
    mod_data = np.copy(data)
    if (mom_type == 'wtd'):
        mod_data[outs] = edge
        mod_data = np.abs(mod_data - edge)
    # If we're finding geometric moments (without density weighting)
    # then make the modified data such that the outside is 0 and
    # the inside is 1 (i.e., a "binary image")
    elif (mom_type == 'geom'):
        mod_data[outs] = 0
        mod_data[ins] = 1

    # if there are no points inside the region, all moments are undefined
    num_points = ins[0].size
    if (num_points == 0):
        m00 = m10 = m01 = np.nan
        xc = yc = np.nan
        j11 = j20 = j02 = np.nan
        j22 = j40 = j04 = np.nan
        angle = asrat = exkur = np.nan

    # In the calculations below, we technically don't need "outs"
    # since we only need to do the integrals using points/weighting from
    # the inside points, as we have made all the outside points equal 0
    else:
        # Calculate the raw moments
        m00 = np.sum(mod_data[ins])
        m10 = np.sum(mod_data[ins]*x[ins])
        m01 = np.sum(mod_data[ins]*y[ins])
        xc = m10/m00 # x-coordinate of centroid
        yc = m01/m00 # y-coordinate of centroid

        # Calculate the central moments
        xmxc = x - xc
        ymyc = y - yc
        j11 = np.sum(mod_data[ins]*xmxc[ins]*ymyc[ins])
        j20 = np.sum(mod_data[ins]*xmxc[ins]*xmxc[ins])
        j02 = np.sum(mod_data[ins]*ymyc[ins]*ymyc[ins])
        j22 = np.sum(mod_data[ins]*xmxc[ins]*xmxc[ins]*ymyc[ins]*ymyc[ins])
        j40 = np.sum(mod_data[ins]*xmxc[ins]*xmxc[ins]*xmxc[ins]*xmxc[ins])
        j04 = np.sum(mod_data[ins]*ymyc[ins]*ymyc[ins]*ymyc[ins]*ymyc[ins])

        # Calculate the equivalent ellipse diagnostics (angle and aspect ratio)
        angle = np.degrees(0.5*np.arctan2((2*j11),(j20-j02)))

        j20_plus_j02 = (j20 + j02)
        sqrt_j2_terms = np.sqrt(4*(j11**2) + (j20-j02)**2)
        asrat = np.sqrt(np.abs((j20_plus_j02 + sqrt_j2_terms) \
                              /(j20_plus_j02 - sqrt_j2_terms)))

        # Calculate the excess kurtosis
        ek1 = m00*((j40 + 2*j22 + j04)/((j20+j02)**2))
        ek2 = (3*(asrat**4)+2*(asrat**2)+3) / (((asrat**2)+1)**2)
        exkur=  ek1 - (2*ek2/3.)

    moments = {'m00':m00, 'm10':m10, 'm01':m01, 'xcent':xc, 'ycent':yc,
               'j11':j11, 'j20':j20, 'j02':j02, 'j40':j40, 'j04':j04,
               'angle':angle, 'asrat':asrat, 'ek':exkur, 'npoints':num_points}
    return moments


def equivalent_ellipse(area, angle, aspect_ratio,
                       centroid_x, centroid_y, npoints=100):
    """ A convenience function that calculates the boundary coordinates of
    the equivalent ellipse of a region with a defined centroid, area, angle,
    and aspect ratio.

    Parameters
    ----------
    area : float
        The area of the region

    angle : float
        The angle of the region with respect to the "x" axis

    aspect_ratio : float
        The aspect ratio of the region

    centroid_x : float
        The x-coordinate of the region's centroid

    centroid_y : float
        The y-coordinate of the region's centroid

    npoints : int
        The number of points to define around the boundary of the
        equivalent ellipse. Defaults to 100.

    Returns
    -------
    tuple containing numpy arrays of the x and y coordinates of the
    equivalent ellipse boundary

    """
    min_length = 2*np.sqrt(area / (np.pi * aspect_ratio))
    maj_length = min_length*aspect_ratio
    t = np.linspace(0, 2*np.pi, npoints)
    a_cos_t = np.cos(t)*(maj_length/2.)
    b_sin_t = np.sin(t)*(min_length/2.)
    x = centroid_x + (a_cos_t*np.cos(np.radians(angle))) - (b_sin_t*np.sin(np.radians(angle)))
    y = centroid_y + (a_cos_t*np.sin(np.radians(angle))) + (b_sin_t*np.cos(np.radians(angle)))

    return np.concatenate((x[:,np.newaxis], y[:,np.newaxis]), axis=1)