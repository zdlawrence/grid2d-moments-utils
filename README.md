# grid2d-moments-utils

## Requirements ##
grid2d_moments_utils.py only requires numpy and scipy

The jupyter notebook examples require
- jupyter
- numpy
- scipy
- xarray
- cartopy
- matplotlib

## To Dos ##
- Provide another example notebook using this code for image characterization tasks with comparisons to scikit-image (to demonstrate the applicability of these techniques for more general purposes) and include resources on relevant computer vision topics (e.g., Hu moments).

## Questions you might ask ##

- **Why do you not provide functions for calculating cartesian coordinates of different 2D map projections?** Because this is Cartopy's job, and it is very good at doing so! There's no need to roll your own functions since you can easily have cartopy translate between the map projections of your choice.
- **Why does your calc_2d_moments function not calculate area?** It's primarily to keep the code as general as possible without being misleading. The problem with calculating area is that it depends on the underlying coordinate system. Since these functions were written with geophysical applications in mind (where a user may be remapping from one set of coordinates to another), I decided it was better to leave the calculation of area to the user. As an exampe, suppose one converts a set of data from a spherical latitude/longitude grid to a north polar stereographic grid. The area of a region in the stereographic coordinates is not equal to the area of the same region if it were calculated in its original spherical coordinates, since the stereographic projection does not preserve area. In the polar vortex example notebook, I provide examples of how you can easily calculate area in the cartesian coordinates


## References ##

Relevant references that use these kinds of methods for atmospheric applications:

Lawrence, Z. D., & Manney, G. L. (2018). Characterizing stratospheric polar vortex variability with computer vision techniques. Journal of Geophysical Research: Atmospheres, 123(3), 1510-1535, https://doi.org/10.1002/2017JD027556

Manney, G. L., Santee, M. L., Lawrence, Z. D., Wargan, K., & Schwartz, M. J. (2021). A moments view of climatology and variability of the Asian summer monsoon anticyclone. Journal of Climate, 34(19), 7821-7841, https://doi.org/10.1175/JCLI-D-20-0729.1

Matthewman, N. J., Esler, J. G., Charlton-Perez, A. J., & Polvani, L. M. (2009). A new look at stratospheric sudden warmings. Part III: Polar vortex evolution and vertical structure. Journal of Climate, 22(6), 1566-1585, https://doi.org/10.1175/2008JCLI2365.1

Mitchell, D. M., Charlton-Perez, A. J., & Gray, L. J. (2011). Characterizing the variability and extremes of the stratospheric polar vortices using 2D moment analysis. Journal of the Atmospheric Sciences, 68(6), 1194-1213, https://doi.org/10.1175/2010JAS3555.1

Seviour, W. J., Mitchell, D. M., & Gray, L. J. (2013). A practical method to identify displaced and split stratospheric polar vortex events. Geophysical Research Letters, 40(19), 5268-5273, https://doi.org/10.1002/grl.50927

Waugh, D. W., & Randel, W. J. (1999). Climatology of Arctic and Antarctic polar vortices using elliptical diagnostics. Journal of the Atmospheric Sciences, 56(11), 1594-1613


## Acknowledgments ##
This code was developed and used for research funded by NASA Earth & Space Science Fellowship NNX16AO19H and the NWS OSTI Weeks 3-4 Program under NOAA Award NA20NWS4680051.