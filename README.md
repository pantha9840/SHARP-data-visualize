# SHARP-data-visualize
#import matplotlib.pyplot as plt

#plt.imshow([[0, 1]], cmap=cmap)
#plt.colorbar()
#plt.title('sdoaia4500 colormap')
#plt.show()

# Integrated Python script to extract SHARP data, visualize it, and compute total magnetic flux,
# tilt angle, and outline for PIL and flare correlation (placeholders for latter analyses)

from __future__ import division, print_function
import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import sunpy.map
from sunpy.visualization.colormaps import color_tables as ct
import drms
from scipy.ndimage import label, center_of_mass
from scipy.stats import linregress

# ===========================
# User Input and Setup
# ===========================

series = 'hmi.sharp_cea_720s'
sharpnum = int(input('Enter SHARP number (e.g. 377): '))
email = 'abcd@gmail.com'
segments = ['magnetogram', 'continuum']
kwlist = ['T_REC', 'LON_FWT', 'OBS_VR', 'CROTA2',
          'CRPIX1', 'CRPIX2', 'CDELT1', 'CDELT2', 'CRVAL1', 'CRVAL2']

c = drms.Client(email=email)
k = c.query(f'{series}[{sharpnum}]', key=kwlist, rec_index=True)

rec_cm = k.LON_FWT.abs().idxmin()
k_cm = k.loc[rec_cm]
t_cm = drms.to_datetime(k_cm.T_REC)
print(rec_cm, '@', k_cm.LON_FWT, 'deg')
print('Timestamp:', t_cm)

t_cm_str = t_cm.strftime('%Y%m%d_%H%M%S_TAI')
fname_mask = '{series}.{sharpnum}.{tstr}.{segment}.fits'
fnames = {
    s: fname_mask.format(
        series=series, sharpnum=sharpnum, tstr=t_cm_str, segment=s)
    for s in segments
}

download_segments = [s for s in segments if not os.path.exists(fnames[s])]
if download_segments:
    exp_query = f'{rec_cm}{{{",".join(download_segments)}}}'
    r = c.export(exp_query, method='url', protocol='fits')
    r.download('.')

# ===========================
# Load Data
# ===========================

def read_fits_data(fname):
    hdulist = fits.open(fname)
    hdulist.verify('silentfix+warn')
    return hdulist[1].data

mag = read_fits_data(fnames['magnetogram'])
cont = read_fits_data(fnames['continuum'])
a1 = sunpy.map.Map(fnames['magnetogram'])

# ===========================
# Coordinate Grid
# ===========================

ny, nx = mag.shape
xmin = (1 - k_cm.CRPIX1) * k_cm.CDELT1 + k_cm.CRVAL1
xmax = (nx - k_cm.CRPIX1) * k_cm.CDELT1 + k_cm.CRVAL1
ymin = (1 - k_cm.CRPIX2) * k_cm.CDELT2 + k_cm.CRVAL2
ymax = (ny - k_cm.CRPIX2) * k_cm.CDELT2 + k_cm.CRVAL2

x_data = np.linspace(xmin, xmax, nx)
y_data = np.linspace(ymin, ymax, ny)
X, Y = np.meshgrid(x_data, y_data)

# ===========================
# Total Unsigned Magnetic Flux
# ===========================

lv1 = 150  # threshold
mask = np.abs(mag) >= lv1
dx = 720  # 1 arcsec = 720 km (approx)
dy = 720
pixel_area_km2 = dx * dy
flux = np.sum(np.abs(mag[mask])) * pixel_area_km2 * 1e8  # convert to Mx
print(f"Total Unsigned Magnetic Flux (|B| > {lv1} G): {flux:.2e} Mx")

# ===========================
# Tilt Angle Estimation
# ===========================

def compute_tilt_angle(B, x, y, threshold=lv1):
    pos_mask = B > threshold
    neg_mask = B < -threshold

    if np.any(pos_mask) and np.any(neg_mask):
        pos_coords = np.array(center_of_mass(pos_mask))
        neg_coords = np.array(center_of_mass(neg_mask))

        y0, x0 = neg_coords
        y1, x1 = pos_coords

        tilt_rad = np.arctan2((y1 - y0), (x1 - x0))
        tilt_deg = np.degrees(tilt_rad)
        return tilt_deg
    return None

tilt_angle = compute_tilt_angle(mag, X, Y)
if tilt_angle is not None:
    print(f"Tilt Angle (Joy's Law angle): {tilt_angle:.2f} degrees")

# ===========================
# Polarity Inversion Line (PIL) - Visualization only
# ===========================

pil_mask = np.zeros_like(mag, dtype=bool)
pil_mask[1:-1, 1:-1] = ((mag[1:-1, 1:-1] > 0) &
                        (mag[:-2, 1:-1] < 0) |
                        (mag[1:-1, 1:-1] < 0) &
                        (mag[:-2, 1:-1] > 0))

# ===========================
# Visualization
# ===========================

plt.rc('font', family='serif')
fig, ax = plt.subplots(2, 1, figsize=(10, 10))

cont_map = plt.get_cmap('sdoaia4500')
img_cont = ax[0].pcolormesh(X, Y, cont/1e3, cmap=cont_map, shading='auto')
fig.colorbar(img_cont, ax=ax[0], label=r'$I_{\mathrm{c}}$ [kDN/s]', shrink=0.7)
ax[0].contour(X, Y, mag, levels=[lv1], colors='b')
ax[0].contour(X, Y, mag, levels=[-lv1], colors='r')
ax[0].set_title(f'NOAA {a1.meta["noaa_ar"]} @ {a1.meta["t_obs"]}')
ax[0].axis('image')

img_mag = ax[1].pcolormesh(X, Y, mag/1e3, cmap='gray', shading='auto', vmin=-1, vmax=1)
fig.colorbar(img_mag, ax=ax[1], label=r'$B_{\mathrm{los}}$ [kG]', shrink=0.7)
ax[1].contour(X, Y, pil_mask, colors='lime', linewidths=0.5)
ax[1].set_title("Magnetogram with Polarity Inversion Line (PIL)")
ax[1].axis('image')

fig.text(0.1, 0.05, f"Total Flux: {flux:.2e} Mx", fontsize=10)
if tilt_angle is not None:
    fig.text(0.1, 0.02, f"Tilt Angle: {tilt_angle:.2f} degrees", fontsize=10)

plt.tight_layout()
plt.show()
