#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from fit_routine_original import WLAX, Lines, lines, sii_doublet, c, z
from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import curve_fit
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

class SII(Lines):
    def __init__(self, name: str, rest, crange, vrange, cube: NDArray, varc: NDArray, contspec: NDArray):
        super().__init__(name, rest, crange, vrange, cube, varc, contspec)
        self.CUBE = cube
        self.fitcube = np.zeros((7, self.cube_x, self.cube_y))
        self.fiterrcube = np.zeros((5, self.cube_x, self.cube_y))

    def plot_spe(self, ax_in: plt.Axes, i: int, j: int):
        self.spec = self.spaxel(self.subcube, i, j)
        self.varspec = self.spaxel(self.errcube, i, j)
        self.remove_nan()
        
        lranges = (self.lranges[0] < self.wlax) & (self.wlax < self.lranges[1])
        fit_spec = self.baseline_subtraction()
        

        popt = list(self.fitcube[0:5, i, j])
        fitsnr1 = self.fitcube[5, i, j]
        fitsnr2 = self.fitcube[6, i, j]
        uncertainty = list(self.fiterrcube[:, i, j])
        
        quicksnr = sum(fit_spec)/np.sqrt(sum(self.varspec))

        
        
        ratioerr = (popt[0]/popt[1])*np.sqrt((uncertainty[0]/popt[0])**2 + (uncertainty[1]/popt[1])**2)
        

        #plot
        ax_in[0].step(self.wlax, fit_spec, where='mid', color='black')
        ax_in[0].step(self.wlax[lranges], fit_spec[lranges], where='mid', color='#1f77b4')
        ax_in[0].axhline(y=np.median(fit_spec[~lranges]), color = 'green', label = 'Median of baseline')
        ax_in[0].plot(self.wlax, sii_doublet(self.wlax, *popt), color='Orange')
        ax_in[0].axhline(y=np.median(fit_spec[lranges]), color='r', label='Median around line ranges')
        ax_in[0].errorbar(self.wlax,fit_spec,yerr=np.sqrt(self.varspec),color='#1f77b4',linestyle='')
        ax_in[0].set_title(f"(i,j): ({j+1}, {i+1}), fitsnr = {fitsnr1}, {fitsnr2}, ratio = {popt[0]/popt[1]} rerr = {ratioerr} ")
        ax_in[0].legend(loc='upper right')
        
        
        linewlax = WLAX[1670:1720]
        contwlax = WLAX[1721:1791]
        linespec = self.CUBE[1670:1720,i,j]
        contspec = self.CUBE[1721:1791,i,j]
        lineavg = np.average(linespec)
        contavg = np.average(contspec) * .98
        ax_in[1].step(linewlax, linespec, where='mid')
        ax_in[1].step(contwlax, contspec, where='mid')
        
        ax_in[1].axhline(y=lineavg, color='orange', label='line average')
        ax_in[1].axhline(y=contavg, color='red', label='cont average')
        ax_in[1].set_title(f"Original Spectrum at ({j+1},{i+1})")
        ax_in[1].legend(loc='upper right')

    
if __name__ == "__main__":
    hdul1 = fits.open("ADP.2016-06-03T11_20_45.461.fits")
    head = hdul1[1].header
    cubehdu = hdul1[1]
    cubehead = cubehdu.header
    cube = cubehdu.data


    
    varc = hdul1[2].data

    hdul2 = fits.open("continuum spectrum.fits")
    basespec = hdul2[0].data

    sii = SII('SII', lines['SII'][0], lines['SII'][1], lines['SII'][2], cube, varc, basespec)

    quickrej = 0
    snrrej = 0
    runerr = 0

    obs_0 = sii.obs[0]
    mask = (sii.lranges[0] < sii.wlax) & (sii.wlax < sii.lranges[1])
    l_wlax = sii.wlax[mask]
    l_lranges = sii.lranges

    stime = time.time()
    for i in tqdm(range(sii.cube_x), smoothing=1):
        

        for j in range(sii.cube_y):
            
            fit_spec, err_spec = sii.get_fit_spaxel(i, j)

            if type(fit_spec) == type(None):
                quickrej +=1
                sii.rejcube[0,i,j] = 1
                continue

            try:
                
                popt, pcov = curve_fit(sii_doublet, l_wlax, fit_spec[mask], p0=[176, 150, obs_0, 2.3], 
                                    bounds=([0,0,l_lranges[0],0], [5e3, 5e3, l_lranges[1], 20]), 
                                    absolute_sigma=True, sigma=err_spec[mask])

                uncertainty = np.diagonal(np.sqrt(np.abs(pcov))[0:5])
                snr1 = popt[0] / uncertainty[0]
                snr2 = popt[1] / uncertainty[1]
                
                if  snr1 > 3. and snr2 > 3.:
                    sii.fitcube[:4,i,j] = popt[:]
                    sii.fitcube[4,i,j] = snr1
                    sii.fitcube[5,i,j] = snr2
                    sii.fiterrcube[:,i,j] = uncertainty
                else:
                    sii.fitcube[:4,i,j] = np.nan
                    sii.fitcube[4,i,j] = snr1
                    sii.fitcube[5,i,j] = snr2
                    
                    sii.rejcube[1,i,j] = 1
                    snrrej += 1
                    
            except (RuntimeError, ValueError):
                sii.set_to_nan(i, j)
                sii.rejcube[2,i,j] = 1
                runerr +=1

    print(quickrej)
    print(snrrej)
    print(runerr)
    print(time.time() - stime)

    newwcs = WCS(cubehead, naxis=2)
    newhead = newwcs.to_header()
    prihdu = fits.PrimaryHDU(sii.fitcube[0], header=newhead)
    newsiihdus = [fits.ImageHDU(sii.fitcube[i]) for i in range(1,sii.fitcube.shape[0])]
    siierrhdus = [fits.ImageHDU(sii.fiterrcube[i]) for i in range(sii.fiterrcube.shape[0])]
    hdul = fits.HDUList([prihdu] + newsiihdus + siierrhdus)
    hdul.writeto('sii_fit_test.fits', overwrite = True)

    rejhdus = fits.PrimaryHDU(sii.rejcube[0], header=newhead)
    otherrejhdus = [fits.ImageHDU(sii.rejcube[i]) for i in range(1,sii.rejcube.shape[0])]
    hdul2 = fits.HDUList([rejhdus]+otherrejhdus)
    hdul2.writeto('sii_rej_test.fits', overwrite = True)

    detectedimg = np.nan_to_num(sii.fitcube[0])
    snrrejimg = np.nan_to_num(sii.rejcube[1])
    evalimg = detectedimg + snrrejimg

    evalhdus = fits.PrimaryHDU(evalimg, header=newhead)
    hdul3 = fits.HDUList([evalhdus])
    hdul3.writeto('sii_eval_test.fits', overwrite = True)

    flux0hdus = [prihdu]
    flux1hdus = [fits.ImageHDU(sii.fitcube[1])]
    velhdus = [fits.ImageHDU(c*(sii.fitcube[2]/sii.rest[0]-1-z))]
    vdisphdus = [fits.ImageHDU(c*(sii.fitcube[3]/sii.rest[0]))]
    resulthdus =  fits.HDUList(flux0hdus+flux1hdus+velhdus+vdisphdus)
    resulthdus.writeto('sii_result_test.fits', overwrite=True)


    check_pixels = [(59,44), (57,41)]
    sii.plot_eval(check_pixels)

        

