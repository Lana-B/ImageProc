import numpy as np
import math
import cv2.ximgproc 
from matplotlib import pylab as plt
import scipy as sp
from scipy import ndimage
from scipy.optimize import curve_fit
from scipy import asarray as ar,exp
from scipy.signal import find_peaks_cwt
import matplotlib.cm as cm

plotIt=False
GBP=False #gain bad pix correction
BLF=False #bilaterial filter and sobel operations
SOB=False
Fields=np.array([8,10,20,25,30,40,50])
pixels_x=4096
pixels_y=4096
pixels=4096*4096

gains=np.fromfile("FlatFieldGains.raw", dtype='float64', sep="")
gains=gains.reshape([1,4096])

dark=np.fromfile("Pedestal/ped_10_27_LE.raw", dtype='<i2', sep="") # little endian (<) int 2 (i2) bytes = 16 bit signed int
dark=dark.reshape(1,pixels)

distanceCalibration=1/0.6072
colors = iter(cm.rainbow(np.linspace(0, 1, 6)))
fig,ax=plt.subplots()
fig2,ax2=plt.subplots()
fig3,ax3=plt.subplots()

leaf_of_choice=10

std_file=np.ones(1024*1024)
std_file=std_file*20

std_file=std_file.reshape(1024,1024)
errorOnPixIntensity=15


# leaf_gradients_dict={}
def line(x,m,c):
  return m*x+c

def gaus(x,a,x0,sigma):
  return a*exp(-(x-x0)**2/(2*sigma**2))

def lognorm(x,a,mu,sigma):
  b = (math.log(x) - mu)/math.sqrt(2*sigma**2)
  p = a*(0.5 + 0.5*math.erf(b))
  return p

def gainsBadPix(origIm, gainIm, pedIm,frames):
  # pedIm=pedIm.reshape(1,pixels)

  origIm=origIm.reshape([frames,pixels])

  #pedestal subtraction
  origIm=origIm-pedIm

  print "ped subtraction"

  origIm=origIm.reshape([frames,pixels_y,pixels_x])
  origIm[:,1022,:]=(origIm[:,1021,:]+origIm[:,1024,:])/2
  origIm[:,1023,:]=(origIm[:,1022,:]+origIm[:,1024,:])/2
  origIm[:,138,:]=(origIm[:,137,:]+origIm[:,141,:])/2
  origIm[:,139,:]=(origIm[:,138,:]+origIm[:,141,:])/2
  origIm[:,140,:]=(origIm[:,139,:]+origIm[:,141,:])/2
  origIm[:,2599,:]=(origIm[:,2598,:]+origIm[:,2602,:])/2
  origIm[:,2600,:]=(origIm[:,2599,:]+origIm[:,2602,:])/2
  origIm[:,2601,:]=(origIm[:,2600,:]+origIm[:,2602,:])/2

  print "bad cols corrected"

  #"firmware lines"
  origIm[:,4095,:]=origIm[:,4094,:]
  origIm[:,3967,:]=(origIm[:,3966,:]+origIm[:,3968,:])/2
  origIm[:,3839,:]=(origIm[:,3838,:]+origIm[:,3840,:])/2
  origIm[:,3711,:]=(origIm[:,3710,:]+origIm[:,3712,:])/2
  origIm[:,3583,:]=(origIm[:,3582,:]+origIm[:,3584,:])/2
  origIm[:,3455,:]=(origIm[:,3454,:]+origIm[:,3456,:])/2
  origIm[:,3327,:]=(origIm[:,3326,:]+origIm[:,3328,:])/2
  origIm[:,3199,:]=(origIm[:,3198,:]+origIm[:,3200,:])/2
  origIm[:,3071,:]=(origIm[:,3070,:]+origIm[:,3072,:])/2
  origIm[:,2943,:]=(origIm[:,2942,:]+origIm[:,2944,:])/2
  origIm[:,2815,:]=(origIm[:,2814,:]+origIm[:,2816,:])/2
  origIm[:,2687,:]=(origIm[:,2686,:]+origIm[:,2688,:])/2
  origIm[:,2559,:]=(origIm[:,2558,:]+origIm[:,2560,:])/2
  origIm[:,2431,:]=(origIm[:,2430,:]+origIm[:,2432,:])/2
  origIm[:,2303,:]=(origIm[:,2302,:]+origIm[:,2304,:])/2
  origIm[:,2175,:]=(origIm[:,2174,:]+origIm[:,2176,:])/2
  origIm[:,2047,:]=(origIm[:,2046,:]+origIm[:,2048,:])/2
  origIm[:,1919,:]=(origIm[:,1918,:]+origIm[:,1920,:])/2
  origIm[:,1791,:]=(origIm[:,1790,:]+origIm[:,1792,:])/2
  origIm[:,1663,:]=(origIm[:,1662,:]+origIm[:,1664,:])/2
  origIm[:,1535,:]=(origIm[:,1534,:]+origIm[:,1536,:])/2
  origIm[:,1407,:]=(origIm[:,1406,:]+origIm[:,1408,:])/2
  origIm[:,1279,:]=(origIm[:,1278,:]+origIm[:,1280,:])/2
  origIm[:,1151,:]=(origIm[:,1150,:]+origIm[:,1152,:])/2
  origIm[:,1023,:]=(origIm[:,1022,:]+origIm[:,1024,:])/2
  origIm[:,895,:]=(origIm[:,894,:]+origIm[:,896,:])/2
  origIm[:,767,:]=(origIm[:,766,:]+origIm[:,768,:])/2
  origIm[:,639,:]=(origIm[:,638,:]+origIm[:,640,:])/2
  origIm[:,511,:]=(origIm[:,510,:]+origIm[:,512,:])/2
  origIm[:,383,:]=(origIm[:,382,:]+origIm[:,384,:])/2
  origIm[:,255,:]=(origIm[:,254,:]+origIm[:,256,:])/2
  origIm[:,127,:]=(origIm[:,126,:]+origIm[:,128,:])/2

  print "firmware lines corrected"

  gainMean=gainIm.mean()
  gainIm=np.tile([gainIm],(frames,1,1))
  origIm=origIm/gains*gainMean
  origIm=origIm.reshape(frames,1024,4,1024,4).mean((2,4))

  print "gain corrected"

  return origIm

def BLFilt(im, frames):
  print im.shape
  for frame in range(0,frames):
    if (frame%20==0):
      print "frame = "+str(frame)

    startFrame=0+(1024*1024)*frame
    endFrame=0+(1024*1024)*(frame+1)

    singleFrameIm=im[startFrame:endFrame]
    singleFrameIm=singleFrameIm.reshape([1024,1024])

    singleFrameIm = np.array(singleFrameIm, dtype = np.float32)

    #bilateral filter to remove noise while preserving edges
    singleFrameIm=cv2.bilateralFilter(singleFrameIm,50,3500,2000)
    singleFrameIm=cv2.bilateralFilter(singleFrameIm,10,1000,2000)
    im[startFrame:endFrame]=singleFrameIm.ravel()
  print im.shape
  return im

def Sob_op(im, frames):
  print im.shape
  for frame in range(0,frames):
    if (frame%20==0):
      print "frame = "+str(frame)

    startFrame=0+(1024*1024)*frame
    endFrame=0+(1024*1024)*(frame+1)

    singleFrameIm=im[startFrame:endFrame]
    singleFrameIm=singleFrameIm.reshape([1024,1024])

    sobel_kernel=np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    sobelSigned = ndimage.convolve(singleFrameIm, sobel_kernel)
    sobelAbs=np.absolute(sobelSigned)
    sobelAbs2 = np.array(sobelAbs, dtype = np.float32)

    sobelAbs2=cv2.bilateralFilter(sobelAbs2,24,5000,5000)
    im[startFrame:endFrame]=sobelAbs2.ravel()
  print im.shape
  return im

for field in range(1,7):#0,len(Fields)
  colorChoice=next(colors)

  # if (Fields[field]==8 or Fields[field]==20 or Fields[field]==30 or Fields[field]==40):
    # continue
  print ("field ",Fields[field])
  frames=0
  if (GBP):
    im=np.fromfile("Z_plan_vmat_E_6_MU_"+str(Fields[field])+"_Sub_LE.raw", dtype='<i2', sep="") # little endian (<) int 2 (i2) bytes = 16 bit signed int
    frames=len(im)/pixels

    im=gainsBadPix(im,gains,dark,frames)
    print "gain and bad pix corrected"
    print ("dtype",im.dtype)
    im.tofile("Z_plan_vmat_E_6_MU_"+str(Fields[field])+"GBPcorr"+"_Sub_LE.raw")
    continue

  #iterate over frames 22 - 41 for 8MU
 
  if BLF:
    im=np.fromfile("Z_plan_vmat_E_6_MU_"+str(Fields[field])+"GBPcorr"+"_Sub_LE.raw", dtype='float64', sep="") # little endian (<) int 2 (i2) bytes = 16 bit signed int
    frames=len(im)/1024/1024
    im=BLFilt(im, frames)
    im.tofile("Z_plan_vmat_E_6_MU_"+str(Fields[field])+"GBPcorrBL"+"_Sub_LE.raw")
    continue
  else:
    im=np.fromfile("Z_plan_vmat_E_6_MU_"+str(Fields[field])+"GBPcorrBL"+"_Sub_LE.raw", dtype='float64', sep="")
    print "loaded Z_plan_vmat_E_6_MU_"+str(Fields[field])+"GBPcorrBL"+"_Sub_LE.raw"
    frames=len(im)/1024/1024

  # if SOB:
  #   im=np.fromfile("Z_plan_vmat_E_6_MU_"+str(Fields[field])+"GBPcorrBL"+"_Sub_LE.raw", dtype='float64', sep="") # little endian (<) int 2 (i2) bytes = 16 bit signed int
  #   frames=len(im)/1024/1024
  #   im=Sob_op(im, frames)
  #   im.tofile("Z_plan_vmat_E_6_MU_"+str(Fields[field])+"GBPcorrBLSOB"+"_Sub_LE.raw")
  #   continue
  # else:
  #   im=np.fromfile("Z_plan_vmat_E_6_MU_"+str(Fields[field])+"GBPcorrBLSOB"+"_Sub_LE.raw", dtype='float64', sep="")
  #   print "loaded Z_plan_vmat_E_6_MU_"+str(Fields[field])+"GBPcorrBLSOB"+"_Sub_LE.raw"
  #   frames=len(im)/1024/1024


  nLeaves=18
  leaf_pos_dict={}
  leaf_pos_error_dict={}
  leaf_n_points_dict={}

  for leaf in range(0,nLeaves):
    leaf_pos_dict[leaf]=np.array([])
    leaf_pos_error_dict[leaf]=np.array([])
    leaf_n_points_dict[leaf]=np.array([])


  for frame in range(0,frames):
    if (frame%15==0):
      print "frame = "+str(frame)

    startFrame=0+(1024*1024)*frame
    endFrame=0+(1024*1024)*(frame+1)

    singleFrameIm=im[startFrame:endFrame]
    singleFrameIm=singleFrameIm.reshape([1024,1024])
    sobel_kernel=np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    sobelSigned = ndimage.convolve(singleFrameIm, sobel_kernel)
    sobelAbs=np.absolute(sobelSigned)
    sobelAbs2 = np.array(sobelAbs, dtype = np.float32)

    sobelAbs2=cv2.bilateralFilter(sobelAbs2,24,5000,5000)
  
    #iterate over each leaf in a frame
    start=54 #196/4
    leafWidth=53
    nSlicesUsed=25


    for leaf in range(0,nLeaves):

      #choose y slices to use for each leaf
      sliceStart=start+leaf*leafWidth
      sliceEnd=sliceStart+nSlicesUsed
      # print "start="+str(sliceStart*4)+"   end="+str(sliceEnd*4)
      sobelSlice=sobelAbs2[...,sliceStart:sliceEnd]
      sobelSlice2=sobelAbs2[...,sliceStart:sliceEnd]
      singleFrameImSlice=singleFrameIm[...,sliceStart:sliceEnd]

      sobelSlice=sobelSlice.mean(axis=1,keepdims=True)
      sobelSlice2=sobelSlice2.mean(axis=1,keepdims=True)
      singleFrameImSlice=singleFrameImSlice.mean(axis=1,keepdims=True)
      std_Slice=std_file[...,sliceStart:sliceEnd]
      # print std_Slice.mean()

      # std_Slice=std_Slice*std_Slice
      # std_Slice=std_Slice.sum(axis=1,keepdims=True)
      # std_Slice=np.sqrt(std_Slice)
      std_Slice=std_file[...,10:11]*np.sqrt(nSlicesUsed)/nSlicesUsed

      # print std_Slice.mean()
      # raw_input("pause")

      sobelSliceSigned=sobelSigned[...,sliceStart:sliceEnd]
      sobelSliceSigned=sobelSliceSigned.mean(axis=1,keepdims=True)

      #turn back to true 1D array
      sobelSlice=sobelSlice.ravel()
      sobelSlice2=sobelSlice2.ravel()
      sobelSliceSigned=sobelSliceSigned.ravel()
      std_Slice=std_Slice.ravel()
      singleFrameImSlice=singleFrameImSlice.ravel()

      xDat=np.arange(sobelSlice.size)

      #peak finder
      indices = find_peaks_cwt(sobelSlice, np.arange(50,80),  noise_perc=10) #np.arange(50,80) range of widths of peaks
      # print("indices",indices)


      if indices.size==0:
        print "continuing"
        continue
      # print "here"
      if (plotIt and leaf==10):
        h1=plt.subplot(2, 1, 1)
        plt.errorbar(xDat,sobelSlice,yerr=std_Slice, color='g')
        plt.errorbar(xDat,sobelSlice2,yerr=std_Slice, color='m')
        # h1=plt.errorbar(xDat,singleFrameImSlice,yerr=std_Slice, color='k')
        # plt.errorbar(xDat,singleFrameImSlice,yerr=std_Slice)
        plt.plot(indices, sobelSlice[indices],'r^')
        plt.pause(0.001)
        raw_input(leaf)

      for ind in range(0,indices.size):
        mean=indices[ind]
        pos_or_neg=sobelSliceSigned[int(np.floor(mean))]

        rangePoN=5
        if (mean>(1024-rangePoN)or mean<rangePoN):
          pos_or_neg=0
        else:
          for pon in range(0,5):
            pos_or_neg+=sobelSliceSigned[int(np.floor(mean))+pon]+sobelSliceSigned[int(np.floor(mean))-pon]

        if (pos_or_neg>0): #left leaf
          lowerWindow=28
          upperWindow=20
        else:
          lowerWindow=14
          upperWindow=28
        if (mean-lowerWindow<0):
          lower=0
        else:
          lower=mean-lowerWindow
        if (mean+upperWindow>1024):
          upper=1024
        else:
          upper=mean+upperWindow

        xDatSub=xDat[lower:upper]
        sobelSliceSub=sobelSlice[lower:upper]
        sigmaParam=5
        windowSize=len(xDatSub)
        sigmaArray=np.ones(windowSize)*15
        try :
          popt,pcov = curve_fit(gaus,xDatSub,sobelSliceSub,p0=[700,mean,sigmaParam],sigma=sigmaArray)
        except:
          popt=[0,0,0]
          pcov=0
        # print "index="+str(ind)
        # print "mean="+str(mean)
        # print ('popt',popt)

        if(abs(popt[2])<70 and abs(popt[2])>10 and abs(popt[0])>200): #
          # print ("popt",popt[1])
          if(plotIt and leaf==10  ):
            plt.plot(xDatSub,gaus(xDatSub,*popt),'bo:',label='fit')
            plt.pause(0.001)
            raw_input(str(frame)+" fit")


            # print sobelSliceSigned[int(np.floor(popt[1]))]
            # print sobelSliceSigned[int(np.floor(popt[1]))+1]
            # print sobelSliceSigned[int(np.floor(popt[1]))-1]
          pos_or_neg=sobelSliceSigned[int(np.floor(popt[1]))]
          for pon in range(0,10):
            pos_or_neg+=sobelSliceSigned[int(np.floor(popt[1]))+pon]+sobelSliceSigned[int(np.floor(popt[1]))-pon]
          
          # print "frame"+str(frame)+" pos "+str(0.0145*4*popt[1]*distanceCalibration/10)
          if (pos_or_neg>0): #for z plan leaves
            # if leaf==1:
              # print popt[1]
              # print 0.0145*4*popt[1]
            # leaf_pos_dict[leaf]=np.append(leaf_pos_dict[leaf],0.0145*4*popt[1]*distanceCalibration/10.0) #*0.0145 for mm
            # leaf_pos_error_dict[leaf]=np.append(leaf_pos_error_dict[leaf],0.0145*4*pcov[1,1]*distanceCalibration/10)
            # leaf_n_points_dict[leaf]=np.append(leaf_n_points_dict[leaf],frame*0.07692) #s *0.023 for seconds
            leaf_pos_dict[leaf]=np.append(leaf_pos_dict[leaf],(0.0145*4*popt[1]*distanceCalibration/10.0)) #*0.0145 for mm

            leaf_pos_error_dict[leaf]=np.append(leaf_pos_error_dict[leaf],0.0145*4*pcov[1,1]*distanceCalibration/10)
            if (field==1):
              leaf_n_points_dict[leaf]=np.append(leaf_n_points_dict[leaf],(frame*0.07692)+0.12) #s *0.023 for seconds
            else:
              leaf_n_points_dict[leaf]=np.append(leaf_n_points_dict[leaf],frame*0.07692) #s *0.023 for seconds

     
            # print ("frame:",frame,"leaf:",leaf,"error on mean",pcov[1,1])
    # h1.cla()
  leaf_of_choice=9
  label_str=str(Fields[field])+"MU"    
  # print leaf_pos_dict[1]    
  # ax.plot(leaf_n_points_dict[leaf_of_choice], leaf_pos_dict[leaf_of_choice],  marker='.', color=next(colors),label=label_str,linestyle="None")
  ax.errorbar(leaf_n_points_dict[leaf_of_choice], leaf_pos_dict[leaf_of_choice], marker='.', color=colorChoice,label=label_str,linestyle="None",yerr=leaf_pos_error_dict[leaf_of_choice])
  ax.legend(loc='lower right',numpoints=1)
  plt.xlabel("Time (s)")
  plt.ylabel("Position (cm)")
  plt.pause(0.01)
  # plt.savefig("LeafTrajectoriesNo9WithFitLines.png")
  # raw_input("continue")

  plotSpeeds=False
  plotTrajectories=True
  speeds_array=np.array([])
  leaves_array = np.array([])
  speeds_array_sigma=np.array([])

  if plotTrajectories:

    for leaf in range(0,nLeaves):

      grad=1
      cept=0

      try :
        popt2,pcov2 = curve_fit(line,leaf_n_points_dict[leaf],leaf_pos_dict[leaf],sigma=leaf_pos_error_dict[leaf],p0=[grad, cept])
      except:
        popt2=[0,0]
        pcov2=[[0,0],[0,0]]

      # leaf_gradients_dict[leaf]=popt2[0]
      leaves_array=np.append(leaves_array,leaf+1)
      speeds_array=np.append(speeds_array,popt2[0])
      speeds_array_sigma=np.append(speeds_array_sigma,pcov2[0,0])
      if Fields[field]==10:
        xDat2=np.arange(frames+2)
      elif Fields[field]==40:
        xDat2=np.arange(math.ceil(frames*1.43))
      else:
        xDat2=np.arange(frames)

      # xDat2=np.arange(int(leaf_n_points_dict[nLeaves-1])+1)
      xDat2=xDat2*0.07692

      if leaf==leaf_of_choice:
        fig1=plt.figure(1)
        plot1=plt.plot(xDat2,line(xDat2,*popt2),color=colorChoice)#,label=label_str+'fit'
        plt.pause(0.01)
        # plt.plot(leaf_n_points_dict[leaf],leaf_pos_dict[leaf],'go')

    speeds_array_STD=np.sqrt(speeds_array_sigma)

    # print leaves_array.size
    # print speeds_array.size
    # print speeds_array_STD.size

    # plt.plot(leaves_array,speeds_array,'go')
    # plt.figure(1)
    # plt.errorbar(leaves_array, speeds_array,fmt='b.',yerr=speeds_array_STD)
    # plt.xlabel("leaf number")
    # plt.ylabel("leaf speed (mm$s^-1$)")
    # plt.pause(0.001)  
    # plt.savefig(str(Fields[field])+"MU_leafVelocities.png")
    # plt.close()
    fig2=plt.figure(2)
    er2=ax2.errorbar(leaves_array, speeds_array,fmt='b.',yerr=speeds_array_STD,label=label_str,color=colorChoice)
    ax2.legend(loc='upper left',numpoints=1)
    ax2.set_xlim(0,18.9)
    ax2.set_ylim(0,4.1)


    plt.xlabel("Leaf number")
    plt.ylabel("Leaf speed (cms$^{-1}$)")
    plt.pause(0.01)
    avSpeedLineBank=(speeds_array[0]+speeds_array[1]+speeds_array[2]+speeds_array[15]+speeds_array[16]+speeds_array[17])/6
    speeds_array=speeds_array/avSpeedLineBank
    speeds_array_STD=speeds_array_STD/avSpeedLineBank

    fig3=plt.figure(3)
    er2=ax3.errorbar(leaves_array, speeds_array,fmt='b.',yerr=speeds_array_STD,label=label_str,color=colorChoice)
    ax3.legend(loc='upper left',numpoints=1)
    ax3.set_xlim(0,18.9)
    ax3.set_ylim(0,1.8)

    plt.ylabel("Normalised leaf speed (cms$^{-1}$)")
    plt.pause(0.01)
    # raw_input('hi')
# plt.figure(2)
fig3.savefig("All_leafVelocities_norm.png")
fig2.savefig("All_leafVelocities.png")
fig1.savefig("LeafTrajectoriesNo9WithFitLinesAfter.png")




# # singleFrameIm.tofile("InputIm4096.raw", sep="")
# # outputFilt1.tofile("Filter1.raw", sep="")
# # outputFilt2.tofile("Filter2.raw", sep="")
# # # sobel.tofile("FilterSobel1.raw", sep="")
# # # sobel2.tofile("FilterSobel2.raw", sep="")
# # sobelAbs.tofile("FilterSobelAbs.raw", sep="")

raw_input("Press enter to continue")

