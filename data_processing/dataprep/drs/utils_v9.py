# ─────────────────────────────────────────────────────────────────────────────
# VENDORED — taken verbatim from drsProcessing for credit/provenance.
#   source : https://github.com/mcitron/drsProcessing
#   file   : utils_v9.py
#   commit : 0b1edca   (git@github.com:mcitron/drsProcessing.git)
#   author : Matthew Citron (mcitron) and drsProcessing contributors
#   vendored: 2026-06-14 into pmtdeconv-dataprep (DRS binary → ROOT step)
# All credit for this code belongs to the drsProcessing authors. No upstream
# LICENSE file was present at the retrieved commit. To update, re-copy from
# upstream (do not hand-edit this vendored copy). See dataprep/drs/NOTICE.md.
# ─────────────────────────────────────────────────────────────────────────────
import sys
import struct
import datetime as dt
import json
import csv
import glob
import argparse
import numpy as np
import ROOT as r
import time
#from scipy.signal import find_peaks
from multiprocessing import Pool
def getStr(fid, length):
    data = fid.read(length)
    if len(data) == 0:
        return None
    res = struct.unpack("c"*length, data)
    res = b"".join(res).decode("utf-8")
    return res


def getShort(fid, num=1):
    data = fid.read(2*num)
    if len(data) == 0:
        return None
    try:
        res = struct.unpack("H"*num, data)
    except:
        res = [0]*num
    return res[0] if num == 1 else res


def getFloat(fid, num=1):
    data = fid.read(4*num)
    if len(data) == 0:
        return None
    res = struct.unpack("f"*num, data)
    return res[0] if num == 1 else res


def getInt(fid, num=1):
    data = fid.read(4*num)
    if len(data) == 0:
        return None
    res = struct.unpack("I"*num, data)
    return res[0] if num == 1 else res


def parseCSV(csvpaths):
    # numpy append may cuase issue, convert at the end
    HV = []
    currs = []
    uts = []
    avgCurr = 0.0
    evts = 0
    for fname in csvpaths:
        print("CSV file: "+fname+" is processed.")
        with open(fname) as csvfile:
            reader = csv.reader(x.replace('\0', '') for x in csvfile)
            #reader = csv.reader(csvfile, delimiter=',')
            reader = list(reader)
            # line 0 is just headers
            # line 1 has starting voltage
            HV.append(-1.0*float(reader[1][0]))
            #currs.append(float(reader[1][1]))
            uts.append(float(reader[1][2]))
            avgCurr += float(reader[1][1])
            evts += 1
            for row in reader[2:]:
                if row:
                # row[0] - first column - voltages
                # row[2] - third column - times
                # when next row has different voltage from
                # previous row, save new voltage to HV
                # and time of voltage change to uts
                    if float(row[0]) != -1.0*HV[-1]:
                        HV.append(-1.0*float(row[0]))
                        uts.append(float(row[2]))
                        currs.append(-1.0*float(avgCurr)/float(evts))
                        avgCurr = 0.0
                        evts = 1
                    else:
                        avgCurr += float(row[1])
                        evts += 1
            currs.append(-1.0*float(avgCurr)/float(evts))
    HV, currs, uts = np.array(HV), np.array(currs), np.array(uts)
    return HV, currs, uts


def getIV(csvpaths, time, start_row):
    for fname in csvpaths:
        with open(fname) as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            reader = list(reader)
            for row in range(start_row,len(reader)):
                if time < float(reader[row][2]) and time > float(reader[row-1][2]):
                    return float(reader[row-1][1]), float(reader[row-1][0]), row
    print("ERROR: No I-V for event! t = ",time)


def postprocess(voltages, times):
    maxtest = max(voltages)
    mintest = min(voltages)
    if abs(maxtest) > abs(mintest):
    	vs = voltages
    else:
        vs = -voltages
#    vs = -voltages
    max_vs = max(vs)
    max_vs_early = max(vs[:300])
    min_vs_early = min(vs[:300])
    # max is just the highet point
    imax = np.argmax(vs)
    spike = 0
    # istart = np.argmax(times_CH1>tstart_1+times_CH1[0])
    # iend = np.argmax(times_CH1>tend_1+times_CH1[0])
    if(imax > 0):
    #if(imax > 100 and imax < 1000):
        # after peak
        offset = np.mean(vs[:int(imax*1/2)])
        vs -= offset
        vs_high = vs[imax:]
        # before peak
        vs_low = vs[:imax]
        # where [axis][index]
        noise = 0.5*(np.percentile(vs[:int(imax*1/2)],95) - np.percentile(vs[:int(imax*1/2)],5))
        try:
            iend = -1
            testindex=0
            while iend < 0:
                it = np.where(vs_high < noise)[0][testindex] + imax
                if(vs[it+1]<noise and vs[it+2]<noise):
                    iend = it
                else:
                    testindex=testindex+1
        except:
            iend = 1000
            #print("end at 1000")
        try:
            istart = -1
            testindex=-1
            while istart < 0:
                it = np.where(vs_low < noise)[0][testindex]
                if(vs[it-1]<noise and vs[it-2]<noise):
                    istart = it
                else:
                    testindex=testindex-1
        except:
            istart = 0
        #    #print("start at 0")
	#print("noise,start,end")
	#print(noise)
	#print(istart)
	#print(iend)
        vMax = max_vs
        vMaxEarly = max_vs_early
        vMinEarly = min_vs_early
        tMax = times[imax]
        tStart = times[istart]
        tEnd = times[iend]
        vs += offset
        offset = np.mean(vs[:int(istart*3/4)])
        vs -= offset
        width = iend-istart
        noise = 0.5*(np.percentile(vs[:int(istart*3/4)],95) - np.percentile(vs[:int(istart*3/4)],5))
        try:
            if(iend - istart >= 20):
                area = np.trapz(vs[istart:iend], times[istart:iend])
            else:
                area = np.trapz(vs[istart:iend], times[istart:iend])
                #area = -999
                spike = spike+1
        except:
            area = -999
        try:
            #vFix10 = vs[istart+10]
            vFix10 = vs[istart+20]
        except:
            vFix10 = -999
        try:
            #vFix20 = vs[istart+80]
            vFix20 = vs[istart+40]
        except:
            vFix20 = -999
        try:
           #vFix30 = vs[istart+100]
            vFix30 = vs[istart+60]
        except:
            vFix30 = -999
        vs += offset
        
        vMaxLate = 0
        tMaxLate = 0
        vMaxLate1 = 0
        tMaxLate1 = 0
        vMaxLate2 = 0
        tMaxLate2 = 0
        npulse = 0
#       use scipy.find_peaks if you're interested in finding multiple peaks (makes the code slightly slower)
############################################################
#       to use multiple peak finding, uncomment from here...
############################################################

        #late_offset = np.mean(vs[int(iend+50):])
        #vs -= late_offset
   
#	using vs_late, if you want to use this peak finding for all, just use 'vs' instead of 'vs[iend:]' and remove "iend" from the tMaxLate calcs below
        #vs_late = vs[iend:]
   
#	modify the args here to customize the type of afterpulse peak you're looking for
        #peaks = find_peaks(vs_late, prominence = 10, width = 20, height = -15, distance = 40)
   
#	get the values
        #height = peaks[1]['peak_heights']
        #prominence = peaks[1]['prominences']
        #widths = peaks[1]['widths']
        #npulse = len(peaks[0])

#	some print statements if you want
#        print(peaks[0])
#        print(height)
#        print(prominence)
#        print(widths)

#	get the parameters but only if the peaks exist        
        #if(len(peaks[0])>0):
        #    tMaxLate=times[peaks[0][0]+iend]
        #    vMaxLate=prominence[0]
             ##vMaxLate=height[0]
        #if(len(peaks[0])>1):
        #    tMaxLate1=times[peaks[0][1]+iend]
        #    vMaxLate1=prominence[1]
            ##vMaxLate1=height[1]
        #if(len(peaks[0])>2):
        #    tMaxLate2=times[peaks[0][2]+iend]
        #    vMaxLate2=prominence[2]
            ##vMaxLate2=height[2]
        #vs += late_offset
############################################################
#       ...to here      
############################################################
        return area, width, offset, noise, tMax, vMax, vMaxEarly, vMinEarly, vMaxLate, tMaxLate, vMaxLate1, tMaxLate1, vMaxLate2, tMaxLate2, npulse, vFix10, vFix20, vFix30, tStart, tEnd
