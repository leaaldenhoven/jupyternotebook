# -----------------------------------------------------------------

#### 1. Import modules

# -----------------------------------------------------------------

import numpy as np
import pandas as pd
from os import listdir
from os.path import join
import mne
import ast
import pickle
import csv

#### to prevent console from being spammed with debug messages:
import logging
logging.getLogger().setLevel(logging.WARNING);


# -----------------------------------------------------------------

#### 2. Set file paths and pre-sets for filtering and resampling

# -----------------------------------------------------------------

programmer = 'Lea'
if programmer == 'Kobus':
    the_folder = 'C:\\internship2\\'
if programmer == 'Katerina':
    the_folder = 'C:\\Users\\katerina.kandylaki\\Documents\\NERHYMUS\\Experiment\\Data\\EEG'
if programmer == 'Lea':
    the_folder = "C:\\Users\\leano\\OneDrive\\Documents\\Master RM CCN NP\\Year 1\\Research Elective NERHYMUS\\EEG"

condition = "EVC" # The options are EVC, SMA, sham 
subject = "M19" # 
# P02, P06, P09, P10, P11a, P14a, P18a, P19a P20a, P27
fullpath = join(the_folder,condition,subject)

hdrfile = [f for f in listdir(fullpath) if f.endswith('.vhdr')][0]

#INSERT LOWPASS VALUE
lp = 30
lb = lp/4

#INSERT HIGHPASS VALUE #1/10th of hp but never lower than 1!
hp = 1
hb = 1

#INSERT RESAMPLE FREQUENCY
sample_freq = 125
sfreq=sample_freq
#ICA SPECS
ICA_NAME_SPECS = '_SELECT_1-30Hz_R'+str(sample_freq)+'_with_interpolation-ica.fif'


# -----------------------------------------------------------------

#### 3. Import data & check up plotting

# -----------------------------------------------------------------

####plotting the easycap montage just for checking
mon = mne.channels.read_montage(kind='easycap-M1', ch_names=None, path=None, unit='m', transform=False)
mne.viz.plot_montage(mon)

# other montage options
# mne.channels.get_builtin_montages()

# Import data and mark eeg channels from others
raw = mne.io.read_raw_brainvision(join(fullpath, hdrfile), preload=True, misc=('Stim',), eog=('64','65'))
picks = mne.pick_types(raw.info, meg=False, eeg=True, misc=False, stim=False)
sound_ch = mne.pick_types(raw.info, meg=False, eeg=False, stim=False, misc=True)
eog = mne.pick_types(raw.info, meg=False, eeg=False, eog=True, misc=False, stim=False)

#Plot to check the messy data
raw.plot()

raw.plot_psd()

####to drop sound channel --> will make basic raw.plot() clearer!
#raw = raw.drop_channels(ch_names=['Stim'])

####plot raw data in a simplistic way
#raw.plot(butterfly=True, group_by='position')

####plot spectrum power with all channels
#raw_plot = raw.plot_psd(area_mode='range', tmax=2200.0, picks=picks, average=False)


# -----------------------------------------------------------------

#### 4. Discard bad channels - check if reference channels are bad

# -----------------------------------------------------------------


####reading in bads
bads_file = open(join(fullpath, 'bads.txt'),'r')
bads = bads_file.readline()
bads_file.close()
bads = ast.literal_eval(bads)
raw.info['bads']=bads

#Loop to check whether Fp1 and / or Fp2 are bads, they are used in eyeblink checking next to the eog!
Fp1_bad = 0
Fp2_bad = 0
eog_bad = 0
TP1_bad = 0
TP2_bad = 0
for x in range(0,len(raw.info['bads'])):
    if raw.info['bads'][x] == 'Fp1':
        Fp1_bad = 1
    if raw.info['bads'][x] == 'Fp2':
        Fp2_bad = 1
    if raw.info['bads'][x] == '63':
        eog_bad = 1
    if raw.info['bads'][x] == '64':
        eog_bad = 1 
    if raw.info['bads'][x] == 'TP9':
        TP1_bad = 1 
    if raw.info['bads'][x] == 'TP10':
        TP2_bad = 1 

if TP1_bad == 0 and TP2_bad == 0:
    REFs = ['TP9','TP10']
if TP1_bad == 0 and TP2_bad == 1:
    REFs = ['TP9']
if TP1_bad == 1 and TP2_bad == 0:
    REFs = ['TP10']
if TP1_bad == 1 and TP2_bad == 1:
    REFs = [None]
    print('NO GOOD REFERENCE! BOTH REFERENCES ARE BAD! USELESS DATASET! *In doubt check TP9 and TP10 quality*')


# -----------------------------------------------------------------

#### 5. Resample data
    
# -----------------------------------------------------------------
#raw = raw.copy().resample(250, npad='auto')
    
raw.resample(sfreq=sample_freq)


# -----------------------------------------------------------------

#### 6. Filter data
    
# -----------------------------------------------------------------

print("Filtering data between 1 and 30 Hz")
low_pass = lp #y
high_pass = hp #x
raw = raw.filter(high_pass, low_pass, picks=picks, n_jobs=-1, fir_design='firwin')

#Plot to check the less-messy data
raw.plot()

raw.plot_psd()


# -----------------------------------------------------------------

#### 7. Crop and concatenate data to include only listening 
    
# -----------------------------------------------------------------

if condition == "SMA" and subject == "P02":
    raw.annotations.delete(1)
    raw.annotations.delete(1)

if condition == "EVC" and subject == "M16":
    raw.annotations.delete(46)

ant = raw.annotations 
timestamps = ant.onset
response = ant.description
responseval = [0]*len(response)
for xc in range(0,len(response)):
    if '4' in response[xc]: 
        responseval[xc] = 4
    elif '8' in response[xc]:
        responseval[xc] = 8
    else:
        responseval[xc] = 0
        
startcounter = 0
for x in range(0,len(responseval)):
    if responseval[x] == 4:
        startcounter +=1
blocks = [0]*startcounter

errorprooftime = [0]*(len(blocks)*2)
xyy = 0
for xy in range (0, len(responseval)):
    if responseval[xy] == 4:
        errorprooftime[xyy] = xy
        xyy +=1
    elif responseval[xy] == 8:
        errorprooftime[xyy] = xy
        xyy +=1
   
Durationblock=[None]*len(blocks)
y2 = 0
for y in range(0,len(blocks)):
    fetch1 = errorprooftime[y2]
    fetch2 = errorprooftime[y2+1]
    Durationblock[y] = timestamps[fetch2] - timestamps[fetch1]
    y2 +=2    

insertedFTTlabel = 'theobeat'

pickle_in = open(join(the_folder +'/details/'+str(sfreq)+'', insertedFTTlabel +"_RZd.pickle"),"rb")
insertedFTT = pickle.load(pickle_in)
pickle_in.close()

#import labels of audio parts played and the order numbers of the audio parts.
pickle_in = open(join(the_folder +'/details/',"labelsandstampsR.pickle"),"rb")
labelsandstampsR = pickle.load(pickle_in)
pickle_in.close()

pickle_in = open(join(the_folder +'/details/',"OrderOfAudio.pickle"),"rb")
OOA = pickle.load(pickle_in)
OrderOfAudio = OOA
pickle_in.close()
# This part loops through the list of condition to check which session the current condition was used in
countnumbercondition = 0

# to find the list used in the current session of the participant!!    
file_name_lists = the_folder + '/details/Lists_for_EEG.xlsx' #has the list codes per session.
xl_file1 = pd.ExcelFile(file_name_lists)
dfs_lists = pd.read_excel(file_name_lists, sheet_name=None)
listchecker = dfs_lists['Sheet1']
participant_session_listinfo =   np.array([[None,None,None,None]]*100)
for x in range(0, len(listchecker)):
    for y in range(0,4):
        participant_session_listinfo[x][y] = listchecker.iat[x,y]

# to find TMS condition used in the session
file_name_TMS = the_folder +'/details/TMS_condition.xlsx' #has the TMS conditions per session
xl_file2 = pd.ExcelFile(file_name_lists)
dfs_lists2 = pd.read_excel(file_name_TMS, sheet_name=None)
listchecker2 = dfs_lists2['Sheet1']
participant_session_tmsinfo =   np.array([[None,None,None,None]]*100)
for x in range(0, len(listchecker2)):
    for y in range(0,4):
        participant_session_tmsinfo[x][y] = listchecker2.iat[x,y]

conditioncode = condition
for xas1 in range (0,len(listchecker2)):
    if participant_session_tmsinfo[xas1][0] == subject:
        for xas2 in range (0,4):
            if participant_session_tmsinfo[xas1][xas2] == conditioncode:
                conditionnumber = countnumbercondition
            if participant_session_tmsinfo[xas1][xas2] != conditioncode:
                countnumbercondition +=1

# This part loops through the list of list versions (eg. B1, C5 etc.) to see which list was used in this session number.
for xas3 in range (0,len(listchecker)):
    if participant_session_listinfo[xas3][0] == subject:
        participantlist = participant_session_listinfo[xas3][conditionnumber]
            
listdetail = participantlist
listnumber = int(listdetail[1])
listversion = listdetail[0]


with open(the_folder +'/details/Lists_' + listversion + '_pseudorandomised.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=';')
    labels = []
    colors = []
    for row in readCSV:
        label = row[listnumber-1]
        labels.append(label)
        
    print(labels)
    

FTTbasedstartandend = [0]*(2*len(labels))
yc=0
for x in range (0, len(labels)):
    
    FTTbasedstartandend [yc] = timestamps[errorprooftime[yc]]
    FTTbasedstartandend [yc+1] = timestamps[errorprooftime[yc]] + labelsandstampsR[labels[x]]
    yc +=2

getnumber = [0]*len(Durationblock)
for x in range(0, len(Durationblock)):
    for y in range(0, 40):
        if labels[x] == OrderOfAudio[y]:
            getnumber[x] = y    
            

insertedFTT_len = []
FTT5_ws_poems = []
FTT5_ws_stories = []          
fillcheck_stories = 0
fillcheck_poems = 0
xcount1 = 0            
rawpartcount = 0
for x in range(0, len(Durationblock)):
    raw_part = raw.copy().crop(FTTbasedstartandend[xcount1], (FTTbasedstartandend[xcount1] + ((len(insertedFTT[getnumber[x]])/sfreq))))

    if rawpartcount == 0:
        insertedFTT_len = list(insertedFTT[getnumber[x]])
        raw_extendhere = raw_part
        rawpartcount += 1
        difflen = len(insertedFTT_len) - len(raw_extendhere)
        while len(raw_extendhere) > len(insertedFTT_len):
            insertedFTT_len.append(sum(insertedFTT[getnumber[x]])/len(insertedFTT[getnumber[x]]))
    if rawpartcount == 1:
        insertedFTT_len.extend(list(insertedFTT[getnumber[x]]))
        raw_extendhere.append([raw_part])
        difflen = len(insertedFTT_len) - len(raw_extendhere)
        while len(raw_extendhere) > len(insertedFTT_len):
            insertedFTT_len.append(sum(insertedFTT[getnumber[x]])/len(insertedFTT[getnumber[x]]))
    xcount1 +=2    
    
    
raw = raw_extendhere   


# -----------------------------------------------------------------

#### 8. Add bad channel names to bads and optionally interpolate
    
# -----------------------------------------------------------------


####writes the bads to a file in the participant's session folder.
thefilez= open(join(fullpath, 'bads.txt'),'w')
thefilez.write(str(raw.info['bads']))
thefilez.close()

####drops the bad channels (only do this for preprocessing ICA!)
#drop_bads = raw.info['bads']
#raw = raw.drop_channels(ch_names=drop_bads)

#### Interpolating bad channels
print("Interpolating bad channels")
raw.interpolate_bads()


# -----------------------------------------------------------------

#### 9. Re-reference to offline
    
# -----------------------------------------------------------------

#### Re-referencing
print("Re-referencing")
#REFs = ['TP9','TP10'] # Refs: FCz (cortical)--> using a cortical reference is suboptimal! 
#--> use the 2 mastoids listed next. Putting both as reference will use their mean as ref., 
#TP9(left mastoid), TP10(right mastoid).  
raw, ref = mne.io.set_eeg_reference(raw, ref_channels=REFs)
#raw.apply_proj()


# -----------------------------------------------------------------

#### 10. Independent Component Analysis (ICA) decomposition
    
# -----------------------------------------------------------------

####ica as by Katerina
#### Run/Load ICA
ica = mne.preprocessing.ICA(n_pca_components=50, max_pca_components=40, n_components=40, random_state=23)
#ica = mne.preprocessing.ICA(n_pca_components=50, n_components=40, random_state=23)
ica.fit(raw)

####Save ICA:
ica.save(join(fullpath, subject + ICA_NAME_SPECS))


# UCOMMENT THIS LINE IF YOU NEED TO LOAD A PREVIOUSLY SAVED ICA
####load ICA: 
#ica = mne.preprocessing.read_ica(join(fullpath, subject + ICA_NAME_SPECS))


####ways to plot ICA: Components displays all the single components and their topo. 
####Properties can be used to zoom in on a single ICA
####Overlay creates an image of data when ICA is substracted and with original raw to see difference
####Sources plots the ICA components over time"""
ica.plot_components()
ica.plot_properties(raw, picks=1)
ica.plot_overlay(raw)
ica.plot_sources(raw)

ica.info["ch_names"]
#### to loop over multiple picks (spares time if you know what you want to look at)
selectpicks = [0, 1] # write component numbers in the [] separated by comma
for x in range(0,len(selectpicks)):
    ica.plot_properties(raw, selectpicks[x])
    
#####for a file with all
#bad_ics_file = open(join(fullpath, 'bad_ics_with_interpolation.txt'),'w')
#bad_ics_file.writelines(["%s\n" % item for item in list(eog_comps)])
#bad_ics_file.close()
    
# -----------------------------------------------------------------

#### 11. Identify EOG components and save externally
    
# -----------------------------------------------------------------


eog_comps = []
###finding ICA's related to eyeblinks through "synthesized" eog (looking at Fp1 and Fp2)
####which ICA correlate with the eog? (Fp1)
if Fp1_bad == 0:
    eog_comps1 = ica.find_bads_eog(raw, ch_name='Fp1')
    thefile= open(join(fullpath, 'eog_comps_basedonFp1_with_interpolation.txt'),'w')
    thefile.writelines(["%s\n" % item for item in list(eog_comps1[0])])
    thefile.close()
    eog_comps = eog_comps1[0]
    
####which ICA correlate with the eog? (Fp2)
if Fp2_bad == 0:
    eog_comps2 = ica.find_bads_eog(raw, ch_name='Fp2')
    thefile2= open(join(fullpath, 'eog_comps_basedonFp2_with_interpolation.txt'),'w')
    thefile2.writelines(["%s\n" % item for item in list(eog_comps2[0])])
    thefile2.close()
    eog_comps += eog_comps2[0]

####which ICA correlate with the eog? EOG channel if used (XX)
if eog_bad == 0:
    eog_comps3 = ica.find_bads_eog(raw)
    thefile3= open(join(fullpath, 'eog_comps_basedonREALEOG_with_interpolation.txt'),'w')
    thefile3.writelines(["%s\n" % item for item in list(eog_comps3[0])])
    thefile3.close()
    eog_comps += eog_comps3[0]
####create a file with all the parts.

####for a file with all
thefile5= open(join(fullpath, 'eog_comps_with_interpolation.txt'),'w')
thefile5.writelines(["%s\n" % item for item in list(eog_comps)])
thefile5.close()

print(eog_comps)

