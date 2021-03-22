import  matplotlib.pyplot as plt, sklearn
import pandas as pd
from numpy import *
import numpy, matplotlib.pyplot as plt
import librosa.display
import IPython.display as ipd
import pyaudio
import wave

plt.rcParams['figure.figsize'] = (15,7)
plt.rcParams['figure.figsize'] = (13, 5)
seed = 1


def main():
    filename = 'ss.wav'
    start = 7
    length = 3

    # open wave file
    #wave_file = wave.open(filename, 'rb')

    # initialize audio
    #py_audio = pyaudio.PyAudio()
    #stream = py_audio.open(format=py_audio.get_format_from_width(wave_file.getsampwidth()),
                           #channels=wave_file.getnchannels(),
                           #rate=wave_file.getframerate(),
                           #output=True)

    # skip unwanted frames
    #n_frames = int(start * wave_file.getframerate())
    #wave_file.setpos(n_frames)

    # write desired frames to audio buffer
    #n_frames = int(length * wave_file.getframerate())
    #frames = wave_file.readframes(n_frames)

    #stream.write(frames)

    # close and terminate everything properly
    #stream.close()
    #py_audio.terminate()
    #wave_file.close()

    list = "all.txt"
    oursong = processFile(filename,list)
    hold1 = createModels(oursong, 3,4)              #centroid and contrast
    hold2= createModels(oursong, 5, 6)             #bandwidth adn rolloff
    hold3 = createModels(oursong, 7, 8)             #rsme and tempo

    similar(hold1, hold2, hold3)



def processFile(filename, list):

    y, sr = librosa.load(filename) #load in file
    ipd.Audio(y, rate = sr)

    #this cell with retrieve all the elements necessary to have a complete entry in the dataset

    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    #print('Estimated tempo: {:.2f} beats per minute'.format(tempo))

    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_avg = numpy.average(zcr)
    #print zcr_avg

    cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    cent_avg = numpy.average(cent)
    #print cent_avg

    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    mfcc_avg = numpy.average(mfcc)
    #print mfcc_avg

    S = numpy.abs(librosa.stft(y))
    contrast = librosa.feature.spectral_contrast(S=S, sr=sr)
    contrast_avg = numpy.average(contrast)
    #print contrast_avg

    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    rolloff_avg = numpy.average(rolloff)
    #print rolloff_avg

    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    bandwidth_avg = numpy.average(bandwidth)
    #print bandwidth_avg

    rmse = librosa.feature.rmse(y=y)
    rmse_avg  = numpy.average(rmse)
    print (rmse_avg)


    # gets the line number entered song will be added on
    countLine = 0
    with open("all.txt") as f:
        for i, l in enumerate(f):
            countLine = i

    countLine = countLine + 1
    print ('countLine: ', countLine)
    songName = filename.split(".")[0] # gets songname to add

    #creates the string that will be added to dataset
    add = str(countLine) + ',' + filename + ',' + str(mfcc_avg) + "," + str(cent_avg) +"," + str(contrast_avg) +"," +str(rolloff_avg) +"," +str(bandwidth_avg) + "," + str(tempo) + ", " + str(rmse_avg)

    checkValid = True
    oursong = countLine - 2 #max line count - 1 to get the data line, -1 for index of features scaled.

    #adds to dataset
    with open(list) as f:
        for i, l in enumerate(f):
            if l.split(",")[1] == filename:
                checkValid = False
                oursong = i - 1
                print (l)

    with open(list, "a") as myfile:
        if checkValid == True:
            myfile.write('\n' +add)

    myfile.close()
    return oursong

def createModels(oursong, f1, f2):
    #reads text file containing dataset
    df = pd.read_csv('all.txt', header = None, low_memory=False)
    x = numpy.asarray(df.loc[1:,f1:f2]) #x = numpy.asarray(df.loc[1:,3:6]) #gets the 1st 2 features

    y = numpy.asarray(df.loc[1:,2:2])#recommandatons based on mfcc
    labels = numpy.asarray(df.loc[:0,]) #label scem-sbwm

    print (labels)
    #print y
    #print x
    min_max_scaler = sklearn.preprocessing.MinMaxScaler(feature_range = (-1,1)) #scales dataset to make it easier to display on graph
    features_scaled = min_max_scaler.fit_transform(x)

    #print features_scaled
    ours = features_scaled[oursong]
    #print ours

    plt.scatter(features_scaled[:,0], features_scaled[:,1]) #1st plot without clustering
    if(f1 == 3 and f2 == 4):
        plt.xlabel('Spectral Centroid (scaled)')
        plt.ylabel('Spectral Constrast (scaled)')
    elif (f1==5 and f2 ==6 ):
        plt.xlabel('Sprectral Rolloff (scaled)')
        plt.ylabel('Spectral Bandwidth (scaled)')
    else:
        plt.xlabel('Tempo (scaled)')
        plt.ylabel('RMSE (scaled)')
    plt.show()

    model = sklearn.cluster.KMeans(n_clusters =10, random_state=seed) #create model with specifications
    labels = model.fit_predict(features_scaled) #fits dataset into model
    print (labels) #shows the cluster number each song belongs to

    #creates cluster color graph now
    plt.scatter(features_scaled[labels==0,0], features_scaled[labels==0,1], c='b')
    plt.scatter(features_scaled[labels==1,0], features_scaled[labels==1,1], c='r')
    plt.scatter(features_scaled[labels==2,0], features_scaled[labels==2,1], c='g')
    plt.scatter(features_scaled[labels==3,0], features_scaled[labels==3,1], c='y')
    plt.scatter(features_scaled[labels==4,0], features_scaled[labels==4,1], c='c')
    plt.scatter(features_scaled[labels==5,0], features_scaled[labels==5,1], c='m')
    plt.scatter(features_scaled[labels==6,0], features_scaled[labels==6,1], c='violet')
    plt.scatter(features_scaled[labels==7,0], features_scaled[labels==7,1], c='lightgreen')
    plt.scatter(features_scaled[labels==8,0], features_scaled[labels==8,1], c='cyan')
    plt.scatter(features_scaled[labels==9,0], features_scaled[labels==9,1], c='salmon')

    if (f1 == 3 and f2 == 4):
        plt.xlabel('Spectral Centroid (scaled)')
        plt.ylabel('Spectral Constrast (scaled)')
    elif (f1 == 5 and f2 == 6):
        plt.xlabel('Sprectral Rolloff (scaled)')
        plt.ylabel('Spectral Bandwidth (scaled)')
    else:
        plt.xlabel('Tempo (scaled)')
        plt.ylabel('RMSE (scaled)')
    plt.legend(('Cluster 0', 'Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4', 'Cluster 5', 'Cluster 6',
                'Cluster 7', 'Cluster 8', 'Cluster 9'))
    plt.show()

    plt.scatter(features_scaled[labels == 0, 0], features_scaled[labels == 0, 1], c='b')
    plt.scatter(features_scaled[labels == 1, 0], features_scaled[labels == 1, 1], c='r')  # first 2 elements in here
    plt.scatter(features_scaled[labels == 2, 0], features_scaled[labels == 2, 1], c='g')
    plt.scatter(features_scaled[labels == 3, 0], features_scaled[labels == 3, 1], c='y')
    plt.scatter(features_scaled[labels == 4, 0], features_scaled[labels == 4, 1], c='c')
    plt.scatter(features_scaled[labels == 5, 0], features_scaled[labels == 5, 1], c='m')
    plt.scatter(features_scaled[labels == 6, 0], features_scaled[labels == 6, 1], c='violet')
    plt.scatter(features_scaled[labels == 7, 0], features_scaled[labels == 7, 1], c='lightgreen')
    plt.scatter(features_scaled[labels == 8, 0], features_scaled[labels == 8, 1], c='cyan')
    plt.scatter(features_scaled[labels == 9, 0], features_scaled[labels == 9, 1], c='salmon')

    # gets the entered song's position on graph and plots it with different color
    for x in numpy.array(features_scaled):
        if (numpy.array_equal(x, ours) == True):
            plt.scatter(x[0], x[1], c='k')

    # marks centroids of each cluster
    centroids = model.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=169, linewidths=3,
                color='k', zorder=10)

    if (f1 == 3 and f2 == 4):
        plt.xlabel('Spectral Centroid (scaled)')
        plt.ylabel('Spectral Constrast (scaled)')
    elif (f1 == 5 and f2 == 6):
        plt.xlabel('Sprectral Rolloff (scaled)')
        plt.ylabel('Spectral Bandwidth (scaled)')
    else:
        plt.xlabel('Tempo (scaled)')
        plt.ylabel('RMSE (scaled)')

    plt.legend(('Cluster 0', 'Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4', 'Cluster 5', 'Cluster 6',
                'Cluster 7', 'Cluster 8', 'Cluster 9'))

    km_labels = model.labels_
    # print km_labels

    # gets the songs in the same cluster as the entered song
    index = [find[0] for find, value in numpy.ndenumerate(km_labels) if value == labels[len(labels) - 1]]
    l2 = [int(v) for v in index]
    # print l2

    print (l2)
    # print model.labels_
    iterHold = 0  # line number in lines
    npLocation = 0  # index in l2
    titles = []  # holds titles

    # returns the titles of the similar songs
    fileOpen = open('all.txt', "r")
    lines = fileOpen.readlines()[1:]

    for look in lines:
        if (l2[npLocation] == iterHold):
            splitting = look.split(",")
            titles.append(splitting[1].split(".")[0])
            npLocation = npLocation + 1
            iterHold = iterHold + 1
        else:
            iterHold = iterHold + 1
    titlesHold = titles;
    print (titles)
    plt.show()
    return titlesHold

#-----------------------------------------------------------------------------------------------------------------------

def similar(titlesHold1, titlesHold2, titlesHold3 ):
    # prints out similar song set from both executions
    print ('Spectral Centroid / Spectral Contrast')
    for i in set(titlesHold1):
        print (i)

    print ('\nSpectral Rolloff / Spectral Bandwidth')
    for j in set(titlesHold2):
        print (j)

    print ('\nTempo / RMSE')
    for j in set(titlesHold3):
        print (j)

        # gets similar songs to show user that despite using different values from same songs, can still get similar recommandations
        # for i in set(titlesHold1).intersection(set(titlesHold2).intersection(set(titlesHold3))):
    print()
    print('Similar songs between sectracl centroid / contrast and rolloff/ bandwidth')
    for i in set(titlesHold1).intersection(set(titlesHold2)):
        print (i)

main()