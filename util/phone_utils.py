import numpy as np

def read_sensorLog(datafile: str):
    # lists for sensor data 
    gyr = []
    acc = []
    mag = []

    # read the data file line by line
    with open(datafile) as f:
        lines = f.readlines()

    # process the data
    for line in lines:
        tokens = line.split()
        match tokens[1]:
            case 'GYR':
                gyr.append(np.array([[float(tokens[2])], [float(tokens[3])], [float(tokens[4])]]))
            case 'ACC':
                a = np.array([[float(tokens[2])], [float(tokens[3])], [float(tokens[4])]])
                acc.append(a/np.linalg.norm(a))
            case 'MAG':
                m = np.array([[float(tokens[2])], [float(tokens[3])], [float(tokens[4])]])
                mag.append(m/np.linalg.norm(m))

    # harmonize list length
    N = min(len(gyr), len(acc), len(mag))
    del gyr[N:]
    del acc[N:]
    del mag[N:]

    return gyr, acc, mag