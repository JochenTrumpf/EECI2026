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

def read_sensorLog_iPhone(dataPath: str):

    # lists for sensor data 
    gyr = []
    acc = []
    mag = []

    # data is read from separate files per sensor
    acc_datafile = f"{dataPath}/Accelerometer.csv"
    gyr_datafile = f"{dataPath}/Gyroscope.csv"
    mag_datafile = f"{dataPath}/Magnetometer.csv"

    files = {
        'GYR': gyr_datafile,
        'ACC': acc_datafile,
        'MAG': mag_datafile
    }

    # read the data file line by line
    for key in files.keys():
        datafile = files[key]
        with open(datafile) as f:
            lines = f.readlines()[1:]

        # process the data
        for line in lines:
            tokens = line.split(",")

            match key:
                case 'GYR':
                    gyr.append(np.array([[float(tokens[1])], [float(tokens[2])], [float(tokens[3])]]))
                case 'ACC':
                    a = np.array([[float(tokens[1])], [float(tokens[2])], [float(tokens[3])]])
                    acc.append(a/np.linalg.norm(a))
                case 'MAG':
                    m = np.array([[float(tokens[1])], [float(tokens[2])], [float(tokens[3])]])
                    mag.append(m/np.linalg.norm(m))

    # harmonize list length
    N = min(len(gyr), len(acc), len(mag))
    del gyr[N:]
    del acc[N:]
    del mag[N:]

    return gyr, acc, mag
