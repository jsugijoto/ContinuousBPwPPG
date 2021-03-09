import glob
import os

def main():
    patient_list = []
    data_dir = r'.\physionet.org\static\published-projects\mimic3wdb-matched\1.0\*'

    for large_group in glob.iglob(data_dir, recursive=True):
        data_dir = large_group + "\\"
        for patient_group in glob.iglob(data_dir, recursive=True): 
            data_dir = patient_group + "\\*"
            for patients in glob.iglob(data_dir, recursive=True):
                data_dir = patients + "\\*"
                for patients1 in glob.iglob(data_dir, recursive=True):
                    if(".hea" in patients1):
                        patient_list = headerCheck(patients1, patient_list)
                        
    createPatientList(patient_list)

def headerCheck(filename, id_list):
    f = open(filename, 'r')
    first, patient_recorded, pleth, abp = True, False, False, False
    for x in f:
        phrase = x.split(' ')
        if(first):
            patient_id = phrase[0]
            if(int(phrase[3]) < 65000):
                break
        first = False
        for x in f:
            phrase2 = x.split(' ')
            for y in phrase2:
                if 'PLETH' in y:
                    pleth = True
                if 'ABP' in y:
                    abp = True
    if(pleth and abp and not patient_recorded):
        id_list.append(patient_id)
        patient_recorded = True
    return id_list

def createPatientList(patient_list):
    f = open("patient_data.txt", "w+")
    for i in range(len(patient_list)):
        f.write(patient_list[i]+".dat\n")
    f.close()

if __name__ == '__main__':
    main()