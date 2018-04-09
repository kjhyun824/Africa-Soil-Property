import os

idFilePath = '/home/students/cs20175052/SJ_Version/Africa_Soil_Property_Prediction/data/sample_submission.csv'
idFile = open(idFilePath,'r')
idData = idFile.read().splitlines()

category = idData[0]
ids = []
for line in idData:
    tokens = line.split(",")
    if tokens[0] == 'PIDN':
        continue
    else:
        ids.append(tokens[0])

Trains=[1]

for elem in Trains:
    outputFilePath = '/home/students/cs20175052/SJ_Version/Africa_Soil_Property_Prediction/result/output_logistic_' + str(elem) + '.csv'
    outputFile = open(outputFilePath,'r')
    outputData = outputFile.read().splitlines()
    
    mergeFilePath = '/home/students/cs20175052/SJ_Version/Africa_Soil_Property_Prediction/result/merge_logistic_' + str(elem) + '.csv'

    if os.path.isfile(mergeFilePath):
        os.remove(mergeFilePath)

    mergeFile = open(mergeFilePath,'w')
    mergeFile.write(category+'\n')
    for i in range(0,len(ids)):
        text = ids[i] + ',' + outputData[i] + '\n'
        mergeFile.write(text)

    outputFile.close()
    mergeFile.close()
