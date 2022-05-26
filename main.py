import os
import operator
import numpy as np
from PIL import Image

def load_matrix(name):
    mat = np.load(os.path.join("./matrix", name+'.npy'))
    return mat


def convert_pgm_P5(f):
    magic_number = f.readline().strip().decode('utf-8')  # P5
    if not operator.eq(magic_number, "P5"):
        raise Exception("Error with magic number.")
    width, height = f.readline().strip().decode('utf-8').split(' ')  # 长宽
    width = int(width)
    height = int(height)
    maxval = f.readline().strip()  # 最大字节数
    if int(maxval) < 256:
        pad = 1
    else:
        pad = 2
    img = np.zeros((height, width))
    img[:, :] = [[ord(f.read(pad)) for j in range(width)]
                 for i in range(height)]
    return img


def process_data(rootDir):
    mat = None
    cnt = 0
    for _, dirs, _ in os.walk(rootDir):
        for dir in dirs:
            for _, _, files in os.walk(os.path.join(rootDir, dir)):
                for file in files:
                    if os.path.splitext(file)[1] == '.pgm':
                        f = open(os.path.join(rootDir, dir, file), 'rb')
                        img = convert_pgm_P5(f)
                        img = img.reshape(1, -1)
                        if cnt == 0:
                            mat = img
                        else:
                            mat = np.concatenate((mat, img), axis=0)
                        cnt += 1
    return [mat, cnt]


def PCA(mat, k):
    mean = np.mean(mat, axis=0)
    np.save('./matrix/mean.npy', mean)
    # meanMat = mean.repeat(col, 1)
    # stdMat = mat - meanMat
    stdMat = mat - mean
    np.save('./matrix/stdMat.npy', stdMat)
    covMat = np.cov(stdMat)
    feaVal, feaVec = np.linalg.eig(covMat)
    np.save('./matrix/feaVal.npy', feaVal)
    np.save('./matrix/feaVec.npy', feaVec)
    index = np.argsort(feaVal)
    sortedFeaVec = feaVec[:, index[:-k-1:-1]]
    eigenface = np.dot(stdMat.T, sortedFeaVec)
    # os.system("pause")
    np.save('./matrix/eigenface.npy', eigenface)
    trainSample = np.dot(stdMat, eigenface)
    np.save('./matrix/trainSample.npy', trainSample)


def load_pic(f):
    img = convert_pgm_P5(f)
    img = img.reshape(1, -1)
    meanMat = load_matrix('mean')
    normMat = img - meanMat
    eigenface = load_matrix('eigenface')
    testSample = np.dot(normMat, eigenface)
    return testSample
    # trainSample = load_matrix('trainSample')
    # minDis = INFINITE
    # ans = 0
    # for i in range(trainSample.shape[0]):
    #     dis = np.linalg.norm(trainSample[i,:] - testSample)
    #     # print("distance isvbcgf %f"%dis)
    #     if dis < minDis:
    #         minDis = dis
    #         ans = i//10+1
    # print("This is person s{}".format(ans))


def batch_test(rootDir):
    # meta = {}
    testMat = None
    cnt = 0
    for _, dirs, _ in os.walk(rootDir):
        for dir in dirs:
            for _, _, files in os.walk(os.path.join(rootDir, dir)):
                for file in files:
                    if os.path.splitext(file)[1] == '.pgm':
                        num = os.path.splitext(file)[0]
                        # print("Start testing pic{} of {}".format(num, dir))
                        f = open(os.path.join(rootDir, dir, file), 'rb')
                        testSample = load_pic(f)
                        # name = dir+'_'+str(num)
                        # meta[name] = testSample
                        if cnt == 0:
                            testMat = testSample
                        else:
                            testMat = np.concatenate(
                                (testMat, testSample), axis=0)
                        cnt += 1
    scale = testMat.shape[0]
    farCnt = 0
    frrCnt = 0
    for i in range(scale):
        mark = []
        for j in range(scale):
            if i == j:
                continue
            dis = np.linalg.norm(testMat[i] - testMat[j])
            mark.append(dis)
        # mark.sort()
        key = sorted(enumerate(mark), key=lambda x: x[1])
        thres = key[1][1]
        for k in range(scale-1):
            if key[k][1] < thres and key[k][0]//10 != i//10:
                farCnt += 1
            if key[k][1] > thres and key[k][0]//10 == i//10:
                frrCnt += 1
    totalNum1 = scale * 9
    totalNum2 = scale * (scale - 1) - totalNum1
    far = farCnt/totalNum1
    frr = frrCnt/totalNum2
    # print(str(totalNum1) + '    '+ str(totalNum2))
    # print("FAR: {:.2%},  FRR: {:.2%}".format(far,frr))
    return far,frr

def display_mean(mat):
    mat = np.reshape(mat,(112,92))
    map = Image.fromarray(mat)
    map.show()

if __name__ == '__main__':
    trainDir = './data/train'
    testDir = './data/test'
    k = 50

    trainMat, num = process_data(trainDir)
    PCA(trainMat, k)
    minfar,minfrr = batch_test(testDir)
    # best = 0
    # for i in range(10,200):
    #     PCA(trainMat, i)
    #     far,frr = batch_test(testDir)
    #     if far<minfar:
    #         minfar = far
    #         minfrr = frr
    #         best = i
    #     print("Iter{}: bestFAR->{:.2%},  bestFRR->{:.2%}".format(i,minfar,minfrr))
    # print(best)
    print("The dimensionality of PCA is: %d"%k)
    print("bestFAR: {:.2%},  bestFRR: {:.2%}".format(minfar,minfrr))
    meanMat = load_matrix('mean')
    display_mean(meanMat)
