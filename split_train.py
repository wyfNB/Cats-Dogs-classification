import os
import random
import shutil

def copyFile(fileDir,tarDir):
    pathDir = os.listdir(fileDir)
    sample = random.sample(pathDir,2400)
    #print(sample)
    for i in range(2000):
        name = sample[i]
        train_path = os.path.join(tarDir,'train')
        if not os.path.exists(train_path):
            os.makedirs(train_path)
        shutil.copy(os.path.join(fileDir,name),train_path)

    for i in range(2000,len(sample)):
        name = sample[i]
        eval_path = os.path.join(tarDir,'eval')
        if not os.path.exists(eval_path):
            os.makedirs(eval_path)
        shutil.copy(os.path.join(fileDir,name),eval_path)



if __name__ == '__main__':
    oldpath = 'C:/Users/test/Desktop/dogs-vs-cats-redux-kernels-edition/train'
    newpath = 'C:/Users/test/Desktop/dogs_cats'
    copyFile(oldpath,newpath)

