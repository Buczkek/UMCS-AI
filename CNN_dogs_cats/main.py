import os, random, shutil
train_path = 'data\\train\\'
files = os.listdir(train_path)
cats = []
dogs = []

for name in files:
    if name in ('dog', 'cat'):
        continue
    if name.startswith("cat"):
        cats.append(name)
    else:
        dogs.append(name)


if 'dog' not in files:
    os.mkdir(train_path + 'dog')

for dogfile in dogs:
    shutil.move(train_path + dogfile, train_path + "dog\\"+dogfile.split('dog.')[1])


if 'cat' not in files:
    os.mkdir(train_path + 'cat')

for catfile in cats:
    shutil.move(train_path + catfile, train_path + "cat\\"+catfile.split('cat.')[1])

if 'valid' not in os.listdir('data\\'):
    os.mkdir()

