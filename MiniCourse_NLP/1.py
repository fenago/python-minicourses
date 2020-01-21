filename = 'sample.txt'
file = open(filename, 'rt')
text = file.read()
file.close()
# split into words by white space
words = text.split()
# convert to lowercase
words = [word.lower() for word in words]