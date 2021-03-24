import re

sentence = "Hallo i bims 1 satz"

print(sentence)

print(re.search("bims", sentence))

print(re.findall("bims", sentence))