loremtext = open("lorem.txt").read()

words = loremtext.split()
word_counts = dict()

for word in words:
    lower_word = word.lower()
    word_counts[lower_word] = word_counts.get(lower_word,0) + 1




value_indexed_word_counts = list()

for k,v in word_counts.items():
    value_indexed_word_counts.append((v,k))

# this works because only the first touple item is compared
sorted_word_counts = sorted(value_indexed_word_counts, reverse=True)

print("Top 10 words",dict(sorted_word_counts[:10]))



comprehended = dict(sorted([ (v,k) for k,v in word_counts.items()], reverse=True)[:10])

print("List comprehension top words", comprehended)
