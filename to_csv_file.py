import csv

text_file = open("data/consys.txt", 'r', encoding='utf-8')
read_file = text_file.read()

read_file = read_file.split(".")
read_file = list(map(lambda x: x.strip(), read_file))


with open('consys.csv', 'w', ) as my_file:
    wr = csv.writer(my_file, quoting=csv.QUOTE_ALL)
    for word in read_file:
        wr.writerow([word])
