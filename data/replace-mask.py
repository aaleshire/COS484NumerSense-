import csv

# define input and output file paths
input_file = "testing/new-test-masked.tsv"
output_file = "testing/new-test-complete-sentences.txt"

# open input and output files
with open(input_file, 'r', newline='', encoding='utf-8') as infile, \
     open(output_file, 'w', encoding='utf-8') as outfile:
    
    # create a TSV reader object
    reader = csv.reader(infile, delimiter='\t')
    
    # loop through each row in the TSV file
    for row in reader:
        
        # find the last word in the row
        last_word = row[-1].split()[-1]
        
        # replace any occurrence of "<mask>" with the last word and delete the last word from the row
        modified_row = [item.replace("<mask>", last_word) if "<mask>" in item else item for item in row[:-1]]
        
        # write the modified row to the output file as a single line of text
        outfile.write("\t".join(modified_row) + "\n")
