input_file = 'new-test-masked.tsv'
output_file = 'new-test-truthword-removed.txt'

with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
    for line in infile:
        # Split the line by period and keep only the part before the first period
        modified_line = line.split('.', 1)[0]
        outfile.write(modified_line + "." + '\n')
