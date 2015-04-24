import nltk
from nltk.corpus import comtrans

NUM_ITERS = 10

# TODO: Initialize IBM Model 1 and return the model.
def create_ibm1(aligned_sents):
    num_iters = NUM_ITERS
    ibm1 = IBMModel1(aligned_sents, num_iters)
    return ibm1

# TODO: Initialize IBM Model 2 and return the model.
def create_ibm2(aligned_sents):
    num_iters = NUM_ITERS
    ibm2 = IBMModel2(aligned_sents, num_iters)
    return ibm2

# TODO: Compute the average AER for the first n sentences
#       in aligned_sents using model. Return the average AER.
def compute_avg_aer(aligned_sents, model, n):
    er = 0
    curr_er = 0

    for i in range(0, n):
        curr_sent = aligned_sents[i]
        aligned_curr_sent = model.align(curr_sent)
        curr_er = curr_sent.alignment_error_rate(aligned_curr_sent)
        er += curr_er

    avg_er = float(er) / n

    return avg_er


# TODO: Computes the alignments for the first 20 sentences in
#       aligned_sents and saves the sentences and their alignments
#       to file_name. Use the format specified in the assignment.
def save_model_output(aligned_sents, model, file_name):
    first_twenty = aligned_sents[:20]
    
    output_file = open(file_name, 'w')

    for curr_sent in aligned_sents:
        aligned_curr_sent = model.align(sent)

        source = ' '.join(aligned_curr_sent.words)
        target = ' '.join(aligned_curr_sent.mots)
        alignments = ' '.join(str(alignm) for alignm in aligned_curr_sent.alignment)
        output = 'Source sentence\t' + source + '\nTarget sentence\t' + target + '\nAlignments\t' + alignments + '\n'

        output_file.write(output)

    output_file.close()

def main(aligned_sents):
    ibm1 = create_ibm1(aligned_sents)
    save_model_output(aligned_sents, ibm1, "ibm1.txt")
    avg_aer = compute_avg_aer(aligned_sents, ibm1, 50)

    print ('IBM Model 1')
    print ('---------------------------')
    print('Average AER: {0:.3f}\n'.format(avg_aer))

    ibm2 = create_ibm2(aligned_sents)
    save_model_output(aligned_sents, ibm2, "ibm2.txt")
    avg_aer = compute_avg_aer(aligned_sents, ibm2, 50)
    
    print ('IBM Model 2')
    print ('---------------------------')
    print('Average AER: {0:.3f}\n'.format(avg_aer))
