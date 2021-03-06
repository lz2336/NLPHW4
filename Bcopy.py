import nltk
import A
from collections import defaultdict
from nltk.align import Alignment, AlignedSent

class BerkeleyAligner():

    def __init__(self, align_sents, num_iter):
        self.t, self.q = self.train(align_sents, num_iter)

    # TODO: Computes the alignments for align_sent, using this model's parameters. Return
    #       an AlignedSent object, with the sentence pair and the alignments computed.
    def align(self, align_sent):
        alignments = []
        german = [None] + align_sent.words
        english = [None] + align_sent.mots
        l = len(german)
        m = len(english)
        p_max = 0
        max_j = 1
        max_i = 0

        for j in range(1, l):
            g_word = german[j]
            for i in range(0, m):
                e_word = english[i]
                print g_word
                print e_word
                print self.t[(g_word, e_word)]
                if p_max < (self.t[(g_word, e_word)] * self.q[(i, j, l, m)]):
                    p_max = self.t[(g_word, e_word)] * self.q[(i, j, l, m)]
                    max_i = i

            if max_i != 0:
                alignments.append((j - 1, max_i - 1))

        return AlignedSent(align_sent.words, align_sent.mots, alignments)

    
    # TODO: Implement the EM algorithm. num_iters is the number of iterations. Returns the 
    # translation and distortion parameters as a tuple.

    def initialize(self, target_sents, source_sents):
        all_source_words = []
        q = {}
        t = {}

        for (target_sent, source_sent) in zip(target_sents, source_sents):
            all_source_words += source_sent[1:] # removing None
            
            # Initialize q. l: length of source sent; m: length of target sent
            l = len(source_sent)
            m = len(target_sent)
            init_prob = 1.0 / m 
            for target_idx in range(0, m):
                for source_idx in range(1, l): # skipping 'NULL' in source_sent
                    q[(target_idx, source_idx, l, m)] = init_prob

        # Initialize t
        t = {}
        all_source_words = set(all_source_words)

        for word in all_source_words:
            possible_translations = []
            for (target_sent, source_sent) in zip(target_sents, source_sents):
                if word in source_sent:
                    possible_translations += target_sent # including 'NULL' in target_sent

            possible_translations = set(possible_translations)
            count = len(possible_translations)
            for possible_translation in possible_translations:
                t[(word, possible_translation)] = 1.0 / count

        return (t,q)


    def train(self, aligned_sents, num_iters):
        gsents = []
        esents = []
        # gsents_N =[]
        # esents_N = []
        for aligned_sent in aligned_sents:
            gsent = [None] + aligned_sent.words
            esent = [None] + aligned_sent.mots
            # gsents.append(aligned_sent.words)
            gsents.append(gsent)
            # esents.append(aligned_sent.mots)
            esents.append(esent)

        # Initialize t_eg, q_eg, t_ge, q_ge
        # Only t_eg and q_eg are returned

        (t_eg,q_eg) = self.initialize(gsents, esents)
        (t_ge,q_ge) = self.initialize(esents, gsents)

        # print t_eg
        # print q_eg
        

        for s in range(0, num_iters):
            c_eg = defaultdict(float)
            c_ge = defaultdict(float)
            # c_q_eg = defaultdict(float)
            # c_q_ge = defaultdict(float)
            normalizer = defaultdict(float)
            # normalizer_ge = defaultdict(float)


            #Calculate counts for e2g:
            for k in range(0, len(aligned_sents)):
                source_sent = esents[k]
                target_sent = gsents[k]
                l = len(source_sent)
                m = len(target_sent)

                for i in range(1, l):
                    source_word = source_sent[i]
                    # print source_word
                    normalizer[source_word] = 0
                    for j in range(0, m):
                        target_word = target_sent[j]
                        # print target_word
                        # print t_eg[(source_word, target_word)]
                        # print q_eg[(j, i, l, m)]
                        normalizer[source_word] += t_eg[(source_word, target_word)] * q_eg[(j, i, l, m)]

                for i in range(1, l):
                    source_word = source_sent[i]
                    for j in range(0, m):
                        target_word = target_sent[j]
                        delta = t_eg[(source_word, target_word)] * q_eg[(j, i, l, m)] / normalizer[source_word]
                        c_eg[(source_word, target_word)] += delta
                        c_eg[source_word] += delta
                        c_eg[(j, i, l, m)] += delta
                        c_eg[(i, l, m)] += delta

            # Calculate counts for g2e:
            for k in range(0, len(aligned_sents)):
                source_sent = gsents[k]
                target_sent = esents[k]
                l = len(source_sent)
                m = len(target_sent)

                for i in range(1, l):
                    source_word = source_sent[i]
                    normalizer[source_word] = 0
                    for j in range(0, m):
                        target_word = target_sent[j]
                        normalizer[source_word] += t_ge[(source_word, target_word)] * q_ge[(j, i, l, m)]

                for i in range(1, l):
                    source_word = source_sent[i]
                    for j in range(0, m):
                        target_word = target_sent[j]
                        delta = t_ge[(source_word, target_word)] * q_ge[(j, i, l, m)] / normalizer[source_word]
                        c_ge[(source_word, target_word)] += delta
                        c_ge[source_word] += delta
                        c_ge[(j, i, l, m)] += delta
                        c_ge[(i, l, m)] += delta

            # Calculate updated t and q
            for (g_sent, e_sent) in zip(gsents, esents):
                source_sent = g_sent
                target_sent = e_sent
                l = len(source_sent)
                m = len(target_sent)

                for j in range(0, m):
                    target_word = target_sent[j]
                    for i in range(0, l):
                        source_word = source_sent[i]
                        # print source_word
                        # print target_word
                        if i * j != 0:
                            q_ge[(j, i, l, m)] = (c_ge[(j, i, l, m)] + c_eg[(i, j, m, l)]) / (c_ge[(i, l, m)] + c_eg[(j, m, l)])
                            t_ge[(source_word, target_word)] = (c_ge[(source_word, target_word)] + c_eg[(target_word, source_word)]) / (c_ge[source_word] + c_eg[target_word])
                            q_eg[(i, j, m, l)] = (c_ge[(j, i, l, m)] + c_eg[(i, j, m, l)]) / (c_ge[(i, l, m)] + c_eg[(j, m, l)])
                            t_eg[(target_word, source_word)] = (c_ge[(source_word, target_word)] + c_eg[(target_word, source_word)]) / (c_ge[source_word] + c_eg[target_word])
                        elif j == 0 and i != 0:
                            q_ge[(j, i, l, m)] = c_ge[(j, i, l, m)] / c_ge[(i, l, m)]
                            t_ge[(source_word, target_word)] = c_ge[(source_word, target_word)] / c_ge[source_word]
                            q_eg[(i, j, m, l)] = (c_ge[(j, i, l, m)] + c_eg[(i, j, m, l)]) / (c_ge[(i, l, m)] + c_eg[(j, m, l)])
                            t_eg[(target_word, source_word)] = (c_ge[(source_word, target_word)] + c_eg[(target_word, source_word)]) / (c_ge[source_word] + c_eg[target_word])
                        elif j != 0 and i == 0:
                            q_ge[(j, i, l, m)] = (c_ge[(j, i, l, m)] + c_eg[(i, j, m, l)]) / (c_ge[(i, l, m)] + c_eg[(j, m, l)])
                            t_ge[(source_word, target_word)] = (c_ge[(source_word, target_word)] + c_eg[(target_word, source_word)]) / (c_ge[source_word] + c_eg[target_word])
                            q_eg[(i, j, m, l)] = c_eg[(i, j, m, l)] / c_eg[(j, m, l)]
                            t_eg[(target_word, source_word)] = c_eg[(target_word, source_word)] / c_eg[target_word]
                        else:
                            pass
        return (t_ge, q_ge)

    
    # def calculate_delta(i, j, souw, tarw, t, q):
    #     return delta

    # def calculate_counts(gsents, esents, t, q):
    #     ge_tcounts1 = {}
    #     ge_tcounts2 = {}
    #     ge_qcounts1 = {}
    #     ge_qcounts2 = {}
    #     eg_tcounts1 = {}
    #     eg_tcounts2 = {}
    #     eg_qcounts1 = {}
    #     eg_qcounts2 = {}

    #     for k in range(0, len(gsents)):
    #         # ge_source_sent = gsents_N[k]
    #         # ge_target_sent = esents[k]
    #         # eg_source_sent = esents_N[k]
    #         # eg_target_sent = gsents[k]
    #         g_sent = gsents[k]
    #         e_sent = esent[k]

    #         e_len = len(e_sent)
    #         g_len = len(g_sent)

    #         for e_idx in range(0, e_len):
    #             for g_idx in range(0, g_len):
    #                 if e_idx == 0: # e_idx at 'NULL' in e_len
    #                     # only update g2e counts
    #                     g_w = g_sent[g_idx]
    #                     e_w = e_sent[e_idx]
    #                     delta = calculate_delta(g_idx, e_idx, g_w, e_w, ge_t, ge_q)

    #                     if (g_w, e_w) in tcounts1:
    #                         tcounts1[(g_w, e_w)] += delta
    #                     else:
    #                         tcounts1[(g_w, e_w)] = delta
    #                     if e_w in tcounts2:
    #                         tcounts2[e_w] += delta
    #                     else:
    #                         tcounts2[e_w] = delta
    #                     if (j, i, l, m) in qcounts1:
    #                         qcounts1[(j, i, l, m)] += delta
    #                     else:
    #                         qcounts1[(j, i, l, m)] = delta
    #                     if (i, l, m) in qcounts2:
    #                         qcounts2[(i, l, m)] += delta
    #                     else:
    #                         qcounts2[(i, l, m)] = delta


    #                 elif g_idx == 0: # g_idx at 'NULL' in g_len
    #                     # only update e2g counts
    #                     sourcew = e_sent[e_idx]
    #                     targetw = g_sent[g_idx]
    #                     delta = calculate_delta(g_idx, e_idx, sourcew, targetw, ge_t, ge_q)

    #                     if (sourcew, targetw) in tcounts1:
    #                         tcounts1[(sourcew, targetw)] += delta
    #                     else:
    #                         tcounts1[(sourcew, targetw)] = delta
    #                     if targetw in tcounts2:
    #                         tcounts2[targetw] += delta
    #                     else:
    #                         tcounts2[targetw] = delta
    #                     if (j, i, l, m) in qcounts1:
    #                         qcounts1[(j, i, l, m)] += delta
    #                     else:
    #                         qcounts1[(j, i, l, m)] = delta
    #                     if (i, l, m) in qcounts2:
    #                         qcounts2[(i, l, m)] += delta
    #                     else:
    #                         qcounts2[(i, l, m)] = delta
    #                 else:


    #                 # souw = source_words[i]
    #                 # tarw = target_words[j]
    #                 # delta = calculate_delta(i, j, souw, tarw, #t_ge, #q_ge)
    #                 # if (souw, tarw) in tcounts1:
    #                 #     tcounts1[(souw, tarw)] += delta
    #                 # else:
    #                 #     tcounts1[(souw, tarw)] = delta
    #                 # if tarw in tcounts2:
    #                 #     tcounts2[tarw] += delta
    #                 # else:
    #                 #     tcounts2 = delta
    #                 # if (j, i, l, m) in qcounts1:
    #                 #     qcounts1[(j, i, l, m)] += delta
    #                 # else:
    #                 #     qcounts1[(j, i, l, m)] = delta
    #                 # if (i, l, m) in qcounts2:
    #                 #     qcounts2[(i, l, m)] += delta
    #                 # else:
    #                 #     qcounts2[(i, l, m)] = delta
    #     return # count dictionary

def main(aligned_sents):
    ba = BerkeleyAligner(aligned_sents, 5)
    A.save_model_output(aligned_sents, ba, "ba.txt")
    avg_aer = A.compute_avg_aer(aligned_sents, ba, 50)

    print ('Berkeley Aligner')
    print ('---------------------------')
    print('Average AER: {0:.3f}\n'.format(avg_aer))
