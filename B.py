import nltk
import A
from collections import defaultdict
from nltk.align import Alignment, AlignedSent
import itertools
import sys

class BerkeleyAligner():

    def __init__(self, align_sents, num_iter):
        self.t, self.q = self.train(align_sents, num_iter)

    # TODO: Computes the alignments for align_sent, using this model's parameters. Return
    #       an AlignedSent object, with the sentence pair and the alignments computed.
    def align(self, align_sent):
        alignments = []
        german = align_sent.words
        english = align_sent.mots
        l_g = len(german)
        l_e = len(english)

        for j in range(0, l_g):
            g_word = german[j]
            p_max = (self.t[(g_word, None)] * self.q[(0, j + 1, l_g, l_e)], None)

            for i in range (0, l_e):
                e_word = english[i]
                p_max = max(p_max, (self.t[(g_word, e_word)] * self.q[(i + 1, j + 1, l_g, l_e)], i))

            if p_max[1] is not None:
                alignments.append((j, p_max[1]))

        return AlignedSent(align_sent.words, align_sent.mots, alignments)


    # TODO: Implement the EM algorithm. num_iters is the number of iterations. Returns the 
    # translation and distortion parameters as a tuple.

    def initialize(self, target_sents, source_sents):
        q = defaultdict(float)
        t = defaultdict(float)
        counts = defaultdict(set)

        # Initialize t
        for (target_sent, source_sent) in zip(target_sents, source_sents):
            target_sent = [None] + target_sent
            for src_word in source_sent:
                counts[src_word].update(target_sent)

        for word in counts.keys():
            tar_words = counts[word]
            for tar_word in tar_words:
                t[(word, tar_word)] = 1.0 / len(tar_words)

        # Initialize q. l: length of source sent; m: length of target sent
        for target_sent, source_sent in zip(target_sents, source_sents):  
            target_sent = [None] + target_sent
            l_src = len(source_sent)
            l_tar = len(target_sent) - 1
            init_prob = 1.0 / (l_tar + 1)

            for i in range(0, l_tar + 1):
                for j in range(1, l_src + 1):
                    q[(i, j, l_src, l_tar)] += init_prob

        return (t,q)


    def train(self, aligned_sents, num_iters):
        gsents = []
        esents = []
        # gsents_N =[]
        # esents_N = []
        for aligned_sent in aligned_sents:
            # gsent = [None] + aligned_sent.words
            # esent = [None] + aligned_sent.mots
            # gsents.append(aligned_sent.words)
            gsents.append(aligned_sent.words)
            # esents.append(aligned_sent.mots)
            esents.append(aligned_sent.mots)

        # Initialize t_eg, q_eg, t_ge, q_ge

        (t_eg,q_eg) = self.initialize(gsents, esents)
        (t_ge,q_ge) = self.initialize(esents, gsents)

        #Calculate counts for e2g:
        for s in range(0, num_iters):
            c_eg = defaultdict(float)
            normalizer = defaultdict(float)

            for k in range(0, len(esents)):
                e_sent = esents[k]
                g_sent = [None] + gsents[k]
                l_e = len(e_sent)
                l_g = len(g_sent) - 1

                # Compute Normalization
                for j in range(1, l_e + 1):
                    e_word = e_sent[j - 1]
                    normalizer[e_word] = 0
                    for i in range(0, l_g + 1):
                        g_word = g_sent[i]
                        normalizer[e_word] += t_eg[(e_word, g_word)] * q_eg[(i, j, l_e, l_g)]

                # Counts
                for j in range(1, l_e + 1):
                    e_word = e_sent[j - 1]
                    for i in range(0, l_g + 1):
                        g_word = g_sent[i]
                        delta = t_eg[(e_word, g_word)] * q_eg[(i, j, l_e, l_g)] / normalizer[e_word]
                        c_eg[(e_word, g_word)] += delta
                        c_eg[g_word] += delta
                        c_eg[(i, j, l_e, l_g)] += delta
                        c_eg[(j, l_e, l_g)] += delta


            # Update t_eg, q_eg values
            for (g_sent, e_sent) in zip(gsents, esents):
                g_sent = [None] + g_sent
                l_e = len(e_sent)
                l_g = len(g_sent) - 1
                for i in range(0, l_g + 1):
                    g_word = g_sent[i]
                    for j in range(1, l_e + 1):
                        e_word = e_sent[j - 1]
                        t_eg[(e_word, g_word)] = c_eg[(e_word, g_word)] / c_eg[g_word]
                        q_eg[(i, j, l_e, l_g)] = c_eg[(i, j, l_e, l_g)] / c_eg[(j, l_e, l_g)]

        # Calculate counts for g2e:
        for s in range(0, num_iters):
            c_ge = defaultdict(float)
            normalizer = defaultdict(float)

            for k in range(0, len(gsents)):
                g_sent = gsents[k]
                e_sent = [None] + esents[k]
                l_g = len(g_sent)
                l_e = len(e_sent) - 1

                # Compute normalizer
                for j in range(1, l_g + 1):
                    g_word = g_sent[j - 1]
                    normalizer[g_word] = 0
                    for i in range(0, l_e + 1):
                        e_word = e_sent[i]
                        normalizer[g_word] += t_ge[(g_word, e_word)] * q_ge[(i, j, l_g, l_e)]

                # Counts
                for j in range(1, l_g + 1):
                    g_word = g_sent[j - 1]
                    for i in range(0, l_e + 1):
                        e_word = e_sent[i]
                        delta = t_ge[(g_word, e_word)] * q_ge[(i, j, l_g, l_e)] / normalizer[g_word]
                        c_ge[(g_word, e_word)] += delta
                        c_ge[e_word] += delta
                        c_ge[(i, j, l_g, l_e)] += delta
                        c_ge[(j, l_g, l_e)] += delta

            # Update t_ge, q_ge values
            for (g_sent, e_sent) in zip(gsents, esents):
                e_sent = [None] + e_sent
                l_g = len(g_sent)
                l_e = len(e_sent) - 1

                for i in range(0, l_e + 1):
                    e_word = e_sent[i]
                    for j in range(1, l_g + 1):
                        g_word = g_sent[j - 1]
                        t_ge[(g_word, e_word)] = c_ge[(g_word, e_word)] / c_ge[e_word]
                        q_ge[(i, j, l_g, l_e)] = c_ge[(i, j, l_g, l_e)] / c_ge[(j, l_g, l_e)]

        # Averaging t, q values
        t = {}
        q = {}
        for (g_sent, e_sent) in zip(gsents, esents):
            e_sent = [None] + e_sent
            l_g = len(g_sent)
            l_e = len(e_sent) - 1

            for i in range(0, l_e + 1):
                e_word = e_sent[i]
                for j in range(1, l_g + 1):
                    g_word = g_sent[j - 1]
                    if (e_word, g_word) in t_eg:
                        t[(g_word, e_word)] = (c_eg[(e_word, g_word)] + c_ge[(g_word, e_word)]) / (c_eg[g_word] + c_ge[e_word])
                        # t[(g_word, e_word)] = (c_eg[(e_word, g_word)] + c_ge[(g_word, e_word)]) / (2 * c_ge[e_word])
                        q[(i, j, l_g, l_e)] = (c_eg[(j, i, l_e, l_g)] + c_ge[(i, j, l_g, l_e)]) / (c_eg[(i, l_e, l_g)] + c_ge[(j, l_g, l_e)])             
                        # q[(i, j, l_g, l_e)] = (c_eg[(j, i, l_e, l_g)] + c_ge[(i, j, l_g, l_e)]) / (2 * c_ge[(j, l_g, l_e)])
                    else:
                        t[(g_word, e_word)] = t_ge[(g_word, e_word)]
                        q[(i, j, l_g, l_e)] = q_ge[(i, j, l_g, l_e)]

        return (t, q)


def main(aligned_sents):
    ba = BerkeleyAligner(aligned_sents, 10)
    A.save_model_output(aligned_sents, ba, "ba.txt")
    avg_aer = A.compute_avg_aer(aligned_sents, ba, 50)

    print ('Berkeley Aligner')
    print ('---------------------------')
    print('Average AER: {0:.3f}\n'.format(avg_aer))
