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
        english = [None] + align_sent.mots
        l_g = len(german)
        l_e = len(english) - 1
        print 'got here'

        for j in range(1, l_g):
            g_word = german[j - 1]
            p_max = (self.t[(g_word, None)] * self.q[(0, j, l_g, l_e)], None)
            print self.t[(g_word, None)]
            print self.q[(0, j, l_g, l_e)]

            for i in range (1, l_e):
                e_word = english[i]
                # print (g_word, e_word)
                # print self.t[(g_word, e_word)]
                # print (i, j, l_g, l_e)
                # print self.q[(i, j, l_g, l_e)]
                p_max = max(p_max, (self.t[(g_word, e_word)] * self.q[(i, j, l_g, l_e)], i))

            if p_max[1] is not None:
                alignments.append((j, p_max[1]))

        return AlignedSent(align_sent.words, align_sent.mots, alignments)


    # TODO: Implement the EM algorithm. num_iters is the number of iterations. Returns the 
    # translation and distortion parameters as a tuple.

    def initialize(self, target_sents, source_sents):
        q = defaultdict(float)
        t = defaultdict(float)
        counts = defaultdict(set)

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
                    q[(i, j, l_src, l_tar)] = init_prob

        # # Initialize t
        # t = {}
        # all_source_words = set(all_source_words)

        # for word in all_source_words:
        #     possible_translations = []
        #     for (target_sent, source_sent) in zip(target_sents, source_sents):
        #         if word in source_sent:
        #             possible_translations += target_sent # including 'NULL' in target_sent

        #     possible_translations = set(possible_translations)
        #     count = len(possible_translations)
        #     for possible_translation in possible_translations:
        #         t[(word, possible_translation)] = 1.0 / count

        return (t,q)


    def train(self, aligned_sents, num_iters):
        gsents = []
        esents = []
        g_vocab = set()
        e_vocab = set()
        # gsents_N =[]
        # esents_N = []
        for aligned_sent in aligned_sents:
            # gsent = [None] + aligned_sent.words
            # esent = [None] + aligned_sent.mots
            # gsents.append(aligned_sent.words)
            gsents.append(aligned_sent.words)
            # esents.append(aligned_sent.mots)
            esents.append(aligned_sent.mots)
            g_vocab.update(aligned_sent.words)
            e_vocab.update(aligned_sent.mots)

        # Initialize t_eg, q_eg, t_ge, q_ge
        # Only t_eg and q_eg are returned

        (t_eg,q_eg) = self.initialize(gsents, esents)
        (t_ge,q_ge) = self.initialize(esents, gsents)

        for s in range(0, num_iters):
            c_eg = defaultdict(float)
            normalizer = defaultdict(float)

            #Calculate counts for e2g:
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
                        # print target_word
                        # print t_eg[(source_word, target_word)]
                        # print q_eg[(j, i, l, m)]
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

            #Update t_ge, q_ge values
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

        # Average between 2 models
        # t = {}
        # e_vocab.add(None)
        # for g in g_vocab:
        #     for e in e_vocab:
        #         if (e, g) in t_eg:
        #             t[(g, e)] = (c_eg[(e, g)] + c_ge[(g, e)]) / (c_eg[g] + c_ge[e])
        #         else:
        #             t[(g, e)] = t_ge[(g, e)]

        # Average q values
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
                    # if (j, i, l_e, l_g) in q_eg:
                    if (e_word, g_word) in t_eg:
                        t[(g_word, e_word)] = (c_eg[(e_word, g_word)] + c_ge[(g_word, e_word)]) / (c_eg[g_word] + c_ge[e_word])
                        q[(i, j, l_g, l_e)] = (c_eg[(j, i, l_e, l_g)] + c_ge[(i, j, l_g, l_e)]) / (c_eg[(i, l_e, l_g)] + c_ge[(j, l_g, l_e)])
                    else:
                        t[(g_word, e_word)] = t_ge[(g_word, e_word)] / 2
                        q[(i, j, l_g, l_e)] = q_ge[(i, j, l_g, l_e)]

                        # print source_word
                        # print target_word
                        # if i * j != 0:
                        
                        
                        #     q_eg[(i, j, m, l)] = (c_ge[(j, i, l, m)] + c_eg[(i, j, m, l)]) / (c_ge[(i, l, m)] + c_eg[(j, m, l)])
                        #     t_eg[(target_word, source_word)] = (c_ge[(source_word, target_word)] + c_eg[(target_word, source_word)]) / (c_ge[source_word] + c_eg[target_word])
                        # elif j == 0 and i != 0:
                        #     q_ge[(j, i, l, m)] = c_ge[(j, i, l, m)] / c_ge[(i, l, m)]
                        #     t_ge[(source_word, target_word)] = c_ge[(source_word, target_word)] / c_ge[source_word]
                        #     q_eg[(i, j, m, l)] = (c_ge[(j, i, l, m)] + c_eg[(i, j, m, l)]) / (c_ge[(i, l, m)] + c_eg[(j, m, l)])
                        #     t_eg[(target_word, source_word)] = (c_ge[(source_word, target_word)] + c_eg[(target_word, source_word)]) / (c_ge[source_word] + c_eg[target_word])
                        # elif j != 0 and i == 0:
                        #     q_ge[(j, i, l, m)] = (c_ge[(j, i, l, m)] + c_eg[(i, j, m, l)]) / (c_ge[(i, l, m)] + c_eg[(j, m, l)])
                        #     t_ge[(source_word, target_word)] = (c_ge[(source_word, target_word)] + c_eg[(target_word, source_word)]) / (c_ge[source_word] + c_eg[target_word])
                        #     q_eg[(i, j, m, l)] = c_eg[(i, j, m, l)] / c_eg[(j, m, l)]
                        #     t_eg[(target_word, source_word)] = c_eg[(target_word, source_word)] / c_eg[target_word]
                        # else:
                        #     pass
        return (t, q)


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
    ba = BerkeleyAligner(aligned_sents, 10)
    A.save_model_output(aligned_sents, ba, "ba.txt")
    avg_aer = A.compute_avg_aer(aligned_sents, ba, 1)

    print ('Berkeley Aligner')
    print ('---------------------------')
    print('Average AER: {0:.3f}\n'.format(avg_aer))
