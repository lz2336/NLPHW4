****************************************************
COMS 4705 Natural Language Processing Spring 2015
		HW4
	Lingzi Zhuang, lz2336
****************************************************

Deliverables:
A.py
ibm1.txt
ibm2.txt
B.py
ba.txt
EC.py
README.txt

Part A

(3) 

We learn from NLTK’s source code that ibm1 optimizes alignment between a pair of words e, f based solely on translation probability t(f|e). By contrast, ibm2 also takes into account a distortion factor q, computed on the position indices of the words e and f in their respective sentences. Using q * t, instead of t, as the parameter, ibm2 is more resistant to incorrect alignments which span a greater distance within the sentences. 

Below is an example which illustrates this.

Sentence: comtrans.aligned_sents()[27]

Dennoch , Frau Präsidentin , wurde meinem Wunsch nicht entsprochen .
But , Madam President , my personal request has not been met .

Alignments:
IBM1:		AER = 0.652
0-11 1-4 2-2 3-2 4-4 5-10 6-11 7-11 8-9 9-11
IBM2:		AER = 0.417
0-6 1-1 2-2 3-6 4-4 5-10 6-6 7-11 8-9 9-11 10-12
Correct alignments:
0-0 1-1 2-2 3-3 4-4 5-8 6-5 6-6 7-7 8-9 9-10 9-11 10-12

In this example, IBM2 successfully aligned 1-1, 6-6, as compared to IBM1’s 1-4, 6-11, which span a greater distance but could not be accounted for by IBM1. IBM2 also succeeded in aligning 10-12 (punctuation).

—————
(4)

Running the two models with incrementing numbers of iterations give the following results:

NUM_ITER	avg_aer(IBM1)	avg_aer(IBM2)
3 		64.1 		64.4
4		63.0		64.2<<
5		62.7		64.4
6		62.6<<		64.7
7		62.9		64.6
10		66.5		65.0
15		66.5		65.0
20		66.1		65.0
30		66.0		64.9

From these results, it is clear that IBM1 performs optimally with NUM_ITER = 6, and IBM2 performs optimally with NUM_ITER =4. 

As NUM_ITER increases, avg_aer for both models converge. avg_aer(IBM1) converges to 66.0 after 20+ iterations. avg_aer(IBM2) converges faster, to 65.0, after 10+ iterations.

————
Part B

(4) 

The average AER for the first 50 sentences is at 57.6.

——-

(5)

The Berkeley aligner has advantage over the IBM models because it takes t and q counts both ways, and thus is able to even out potential errors caused by a simple one-way perspective. 

Below is an example of a sentence on which Berkeley aligner outperforms the IBM models.

Vielen Dank , Herr Segni , das will ich gerne tun .
Thank you , Mr Segni , I shall do so gladly .

Berkeley:	AER = 0.280
0-0 1-0 2-2 3-3 4-4 5-2 6-4 7-8 8-6 9-10 10-8 11-11
IBM1:		AER = 0.500
0-0 1-0 2-5 3-3 4-10 5-5 6-9 7-10 8-6 9-10 10-10
IBM2: 		AER = 0.583
0-0 1-0 2-5 3-3 4-10 5-5 6-4 7-4 8-9 9-10 10-10
Correct alignments:
0-0 1-0 1-1 2-2 3-3 4-4 5-5 7-7 8-6 9-10 10-8 10-9 11-11

In this example, Berkeley aligner succeeded in aligning 2-2, 4-4, 8-6, 10-8, which are missed by the IBM models. In particular, word 4 ‘Segni’ in German is incorrectly aligned with 10 ‘gladly’ by both IBM models, while being a proper name it has an exact counterpart in English. The Berkeley aligner is better at evening out this one-way bias.

In addition, the Berkeley aligner does well in correctly aligning punctuation, possibly due to the same ability to even out one-way biases.

——
Note: 

* I referred extensively to NLTK’s implementation of IBM2 both when learning and when writing Berkeley aligner. 
