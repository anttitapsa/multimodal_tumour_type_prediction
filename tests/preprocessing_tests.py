"""Unittest for module src/annotate_mutations.py
"""

import sys
import unittest

sys.path.insert(0, '../')
from src.annotate_mutations import read_codes, get_mutation_seq 


# line, reverse_code, mutation_code, motif_length=3
class PreprocessTest(unittest.TestCase):
    mutation_code, reverse_code = read_codes('/csc/epitkane/home/ahuttun/muat-github/extfile/mutation_codes_sv.tsv')
    
    def test_00_SNV(self):
        # The pyrimidine based mutations only occur in the data 
        # line contains following information: 
        # CHROM, POS, REF, ALT, sample, seq, genic, exonic, strand, rt, sex, age, histology
        line = ['1,7,T,A,test-case,GAATCGA:GCCATA,0,0,=,0.0,test,0,test-case',
                '1,7,T,C,test-case,GAATCGA;GCCATA,0,0,=,0.0,test,0,test-case',
                '1,7,T,G,test-case,GAATCGA?GCCATA,0,0,=,0.0,test,0,test-case',
                '1,7,C,A,test-case,GAATCGA$GCCATA,0,0,=,0.0,test,0,test-case',
                '1,7,C,G,test-case,GAATCGA%GCCATA,0,0,=,0.0,test,0,test-case',
                '1,7,C,T,test-case,GAATCGA^GCCATA,0,0,=,0.0,test,0,test-case']

        seq, ref_seq, ref, alt = get_mutation_seq(line[0].split(sep = ','), self.reverse_code)
        self.assertEqual(seq, 'AAG', msg='Fail on sequence, case T > A')
        self.assertEqual(ref_seq, 'ATG', msg='Fail on reference sequence, case T > A')
        
        seq, ref_seq, ref, alt = get_mutation_seq(line[1].split(sep = ','), self.reverse_code)
        self.assertEqual(seq, 'ACG', msg='Fail on sequence, case T > C')
        self.assertEqual(ref_seq, 'ATG', msg='Fail on reference sequence, case T > C')

        seq, ref_seq, ref, alt = get_mutation_seq(line[2].split(sep = ','), self.reverse_code)
        self.assertEqual(seq, 'AGG', msg='Fail on sequence, case T > G')
        self.assertEqual(ref_seq, 'ATG', msg='Fail on reference sequence, case T > G')

        seq, ref_seq, ref, alt = get_mutation_seq(line[3].split(sep = ','), self.reverse_code)
        self.assertEqual(seq, 'AAG', msg='Fail on sequence, case C > A')
        self.assertEqual(ref_seq, 'ACG', msg='Fail on reference sequence, case C > A')

        seq, ref_seq, ref, alt = get_mutation_seq(line[4].split(sep = ','), self.reverse_code)
        self.assertEqual(seq, 'AGG', msg='Fail on sequence, case C > G')
        self.assertEqual(ref_seq, 'ACG', msg='Fail on reference sequence, case C > G')

        seq, ref_seq, ref, alt = get_mutation_seq(line[5].split(sep = ','), self.reverse_code)
        self.assertEqual(seq, 'ATG', msg='Fail on sequence, case C > T')
        self.assertEqual(ref_seq, 'ACG', msg='Fail on reference sequence, case C > T')


    def test_01_simple_deletions(self):

        line = ['1,7,A,,test-case,GAATCGA1GCCATA,0,0,=,0.0,test,0,test-case',
                '1,7,C,,test-case,GAATCGA2GCCATA,0,0,=,0.0,test,0,test-case',
                '1,7,G,,test-case,GAATCGA3GCCATA,0,0,=,0.0,test,0,test-case',
                '1,7,T,,test-case,GAATCGA4GCCATA,0,0,=,0.0,test,0,test-case']
        
        # One deletion
        seq, ref_seq, ref, alt = get_mutation_seq(line[0].split(sep = ','), self.reverse_code)
        self.assertEqual(seq, 'AG', msg='Fail on sequence, case A > -')
        self.assertEqual(ref_seq, 'AAG', msg='Fail on reference sequence, case A > -')

        seq, ref_seq, ref, alt = get_mutation_seq(line[1].split(sep = ','), self.reverse_code)
        self.assertEqual(seq, 'AG', msg='Fail on sequence, case C > -')
        self.assertEqual(ref_seq, 'ACG', msg='Fail on reference sequence, case C > -')

        seq, ref_seq, ref, alt = get_mutation_seq(line[2].split(sep = ','), self.reverse_code)
        self.assertEqual(seq, 'AG', msg='Fail on sequence, case G > -')
        self.assertEqual(ref_seq, 'AGG', msg='Fail on reference sequence, case G > -')

        seq, ref_seq, ref, alt = get_mutation_seq(line[3].split(sep = ','), self.reverse_code)
        self.assertEqual(seq, 'AG', msg='Fail on sequence, case T > -')
        self.assertEqual(ref_seq, 'ATG', msg='Fail on reference sequence, case T > -')


    def test_02_multiple_deletions_even(self):
        
        line = ['1,7,TA,,test-case,GAATCGA41CCATA,0,0,=,0.0,test,0,test-case',
                '1,7,CTAG,,test-case,GAATCG2413CATA,0,0,=,0.0,test,0,test-case',
                '1,7,CGATAC,,test-case,GAAT231412CATA,0,0,=,0.0,test,0,test-case']
        
        # even number of deletions 2, 4, and 6
        seq, ref_seq, ref, alt = get_mutation_seq(line[0].split(sep = ','), self.reverse_code)
        self.assertEqual(seq, 'AC', msg='Fail on sequence, case of two bases deletion')
        self.assertEqual(ref_seq, 'ATAC', msg='Fail on reference sequence, case of two bases deletion')
        
        seq, ref_seq, ref, alt = get_mutation_seq(line[1].split(sep = ','), self.reverse_code)
        self.assertEqual(seq, 'GC', msg='Fail on sequence, case of four bases deletion')
        self.assertEqual(ref_seq, 'GCTAGC', msg='Fail on reference sequence, case of four bases deletion')
        
        seq, ref_seq, ref, alt = get_mutation_seq(line[2].split(sep = ','), self.reverse_code)
        self.assertEqual(seq, 'TC', msg='Fail on sequence, case of six bases deletion')
        self.assertEqual(ref_seq, 'TCGATACC', msg='Fail on reference sequence, case of six bases deletion')


    def test_03_multiple_deletions_odd(self):

        line = ['1,7,ATG,,test-case,GAATCG143CCATA,0,0,=,0.0,test,0,test-case',
                '1,7,GATGC,,test-case,GAATC31432CATA,0,0,=,0.0,test,0,test-case']
        
        # uneven number of deletions 3 and 5
        seq, ref_seq, ref, alt = get_mutation_seq(line[0].split(sep = ','), self.reverse_code)
        self.assertEqual(seq, 'GC', msg='Fail on sequence, case of three base deletion')
        self.assertEqual(ref_seq, 'GATGC', msg='Fail on reference sequence, case of three bases deletion')
        
        seq, ref_seq, ref, alt = get_mutation_seq(line[1].split(sep = ','), self.reverse_code)
        self.assertEqual(seq, 'CC', msg='Fail on sequence, case of five bases deletion')
        self.assertEqual(ref_seq, 'CGATGCC', msg='Fail on reference sequence, case of five bases deletion')


    def test_04_simple_insertions(self):
        
        line = ['1,7,,A,test-case,GAATCGA5GCCATA,0,0,=,0.0,test,0,test-case',
                '1,7,,C,test-case,GAATCGA6GCCATA,0,0,=,0.0,test,0,test-case',
                '1,7,,G,test-case,GAATCGA7GCCATA,0,0,=,0.0,test,0,test-case',
                '1,7,,T,test-case,GAATCGA8GCCATA,0,0,=,0.0,test,0,test-case']
        
        seq, ref_seq, ref, alt = get_mutation_seq(line[0].split(sep = ','), self.reverse_code)
        self.assertEqual(seq, 'AAG', msg='Fail on sequence, case - > A')
        self.assertEqual(ref_seq, 'AG', msg='Fail on reference sequence, case - > A')

        seq, ref_seq, ref, alt = get_mutation_seq(line[1].split(sep = ','), self.reverse_code)
        self.assertEqual(seq, 'ACG', msg='Fail on sequence, case - > C')
        self.assertEqual(ref_seq, 'AG', msg='Fail on reference sequence, case - > C')

        seq, ref_seq, ref, alt = get_mutation_seq(line[2].split(sep = ','), self.reverse_code)
        self.assertEqual(seq, 'AGG', msg='Fail on sequence, case - > G')
        self.assertEqual(ref_seq, 'AG', msg='Fail on reference sequence, case - > G')

        seq, ref_seq, ref, alt = get_mutation_seq(line[3].split(sep = ','), self.reverse_code)
        self.assertEqual(seq, 'ATG', msg='Fail on sequence, case - > T')
        self.assertEqual(ref_seq, 'AG', msg='Fail on reference sequence, case - > T')


    def test_05_multiple_insertions_even(self):

        line =['1,7,,TA,test-case,GAATCGA85CCATA,0,0,=,0.0,test,0,test-case',
               '1,7,,CTAG,test-case,GAATCG6857CATA,0,0,=,0.0,test,0,test-case',
               '1,7,,CGATAC,test-case,GAAT675856CATA,0,0,=,0.0,test,0,test-case']

        # even number of insertions: 2, 4, and 6
        seq, ref_seq, ref, alt = get_mutation_seq(line[0].split(sep = ','), self.reverse_code)
        self.assertEqual(seq, 'ATAC', msg='Fail on sequence, case of two bases insertion')
        self.assertEqual(ref_seq, 'AC', msg='Fail on reference sequence, case of two bases insertion')
        
        seq, ref_seq, ref, alt = get_mutation_seq(line[1].split(sep = ','), self.reverse_code)
        self.assertEqual(seq, 'GCTAGC', msg='Fail on sequence, case of four bases insertion')
        self.assertEqual(ref_seq, 'GC', msg='Fail on reference sequence, case of four bases insertion')
        
        seq, ref_seq, ref, alt = get_mutation_seq(line[2].split(sep = ','), self.reverse_code)
        self.assertEqual(seq, 'TCGATACC', msg='Fail on sequence, case of six bases insertion')
        self.assertEqual(ref_seq, 'TC', msg='Fail on reference sequence, case of six bases insertion')


    def test_06_multiple_insertions_odd(self):

        line = ['1,7,,ATG,test-case,GAATCG587CCATA,0,0,=,0.0,test,0,test-case',
                '1,7,,GATGC,test-case,GAATC75876CATA,0,0,=,0.0,test,0,test-case']
        
        # uneven number of insertions 3 and 5
        seq, ref_seq, ref, alt = get_mutation_seq(line[0].split(sep = ','), self.reverse_code)
        self.assertEqual(seq, 'GATGC', msg='Fail on sequence, case of three base deletion')
        self.assertEqual(ref_seq, 'GC', msg='Fail on reference sequence, case of three bases insertion')
        
        seq, ref_seq, ref, alt = get_mutation_seq(line[1].split(sep = ','), self.reverse_code)
        self.assertEqual(seq, 'CGATGCC', msg='Fail on sequence, case of five bases deletion')
        self.assertEqual(ref_seq, 'CC', msg='Fail on reference sequence, case of five bases insertion')
    

    def test_07_no_mutation(self):

        line = ['1,7,A,A,test-case,GAATCGAAGCCATA,0,0,=,0.0,test,0,test-case',
                '1,7,C,C,test-case,GAATCGACGCCATA,0,0,=,0.0,test,0,test-case',
                '1,7,G,G,test-case,GAATCGAGGCCATA,0,0,=,0.0,test,0,test-case',
                '1,7,T,T,test-case,GAATCGATGCCATA,0,0,=,0.0,test,0,test-case']
        
        seq, ref_seq, ref, alt = get_mutation_seq(line[0].split(sep = ','), self.reverse_code)
        self.assertEqual(seq, 'AAG', msg='Fail on sequence, case A > A')
        self.assertEqual(ref_seq, 'AAG', msg='Fail on reference sequence, case A > A')

        seq, ref_seq, ref, alt = get_mutation_seq(line[1].split(sep = ','), self.reverse_code)
        self.assertEqual(seq, 'ACG', msg='Fail on sequence, case C > C')
        self.assertEqual(ref_seq, 'ACG', msg='Fail on reference sequence, case C > C')

        seq, ref_seq, ref, alt = get_mutation_seq(line[2].split(sep = ','), self.reverse_code)
        self.assertEqual(seq, 'AGG', msg='Fail on sequence, case G > G')
        self.assertEqual(ref_seq, 'AGG', msg='Fail on reference sequence, case G > G')

        seq, ref_seq, ref, alt = get_mutation_seq(line[3].split(sep = ','), self.reverse_code)
        self.assertEqual(seq, 'ATG', msg='Fail on sequence, case T > T')
        self.assertEqual(ref_seq, 'ATG', msg='Fail on reference sequence, case T > T')

    def test_08_invalid_token(self):
        # get_mutation_seq should return Nones if there is invalid mutation token 

        line = '1,7,A,A,test-case,GAATCGA£GCCATA,0,0,=,0.0,test,0,test-case'

        seq, ref_seq, ref, alt = get_mutation_seq(line.split(sep = ','), self.reverse_code)
        self.assertIsNone(seq)
        self.assertIsNone(ref_seq)
        self.assertIsNone(ref)
        self.assertIsNone(alt)

    def test_09_longer_motif_even(self):

        line = ['1,7,T,T,test-case,GAATCGATGCCATA,0,0,=,0.0,test,0,test-case',
               '1,7,T,A,test-case,GAATCGA:GCCATA,0,0,=,0.0,test,0,test-case',
               '1,7,A,,test-case,GAATCGA1GCCATA,0,0,=,0.0,test,0,test-case',
               '1,7,CGATAC,,test-case,GAAT231412CATA,0,0,=,0.0,test,0,test-case',
               '1,7,GATGC,,test-case,GAATC31432CATA,0,0,=,0.0,test,0,test-case',
               '1,7,,A,test-case,GAATCGA5GCCATA,0,0,=,0.0,test,0,test-case',
               '1,7,,CGATAC,test-case,GAAT675856CATA,0,0,=,0.0,test,0,test-case',
               '1,7,,GATGC,test-case,GAATC75876CATA,0,0,=,0.0,test,0,test-case']
        
        # no mutation
        motif = 6
        seq, ref_seq, ref, alt = get_mutation_seq(line[0].split(sep = ','), self.reverse_code, motif_length=motif)
        self.assertEqual(seq, 'GATGCC', msg='Fail on sequence, case T > T')
        self.assertEqual(ref_seq, 'GATGCC', msg='Fail on reference sequence, case T > T')

        # SNV
        seq, ref_seq, ref, alt = get_mutation_seq(line[1].split(sep = ','), self.reverse_code, motif_length=motif)
        self.assertEqual(seq, 'GAAGCC', msg='Fail on sequence, case T > A')
        self.assertEqual(ref_seq, 'GATGCC', msg='Fail on reference sequence, case T > A')

        # deletion
        seq, ref_seq, ref, alt = get_mutation_seq(line[2].split(sep = ','), self.reverse_code, motif_length=motif)
        self.assertEqual(seq, 'GAGCC', msg='Fail on sequence, case A > -')
        self.assertEqual(ref_seq, 'GAAGCC', msg='Fail on reference sequence, case A > -')

        # multiple deletions even
        seq, ref_seq, ref, alt = get_mutation_seq(line[3].split(sep = ','), self.reverse_code, motif_length=motif)
        self.assertEqual(seq, 'ATCAT', msg='Fail on sequence, case of six bases deletion')
        self.assertEqual(ref_seq, 'ATCGATACCAT', msg='Fail on reference sequence, case of six bases deletion')

        # multiple deletions odd
        seq, ref_seq, ref, alt = get_mutation_seq(line[4].split(sep = ','), self.reverse_code, motif_length=motif)
        self.assertEqual(seq, 'TCCAT', msg='Fail on sequence, case of five bases deletion')
        self.assertEqual(ref_seq, 'TCGATGCCAT', msg='Fail on reference sequence, case of five bases deletion')

        # simple insertion
        seq, ref_seq, ref, alt = get_mutation_seq(line[5].split(sep = ','), self.reverse_code, motif_length=motif)
        self.assertEqual(seq, 'GAAGCC', msg='Fail on sequence, case - > A')
        self.assertEqual(ref_seq, 'GAGCC', msg='Fail on reference sequence, case - > A')

        # multiple insertions even
        seq, ref_seq, ref, alt = get_mutation_seq(line[6].split(sep = ','), self.reverse_code, motif_length=motif)
        self.assertEqual(seq, 'ATCGATACCAT', msg='Fail on sequence, case of six bases insertion')
        self.assertEqual(ref_seq, 'ATCAT', msg='Fail on reference sequence, case of six bases insertion')

        # multiple insertions odd
        seq, ref_seq, ref, alt = get_mutation_seq(line[7].split(sep = ','), self.reverse_code, motif_length=motif)
        self.assertEqual(seq, 'TCGATGCCAT', msg='Fail on sequence, case of five bases deletion')
        self.assertEqual(ref_seq, 'TCCAT', msg='Fail on reference sequence, case of five bases insertion')

    def test_10_longer_motif_even(self):

        line = ['1,7,T,T,test-case,GAATCGATGCCATA,0,0,=,0.0,test,0,test-case',
               '1,7,T,A,test-case,GAATCGA:GCCATA,0,0,=,0.0,test,0,test-case',
               '1,7,A,,test-case,GAATCGA1GCCATA,0,0,=,0.0,test,0,test-case',
               '1,7,CGATAC,,test-case,GAAT231412CATA,0,0,=,0.0,test,0,test-case',
               '1,7,GATGC,,test-case,GAATC31432CATA,0,0,=,0.0,test,0,test-case',
               '1,7,,A,test-case,GAATCGA5GCCATA,0,0,=,0.0,test,0,test-case',
               '1,7,,CGATAC,test-case,GAAT675856CATA,0,0,=,0.0,test,0,test-case',
               '1,7,,GATGC,test-case,GAATC75876CATA,0,0,=,0.0,test,0,test-case']
        
        # no mutation
        motif = 7
        seq, ref_seq, ref, alt = get_mutation_seq(line[0].split(sep = ','), self.reverse_code, motif_length=motif)
        self.assertEqual(seq, 'CGATGCC', msg='Fail on sequence, case T > T')
        self.assertEqual(ref_seq, 'CGATGCC', msg='Fail on reference sequence, case T > T')

        # SNV
        seq, ref_seq, ref, alt = get_mutation_seq(line[1].split(sep = ','), self.reverse_code, motif_length=motif)
        self.assertEqual(seq, 'CGAAGCC', msg='Fail on sequence, case T > A')
        self.assertEqual(ref_seq, 'CGATGCC', msg='Fail on reference sequence, case T > A')

        # deletion
        seq, ref_seq, ref, alt = get_mutation_seq(line[2].split(sep = ','), self.reverse_code, motif_length=motif)
        self.assertEqual(seq, 'CGAGCC', msg='Fail on sequence, case A > -')
        self.assertEqual(ref_seq, 'CGAAGCC', msg='Fail on reference sequence, case A > -')

        # multiple deletions even
        seq, ref_seq, ref, alt = get_mutation_seq(line[3].split(sep = ','), self.reverse_code, motif_length=motif)
        self.assertEqual(seq, 'AATCAT', msg='Fail on sequence, case of six bases deletion')
        self.assertEqual(ref_seq, 'AATCGATACCAT', msg='Fail on reference sequence, case of six bases deletion')

        # multiple deletions odd
        seq, ref_seq, ref, alt = get_mutation_seq(line[4].split(sep = ','), self.reverse_code, motif_length=motif)
        self.assertEqual(seq, 'ATCCAT', msg='Fail on sequence, case of five bases deletion')
        self.assertEqual(ref_seq, 'ATCGATGCCAT', msg='Fail on reference sequence, case of five bases deletion')

        # simple insertion
        seq, ref_seq, ref, alt = get_mutation_seq(line[5].split(sep = ','), self.reverse_code, motif_length=motif)
        self.assertEqual(seq, 'CGAAGCC', msg='Fail on sequence, case - > A')
        self.assertEqual(ref_seq, 'CGAGCC', msg='Fail on reference sequence, case - > A')

        # multiple insertions even
        seq, ref_seq, ref, alt = get_mutation_seq(line[6].split(sep = ','), self.reverse_code, motif_length=motif)
        self.assertEqual(seq, 'AATCGATACCAT', msg='Fail on sequence, case of six bases insertion')
        self.assertEqual(ref_seq, 'AATCAT', msg='Fail on reference sequence, case of six bases insertion')

        # multiple insertions odd
        seq, ref_seq, ref, alt = get_mutation_seq(line[7].split(sep = ','), self.reverse_code, motif_length=motif)
        self.assertEqual(seq, 'ATCGATGCCAT', msg='Fail on sequence, case of five bases deletion')
        self.assertEqual(ref_seq, 'ATCCAT', msg='Fail on reference sequence, case of five bases insertion')

    def test_11_MNV(self):
    # The pyrimidine based mutations only occur in the data 
    # line contains following information: 
    # CHROM, POS, REF, ALT, sample, seq, genic, exonic, strand, rt, sex, age, histology
        line = ['1,7,T,A,test-case,GAATCGA:@CCATA,0,0,=,0.0,test,0,test-case',
                '1,7,T,C,test-case,GAATCG@;%CCATA,0,0,=,0.0,test,0,test-case',
                '1,7,T,G,test-case,GAATCG5?GCCATA,0,0,=,0.0,test,0,test-case',
                '1,7,C,A,test-case,GAATCGA$6CCATA,0,0,=,0.0,test,0,test-case',
                '1,7,C,G,test-case,GAATCG5%6CCATA,0,0,=,0.0,test,0,test-case',
                '1,7,C,T,test-case,GAATCG5^@CCATA,0,0,=,0.0,test,0,test-case',
                '1,7,T,G,test-case,GAATCG1?GCCATA,0,0,=,0.0,test,0,test-case',
                '1,7,C,A,test-case,GAATCGA$2CCATA,0,0,=,0.0,test,0,test-case',
                '1,7,C,G,test-case,GAATCG1%2CCATA,0,0,=,0.0,test,0,test-case',
                '1,7,C,T,test-case,GAATCG1^@CCATA,0,0,=,0.0,test,0,test-case']

        seq, ref_seq, ref, alt = get_mutation_seq(line[0].split(sep = ','), self.reverse_code)
        self.assertEqual(seq, 'AAGC', msg='Fail on sequence, MNV')
        self.assertEqual(ref_seq, 'ATAC', msg='Fail on sequence, MNV')
        
        seq, ref_seq, ref, alt = get_mutation_seq(line[1].split(sep = ','), self.reverse_code)
        self.assertEqual(seq, 'GGCGC', msg='Fail on sequence, MNV')
        self.assertEqual(ref_seq, 'GATCC', msg='Fail on sequence, MNV')

        seq, ref_seq, ref, alt = get_mutation_seq(line[2].split(sep = ','), self.reverse_code)
        self.assertEqual(seq, 'GAGG', msg='Fail on sequence, MNV')
        self.assertEqual(ref_seq, 'GTG', msg='Fail on sequence, MNV')

        seq, ref_seq, ref, alt = get_mutation_seq(line[3].split(sep = ','), self.reverse_code)
        self.assertEqual(seq, 'AACC', msg='Fail on sequence, MNV')
        self.assertEqual(ref_seq, 'ACC', msg='Fail on sequence, MNV')

        seq, ref_seq, ref, alt = get_mutation_seq(line[4].split(sep = ','), self.reverse_code)
        self.assertEqual(seq, 'GAGCC', msg='Fail on sequence, MNV')
        self.assertEqual(ref_seq, 'GCC', msg='Fail on sequence, MNV')

        seq, ref_seq, ref, alt = get_mutation_seq(line[5].split(sep = ','), self.reverse_code)
        self.assertEqual(seq, 'GATGC', msg='Fail on sequence, MNV')
        self.assertEqual(ref_seq, 'GCAC', msg='Fail on sequence, MNV')

        seq, ref_seq, ref, alt = get_mutation_seq(line[6].split(sep = ','), self.reverse_code)
        self.assertEqual(seq, 'GGG', msg='Fail on sequence, MNV')
        self.assertEqual(ref_seq, 'GATG', msg='Fail on sequence, MNV')

        seq, ref_seq, ref, alt = get_mutation_seq(line[7].split(sep = ','), self.reverse_code)
        self.assertEqual(seq, 'AAC', msg='Fail on sequence, MNV')
        self.assertEqual(ref_seq, 'ACCC', msg='Fail on sequence, MNV')

        seq, ref_seq, ref, alt = get_mutation_seq(line[8].split(sep = ','), self.reverse_code)
        self.assertEqual(seq, 'GGC', msg='Fail on sequence, MNV')
        self.assertEqual(ref_seq, 'GACCC', msg='Fail on sequence, MNV')

        seq, ref_seq, ref, alt = get_mutation_seq(line[9].split(sep = ','), self.reverse_code)
        self.assertEqual(seq, 'GTGC', msg='Fail on sequence, MNV')
        self.assertEqual(ref_seq, 'GACAC', msg='Fail on sequence, MNV')

if __name__ =='__main__':
    unittest.main()